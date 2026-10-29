"""Timed or matched-update PPO, full validation, rollout-boundary resume.

Default runs fill their wall allowance. Paired experiments can set --updates
and --validate-updates to match training and selection work, with a wall guard.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import signal
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from env_r5 import CaveEnv, make_pool, assert_disjoint, manifest_digest, OBS_DIM
from eval_r5 import suite
from io_r5 import Deadline, save_checkpoint, load_checkpoint, write_json
from policy_r5 import Policy, LearnedAgent, as_mlx, restore_mlx_rng
from math_r5 import gae, validation_due


def log_softmax(x):
    return x-mx.logsumexp(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=24)
    ap.add_argument("--out", default="checkpoints/r5_main.npz")
    ap.add_argument("--work", default="logs/r5/main")
    ap.add_argument("--resume")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=.015)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--train-per-group", type=int, default=256)
    ap.add_argument("--val-per-group", type=int, default=32)
    ap.add_argument("--validate-seconds", type=float, default=120)
    ap.add_argument("--validate-updates", type=int, default=0, help="0 uses timed validation; positive interval saved in config")
    ap.add_argument("--checkpoint-seconds", type=float, default=60)
    ap.add_argument("--updates", type=int, default=0, help="additional updates, 0 fills allowance")
    ap.add_argument("--save-every-update", action="store_true", help="smoke kill/resume probe")
    a = ap.parse_args()
    clock = Deadline(a.max_minutes)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    latest = work/"latest.npz"
    if not a.resume and (latest.exists() or Path(a.out).exists()):
        raise FileExistsError("use a new work/output path or --resume; do not overwrite an earlier run")
    config = {k: getattr(a, k) for k in ("envs", "horizon", "hidden", "epochs", "minibatches", "lr",
                                      "entropy", "seed", "train_per_group", "val_per_group", "validate_updates")}
    saved = load_checkpoint(a.resume) if a.resume else None
    if saved:
        config = saved["config"]
    if a.updates < 0 or config.get("validate_updates", 0) < 0:
        raise ValueError("update counts must be nonnegative")
    for key in ("envs", "horizon", "hidden", "epochs", "minibatches", "train_per_group", "val_per_group"):
        if config[key] <= 0:
            raise ValueError(key)
    if config["envs"] % config["minibatches"]:
        raise ValueError("envs must be divisible by minibatches")
    mx.random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    train = make_pool("train", config["train_per_group"])
    val = make_pool("val", config["val_per_group"])
    assert_disjoint(train, val)
    write_json(work/"manifest.json", dict(train=train.manifest(), validation=val.manifest(), config=config))
    n, T = config["envs"], config["horizon"]
    env = CaveEnv(train, n, seed=config["seed"])
    model = Policy(config["hidden"])
    optimizer = optim.Adam(learning_rate=config["lr"])
    optimizer.init(model.trainable_parameters())
    obs = env.reset()
    h = np.zeros((n, model.hidden), np.float32)
    starts = np.ones(n, np.float32)
    update = 0; best = -1e30; previous_seconds = 0.; last_loss = None
    if saved:
        if saved["train_digest"] != manifest_digest(train) or saved["val_digest"] != manifest_digest(val):
            raise ValueError("checkpoint/data mismatch")
        model.update(as_mlx(saved["weights"]))
        optimizer.state = as_mlx(saved["optimizer"])
        optimizer.init(model.trainable_parameters())
        env.load_state_dict(saved["environment"])
        rng.bit_generator.state = saved["numpy_rng"]
        restore_mlx_rng(saved["mlx_rng"])
        obs, h, starts = saved["obs"], saved["hidden_state"], saved["starts"]
        update, best, previous_seconds = saved["update"], saved["best_score"], saved["total_seconds"]
        if not Path(a.out).exists():
            # A new destination cannot preserve an unavailable old champion.
            # Normal resume uses the original --out and preserves selection.
            best = -1e30
        else:
            selected = load_checkpoint(a.out)
            if selected["config"] != config or selected["val_digest"] != manifest_digest(val):
                raise ValueError("selected checkpoint belongs to a different experiment")
            # A crash may occur after selection but before latest is refreshed.
            # Never let a stale latest.best_score replace a better champion.
            best = max(best, selected["best_score"])
        last_loss = saved["last_loss"]
        print(json.dumps(dict(resumed_update=update, optimizer_step=int(optimizer.step.item()),
                              previous_loss=last_loss)), flush=True)
    mx.eval(model.parameters(), optimizer.state)

    def loss_fn(model, observations, h0, reset, move, pulse, old_logp, advantage, returns):
        lm, lp, value = model.sequence(observations, h0, reset)
        lm, lp = log_softmax(lm), log_softmax(lp)
        logp = mx.take_along_axis(lm, move[..., None], -1)[..., 0]
        logp += mx.take_along_axis(lp, pulse[..., None], -1)[..., 0]
        ratio = mx.exp(mx.clip(logp-old_logp, -20., 20.))
        pg = -mx.minimum(ratio*advantage, mx.clip(ratio, .8, 1.2)*advantage).mean()
        vl = ((value-returns)**2).mean()
        entropy = -(mx.exp(lm)*lm).sum(-1).mean()-(mx.exp(lp)*lp).sum(-1).mean()
        return pg + .25*vl - config["entropy"]*entropy, (pg, vl, entropy)

    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def optimize(*args):
        (loss, aux), grads = vg(model, *args)
        # Guard zero BEFORE sqrt. No norm derivative through a bare sqrt(0).
        squared = sum(mx.sum(g*g) for _, g in tree_flatten(grads))
        norm = mx.sqrt(mx.maximum(squared, 1e-12))
        scale = mx.minimum(1., .5/norm)
        grads = tree_map(lambda g: g*scale, grads)
        optimizer.update(model, grads)
        return loss, aux[0], aux[1], aux[2], norm

    @partial(mx.compile, inputs=[model.state])
    def act(observations, hidden):
        return model.step(observations, hidden)

    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def payload():
        mx.eval(model.parameters(), optimizer.state, mx.random.state)
        return dict(format="bat-r5-v1", config=config, weights=model.parameters(), optimizer=optimizer.state,
                    update=update, env_steps=update*n*T, best_score=best, last_loss=last_loss,
                    total_seconds=previous_seconds+clock.elapsed, numpy_rng=rng.bit_generator.state,
                    mlx_rng=[np.asarray(key) for key in mx.random.state], environment=env.state_dict(),
                    obs=obs, hidden_state=h, starts=starts, train_digest=manifest_digest(train),
                    val_digest=manifest_digest(val))

    log = (work/"train.jsonl").open("a")
    initial_update = update
    update_times = []
    validation_seconds = 0.
    next_validation = 0.
    last_validation_update = -1
    last_save = clock.elapsed
    episode_stats = []
    reason = "wall-clock guard"
    while True:
        if stop_requested:
            reason = "signal at update boundary"; break
        if a.updates and update-initial_update >= a.updates:
            reason = "update target"; break
        expected = max(update_times[-10:], default=2.)
        if clock.remaining < expected*1.3 + 2:
            break
        begun = time.monotonic()
        bo = np.zeros((T, n, OBS_DIM), np.float32)
        bs = np.zeros((T, n), np.float32)
        bm = np.zeros((T, n), np.int32); bp = bm.copy()
        bl = bs.copy(); bv = bs.copy(); br = bs.copy(); bd = bs.copy()
        h0 = h.copy()
        hidden = mx.array(h)
        # Once begun, finish this bounded update before saving signal state.
        for t in range(T):
            bo[t], bs[t] = obs, starts
            lm, lp, value, hidden = act(mx.array(obs), hidden*mx.array(1-starts)[:, None])
            pm, pp = np.asarray(mx.softmax(lm, -1)), np.asarray(mx.softmax(lp, -1))
            if not (np.isfinite(pm).all() and np.isfinite(pp).all()):
                raise FloatingPointError("nonfinite action probabilities")
            move = np.minimum((rng.random((n, 1)) > np.cumsum(pm, -1)).sum(-1), 4)
            pulse = (rng.random(n) < pp[:, 1]).astype(int)
            bl[t] = np.log(np.maximum(pm[np.arange(n), move], 1e-12)) + np.log(np.maximum(pp[np.arange(n), pulse], 1e-12))
            bm[t], bp[t], bv[t] = move, pulse, np.asarray(value)
            obs, reward, done, info = env.step(move, pulse)
            br[t], bd[t] = reward, done
            starts = done.astype(np.float32)
            for i in np.flatnonzero(done):
                episode_stats.append([info["success"][i], info["pulses"][i], info["collisions"][i]])
        h = np.asarray(hidden).copy()
        _, _, final_value, _ = act(mx.array(obs), hidden*mx.array(1-starts)[:, None])
        advantage, returns = gae(br, bv, bd, np.asarray(final_value))
        advantage = (advantage-advantage.mean()) / np.sqrt(max(float(advantage.var()), 1e-8))
        stats = []
        for _ in range(config["epochs"]):
            for chunk in np.array_split(rng.permutation(n), config["minibatches"]):
                values = optimize(mx.array(bo[:, chunk]), mx.array(h0[chunk]), mx.array(bs[:, chunk]),
                                  mx.array(bm[:, chunk]), mx.array(bp[:, chunk]), mx.array(bl[:, chunk]),
                                  mx.array(advantage[:, chunk]), mx.array(returns[:, chunk]))
                mx.eval(values, model.parameters(), optimizer.state)
                stats.append([float(x.item()) for x in values])
        if not np.isfinite(stats).all():
            raise FloatingPointError("nonfinite training loss/gradient; previous generation preserved")
        update += 1
        last_loss = float(np.mean(np.asarray(stats)[:, 0]))
        update_times.append(time.monotonic()-begun)
        record = dict(update=update, env_steps=update*n*T, seconds=clock.elapsed, loss=last_loss,
                      entropy=float(np.mean(np.asarray(stats)[:, 3])), grad_norm=float(np.mean(np.asarray(stats)[:, 4])),
                      policy_loss=float(np.mean(np.asarray(stats)[:, 1])),
                      value_loss=float(np.mean(np.asarray(stats)[:, 2])), reward_mean=float(br.mean()),
                      action_fractions=(np.bincount(bm.ravel(), minlength=5)/bm.size).tolist(),
                      pulse_request_fraction=float(bp.mean()), completed_episodes=int(bd.sum()),
                      optimizer_step=int(optimizer.step.item()))
        if episode_stats:
            record["completion_window_n"] = min(400, len(episode_stats))
            record["train_completion_last400"] = float(np.asarray(episode_stats[-400:])[:, 0].mean())
            episode_stats = episode_stats[-400:]
        if len(update_times) >= 4:
            rate = 1/np.mean(update_times[-20:])
            record.update(measured_updates_per_second=float(rate), measured_env_steps_per_second=float(rate*n*T),
                          projected_additional_updates=int(max(0, clock.remaining)*rate),
                          budget_mode="matched updates with wall guard" if a.updates else "fill wall allowance")
        log.write(json.dumps(record, allow_nan=False)+"\n"); log.flush()
        if update % 10 == 0 or a.updates or a.save_every_update:
            print(json.dumps(record), flush=True)
        # Save before validation too: kill/resume smoke never waits on selection.
        if a.save_every_update or clock.elapsed-last_save >= a.checkpoint_seconds:
            save_checkpoint(latest, payload()); last_save = clock.elapsed
        can_validate = clock.remaining > max(validation_seconds*1.3+2, 4)
        if validation_due(update, clock.elapsed, next_validation, config.get("validate_updates", 0)) and can_validate and not stop_requested:
            try:
                result = suite(val, lambda count: LearnedAgent(model, count), deadline=clock)
            except TimeoutError:
                break
            validation_seconds = result["seconds"]
            last_validation_update = update
            validation = dict(update=update, score=result["score"], seconds=result["seconds"], rows=result["rows"])
            with (work/"validation.jsonl").open("a") as f:
                f.write(json.dumps(validation)+"\n")
            if result["score"] > best:
                best = result["score"]
                selected = payload(); selected["validation"] = validation
                save_checkpoint(a.out, selected)
            print(json.dumps(dict(validation_update=update, score=result["score"], best_score=best,
                                  rows=len(result["rows"]), complete=True)), flush=True)
            next_validation = clock.elapsed+a.validate_seconds
    # Last selection uses the same full suite, never a cheaper/easier subset.
    if update != last_validation_update and not stop_requested and clock.remaining > max(1.3*validation_seconds+2, 4):
        try:
            result = suite(val, lambda count: LearnedAgent(model, count), deadline=clock)
            last_validation_update = update
            validation = dict(update=update, score=result["score"], seconds=result["seconds"], rows=result["rows"])
            with (work/"validation.jsonl").open("a") as f:
                f.write(json.dumps(validation)+"\n")
            if result["score"] > best:
                best = result["score"]
                selected = payload(); selected["validation"] = validation
                save_checkpoint(a.out, selected)
        except TimeoutError:
            pass
    save_checkpoint(latest, payload())
    summary = dict(update=update, invocation_updates=update-initial_update, seconds=clock.elapsed,
                   total_seconds=previous_seconds+clock.elapsed, best_score=best, stop=reason,
                   selected=str(a.out) if Path(a.out).exists() else None, latest=str(latest),
                   utilization=clock.elapsed/(a.max_minutes*60),
                   target_update=initial_update+a.updates if a.updates else None,
                   target_reached=update-initial_update == a.updates if a.updates else None,
                   last_validation_update=last_validation_update,
                   measured_update_seconds=float(np.mean(update_times[-20:])) if update_times else None)
    write_json(work/"summary.json", summary)
    log.close()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
