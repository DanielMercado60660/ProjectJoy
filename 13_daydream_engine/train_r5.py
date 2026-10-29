"""Train/resume bootstrapped hybrid ensembles. No test split is opened here."""
import argparse
import signal
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from r5_core import Budget, BudgetExpired, SCALE, TRAIN_DEFAULTS, digest, sample_windows, write_json, learning_rate_at
from r5_model import HybridEnsemble, Predictor, save_checkpoint, load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=20000, help="absolute total step target")
    ap.add_argument("--max-minutes", type=float, required=True)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128, help="per ensemble member")
    ap.add_argument("--unroll", type=int, default=3)
    ap.add_argument("--seed", type=int, default=51010)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--features", choices=("raw", "geometry"), default="raw")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--collision-weight", type=float, default=1.)
    ap.add_argument("--contact-fraction", type=float, default=.5)
    ap.add_argument("--decay-steps", type=int, default=0)
    ap.add_argument("--min-lr-fraction", type=float, default=.1)
    ap.add_argument("--history-every", type=int, default=1, help="loss sample interval; optimizer still updates every step")
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--stop-after", type=int, default=0, help="absolute step cap for resume test")
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    if min(a.members, a.hidden, a.depth, a.batch, a.unroll, a.steps, a.checkpoint_every, a.history_every) < 1:
        ap.error("counts must be positive")
    if not 0 <= a.contact_fraction <= 1 or a.collision_weight < 1 or a.decay_steps < 0 or not 0 < a.min_lr_fraction <= 1:
        ap.error("invalid contact fraction, collision weight or LR schedule")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume:
        ap.error("output already exists; use --resume or a fresh r5_* path")
    data = Path(a.data)
    tr = np.load(data / "train.npz")
    val = np.load(data / "val.npz")
    states, flags = tr["states"], tr["flags"]
    c = {k: getattr(a, k) for k in ("members", "hidden", "batch", "unroll", "seed", "lr")}
    c.update({k: getattr(a, k) for k in TRAIN_DEFAULTS})
    c["train_sha256"], c["val_sha256"] = digest(data / "train.npz"), digest(data / "val.npz")
    optimizer = optim.Adam(learning_rate=a.lr)
    rngs = [np.random.default_rng(a.seed + 10000 + e) for e in range(a.members)]
    pools = [r.integers(0, len(states), len(states)) for r in rngs]
    step, history = 0, []
    if a.resume:
        model, meta = load_checkpoint(out, optimizer)
        if {**TRAIN_DEFAULTS, **meta["config"]} != c:
            raise ValueError("resume config/data mismatch; pass the original model, sampler and LR flags")
        step, history = meta["step"], meta["history"]
        for r, state in zip(rngs, meta["numpy_rng_states"]):
            r.bit_generator.state = state
        pools = [np.array(p) for p in meta["bootstrap_scenes"]]
        print(f"RESUMED step={step} optimizer_step={int(optimizer.state['step'])}", flush=True)
    else:
        model = HybridEnsemble(a.members, a.hidden, a.seed, a.features, a.depth)
        optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    # Cache collision-window indices per member; no full-dataset scan in the loop.
    eligible = []
    room = states.shape[1] - a.unroll
    if room < 1:
        raise ValueError("training unroll longer than dataset")
    for pool in pools:
        hits = np.zeros((len(pool), room), bool)
        for t in range(a.unroll):
            hits |= flags[pool, t:t + room, 1]
        eligible.append(np.nonzero(hits))

    def get_batch():
        windows, events = [], []
        for r, pool, (ci, ct) in zip(rngs, pools, eligible):
            ids, times = r.choice(pool, a.batch), r.integers(0, room, a.batch)
            if len(ci):
                ncontact = int(a.batch * a.contact_fraction)
                chosen = r.integers(0, len(ci), ncontact)
                ids[:ncontact], times[:ncontact] = pool[ci[chosen]], ct[chosen]
            windows.append(states[ids[:, None], times[:, None] + np.arange(a.unroll + 1)])
            events.append(flags[ids[:, None], times[:, None] + np.arange(a.unroll), 1])
        return mx.array(np.stack(windows)), mx.array(np.stack(events))

    scale = mx.array(SCALE)

    def loss_fn(net, windows, events):
        s, loss = windows[:, :, 0], mx.array(0.0)
        for t in range(a.unroll):
            s = net(s)
            error = mx.mean(((s - windows[:, :, t + 1]) / scale) ** 2, axis=-1)
            weight = 1 + (a.collision_weight - 1) * events[:, :, t]
            loss = loss + mx.sum(weight * error) / mx.sum(weight)
        return loss / a.unroll

    vg = nn.value_and_grad(model, loss_fn)

    def update(windows, events):
        loss, grads = vg(model, windows, events)
        grads, _ = optim.clip_grad_norm(grads, 5.0)
        optimizer.update(model, grads)
        return loss

    update = mx.compile(update, inputs=[model.state, optimizer.state],
                        outputs=[model.state, optimizer.state])
    # Fixed validation windows, available equally to the classical fit.
    monitor, monitor_events = sample_windows(val["states"], val["flags"], min(128, len(val["states"])),
                                             a.unroll, np.random.default_rng(51012), return_flags=True)

    def validate():
        # Eager forward, deliberately independent from compiled update.
        win = mx.array(np.broadcast_to(monitor, (a.members,) + monitor.shape).copy())
        events = mx.array(np.broadcast_to(monitor_events, (a.members,) + monitor_events.shape).copy())
        value = loss_fn(model, win, events)
        mx.eval(value)
        return float(value)

    initial_val = validate()
    print(f"INITIAL step={step} val_balanced_loss={initial_val:.8g}", flush=True)

    last_loss = meta.get("last_loss") if a.resume else None

    def checkpoint():
        meta = dict(config=c, step=step, history=history, numpy_rng_states=[r.bit_generator.state for r in rngs],
                    bootstrap_scenes=[p.tolist() for p in pools],
                    rng_note="MLX used only for initialization with seed+member; all training draws use saved NumPy states",
                    invocation_seconds=budget.elapsed, validation_balanced_loss=validate(),
                    last_loss=last_loss, history_every=a.history_every)
        save_checkpoint(out, model, optimizer, meta)

    checkpoint()
    reason = "steps"
    try:
        while step < a.steps:
            budget.check()
            if stop[0] or (a.stop_after and step >= a.stop_after):
                reason = "signal" if stop[0] else "stop_after"
                break
            current_lr = learning_rate_at(step, a.lr, a.decay_steps, a.min_lr_fraction)
            optimizer.learning_rate = current_lr
            value = update(*get_batch())
            mx.eval(value, model.parameters(), optimizer.state)
            loss = float(value)
            if not np.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step + 1}; last committed checkpoint retained")
            step += 1
            last_loss = loss
            if step == 1 or step % a.history_every == 0:
                history.append(dict(step=step, loss=loss, lr=current_lr))
            if step == 1 or step % a.checkpoint_every == 0:
                print(f"TRAIN step={step} loss={loss:.8g} lr={current_lr:.7g} seconds={budget.elapsed:.1f}", flush=True)
            if step % a.checkpoint_every == 0:
                checkpoint()
    except BudgetExpired:
        reason = "wall_guard"
    checkpoint()
    write_json(out / "run_status.json", dict(step=step, reason=reason, seconds=budget.elapsed,
                                             requested_steps=a.steps, initial_val=initial_val))
    print(f"TRAIN_DONE step={step} reason={reason} seconds={budget.elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
