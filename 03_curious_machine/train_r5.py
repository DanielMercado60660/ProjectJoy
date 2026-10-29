"""R5 MLX PPO. Reuses the exhibit's networks/loss; keeps all old commands unchanged.
Only NumPy RNG is consumed after network initialisation; its full state is checkpointed.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from agents import MLP
from env import VecRooms, OBS_DIM, N_ACTIONS
from train import (Explorer, DEFAULTS, ppo_loss, gae, norm_fn, intrinsic_rewards,
                   to_np_tree, from_np_tree)
from r5_core import signed_progress
from progress_r5_t2 import ErrorQueue, RewardScale

METHODS = ("lp", "rnd_ep", "disagree", "ngu", "lpm")
CONFIG = dict(DEFAULTS, agent="lp", n_envs=32, rollout=128, total_steps=1048576,
              ckpt_every=8, lp_lag=4, lp_clip=1.0, max_minutes=25.0)


class ErrorProgress(nn.Module):
    """LPM log-error adaptation: g(o,a) - log(MSE(f(o,a),phi(o')) + eps).
    g fits a bounded queue of PRE-update errors. All target labels are detached.
    Fixed random features replace the paper's observation predictor; no immunity claim.
    """
    def __init__(self):
        super().__init__()
        self.phi = MLP([OBS_DIM, 128, 64]); self.phi.freeze()
        self.current = MLP([OBS_DIM + N_ACTIONS, 128, 128, 64])
        self.error_model = MLP([OBS_DIM + N_ACTIONS, 128, 128, 1])

    def inputs(self, x, a):
        assert x.ndim == 2 and a.shape == (x.shape[0],)
        onehot = (mx.arange(N_ACTIONS)[None] == a[:, None]).astype(mx.float32)
        return mx.concatenate([x, onehot], axis=1)

    def errors(self, x, a, xn):
        return mx.mean((self.current(self.inputs(x, a)) - mx.stop_gradient(self.phi(xn))) ** 2, axis=1)

    def expected(self, x, a):
        return self.error_model(self.inputs(x, a))[:, 0]

    def loss(self, x, a, xn, mask, qx, qa, qt):
        # Fixed multiplier puts the dynamics loss near the error regression's scale;
        # no gradients pass into the queued labels or the frozen random features.
        dynamics = 100.0 * mx.sum(self.errors(x, a, xn) * mask) / mx.maximum(mx.sum(mask), 1.)
        calibration = mx.mean((self.expected(qx, qa) - mx.stop_gradient(qt)) ** 2)
        return dynamics + calibration


class Progress(nn.Module):
    """Lagged-predictor progress, an LPM-inspired ablation, not the paper's error model.

    Both predictors score the SAME fresh target before any fitting to that rollout.
    Fixed raw input/target scales eliminate moving-normaliser pseudo-progress.
    """
    def __init__(self):
        super().__init__()
        self.phi = MLP([OBS_DIM, 128, 64]); self.phi.freeze()
        self.current = MLP([OBS_DIM + N_ACTIONS, 128, 128, 64])
        self.reference = MLP([OBS_DIM + N_ACTIONS, 128, 128, 64]); self.reference.freeze()
        self.snapshot()

    def snapshot(self):
        # Materialise independent arrays; no shared mutable references.
        self.reference.update(from_np_tree(to_np_tree(self.current.parameters())))

    def errors(self, x, a, xn, old=False):
        assert x.ndim == xn.ndim == 2 and x.shape == xn.shape and a.shape == (x.shape[0],)
        onehot = (mx.arange(N_ACTIONS)[None] == a[:, None]).astype(mx.float32)
        pred = (self.reference if old else self.current)(mx.concatenate([x, onehot], axis=1))
        target = mx.stop_gradient(self.phi(xn))
        return mx.mean((pred - target) ** 2, axis=1)

    def loss(self, x, a, xn, mask):
        return mx.sum(self.errors(x, a, xn) * mask) / mx.maximum(mx.sum(mask), 1.0)


class R5Explorer(Explorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg["agent"] == "lp":
            self.aux = Progress(); self.aux_opt = optim.Adam(learning_rate=cfg["lr"])
            mx.eval(self.aux.parameters())
        elif cfg["agent"] == "lpm":
            self.aux = ErrorProgress(); self.aux_opt = optim.Adam(learning_rate=cfg["lr"])
            self.error_queue = ErrorQueue(cfg["lpm_window"] * cfg["n_envs"] * cfg["rollout"], OBS_DIM)
            self.reward_scale = RewardScale()
            mx.eval(self.aux.parameters())

    def save(self, path, env, extra=None):
        extra = dict(extra or {})
        if self.cfg["agent"] == "lpm":
            extra.update(error_queue=self.error_queue.state(), reward_scale=self.reward_scale.__dict__)
        super().save(path, env, extra=extra)

    def act(self, obs, greedy=False):
        output = super().act(obs, greedy)
        check_finite(output, "policy action/log-probability/values")
        return output

    @staticmethod
    def load(path, env=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("extra", {}).get("round") != 5:
            raise ValueError("Expected an R5 checkpoint; legacy runs are not warm starts")
        ex = R5Explorer(d["cfg"])
        for name in ("policy", "aux"):
            getattr(ex, name).update(from_np_tree(d[name]))
        for name in ("opt", "aux_opt"):
            getattr(ex, name).state = from_np_tree(d[name])
        for name in ("obs_rms", "ret_rms"):
            getattr(ex, name).load(d[name])
        if ex.cfg["agent"] == "ngu":
            ex.knn.load(d["knn"]); ex.err_rms.load(d["err_rms"])
        if ex.cfg["agent"] == "rnd_ep":
            ex.ep_counts = d["ep_counts"]; ex.err_rms.load(d["err_rms"])
        if ex.cfg["agent"] == "lpm":
            ex.error_queue.load(d["extra"]["error_queue"])
            ex.reward_scale.__dict__.update(d["extra"]["reward_scale"])
        ex.rewems = d["rewems"]
        ex.rng.bit_generator.state = d["rng"]
        ex.update, ex.steps = d["update"], d["steps"]
        if env is not None:
            env.set_state(d["env"])
        return ex, d


def safe_clip(grads, max_norm=0.5):
    from mlx.utils import tree_flatten, tree_map
    # Epsilon INSIDE sqrt: exact zero gradients must have a finite derivative.
    norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)) + 1e-12)
    scale = mx.minimum(1.0, max_norm / norm)
    return tree_map(lambda g: g * scale, grads)


def check_finite(values, where):
    for value in values:
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError(f"Nonfinite {where}")


def train_r5(cfg, checkpoint, log_dir, resume=False, stop_after_updates=None, barrier_update=None):
    cfg = dict(cfg)
    assert cfg["agent"] in METHODS
    N, T = cfg["n_envs"], cfg["rollout"]; B = N * T
    if cfg["total_steps"] % B or B % cfg["minibatches"] or cfg["lp_lag"] < 1:
        raise ValueError("Use exact rollout/minibatch multiples and positive lp_lag")
    if cfg['agent']=='lpm' and cfg.get('lpm_window',0)<1:
        raise ValueError('lpm_window must be positive')
    if not 0 < cfg["max_minutes"] <= 45:
        raise ValueError("--max-minutes must be in (0,45]")
    started = time.monotonic(); deadline = started + cfg["max_minutes"] * 60
    checkpoint = str(checkpoint); log_dir = Path(log_dir)
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (Path(checkpoint).exists() or (log_dir / "updates.jsonl").exists()):
        raise FileExistsError("Refusing to overwrite a run; resume it or use a fresh tag")
    env = VecRooms(N, cfg["max_steps"], cfg["noisy_tv"], seed=cfg["seed"])
    if resume:
        ex, _ = R5Explorer.load(checkpoint, env)
        mutable = {"total_steps", "max_minutes", "ckpt_every"}
        for k, v in cfg.items():
            if k not in mutable and ex.cfg.get(k) != v:
                raise ValueError(f"Resume configuration mismatch: {k}")
        ex.cfg.update(cfg)
        print(f"RESUME update={ex.update} steps={ex.steps}", flush=True)
    else:
        ex = R5Explorer(cfg)
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    obs = env.obs.copy(); initial_steps = ex.steps
    ppo_grad = nn.value_and_grad(ex.policy, ppo_loss)
    if cfg["agent"] == "rnd_ep":
        aux_grad = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mask: m.loss(xn, mask))
    elif cfg["agent"] == "ngu":
        aux_grad = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mask: m.loss(x, xn, a, mask[0], mask[1]))
    elif cfg["agent"] == "lpm":
        aux_grad = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mask, qx, qa, qt: m.loss(x, a, xn, mask, qx, qa, qt))
    else:
        aux_grad = nn.value_and_grad(ex.aux, lambda m, x, a, xn, mask: m.loss(x, a, xn, mask))
    durations = []; recent = []; status = "complete"
    def save():
        for model in (ex.policy, ex.aux):
            check_finite(list(to_np_tree(model.parameters()).values()), "checkpoint weights")
        ex.save(checkpoint, env, extra={"round": 5, "mlx_random": "seed-only initialisation; no draws after init"})
    while ex.steps < cfg["total_steps"]:
        # Reserve time to finish the next atomic PPO update and save. The suite adds
        # a process-level deadline as a second guard against a hung backend.
        reserve = max(3.0, max(durations[-3:], default=0.0) * 1.5)
        if time.monotonic() + reserve >= deadline:
            status = "partial_wall_guard"; break
        u0 = time.monotonic()
        O = np.zeros((T, N, OBS_DIM), np.float32); NX = np.zeros_like(O)
        A = np.zeros((T, N), np.int32); LP = np.zeros((T, N), np.float32)
        VI = np.zeros_like(LP); VE = np.zeros_like(LP); D = np.zeros_like(LP); EP = np.zeros_like(LP)
        norm_roll = norm_fn(ex.obs_rms)
        for t in range(T):
            a, lp, vi, ve = ex.act(obs)
            nxt, done, episodes = env.step(a)
            O[t], NX[t], A[t], LP[t], VI[t], VE[t], D[t] = obs, nxt, a, lp, vi, ve, done
            recent = (recent + episodes)[-2 * N:]
            if cfg["agent"] == "rnd_ep":
                for i, h in enumerate(ex.hasher.hash64(nxt)):
                    if done[i]:
                        ex.ep_counts[i] = {}
                    c = ex.ep_counts[i].get(int(h), 0) + 1
                    ex.ep_counts[i][int(h)] = c; EP[t, i] = 1 / np.sqrt(c)
            elif cfg["agent"] == "ngu":
                embedding = ex.aux.embed(norm_roll(mx.array(nxt))); mx.eval(embedding)
                EP[t] = ex.knn.reward_and_add(np.array(embedding), done)
            obs = nxt
        X, XN, AG = mx.array(O.reshape(B, OBS_DIM)), mx.array(NX.reshape(B, OBS_DIM)), mx.array(A.ravel())
        valid = 1 - D.ravel()
        stats = {}
        if cfg["agent"] == "lpm":
            XO, XNN = X, XN
            log_error = mx.log(ex.aux.errors(X, AG, XN) + 1e-6)
            expected = ex.aux.expected(X, AG)
            mx.eval(log_error, expected)
            le, ge = np.array(log_error), np.array(expected)
            ready = ex.update >= cfg["lpm_window"] + 2
            delta = np.clip(ge - le, -5., 5.)
            reward = (delta * valid * ready).reshape(T, N)
            # Queue labels precede all fitting to this rollout and include only physical transitions.
            keep = valid > 0
            ex.error_queue.add(O.reshape(B, OBS_DIM)[keep], A.ravel()[keep], le[keep])
            stats.update(log_error=float(le[keep].mean()), expected_log_error=float(ge[keep].mean()),
                         calibration_rmse=float(np.sqrt(np.mean((ge[keep] - le[keep]) ** 2) + 1e-12)),
                         progress_positive=float(np.mean(reward > 0)), progress_negative=float(np.mean(reward < 0)),
                         progress_ready=ready, queue_count=ex.error_queue.count)
        elif cfg["agent"] == "lp":
            XO, XNN = X, XN
            old = ex.aux.errors(X, AG, XN, old=True); new = ex.aux.errors(X, AG, XN)
            mx.eval(old, new)
            reward = signed_progress(np.array(old), np.array(new), valid, cfg["lp_clip"]).reshape(T, N)
            stats.update(progress_old=float(mx.mean(old)), progress_new=float(mx.mean(new)),
                         progress_positive=float(np.mean(reward > 0)), progress_negative=float(np.mean(reward < 0)))
            # Snapshot AFTER scoring, BEFORE fitting this rollout.
            if ex.update % cfg["lp_lag"] == 0:
                ex.aux.snapshot()
        else:
            norm = norm_fn(ex.obs_rms); XO, XNN = norm(X), norm(XN)
            reward = intrinsic_rewards(ex, XO, AG, XNN).reshape(T, N)
            if cfg["agent"] in ("rnd_ep", "ngu"):
                ex.err_rms.update(reward.ravel())
                alpha = 1 + (reward - ex.err_rms.mean) / np.sqrt(ex.err_rms.var + 1e-12)
                reward = EP * np.clip(alpha, 1, 5)
            # Auto-reset observations are not physical transitions. Apply equally.
            reward *= valid.reshape(T, N)
        check_finite([reward], "intrinsic reward")
        returns = np.zeros_like(reward)
        for t in range(T):
            ex.rewems = ex.rewems * cfg["gamma_int"] + reward[t]; returns[t] = ex.rewems
        ex.ret_rms.update(returns.ravel())
        if cfg["agent"] == "lpm":
            normalized, scale = ex.reward_scale.normalize(reward, valid.reshape(T, N))
            stats["reward_scale"] = scale
        else:
            normalized = reward / np.sqrt(ex.ret_rms.var + 1e-8)
            stats["reward_scale"] = float(np.sqrt(ex.ret_rms.var + 1e-8))
        stats["normalized_reward_std"] = float(normalized.std())
        stats["sampled_entropy"] = float(-LP.mean())
        from env import VIEW, CELL_FEAT, TV
        tv_mask = (
            NX.reshape(B, OBS_DIM)[:, :-5].reshape(B, VIEW * VIEW, CELL_FEAT)[:, :, TV] > .5).any(1)
        for label, subset in (("tv", tv_mask), ("non_tv", ~tv_mask)):
            subset = subset & (valid > 0)
            stats[f"reward_{label}_mean"] = float(reward.ravel()[subset].mean()) if subset.any() else None
        # Value-only forward avoids consuming a spurious action RNG draw.
        _, lv, le = ex.policy(mx.array(obs)); mx.eval(lv, le)
        ai, ri = gae(normalized, VI, np.array(lv), None, cfg["gamma_int"], cfg["lam"])
        _, re = gae(np.zeros_like(D), VE, np.array(le), D, cfg["gamma_ext"], cfg["lam"])
        adv = ai.ravel().astype(np.float32)
        if cfg["agent"] not in ("lp", "lpm"):
            bm, bv = mx.mean(XN, axis=0), mx.var(XN, axis=0); mx.eval(bm, bv)
            ex.obs_rms.update_moments(np.array(bm), np.array(bv), B)
        losses = []; mb = B // cfg["minibatches"]
        for _ in range(cfg["epochs"]):
            perm = ex.rng.permutation(B)
            for k in range(cfg["minibatches"]):
                idx = perm[k * mb:(k + 1) * mb]; ig = mx.array(idx)
                ad = adv[idx]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                (loss, parts), grad = ppo_grad(ex.policy, X[ig], AG[ig], mx.array(LP.ravel()[idx]), mx.array(ad),
                                          mx.array(ri.ravel()[idx].astype(np.float32)),
                                          mx.array(re.ravel()[idx].astype(np.float32)), cfg["clip"], cfg["ent_coef"])
                mask = (ex.rng.random(mb) < cfg["aux_frac"]).astype(np.float32) * valid[idx]
                if cfg["agent"] == "ngu":
                    mask = (mx.array(mask), mx.array(valid[idx]))
                elif cfg["agent"] == "disagree":
                    mask = mx.array((ex.rng.random((cfg["ens_k"], mb)) < 0.5).astype(np.float32) * valid[idx][None])
                else:
                    mask = mx.array(mask)
                if cfg["agent"] == "lpm":
                    qx, qa, qt = ex.error_queue.sample(ex.rng, mb)
                    al, ag = aux_grad(ex.aux, XO[ig], AG[ig], XNN[ig], mask, mx.array(qx), mx.array(qa), mx.array(qt))
                else:
                    al, ag = aux_grad(ex.aux, XO[ig], AG[ig], XNN[ig], mask)
                mx.eval(loss, al, parts); check_finite([loss, al, parts[2]], "loss/entropy before update")
                ex.opt.update(ex.policy, safe_clip(grad)); ex.aux_opt.update(ex.aux, safe_clip(ag, 5.0))
                mx.eval(ex.policy.parameters(), ex.opt.state, ex.aux.parameters(), ex.aux_opt.state)
                losses.append((float(loss), float(al), float(parts[2])))
        ex.update += 1; ex.steps += B
        rec = dict(update=ex.update, steps=ex.steps, elapsed_seconds=time.monotonic() - started,
                   ppo_loss=float(np.mean(losses, axis=0)[0]), aux_loss=float(np.mean(losses, axis=0)[1]),
                   policy_entropy=float(np.mean(losses, axis=0)[2]),
                   reward_mean=float(reward.mean()), reward_std=float(reward.std()), **stats)
        if recent:
            for field in ("rooms", "interactions", "coverage", "tv_steps"):
                rec[field] = float(np.mean([e[field] for e in recent]))
        with (log_dir / "updates.jsonl").open("a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        if ex.update == 1 or ex.update % 8 == 0:
            print(json.dumps(rec), flush=True)
        if ex.update % cfg["ckpt_every"] == 0 or ex.steps == cfg["total_steps"] or ex.update == barrier_update:
            save()
        if ex.update == barrier_update:  # subprocess kill/resume smoke only
            Path(checkpoint + ".ready").write_text("saved")
            while time.monotonic() < deadline:
                time.sleep(0.05)
            status = "partial_barrier"; break
        durations.append(time.monotonic() - u0)
        if stop_after_updates is not None and ex.update >= stop_after_updates:
            status = "partial_test_stop"; break
    save()
    outcome = dict(status=status, steps=ex.steps, target_steps=cfg["total_steps"], update=ex.update,
                   invocation_seconds=time.monotonic() - started, new_steps=ex.steps - initial_steps)
    (log_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    print(json.dumps(outcome), flush=True)
    return ex, env, outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, default="lp"); p.add_argument("--seed", type=int, default=510)
    p.add_argument("--world", choices=("plain", "tv"), default="tv")
    p.add_argument("--steps", type=int, default=1048576); p.add_argument("--max-minutes", type=float, default=25)
    p.add_argument("--checkpoint", required=True); p.add_argument("--log-dir", required=True)
    p.add_argument("--resume", action="store_true"); p.add_argument("--lp-lag", type=int, default=4)
    p.add_argument("--lpm-window", type=int, default=1); p.add_argument("--ent-coef", type=float, default=.001)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--tiny", action="store_true"); p.add_argument("--barrier-update", type=int)
    a = p.parse_args()
    cfg = dict(CONFIG, agent=a.method, seed=a.seed, noisy_tv=a.world == "tv", total_steps=a.steps,
               max_minutes=a.max_minutes, lp_lag=a.lp_lag)
    if a.method == "lpm":
        cfg["lpm_window"] = a.lpm_window
    cfg.update(ent_coef=a.ent_coef, n_envs=a.n_envs)
    if a.tiny:
        cfg.update(n_envs=4, rollout=8, hidden=32, epochs=1, minibatches=2, max_steps=32, ckpt_every=1)
    train_r5(cfg, a.checkpoint, a.log_dir, a.resume, barrier_update=a.barrier_update)


if __name__ == "__main__":
    main()
