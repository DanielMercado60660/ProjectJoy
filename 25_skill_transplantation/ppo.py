"""PPO (MLX updates, numpy CPU rollouts) for task-conditioned HalfCheetah policies.

One function, `train`, used for every phase: shared base (multitask), specialist branches
(single task), multitask fine-tuning of merged policies, and transfer runs. Envs are a plain
list of gymnasium HalfCheetah-v5 instances; env i optimises task `env_tasks[i]`.

Old log-probs are recomputed with the MLX network before each update, so the PPO ratio starts
at exactly 1 on the update device (CPU-fp32 rollout vs GPU-TF32 update lesson). The action mean
is tanh-bounded. Approx-KL and clip fraction are logged.

Checkpoints (`<ckpt>.npz` + `<ckpt>_state.pkl`) hold weights, Adam state, step count, RNG state,
normalizer and config; `train(..., resume=True)` continues from them.
"""
import json
import os
import pickle
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from common import (ACT_DIM, HID, IN_DIM, LOGSTD_MAX, LOGSTD_MIN, PARAM_KEYS, TASKS, Normalizer,
                    make_env, np_mean, np_value, policy_input, save_params, task_reward)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1, self.a2, self.a3 = nn.Linear(IN_DIM, HID), nn.Linear(HID, HID), nn.Linear(HID, ACT_DIM)
        self.c1, self.c2, self.c3 = nn.Linear(IN_DIM, HID), nn.Linear(HID, HID), nn.Linear(HID, 1)
        self.log_std = mx.zeros((ACT_DIM,))

    def mean(self, x):
        return mx.tanh(self.a3(mx.tanh(self.a2(mx.tanh(self.a1(x))))))

    def value(self, x):
        return self.c3(mx.tanh(self.c2(mx.tanh(self.c1(x)))))[:, 0]

    def ls(self):
        return mx.clip(self.log_std, LOGSTD_MIN, LOGSTD_MAX)


def model_from_np(p):
    m = ActorCritic()
    m.update(tree_unflatten([(k, mx.array(p[k])) for k in PARAM_KEYS]))
    mx.eval(m.parameters())
    return m


def np_from_model(m):
    return {k: np.array(v, dtype=np.float32) for k, v in tree_flatten(m.parameters())}


def logp(a, mu, ls):
    return (-0.5 * ((a - mu) / mx.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)


DEFAULTS = dict(T=256, epochs=10, mb=1024, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, vf=0.5,
                max_grad=0.5, rew_scale=0.1, ckpt_every=25)


def train(params, norm, env_tasks, steps, seed, ckpt, update_norm=False, resume=True,
          eval_every=0, eval_cb=None, log_every=10, quiet=False, **kw):
    """Run PPO for `steps` env steps. Returns (params, norm, history).

    eval_cb(step, params, norm) -> dict is called every `eval_every` steps (and at step 0
    and the end) and its output stored in history["evals"]."""
    cfg = dict(DEFAULTS, **kw)
    n_env = len(env_tasks)
    T = cfg["T"]
    per_update = T * n_env
    n_updates = max(1, steps // per_update)
    task_ids = np.array([TASKS.index(t) for t in env_tasks])
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = model_from_np(params)
    opt = optim.Adam(learning_rate=cfg["lr"])
    update, hist = 0, {"log": [], "evals": [], "cfg": dict(cfg, steps=steps, seed=seed,
                                                           env_tasks=list(env_tasks))}
    st_path = ckpt + "_state.pkl"
    if resume and os.path.exists(st_path):
        st = pickle.load(open(st_path, "rb"))
        if st.get("done"):
            p = {k: v for k, v in st["params"].items()}
            return p, Normalizer.from_state(st["norm"]), st["hist"]
        model = model_from_np(st["params"])
        opt.state = tree_unflatten([(k, mx.array(v)) for k, v in st["opt"]])
        update, hist = st["update"], st["hist"]
        rng.bit_generator.state = st["rng"]
        norm = Normalizer.from_state(st["norm"])
        if not quiet:
            print(f"[ppo] resumed {ckpt} at update {update}/{n_updates}", flush=True)
    else:
        opt.init(model.trainable_parameters())

    def save(done=False):
        st = {"params": np_from_model(model), "opt": [(k, np.array(v)) for k, v in tree_flatten(opt.state)],
              "update": update, "hist": hist, "rng": rng.bit_generator.state, "norm": norm.state(),
              "done": done, "cfg": hist["cfg"]}
        tmp = st_path + ".tmp"
        pickle.dump(st, open(tmp, "wb"))
        os.replace(tmp, st_path)
        save_params(ckpt + ".npz", np_from_model(model), norm, step=update * per_update)

    def loss_fn(m, x, a, old_lp, adv, ret):
        mu = m.mean(x)
        ls = m.ls()
        lp = logp(a, mu, ls)
        ratio = mx.exp(lp - old_lp)
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * adv).mean()
        vl = ((m.value(x) - ret) ** 2).mean()
        kl = ((ratio - 1) - mx.log(ratio)).mean()
        cf = (mx.abs(ratio - 1) > cfg["clip"]).mean()
        return pg + cfg["vf"] * vl, (pg, vl, kl, cf)

    grad_fn = nn.value_and_grad(model, loss_fn)

    envs = [make_env() for _ in range(n_env)]
    obs = np.stack([e.reset(seed=int(seed * 1000 + i + 7919 * update))[0] for i, e in enumerate(envs)])
    ep_ret = np.zeros(n_env)
    ep_returns = []
    t0 = time.time()
    if eval_cb and update == 0 and eval_every:
        hist["evals"].append(dict(step=0, **eval_cb(0, np_from_model(model), norm)))

    while update < n_updates:
        p = np_from_model(model)
        std = np.exp(np.clip(p["log_std"], LOGSTD_MIN, LOGSTD_MAX))
        B_x = np.zeros((T, n_env, IN_DIM), np.float32)
        B_a = np.zeros((T, n_env, ACT_DIM), np.float32)
        B_r = np.zeros((T, n_env), np.float32)
        B_v = np.zeros((T + 1, n_env), np.float32)
        B_done = np.zeros((T, n_env), np.float32)
        for t in range(T):
            if update_norm:
                norm.update(obs)
            x = policy_input(norm, obs, task_ids)
            mu = np_mean(p, x)
            a = (mu + std * rng.standard_normal(mu.shape)).astype(np.float32)
            B_x[t], B_a[t], B_v[t] = x, a, np_value(p, x)
            ac = np.clip(a, -1, 1)
            for i, e in enumerate(envs):
                o, _, term, trunc, info = e.step(ac[i])
                r = task_reward(env_tasks[i], info["x_velocity"], ac[i])
                ep_ret[i] += r
                r = r * cfg["rew_scale"]
                if trunc or term:
                    if trunc and not term:   # time-limit: bootstrap from the final obs
                        xf = policy_input(norm, o[None], task_ids[i:i + 1])
                        r += cfg["gamma"] * float(np_value(p, xf)[0])
                    B_done[t, i] = 1.0
                    ep_returns.append((env_tasks[i], ep_ret[i]))
                    ep_ret[i] = 0.0
                    o, _ = e.reset()
                B_r[t, i] = r
                obs[i] = o
        B_v[T] = np_value(p, policy_input(norm, obs, task_ids))
        adv = np.zeros((T, n_env), np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nd = 1.0 - B_done[t]
            delta = B_r[t] + cfg["gamma"] * B_v[t + 1] * nd - B_v[t]
            last = delta + cfg["gamma"] * cfg["lam"] * nd * last
            adv[t] = last
        ret = adv + B_v[:T]
        X = mx.array(B_x.reshape(-1, IN_DIM))
        A = mx.array(B_a.reshape(-1, ACT_DIM))
        ADV = adv.reshape(-1)
        RET = mx.array(ret.reshape(-1))
        old_lp = logp(A, model.mean(X), model.ls())
        mx.eval(old_lp)
        frac = 1.0 - update / n_updates
        opt.learning_rate = cfg["lr"] * frac
        N = X.shape[0]
        stats = []
        for _ in range(cfg["epochs"]):
            perm = rng.permutation(N)
            for s in range(0, N, cfg["mb"]):
                idx = perm[s:s + cfg["mb"]]
                ad = ADV[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                ix = mx.array(idx)
                (_, aux), g = grad_fn(model, X[ix], A[ix], old_lp[ix], mx.array(ad), RET[ix])
                g, _ = optim.clip_grad_norm(g, cfg["max_grad"])
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state, aux)
                stats.append([float(v) for v in aux])
        update += 1
        step = update * per_update
        s = np.mean(stats, 0)
        recent = {}
        for tk in TASKS:
            rr = [r for (k, r) in ep_returns[-2 * n_env:] if k == tk]
            if rr:
                recent[tk] = float(np.mean(rr))
        rec = dict(update=update, step=step, pg=s[0], vl=s[1], approx_kl=s[2], clip_frac=s[3],
                   std=float(np.mean(std)), ep_ret=recent, wall=time.time() - t0)
        hist["log"].append(rec)
        if not quiet and (update % log_every == 0 or update == n_updates):
            print(f"[ppo {os.path.basename(ckpt)}] {step:>8d} steps  ret {json.dumps({k: round(v) for k, v in recent.items()})}"
                  f"  kl {s[2]:.4f} clip {s[3]:.3f} std {np.mean(std):.3f}  {step / max(1e-9, time.time() - t0):.0f} sps",
                  flush=True)
        if eval_cb and eval_every and (step // eval_every) != ((step - per_update) // eval_every):
            hist["evals"].append(dict(step=step, **eval_cb(step, np_from_model(model), norm)))
        if update % cfg["ckpt_every"] == 0 and update < n_updates:
            save()
    for e in envs:
        e.close()
    save(done=True)
    return np_from_model(model), norm, hist
