"""Evaluate an explorer (or the random walk) under a fixed per-episode action budget.

Each evaluation runs `n_episodes` fresh episodes of `max_steps` actions (default 512) from the
start cell, with a fixed evaluation seed that differs from all training seeds. The learned policy
samples stochastically from its action distribution (that is how it explores while training).
Metrics per episode: rooms reached (/9), useful interactions (/8), coverage (cells visited /233),
unproductive steps (no new cell and no new interaction), noisy-TV steps (TV screen in view),
repeated interactions (re-toggling an already-discovered lever/button).

  ../.venv/bin/python evaluate.py --ckpt checkpoints/rnd_s0.pkl --out runs/eval_rnd_s0.json
  ../.venv/bin/python evaluate.py --random --out runs/eval_random.json [--noisy_tv]
"""
import argparse
import json

import numpy as np

from env import VecRooms, N_ACTIONS, EP_FIELDS, INTER_NAMES


def run_eval(ex=None, n_episodes=64, max_steps=512, noisy_tv=False, seed=10_000, record=1):
    """Returns (list of per-episode stats, recordings for the first `record` envs)."""
    env = VecRooms(n_episodes, max_steps, noisy_tv, seed=seed)
    rng = np.random.default_rng(seed)
    if ex is not None:
        ex.rng = np.random.default_rng(seed)  # evaluation randomness independent of training
    obs = env.obs.copy()
    rec = [dict(pos=[env.pos[i].copy()], grid=[env.grid[i].copy()], tv=[env.tv_val[i].copy()],
                events=[]) for i in range(record)]
    eps = []
    for t in range(max_steps):
        a = rng.integers(0, N_ACTIONS, n_episodes) if ex is None else ex.act(obs)[0]
        before = env.inter[:record].copy()
        obs, done, e = env.step(a)
        eps += e
        for i in range(record):
            r = rec[i]
            if not done[i]:
                r["pos"].append(env.pos[i].copy()); r["grid"].append(env.grid[i].copy())
                r["tv"].append(env.tv_val[i].copy())
                for k in np.nonzero(env.inter[i] & ~before[i])[0]:
                    r["events"].append((t, int(k)))
    for i in range(record):
        rec[i] = {k: (np.array(v) if k != "events" else v) for k, v in rec[i].items()}
        rec[i]["inter_t"] = eps[i]["inter_t"]
    return eps, rec


def summarize(eps):
    out = {}
    for f in EP_FIELDS:
        v = np.array([e[f] for e in eps], float)
        out[f] = dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()))
    it = np.array([e["inter_t"] for e in eps])
    out["interaction_rate"] = {INTER_NAMES[k]: float((it[:, k] >= 0).mean()) for k in range(it.shape[1])}
    rooms_all = np.array([e["rooms"] for e in eps])
    out["frac_episodes_all_9_rooms"] = float((rooms_all == 9).mean())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--noisy_tv", action="store_true")
    p.add_argument("--n_episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ex = None
    if not a.random:
        from train import Explorer
        ex, _ = Explorer.load(a.ckpt)
    eps, _ = run_eval(ex, a.n_episodes, noisy_tv=a.noisy_tv, seed=a.seed, record=0)
    s = summarize(eps)
    s["episodes"] = eps
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps({k: v["mean"] for k, v in s.items() if isinstance(v, dict) and "mean" in v}))
