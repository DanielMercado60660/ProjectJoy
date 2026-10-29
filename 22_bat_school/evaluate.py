"""Held-out evaluation: learned bat vs wall-follower (fixed-rate pulses) vs random walk.

  ../.venv/bin/python evaluate.py --ckpt checkpoints/bat            # writes results/eval.json
  ../.venv/bin/python evaluate.py --ckpt checkpoints/smoke --n 20 --out results/smoke_eval.json
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from bat_env import BatEnv, MazePool, RandomWalk, WallFollower, WallFollowerEvent
from model import load

HELDOUT = {5: 5005, 6: 6006, 7: 7007}   # pool seeds; training uses seed 1000 (sizes 4,5,6)


class LearnedAgent:
    def __init__(self, model, n, greedy=False, pulse_k=None, seed=0, amnesia=False):
        self.model, self.n, self.greedy, self.pulse_k = model, n, greedy, pulse_k
        self.amnesia = amnesia                # ablation: wipe the GRU state before every step
        self.h = mx.zeros((n, model.hidden))
        self.rng = np.random.default_rng(seed)
        self.t = np.zeros(n, int)

    def act(self, obs):
        if self.amnesia:
            self.h = mx.zeros_like(self.h)
        lm, lp, _, self.h = self.model.step(mx.array(obs), self.h)
        pm = np.array(mx.softmax(lm, -1)); pp = np.array(mx.softmax(lp, -1))
        if self.greedy:
            mv = pm.argmax(-1); pu = pp[:, 1] > 0.5
        else:
            mv = (self.rng.random((self.n, 1)) > np.cumsum(pm, -1)).sum(-1).clip(0, 2)
            pu = self.rng.random(self.n) < pp[:, 1]
        self.t += 1
        if self.pulse_k is not None:          # override: fixed-rate sensing, same move policy
            pu = (self.t % self.pulse_k) == 0
        self.last_pulse_prob = pp[:, 1]
        return mv, pu

    def on_done(self, done):
        if done.any():
            mask = mx.array((~done).astype(np.float32))[:, None]
            self.h = self.h * mask
            self.t[done] = 0


def run(agent_fn, pool, noise=0.03, dropout=0.05, pulse_budget=None, seed=0):
    n = pool.n
    env = BatEnv(pool, n, seed=seed, noise=noise, dropout=dropout, pulse_budget=pulse_budget, fixed_order=True)
    agent = agent_fn(n)
    obs = env.reset()
    first = env.ep_id.copy()
    res = {}
    while len(res) < n:
        mv, pu = agent.act(obs)
        obs, r, d, info = env.step(mv, pu)
        agent.on_done(d)
        for i in np.nonzero(d)[0]:
            e = int(info["ep_id"][i])
            if e == first[i] and e not in res:
                res[e] = (bool(info["success"][i]), int(info["t"][i]), int(info["pulses"][i]), int(info["collisions"][i]))
    a = np.array([res[e] for e in sorted(res)], float)
    succ, T, P, C = a.T
    return dict(n=n, success=float(succ.mean()), success_se=float(succ.std() / np.sqrt(n)),
                steps=float(T.mean()), steps_on_success=float(T[succ > 0].mean()) if succ.any() else None,
                pulses_per_ep=float(P.mean()), pulses_per_step=float((P / T).mean()),
                collisions_per_ep=float(C.mean()), collision_rate=float((C / T).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bat")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--quick", action="store_true", help="main table only (smoke)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model, cfg = load(a.ckpt)
    t0 = time.time()
    pools = {s: MazePool([s], a.n, seed=seed) for s, seed in HELDOUT.items()}
    out = {"ckpt": a.ckpt, "train_cfg": cfg, "n_mazes_per_size": a.n, "tables": {}}

    agents = {
        "random_walk": lambda n: RandomWalk(n, seed=1),
        "wallfollow_k1": lambda n: WallFollower(n, 1),
        "wallfollow_k2": lambda n: WallFollower(n, 2),
        "wallfollow_k3": lambda n: WallFollower(n, 3),
        "wallfollow_k4": lambda n: WallFollower(n, 4),
        "wallfollow_event": lambda n: WallFollowerEvent(n),   # reviewer: pulse only when a decision is due
        "bat_learned": lambda n: LearnedAgent(model, n, seed=1),
        "bat_learned_greedy": lambda n: LearnedAgent(model, n, greedy=True),
        "bat_no_memory": lambda n: LearnedAgent(model, n, seed=1, amnesia=True),
        "bat_no_memory_forced_k1": lambda n: LearnedAgent(model, n, seed=1, amnesia=True, pulse_k=1),
        "bat_forced_k1": lambda n: LearnedAgent(model, n, pulse_k=1, seed=1),
        "bat_forced_k2": lambda n: LearnedAgent(model, n, pulse_k=2, seed=1),
        "bat_forced_k4": lambda n: LearnedAgent(model, n, pulse_k=4, seed=1),
    }
    # 1) main table: every agent on every held-out size, training noise level
    for s, pool in pools.items():
        for name, fn in agents.items():
            r = run(fn, pool)
            out["tables"].setdefault(f"size{s}", {})[name] = r
            print(f"size {s} {name:20s} succ {r['success']:.3f} coll/step {r['collision_rate']:.3f} "
                  f"pulses/ep {r['pulses_per_ep']:.1f} steps {r['steps']:.0f}", flush=True)
    if a.quick:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        return
    # 2) robustness: noisier echoes and more dropout (6x6 held-out)
    rob = {}
    for noise, drop in [(0.03, 0.05), (0.1, 0.05), (0.2, 0.05), (0.03, 0.3), (0.1, 0.3)]:
        key = f"noise{noise}_drop{drop}"
        rob[key] = {}
        for name in ["wallfollow_k1", "wallfollow_k2", "bat_learned", "bat_forced_k1", "bat_forced_k2"]:
            rob[key][name] = run(agents[name], pools[6], noise=noise, dropout=drop)
            print(f"robust {key} {name:16s} succ {rob[key][name]['success']:.3f} "
                  f"coll/step {rob[key][name]['collision_rate']:.3f}", flush=True)
    out["robustness_size6"] = rob
    # 3) hard pulse budget (6x6 held-out)
    bud = {}
    for B in [10, 20, 40]:
        bud[f"budget{B}"] = {name: run(agents[name], pools[6], pulse_budget=B)
                             for name in ["wallfollow_k1", "wallfollow_k2", "bat_learned"]}
        print(f"budget {B}: " + ", ".join(f"{k} {v['success']:.3f}" for k, v in bud[f'budget{B}'].items()), flush=True)
    out["pulse_budget_size6"] = bud
    out["eval_sec"] = time.time() - t0
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
