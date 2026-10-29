"""Scripted baseline: an open-loop sinusoidal gait (central pattern generator) per task.

a_j(t) = clip(b_j + A_j * sin(2*pi*f*t*dt + phi_j), -1, 1), 19 parameters (f, 6 A, 6 phi, 6 b).
Tuned with the cross-entropy method on TRAINING seeds only (reset seeds 500..), then scored on the
held-out eval seeds with the same reward as the learned policies. Like the switching baseline it
is told the task label (one tuned gait per task).

  ../bin/gpu_run ../.venv/bin/python cpg.py --iters 40 --pop 48
"""
import argparse
import json
import os
import time

import numpy as np

from common import ACT_DIM, EP_LEN, EVAL_SEEDS, TASKS, make_env, task_reward

DT = 0.05
LO = np.array([0.5] + [0.0] * 6 + [-np.pi] * 6 + [-0.5] * 6)
HI = np.array([4.0] + [1.0] * 6 + [np.pi] * 6 + [0.5] * 6)


def gait(theta, t):
    f, A, phi, b = theta[0], theta[1:7], theta[7:13], theta[13:19]
    return np.clip(b + A * np.sin(2 * np.pi * f * t * DT + phi), -1, 1)


def rollout(thetas, task, seeds):
    envs = [make_env() for _ in thetas]
    for e, s in zip(envs, seeds):
        e.reset(seed=int(s))
    ret = np.zeros(len(thetas))
    for t in range(EP_LEN):
        for i, e in enumerate(envs):
            a = gait(thetas[i], t)
            _, _, _, _, info = e.step(a)
            ret[i] += task_reward(task, info["x_velocity"], a)
    for e in envs:
        e.close()
    return ret


def cem(task, iters, pop, seed, n_elite=None):
    rng = np.random.default_rng(seed)
    n_elite = n_elite or max(4, pop // 6)
    mu, sd = (LO + HI) / 2, (HI - LO) / 3
    best, best_r, curve = None, -np.inf, []
    for it in range(iters):
        th = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), LO, HI)
        r = rollout(th, task, 500 + rng.integers(0, 1000, size=pop))
        el = th[np.argsort(r)[-n_elite:]]
        mu, sd = el.mean(0), el.std(0) + 0.02 * (HI - LO)
        if r.max() > best_r:
            best_r, best = float(r.max()), th[np.argmax(r)].copy()
        curve.append(float(np.sort(r)[-n_elite:].mean()))
    # re-score the final mean and the best sample on 3 more training seeds, keep the better
    cand = np.stack([mu, best])
    rr = np.stack([rollout(cand, task, [900 + k] * 2) for k in range(3)]).mean(0)
    return cand[int(np.argmax(rr))], curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cpg.json")
    args = ap.parse_args()
    t0 = time.time()
    out = {"tuning": f"CEM {args.iters} iters x pop {args.pop} x 1 episode on training seeds "
                     f"({args.iters * args.pop * EP_LEN} env steps per task); ranges f in [0.5,4] Hz, "
                     f"A in [0,1], phi in [-pi,pi], b in [-0.5,0.5]"}
    for task in TASKS:
        th, curve = cem(task, args.iters, args.pop, args.seed)
        ev = {tk: rollout(np.stack([th] * len(EVAL_SEEDS)), tk, EVAL_SEEDS).tolist() for tk in TASKS}
        out[task] = dict(theta=th.tolist(), elite_curve=curve, eval=ev)
        print(f"[cpg] {task}: eval own {np.mean(ev[task]):.0f}  other {np.mean(ev[TASKS[1 - TASKS.index(task)]]):.0f}",
              flush=True)
    out["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
