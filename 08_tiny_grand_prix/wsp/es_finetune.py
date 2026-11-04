"""Matched-budget ES fine-tuning of the same start policy (the gradient-free learning baseline).

Same OpenAI-ES update as train_es.py (antithetic pairs, centered ranks, Adam, sigma 0.04, lr 0.01,
wd 0.005) but small populations so a tiny episode budget still buys some generations, and only the
32 train tracks Astra's wrapper uses (ids 0..31, random start points, 60 s episodes, strict walls;
task=grip: half the episodes at grip 0.5, half at 1.0). Stops when the episode budget is spent; the final weights
are the result (no validation selection: the wrapper's tracks are the only data either side gets).

    python wsp/es_finetune.py --task hack --start ../checkpoints/es_soft_best.npz --episodes_budget 1600 --tag x
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WSP, dict_to_theta, load_weights, save_weights, theta_to_dict  # noqa: E402
sys.path.insert(0, ROOT)
from train_es import centered_ranks, run_pop  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hack")
    ap.add_argument("--start", required=True)
    ap.add_argument("--episodes_budget", type=int, required=True)
    ap.add_argument("--half", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="episodes per member per generation")
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    bank = pack(make_tracks("train", 32))
    # task=grip: like the wrapper, which scores grip 0.5 under a grip-1.0 constraint, half of each
    # generation's episodes run at grip 0.5 and half at grip 1.0
    grips = np.array([0.5, 1.0] * (a.k // 2)) if a.task == "grip" else np.ones(a.k)
    rng = np.random.default_rng(a.seed)
    theta = dict_to_theta(load_weights(a.start))
    P = len(theta)
    m = np.zeros(P); v = np.zeros(P)
    eps_used, steps_used, gen = 0, 0, 0
    per_gen = (2 * a.half + 1) * a.k
    t0 = time.time()
    hist = []
    while eps_used + per_gen <= a.episodes_budget:
        eps = rng.standard_normal((a.half, P)).astype(np.float32)
        thetas = np.concatenate([theta + a.sigma * eps, theta - a.sigma * eps, theta[None]], 0)
        tids = rng.integers(0, 32, a.k)
        starts = rng.random(a.k)
        s, crashed, n = run_pop(bank, thetas, tids, starts, grips, 600, False)
        eps_used += per_gen; steps_used += n
        f = s.mean(1)
        r = centered_ranks(f[:-1])
        w = r[: a.half] - r[a.half: 2 * a.half]
        g = -(w @ eps) / (2 * a.half * a.sigma) + a.wd * theta
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        gen += 1
        theta = (theta - a.lr * (m / (1 - 0.9**gen)) / (np.sqrt(v / (1 - 0.999**gen)) + 1e-8)).astype(np.float32)
        hist.append(dict(gen=gen, episodes=eps_used, steps=steps_used, center=float(f[-1]),
                         center_crash=float(crashed[-1].mean())))
    out_dir = os.path.join(WSP, "baselines")
    os.makedirs(out_dir, exist_ok=True)
    save_weights(os.path.join(out_dir, f"es_ft_{a.tag}.npz"), theta_to_dict(theta))
    info = dict(cfg=vars(a), gens=gen, episodes=eps_used, env_steps=steps_used, seconds=time.time() - t0,
                last=hist[-5:] if hist else [])
    json.dump(dict(info=info, hist=hist), open(os.path.join(out_dir, f"es_ft_{a.tag}.json"), "w"))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
