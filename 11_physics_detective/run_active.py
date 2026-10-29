"""Active (EIG) vs random vs fixed-protocol experiments over many hidden worlds.

For every hidden world (mu_k drawn from the prior) and every strategy, run N pushes, and after
each push record the exact Bayesian posterior, the least-squares baseline estimate, and the
prediction error on held-out new pushes. Everything with fixed seeds.

    ../.venv/bin/python run_active.py --worlds 300 --steps 12 --out results/active.npz
"""
import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from inference import (GridPosterior, choose_design, lsq_fit, lsq_interval, run_experiment)
from physics import MU_HI, MU_LO, NT, world1_positions

STRATS = ["active", "random", "fixed"]
LEVELS = [0.5, 0.9]
TABLE = 2.0    # prediction targets are clipped here: past 2 m the block has left the table


def heldout_designs(rng, n=30):
    return np.stack([rng.uniform(1.0, 15.0, n), rng.uniform(0.1, 1.0, n)], 1)


def one_world(args):
    w, steps, seed = args
    rng_w = np.random.default_rng([seed, w])
    mu = rng_w.uniform(MU_LO, MU_HI)
    H = heldout_designs(rng_w)
    truth = np.minimum(world1_positions(H[:, 0], H[:, 1], mu), TABLE)        # (30, NT)
    rec = {"mu": mu}
    for si, strat in enumerate(STRATS):
        rng = np.random.default_rng([seed, w, si])
        post = GridPosterior()
        data = []
        r = {k: np.zeros(steps) for k in ["mean", "sd", "lsq", "lsq_se", "pred_bayes", "pred_lsq", "t_choose"]}
        for lv in LEVELS:
            r[f"lo{int(lv*100)}"] = np.zeros(steps); r[f"hi{int(lv*100)}"] = np.zeros(steps)
        r["designs"] = np.zeros((steps, 2)); r["y"] = np.zeros((steps, NT)); r["cens"] = np.zeros((steps, NT), bool)
        for n in range(steps):
            t0 = time.perf_counter()
            d, _ = choose_design(strat, post, rng)
            r["t_choose"][n] = time.perf_counter() - t0
            y, c = run_experiment(mu, d[0], d[1], rng)
            post.update(d[0], d[1], y, c)
            data.append((d[0], d[1], y, c))
            r["designs"][n] = d; r["y"][n] = y; r["cens"][n] = c
            r["mean"][n] = post.mean(); r["sd"][n] = post.sd()
            for lv in LEVELS:
                r[f"lo{int(lv*100)}"][n], r[f"hi{int(lv*100)}"][n] = post.interval(lv)
            m_ls, se = lsq_fit(data)
            r["lsq"][n], r["lsq_se"][n] = m_ls, se
            pb = np.minimum(post.predict(H[:, 0], H[:, 1]), TABLE)
            pl = np.minimum(world1_positions(H[:, 0], H[:, 1], m_ls), TABLE)
            r["pred_bayes"][n] = np.sqrt(np.mean((pb - truth) ** 2))
            r["pred_lsq"][n] = np.sqrt(np.mean((pl - truth) ** 2))
        rec[strat] = r
    print(f"world {w} done (mu={mu:.3f})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=300)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="results/active.npz")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(w, a.steps, a.seed) for w in range(a.worlds)]
    if a.procs > 1:
        with Pool(a.procs) as p:
            recs = p.map(one_world, jobs, chunksize=4)
    else:
        recs = [one_world(j) for j in jobs]
    out = {"mu": np.array([r["mu"] for r in recs])}
    for s in STRATS:
        for k in recs[0][s]:
            out[f"{s}_{k}"] = np.stack([r[s][k] for r in recs])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"saved {a.out}: {a.worlds} worlds x {a.steps} steps in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
