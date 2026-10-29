"""Compare the amortized MLX posterior network against the exact grid posterior.

Test sets (none seen in training; training draws fresh simulations from seed 0):
  A. 3000 fresh random-design datasets (1..15 pushes, seed 999).
  B. The experiment sequences actually collected by the active / random / fixed detectives in
     results/active.npz (500 hidden worlds x 15 pushes). Active sequences are a design
     distribution shift for the net (EIG keeps repeating a few very informative pushes).

    ../.venv/bin/python eval_sbi.py --ckpt checkpoints/sbi
"""
import argparse
import json
import time

import mlx.core as mx
import numpy as np

from inference import GRID, GridPosterior
from physics import observe, world1_positions
from sbi import MAXN, featurize, load, posterior_on_grid, sample_designs


def stats_from_grid(W, grid=GRID):
    mean = W @ grid
    sd = np.sqrt(np.maximum(W @ grid ** 2 - mean ** 2, 0))
    C = np.cumsum(W, 1); C /= C[:, -1:]
    q = {}
    for p in (0.05, 0.25, 0.75, 0.95):
        q[p] = np.array([np.interp(p, c, grid) for c in C])
    return mean, sd, q, C


def summarize(name, mu, mean_n, sd_n, q_n, mean_e, sd_e, q_e, C_n, C_e):
    dx = GRID[1] - GRID[0]
    w1 = np.abs(C_n - C_e).sum(1) * dx                         # Wasserstein-1 between posteriors
    r = {
        "n": int(len(mu)),
        "rmse_net": float(np.sqrt(np.mean((mean_n - mu) ** 2))),
        "rmse_exact": float(np.sqrt(np.mean((mean_e - mu) ** 2))),
        "median_abs_mean_gap_over_exact_sd": float(np.median(np.abs(mean_n - mean_e) / sd_e)),
        "median_sd_ratio_net_over_exact": float(np.median(sd_n / sd_e)),
        "median_W1_over_exact_sd": float(np.median(w1 / sd_e)),
        "cov50_net": float(np.mean((mu >= q_n[0.25]) & (mu <= q_n[0.75]))),
        "cov90_net": float(np.mean((mu >= q_n[0.05]) & (mu <= q_n[0.95]))),
        "cov50_exact": float(np.mean((mu >= q_e[0.25]) & (mu <= q_e[0.75]))),
        "cov90_exact": float(np.mean((mu >= q_e[0.05]) & (mu <= q_e[0.95]))),
    }
    print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
    return r


def run_net(model, feats, mask, bs=1000):
    out = []
    for s in range(0, len(feats), bs):
        out.append(posterior_on_grid(model, feats[s:s + bs], mask[s:s + bs], GRID))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sbi")
    ap.add_argument("--active", default="results/active.npz")
    ap.add_argument("--n_fresh", type=int, default=3000)
    ap.add_argument("--out", default="results/sbi_eval.json")
    a = ap.parse_args()
    model, cfg = load(a.ckpt)
    res = {"config": cfg}

    # ---------------- A: fresh random-design datasets
    rng = np.random.default_rng(999)
    B = a.n_fresh
    mu = rng.uniform(0.05, 0.8, B)
    d = sample_designs(rng, B, MAXN)
    x = world1_positions(d[..., 0], d[..., 1], mu[:, None])
    y, c = observe(x, rng)
    n = rng.integers(1, MAXN + 1, B)
    mask = (np.arange(MAXN)[None] < n[:, None]).astype(np.float32)
    feats = featurize(d, y, c)
    t0 = time.perf_counter()
    W_net = run_net(model, feats, mask)
    t_net = time.perf_counter() - t0
    t0 = time.perf_counter()
    W_ex = []
    for i in range(B):
        p = GridPosterior()
        for j in range(n[i]):
            p.update(d[i, j, 0], d[i, j, 1], y[i, j], c[i, j])
        W_ex.append(p.w)
    W_ex = np.array(W_ex)
    t_ex = time.perf_counter() - t0
    mn, sn, qn, Cn = stats_from_grid(W_net)
    me, se, qe, Ce = stats_from_grid(W_ex)
    pit_n = np.array([np.interp(m_, GRID, c_ / c_[-1]) for m_, c_ in zip(mu, Cn)])
    pit_e = np.array([np.interp(m_, GRID, c_ / c_[-1]) for m_, c_ in zip(mu, Ce)])
    np.savez_compressed(a.out.replace(".json", "_pit.npz").replace("sbi_eval", "sbi"), pit_net=pit_n, pit_exact=pit_e, n=n, mu=mu)
    res["fresh_random"] = summarize("fresh", mu, mn, sn, qn, me, se, qe, Cn, Ce)
    # by number of pushes
    res["fresh_by_n"] = {}
    for k in (1, 3, 8, 15):
        sel = n == k
        res["fresh_by_n"][k] = summarize(f"fresh n={k}", mu[sel], mn[sel], sn[sel], {p: v[sel] for p, v in qn.items()},
                                         me[sel], se[sel], {p: v[sel] for p, v in qe.items()}, Cn[sel], Ce[sel])

    # speed: single-dataset latency and batched throughput
    f1, m1 = feats[:1], mask[:1]
    for _ in range(3):
        posterior_on_grid(model, f1, m1, GRID)
    t0 = time.perf_counter()
    for _ in range(50):
        lw, lc, sc = model(mx.array(f1), mx.array(m1)); mx.eval(lw, lc, sc)
    lat_net = (time.perf_counter() - t0) / 50
    fb, mb = mx.array(feats), mx.array(mask)
    t0 = time.perf_counter()
    for _ in range(5):
        lw, lc, sc = model(fb, mb); mx.eval(lw, lc, sc)
    t_raw = (time.perf_counter() - t0) / 5
    res["speed"] = {
        "net_batched_ms_per_dataset_incl_grid_density": 1000 * t_net / B,
        "exact_grid_ms_per_dataset": 1000 * t_ex / B,
        "net_single_forward_ms": 1000 * lat_net,
        "net_batched_forward_only_ms_per_dataset": 1000 * t_raw / B,
        "mean_pushes_per_dataset": float(n.mean()),
    }
    print("speed", res["speed"])

    # ---------------- B: sequences collected by the detectives
    D = np.load(a.active)
    mu_w = D["mu"]
    res["detective_sequences"] = {}
    for strat in ["active", "random", "fixed"]:
        des, Y, Cc = D[f"{strat}_designs"], D[f"{strat}_y"], D[f"{strat}_cens"]
        nw, steps = des.shape[:2]
        feats = featurize(des, Y, Cc)
        if steps < MAXN:
            feats = np.concatenate([feats, np.zeros((nw, MAXN - steps, feats.shape[-1]), np.float32)], 1)
        rows = {}
        for k in [k for k in (1, 2, 4, 8, 15) if k <= steps]:
            mask = (np.arange(MAXN)[None] < k).astype(np.float32).repeat(nw, 0)
            W = run_net(model, feats, mask)
            mn, sn, qn, Cn = stats_from_grid(W)
            # exact posterior recomputed (cheap) to get the full CDF for W1
            We = []
            for i in range(nw):
                p = GridPosterior()
                for j in range(k):
                    p.update(des[i, j, 0], des[i, j, 1], Y[i, j], Cc[i, j])
                We.append(p.w)
            me, se, qe, Ce = stats_from_grid(np.array(We))
            rows[k] = summarize(f"{strat} n={k}", mu_w, mn, sn, qn, me, se, qe, Cn, Ce)
        res["detective_sequences"][strat] = rows
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
