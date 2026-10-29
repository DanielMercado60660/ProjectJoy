"""Main run: evolve gliders with CMA-ES and MAP-Elites, against random search at matched budget.

Arms (all maximise mean range over the 6 fixed training launches in sim.training_conditions):
  cma      CMA-ES, RK4 dt=5 ms simulator               (5 seeds x BUDGET evals)
  random   uniform random designs, same simulator      (5 seeds x BUDGET evals)  <- cheap baseline
  mapel    MAP-Elites archive, descriptors = (nominal flight time, log10 pitch-rate RMS)
  cheap    CMA-ES on a "cheap" simulator: explicit Euler dt=20 ms (16x less compute). This arm
           exists to see whether evolution exploits integrator error; its winners are audited later.

Usage:  NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python search.py [--budget 4000] [--me-evals 120000]
Writes checkpoints/search.npz (+ json summary) and results/search_log.json.
"""
import argparse
import json
import os
import time

import numpy as np

import sim

HERE = os.path.dirname(os.path.abspath(__file__))
DT_SOUND, DT_CHEAP = 0.005, 0.02
TMAX = 60.0


class Fitness:
    def __init__(self, dt, method):
        self.conds, self.gusts = sim.training_conditions(seed=0)
        self.dt, self.method = dt, method
        self.n_evals = 0

    def __call__(self, genes):
        genes = np.atleast_2d(genes)
        clipped = np.clip(genes, 0.0, 1.0)
        stats = sim.evaluate_batch(clipped, self.conds, self.gusts, self.dt, TMAX, self.method)
        self.n_evals += len(genes)
        fit = stats[:, :, 0].mean(1)
        fit = fit - 100.0 * ((genes - clipped) ** 2).sum(1)      # soft box penalty
        return fit, stats


def cma_es(f, n=8, budget=4000, sigma0=0.25, popsize=16, seed=0):
    """Plain (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial defaults), maximising f."""
    rng = np.random.default_rng(seed)
    lam = popsize
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    mean = rng.uniform(0.2, 0.8, n)
    sigma = sigma0
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    best_x, best_f = None, -np.inf
    hist = []
    evals = 0
    gen = 0
    while evals + lam <= budget:
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-20))
        z = rng.standard_normal((lam, n))
        y = z * D @ B.T
        x = mean + sigma * y
        fit, _ = f(x)
        evals += lam
        order = np.argsort(-fit)
        if fit[order[0]] > best_f:
            best_f, best_x = float(fit[order[0]]), np.clip(x[order[0]], 0, 1).copy()
        yw = (w[:, None] * y[order[:mu]]).sum(0)
        mean = mean + sigma * yw
        invsqrtC = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ yw
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        yk = y[order[:mu]]
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
             + cmu * (w[:, None, None] * yk[:, :, None] * yk[:, None, :]).sum(0))
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        sigma = min(sigma, 1.0)
        hist.append((evals, best_f))
        gen += 1
    return dict(best_x=best_x, best_f=best_f, hist=np.array(hist), mean=mean, sigma=sigma, C=C)


def random_search(f, n=8, budget=4000, seed=0, batch=500):
    rng = np.random.default_rng(1000 + seed)
    best_x, best_f, hist, evals = None, -np.inf, [], 0
    while evals < budget:
        x = rng.uniform(0, 1, (min(batch, budget - evals), n))
        fit, _ = f(x)
        evals += len(x)
        i = int(np.argmax(fit))
        if fit[i] > best_f:
            best_f, best_x = float(fit[i]), x[i].copy()
        hist.append((evals, best_f))
    return dict(best_x=best_x, best_f=best_f, hist=np.array(hist))


# MAP-Elites grid: x = nominal still-air flight time (s), y = log10 pitch-rate RMS after 2 s (rad/s)
ME_BINS = (24, 20)
ME_RANGE = ((0.0, 24.0), (-3.0, 1.0))


def me_cell(stats):
    t = stats[:, 0, 1]
    q = np.log10(np.maximum(stats[:, 0, 6], 1e-6))
    ix = np.clip(((t - ME_RANGE[0][0]) / (ME_RANGE[0][1] - ME_RANGE[0][0]) * ME_BINS[0]).astype(int), 0, ME_BINS[0] - 1)
    iy = np.clip(((q - ME_RANGE[1][0]) / (ME_RANGE[1][1] - ME_RANGE[1][0]) * ME_BINS[1]).astype(int), 0, ME_BINS[1] - 1)
    return ix, iy


def map_elites(f, n=8, evals_total=120000, init=4000, batch=400, seed=0):
    rng = np.random.default_rng(2000 + seed)
    fit_grid = np.full(ME_BINS, -np.inf)
    x_grid = np.zeros(ME_BINS + (n,))
    evals = 0
    hist = []

    def insert(x, fit, stats):
        ix, iy = me_cell(stats)
        for k in range(len(x)):
            if fit[k] > fit_grid[ix[k], iy[k]]:
                fit_grid[ix[k], iy[k]] = fit[k]
                x_grid[ix[k], iy[k]] = np.clip(x[k], 0, 1)

    x = rng.uniform(0, 1, (init, n))
    fit, stats = f(x)
    evals += init
    insert(x, fit, stats)
    while evals < evals_total:
        filled = np.argwhere(np.isfinite(fit_grid))
        pa = filled[rng.integers(0, len(filled), batch)]
        pb = filled[rng.integers(0, len(filled), batch)]
        xa = x_grid[pa[:, 0], pa[:, 1]]
        xb = x_grid[pb[:, 0], pb[:, 1]]
        # iso+line variation (Vassiliades & Mouret 2018)
        x = xa + 0.03 * rng.standard_normal((batch, n)) + 0.2 * rng.standard_normal((batch, 1)) * (xb - xa)
        x = np.clip(x, 0, 1)
        fit, stats = f(x)
        evals += batch
        insert(x, fit, stats)
        ok = np.isfinite(fit_grid)
        hist.append((evals, int(ok.sum()), float(np.where(ok, np.maximum(fit_grid, 0), 0).sum()),
                     float(fit_grid[ok].max())))
    return dict(fit_grid=fit_grid, x_grid=x_grid, hist=np.array(hist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4000, help="evals per CMA/random seed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--me-evals", type=int, default=120000)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints", "search.npz"))
    args = ap.parse_args()
    t0 = time.time()
    log = {"args": vars(args), "timing_s": {}}
    sound = Fitness(DT_SOUND, 0)
    cheap = Fitness(DT_CHEAP, 1)
    sound(np.full((2, 8), 0.5)); cheap(np.full((2, 8), 0.5))   # jit warmup
    sound.n_evals = cheap.n_evals = 0

    res = {}
    for arm, f, fn in [("cma", sound, cma_es), ("random", sound, random_search), ("cheap", cheap, cma_es)]:
        t = time.time()
        runs = [fn(f, budget=args.budget, seed=s) for s in range(args.seeds)]
        res[arm] = runs
        log["timing_s"][arm] = time.time() - t
        print(f"{arm:7s} best per seed: " + " ".join(f"{r['best_f']:.1f}" for r in runs)
              + f"   ({time.time() - t:.1f}s)", flush=True)
    t = time.time()
    me = map_elites(sound, evals_total=args.me_evals)
    log["timing_s"]["mapel"] = time.time() - t
    ok = np.isfinite(me["fit_grid"])
    print(f"mapel   cells {ok.sum()}/{ok.size}  best {me['fit_grid'][ok].max():.1f}  ({time.time() - t:.1f}s)")
    log["timing_s"]["total"] = time.time() - t0
    log["uptime"] = os.popen("uptime").read().strip()
    log["sound_evals"] = sound.n_evals
    log["cheap_evals"] = cheap.n_evals

    save = {}
    for arm in ("cma", "random", "cheap"):
        save[f"{arm}_best_x"] = np.array([r["best_x"] for r in res[arm]])
        save[f"{arm}_best_f"] = np.array([r["best_f"] for r in res[arm]])
        save[f"{arm}_hist"] = np.array([r["hist"][:, 1] for r in res[arm]])
        save[f"{arm}_hist_evals"] = res[arm][0]["hist"][:, 0]
    for arm in ("cma", "cheap"):
        save[f"{arm}_mean"] = np.array([r["mean"] for r in res[arm]])
        save[f"{arm}_sigma"] = np.array([r["sigma"] for r in res[arm]])
        save[f"{arm}_C"] = np.array([r["C"] for r in res[arm]])
    save["me_fit"] = me["fit_grid"]
    save["me_x"] = me["x_grid"]
    save["me_hist"] = me["hist"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **save)
    for arm in ("cma", "random", "cheap"):
        log[arm] = {"best_f_per_seed": [r["best_f"] for r in res[arm]]}
    log["mapel"] = {"cells": int(ok.sum()), "best": float(me["fit_grid"][ok].max())}
    default_out = os.path.abspath(args.out) == os.path.join(HERE, "checkpoints", "search.npz")
    log_path = (os.path.join(HERE, "results", "search_log.json") if default_out
                else os.path.splitext(args.out)[0] + "_log.json")
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=1)
    print(f"saved {args.out}  total {time.time() - t0:.1f}s  | {log['uptime']}")


if __name__ == "__main__":
    main()
