"""Reviewer check (REVIEW.md): does ANY fast open-loop CPG gait stand up, even on a flat walkway?

Planner round 5 (review_tune_wide.py) was a warm-started local search that never left ~0.5 m/s
nominal speed, so it says little about fast gaits. This searches only the fast corner of the gait
box (freq 3.2-5.0 Hz, stride 0.22-0.45 m, i.e. nominal freq x stride >= 0.7 m/s; every other CPG
parameter over its full range) on FLAT training walkways (no gaps), where only gait stability
matters. Random search, then CEM. Objective = flat completion + 0.3 x progress.
Note: at stride >= ~0.37 m with the heading term saturated (x1.5), the swing reach exceeds the leg's
0.425 m and leg_ik clips the target, so the very longest strides are not executed as commanded.

Usage: ../bin/gpu_run ../.venv/bin/python review_fast_corner.py --out checkpoints/review_fast_corner_params.json
"""
import argparse, json, os, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAST = dict(freq=(3.2, 5.0), stride=(0.22, 0.45))
_task = None


def _init():
    global _task
    import sys
    sys.path.insert(0, HERE)
    import task as T
    _task = T.LavaTask()


def _eval(args):
    params, lseed = args
    import task as T, lava_env as E
    _task.cpg = params
    r = T.run_episode(_task, E.make_layout(lseed, flat=True), rng=np.random.default_rng(lseed + 777), init_noise=0.05)
    return r["success"], r["frac"], r["t"]


def score(pool, plist, seeds):
    res = pool.map(_eval, [(p, s) for p in plist for s in seeds], chunksize=4)
    n = len(seeds)
    out = []
    for i in range(len(plist)):
        rr = res[i * n:(i + 1) * n]
        su, fr = np.mean([r[0] for r in rr]), np.mean([r[1] for r in rr])
        out.append((su + 0.3 * fr, su, fr))
    return out, sum(r[2] for r in res) / 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=240)
    ap.add_argument("--n_layouts", type=int, default=6)
    ap.add_argument("--cem_gens", type=int, default=8)
    ap.add_argument("--cem_pop", type=int, default=48)
    ap.add_argument("--final_layouts", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="checkpoints/review_fast_corner_params.json")
    a = ap.parse_args()
    os.chdir(HERE)
    import task as T
    ranges = dict(T.CPG_RANGES, **FAST)
    keys = list(ranges)
    lo, hi = np.array([ranges[k][0] for k in keys]), np.array([ranges[k][1] for k in keys])
    rng = np.random.default_rng(a.seed)
    to_p = lambda v: {k: float(x) for k, x in zip(keys, v)}
    seeds = lambda n: list(rng.integers(0, 9_000_000, n))   # training seeds only (< 10M)
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init)
    t0, steps, hist = time.time(), 0.0, []
    V = [rng.uniform(lo, hi) for _ in range(a.n_random)]
    sc, st = score(pool, [to_p(v) for v in V], seeds(a.n_layouts)); steps += st
    hist.append(dict(stage="random", best_succ=max(s[1] for s in sc), mean_succ=float(np.mean([s[1] for s in sc])),
                     n_any_success=int(sum(s[1] > 0 for s in sc))))
    print(hist[-1], f"{time.time()-t0:.0f}s", flush=True)
    order = np.argsort([-s[0] for s in sc])
    el = np.array([V[i] for i in order[:10]])
    mu, sd = el.mean(0), el.std(0) + 0.05 * (hi - lo)
    best = [(sc[i][0], V[i]) for i in order[:6]]
    for g in range(a.cem_gens):
        P = [np.clip(mu + sd * rng.standard_normal(len(keys)), lo, hi) for _ in range(a.cem_pop - 1)] + [mu.copy()]
        sc, st = score(pool, [to_p(v) for v in P], seeds(a.n_layouts)); steps += st
        order = np.argsort([-s[0] for s in sc])
        el = np.array([P[i] for i in order[:8]])
        mu = 0.3 * mu + 0.7 * el.mean(0)
        sd = 0.3 * sd + 0.7 * (el.std(0) + 0.01 * (hi - lo))
        best += [(sc[i][0], P[i]) for i in order[:3]]
        hist.append(dict(stage=f"cem{g}", best_succ=float(sc[order[0]][1]), mean_succ=float(np.mean([s[1] for s in sc])),
                         mu_freq=float(mu[keys.index("freq")]), mu_stride=float(mu[keys.index("stride")])))
        print(hist[-1], f"{time.time()-t0:.0f}s", flush=True)
    best.sort(key=lambda t: -t[0])
    cands = [v for _, v in best[:6]] + [mu]
    sc, st = score(pool, [to_p(v) for v in cands], seeds(a.final_layouts)); steps += st
    bi = int(np.argmax([s[0] for s in sc]))
    pool.close()
    p = to_p(cands[bi])
    out = dict(params=p, final_flat_succ=sc[bi][1], final_flat_frac=sc[bi][2], final_layouts=a.final_layouts,
               nominal_speed=p["freq"] * p["stride"], env_steps=steps, wall_s=time.time() - t0, ranges=ranges, history=hist)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("ranges", "history")}, indent=1))


if __name__ == "__main__":
    main()
