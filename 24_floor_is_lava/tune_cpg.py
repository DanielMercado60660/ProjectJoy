"""Tune the open-loop CPG trot baseline on TRAINING layouts only.

Stage 1: random search over CPG_RANGES (uniform).  Stage 2: cross-entropy method (CEM) around the
best configs.  Stage 3: re-evaluate the top configs on fresh training layouts and keep the best
(avoids picking a lucky config).  Objective = completion rate + 0.3 * mean progress fraction.

Usage:  ../bin/gpu_run ../.venv/bin/python tune_cpg.py --out checkpoints/cpg_params.json
        ../.venv/bin/python tune_cpg.py --tiny       (smoke)
"""
import argparse, json, os, time
import multiprocessing as mp
import numpy as np

KEYS = None
_task = None


def _init(planner=False):
    global _task
    import task as T
    _task = T.LavaTask(planner=planner)


def _eval(args):
    params, lseed, nseed = args
    import task as T, lava_env as E
    _task.cpg = params
    r = T.run_episode(_task, E.make_layout(lseed), rng=np.random.default_rng(nseed), init_noise=0.05)
    return r["success"], r["frac"], r["t"]


def score(pool, plist, layout_seeds):
    jobs = [(p, ls, ls + 777) for p in plist for ls in layout_seeds]
    res = pool.map(_eval, jobs, chunksize=4)
    n = len(layout_seeds)
    out = []
    for i in range(len(plist)):
        rr = res[i * n:(i + 1) * n]
        succ = np.mean([r[0] for r in rr])
        frac = np.mean([r[1] for r in rr])
        out.append((succ + 0.3 * frac, succ, frac))
    steps = sum(r[2] for r in res) / 0.02
    return out, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/cpg_params.json")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--n_random", type=int, default=320)
    ap.add_argument("--n_layouts", type=int, default=16)
    ap.add_argument("--cem_gens", type=int, default=10)
    ap.add_argument("--cem_pop", type=int, default=40)
    ap.add_argument("--final_layouts", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", action="store_true", help="tune CPG + scripted foothold planner")
    ap.add_argument("--warm", default="", help="json of tuned params to include in the initial population")
    a = ap.parse_args()
    if a.tiny:
        a.n_random, a.n_layouts, a.cem_gens, a.cem_pop, a.final_layouts = 6, 2, 1, 4, 4
    import task as T
    ranges = dict(T.CPG_RANGES, **(T.PLAN_RANGES if a.planner else {}))
    default = dict(T.DEFAULT_CPG, **(T.PLAN_DEFAULT if a.planner else {}))
    keys = list(ranges)
    lo = np.array([ranges[k][0] for k in keys])
    hi = np.array([ranges[k][1] for k in keys])
    rng = np.random.default_rng(a.seed)
    to_p = lambda v: {k: float(x) for k, x in zip(keys, v)}
    log = []
    t0 = time.time()
    total_steps = 0
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.planner,))

    def batch_layouts():   # fresh training layouts each batch (training seeds are < 10M)
        return list(rng.integers(0, 5_000_000, a.n_layouts))

    # stage 1: random search (plus the hand-picked default)
    V = [np.array([default[k] for k in keys])] + [rng.uniform(lo, hi) for _ in range(a.n_random)]
    if a.warm:
        w = dict(default, **json.load(open(a.warm))["params"])
        V.append(np.array([w[k] for k in keys]))
    lay = batch_layouts()
    sc, st = score(pool, [to_p(v) for v in V], lay)
    total_steps += st
    for v, s in zip(V, sc):
        log.append(dict(stage="random", params=to_p(v), score=s[0], succ=s[1], frac=s[2]))
    print(f"[random] best {max(s[0] for s in sc):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    # stage 2: CEM, elites from random search seed the mean
    order = np.argsort([-s[0] for s in sc])
    elite = np.array([V[i] for i in order[:10]])
    mu, sd = elite.mean(0), elite.std(0) + 0.05 * (hi - lo)
    if a.warm:
        # warm-started rounds: centre the CEM on the warm parameters with a modest spread
        # (the mean of a mostly-random elite set would throw the warm start away)
        mu, sd = V[-1].copy(), 0.08 * (hi - lo)
    pool_best = [(sc[i][0], V[i]) for i in order[:10]]
    for g in range(a.cem_gens):
        P = [np.clip(mu + sd * rng.standard_normal(len(keys)), lo, hi) for _ in range(a.cem_pop - 1)] + [mu.copy()]
        lay = batch_layouts()
        sc, st = score(pool, [to_p(v) for v in P], lay)
        total_steps += st
        for v, s in zip(P, sc):
            log.append(dict(stage=f"cem{g}", params=to_p(v), score=s[0], succ=s[1], frac=s[2]))
        order = np.argsort([-s[0] for s in sc])
        el = np.array([P[i] for i in order[:8]])
        mu = 0.3 * mu + 0.7 * el.mean(0)
        sd = 0.3 * sd + 0.7 * (el.std(0) + 0.01 * (hi - lo))
        pool_best += [(sc[i][0], P[i]) for i in order[:3]]
        print(f"[cem {g}] best {sc[order[0]][0]:.3f} succ {sc[order[0]][1]:.2f} mean-of-pop {np.mean([s[0] for s in sc]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    # stage 3: re-evaluate the top candidates (+ CEM mean) on fresh training layouts
    pool_best.sort(key=lambda t: -t[0])
    cands = [v for _, v in pool_best[:8]] + [mu]
    lay = list(rng.integers(5_000_000, 9_000_000, a.final_layouts))
    sc, st = score(pool, [to_p(v) for v in cands], lay)
    total_steps += st
    for v, s in zip(cands, sc):
        log.append(dict(stage="final", params=to_p(v), score=s[0], succ=s[1], frac=s[2]))
    bi = int(np.argmax([s[0] for s in sc]))
    best = to_p(cands[bi])
    pool.close()
    elapsed = time.time() - t0
    print(f"[final] best score {sc[bi][0]:.3f} succ {sc[bi][1]:.3f} frac {sc[bi][2]:.3f}  configs={len(log)}  "
          f"env steps={total_steps:.0f}  {elapsed:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(params=best, final_succ=sc[bi][1], final_frac=sc[bi][2], final_layouts=a.final_layouts,
                   env_steps=total_steps, wall_s=elapsed, ranges=ranges, planner=a.planner,
                   n_configs=len(log)), open(a.out, "w"), indent=1)
    json.dump(log, open(a.out.replace(".json", "_log.json"), "w"))
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
