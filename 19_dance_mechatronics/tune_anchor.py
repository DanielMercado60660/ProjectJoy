"""Review addition: tune the anchored baseline (baseline_anchor.py) with the same protocol as tune_baseline.py.

Differences from tune_baseline.py, nothing else: 3 extra parameters (k_anchor, delta, k_sw), com_rel always
on, a warm start (default: the tuned baseline_rel solution, anchors at 0) is injected into the stage-1 pool as a warm start,
and --final_loops sets the stage-3 rescoring length (default 5 loops, as in tune_baseline.py).

Original docstring:
Tune the PD + feed-forward + balance baseline on TRAINING seeds only.

Objective = the same per-step DeepMimic tracking reward the RL policy maximises, with early termination
(a fall ends the run and every remaining step scores 0), averaged over seeds and start phases.
Stage 1: log-uniform random search. Stage 2: (mu, lambda) evolution strategy around the best.
Stage 3: re-score the top candidates on more training seeds/loops and keep the best.

  NUMBA_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python tune_baseline.py --n_random 3000 --gens 40
"""
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool

RANGES = {  # name: (low, high, log-scale)
    "kp_h": (50, 6000, True), "kp_k": (50, 6000, True), "kp_a": (20, 4000, True),
    "z_h": (0.003, 0.3, True), "z_k": (0.003, 0.3, True), "z_a": (0.003, 0.3, True),
    "ff": (0.0, 1.2, False), "k_com": (-2000, 2000, False), "k_comv": (-500, 500, False),
    "k_pitch": (-1000, 1000, False), "k_pitchv": (-200, 200, False), "lead": (0.0, 3.0, False),
    "k_anchor": (-5.0, 5.0, False), "delta": (0.005, 0.08, True), "k_sw": (-3.0, 3.0, False),
}
NAMES = list(RANGES)
_env = None


def to_unit(p):
    u = []
    for k in NAMES:
        lo, hi, lg = RANGES[k]
        u.append((np.log(p[k]) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (p[k] - lo) / (hi - lo))
    return np.array(u)


def from_unit(u):
    u = np.clip(u, 0, 1)
    p = {}
    for k, x in zip(NAMES, u):
        lo, hi, lg = RANGES[k]
        p[k] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if lg else float(lo + x * (hi - lo))
    return p


def score(args):
    params, seeds, loops, push, com_rel = args
    params = dict(params, com_rel=com_rel)
    global _env
    from dance_env import DanceEnv, run_clip
    from baseline_anchor import AnchorBaselineController as BaselineController
    if _env is None:
        _env = DanceEnv(seed=0)
    ctrl = BaselineController(params)
    tot, n = 0.0, 0
    falls = 0
    for s in seeds:
        _env.rng = np.random.default_rng(s)
        ph = _env.rng.uniform()
        _env.push = _env.rng.uniform(0, push) if push > 0 else 0.0   # same training pushes as PPO --push
        log = run_clip(_env, ctrl, loops=loops, phase=ph, recover=False)
        n_full = int(round(loops * _env.clip.T / 0.02))
        tot += log["rew"].sum()
        n += n_full
        falls += log["falls"]
    return tot / n, falls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=3000)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/baseline_tuned_anchor.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", type=float, default=0.0, help="training pushes: per-run size U(0, push) N")
    ap.add_argument("--train_seeds", type=int, default=4)
    ap.add_argument("--com_rel", action="store_true", default=True, help="(always on here)")
    ap.add_argument("--warm", default="results/baseline_tuned_rel.json", help="warm start (anchors at 0)")
    ap.add_argument("--final_loops", type=int, default=5)
    ap.add_argument("--k_com_max", type=float, default=2000.0, help="|k_com| range (review: 5000 to check the edge)")
    args = ap.parse_args()
    RANGES["k_com"] = (-args.k_com_max, args.k_com_max, False)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    train_seeds = list(range(100, 100 + args.train_seeds))   # training seeds; evaluation uses 1000+
    hist = []
    with Pool(args.workers) as pool:
        # stage 1: random search, 2 seeds x 2 loops
        U = rng.uniform(size=(args.n_random, len(NAMES)))
        if args.warm:
            U[0] = to_unit(dict(dict(k_anchor=0.0, delta=0.02, k_sw=0.0), **json.load(open(args.warm))["params"]))
        res = pool.map(score, [(from_unit(u), train_seeds[:2], 2, args.push, args.com_rel) for u in U], chunksize=8)
        S = np.array([r[0] for r in res])
        order = np.argsort(-S)
        print(f"[stage1] {args.n_random} configs in {time.time()-t0:.0f}s, best {S[order[0]]:.4f}, "
              f"top5 {S[order[:5]].round(4).tolist()}", flush=True)
        hist.append(dict(stage=1, best=float(S[order[0]])))
        # stage 2: ES from the top 8, 4 seeds x 2 loops
        mu = 8
        parents = U[order[:mu]]
        pres = pool.map(score, [(from_unit(u), train_seeds, 2, args.push, args.com_rel) for u in parents])
        pS = np.array([r[0] for r in pres])
        sigma = 0.08
        for g in range(args.gens):
            kids = parents[rng.integers(0, mu, args.pop)] + sigma * rng.normal(size=(args.pop, len(NAMES)))
            kids = np.clip(kids, 0, 1)
            kres = pool.map(score, [(from_unit(u), train_seeds, 2, args.push, args.com_rel) for u in kids], chunksize=2)
            kS = np.array([r[0] for r in kres])
            allU, allS = np.vstack([parents, kids]), np.r_[pS, kS]
            o = np.argsort(-allS)[:mu]
            parents, pS = allU[o], allS[o]
            sigma = max(0.02, sigma * 0.97)
            print(f"[stage2] gen {g} best {pS[0]:.4f} mean(mu) {pS.mean():.4f} sigma {sigma:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            hist.append(dict(stage=2, gen=g, best=float(pS[0])))
        # stage 3: re-score top candidates (stage-1 top 4 + ES parents) on 8 training seeds x 5 loops
        cands = np.vstack([U[order[:4]], parents] + ([U[:1]] if args.warm else []))
        final_seeds = list(range(100, 100 + 2 * args.train_seeds))
        fres = pool.map(score, [(from_unit(u), final_seeds, args.final_loops, args.push, args.com_rel) for u in cands])
        fS = np.array([r[0] for r in fres])
        fF = np.array([r[1] for r in fres])
        b = int(np.argmax(fS))
        best = dict(from_unit(cands[b]), com_rel=args.com_rel)
        print(f"[stage3] final scores {fS.round(4).tolist()} falls {fF.tolist()} -> best {fS[b]:.4f}", flush=True)
    n_evals = args.n_random + mu + args.gens * args.pop + len(cands)
    out = dict(params=best, score=float(fS[b]), falls_in_final=int(fF[b]), n_evals=n_evals,
               wall_s=time.time() - t0, ranges={k: list(v) for k, v in RANGES.items()},
               train_seeds=final_seeds, push=args.push, controller="anchor", warm=args.warm,
               final_loops=args.final_loops, history=hist, uptime=os.popen("uptime").read().strip())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["params"], indent=1))


if __name__ == "__main__":
    main()
