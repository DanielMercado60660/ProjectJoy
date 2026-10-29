"""Evolutionary dungeon designer with a hard solvability constraint.

Genome = (walls, start, goal). Mutation = a few edits (flip an interior cell, move start, move
goal). With `constrained=True` every child is BFS-checked and an invalid child is redrawn (up to
`tries` times, else the parent is copied), so the designer can never wall off the exit.
Fitness = regret vs the BFS oracle, averaged over K stochastic solver runs, in env-return units
(R = 1 - 0.9 t/T on success, 0 on failure): regret = R(BFS path) - R(solver). A failure scores
about 0.9, a BFS-optimal escape 0. (An earlier 1 - SPL fitness was gamed with very short paths;
see RESULTS.md.)
"""
import json, time
import numpy as np

from .mazes import SIZE, bfs_dist, level_stats
from .scripted import run_batch, summarize
from .env import MAX_STEPS

MIN_DIST = 6


def solvable(w, s, g, min_dist=None):
    min_dist = MIN_DIST if min_dist is None else min_dist
    if w[s[0], s[1]] or w[g[0], g[1]]:
        return False
    d = bfs_dist(w, s[0], s[1])[g[0], g[1]]
    return d >= min_dist


def mutate(w, s, g, rng):
    w, s, g = w.copy(), s.copy(), g.copy()
    n_edits = min(6, int(rng.geometric(0.5)))
    for _ in range(n_edits):
        u = rng.random()
        if u < 0.8:
            r, c = rng.integers(1, SIZE - 1, size=2)
            if (r, c) != tuple(s) and (r, c) != tuple(g):
                w[r, c] ^= 1
        else:
            floor = np.argwhere(w == 0)
            p = floor[rng.integers(len(floor))]
            if u < 0.9:
                if tuple(p) != tuple(g):
                    s = p.astype(np.int64)
            else:
                if tuple(p) != tuple(s):
                    g = p.astype(np.int64)
    return w, s, g


class Stats:
    def __init__(self):
        self.proposed = 0
        self.valid = 0


def make_child(w, s, g, rng, constrained, stats, tries=20):
    for _ in range(tries):
        cw, cs, cg = mutate(w, s, g, rng)
        stats.proposed += 1
        ok = solvable(cw, cs, cg)
        stats.valid += int(ok)
        if ok or not constrained:
            return cw, cs, cg
    return w.copy(), s.copy(), g.copy()


def policy_eval_fn(policy, K):
    """fitness function backed by the learned solver (stochastic sampling)."""
    from .rollout import run_policy

    def f(levels, seed):
        n = len(levels["walls"])
        rep = {k: np.repeat(v, K, axis=0) for k, v in levels.items()}
        o = run_policy(policy, rep, greedy=False, seed=seed)
        return _score(levels, o, n, K)
    return f


def scripted_eval_fn(algo, K):
    def f(levels, seed):
        n = len(levels["walls"])
        rep = {k: np.repeat(v, K, axis=0) for k, v in levels.items()}
        o = run_batch(algo, rep, seed=seed)
        return _score(levels, o, n, K)
    return f


def _score(levels, o, n, K, T=MAX_STEPS):
    """Regret vs the BFS oracle in env-return units: R(oracle) - R(solver), R = 1 - 0.9 t/T
    on success, 0 on failure. Failing scores ~0.9, a wasted step scores 0.9/T."""
    opt = np.array([max(bfs_dist(w, s[0], s[1])[g[0], g[1]], 0)
                    for w, s, g in zip(levels["walls"], levels["start"], levels["goal"])], float)
    succ = o["success"].reshape(n, K)
    steps = o["steps"].reshape(n, K).astype(float)
    r_oracle = np.where(opt > 0, 1 - 0.9 * opt / T, 0.0)[:, None]
    r_solver = np.where(succ, 1 - 0.9 * steps / T, 0.0)
    regret = np.maximum(r_oracle - r_solver, 0.0)
    spl = succ * opt[:, None] / np.maximum(steps, np.maximum(opt[:, None], 1))
    solv = opt > 0
    # an unsolvable level (only possible without the constraint) gets the maximum score
    fitness = np.where(solv, regret.mean(1), 1.0)
    return dict(fitness=fitness, fail=1 - succ.mean(1), spl=spl.mean(1), solvable=solv)


def evolve(eval_fn, init_levels, gens=200, pop=128, elite=16, tourn=4, seed=0,
           constrained=True, select=True, archive_thresh=0.5, log=None, verbose=True):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(init_levels["walls"]), pop, replace=False)
    P = {k: init_levels[k][idx].copy() for k in ("walls", "start", "goal")}
    stats = Stats()
    archive = {}
    history = []
    t0 = time.time()
    for gen in range(gens + 1):
        sc = eval_fn(P, seed=seed * 100000 + gen)
        fit = sc["fitness"]
        # archive (keyed by layout)
        for i in np.nonzero(fit >= archive_thresh)[0]:
            key = P["walls"][i].tobytes() + P["start"][i].tobytes() + P["goal"][i].tobytes()
            if key not in archive:
                archive[key] = (P["walls"][i].copy(), P["start"][i].copy(), P["goal"][i].copy(),
                                gen, float(fit[i]))
        st = [level_stats(w, s, g) for w, s, g in zip(P["walls"], P["start"], P["goal"])]
        rec = dict(gen=gen, fit_mean=float(fit.mean()), fit_max=float(fit.max()),
                   fail_mean=float(sc["fail"].mean()), solvable=float(sc["solvable"].mean()),
                   valid_proposal_rate=stats.valid / max(1, stats.proposed), archive=len(archive),
                   wall=time.time() - t0,
                   **{k: float(np.mean([r[k] for r in st])) for k in ("path_len", "dead_ends", "loops", "open2x2", "floor")})
        history.append(rec)
        if log is not None:
            log.write(json.dumps(rec) + "\n"); log.flush()
        if verbose and (gen % 10 == 0 or gen == gens):
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if gen == gens:
            break
        # next generation
        order = np.argsort(-fit)
        elite = min(elite, pop // 4)
        newP = {k: np.empty_like(v) for k, v in P.items()}
        if select:
            for j in range(elite):
                for k in P:
                    newP[k][j] = P[k][order[j]]
            start_j = elite
        else:
            start_j = 0
        for j in range(start_j, pop):
            if select:
                cand = rng.integers(pop, size=tourn)
                p = cand[np.argmax(fit[cand])]
            else:
                p = j  # neutral drift: every individual mutates in place, no selection
            cw, cs, cg = make_child(P["walls"][p], P["start"][p], P["goal"][p], rng, constrained, stats)
            newP["walls"][j], newP["start"][j], newP["goal"][j] = cw, cs, cg
        P = newP
    arch = list(archive.values())
    A = dict(walls=np.array([a[0] for a in arch], np.uint8).reshape(-1, SIZE, SIZE),
             start=np.array([a[1] for a in arch], np.int64).reshape(-1, 2),
             goal=np.array([a[2] for a in arch], np.int64).reshape(-1, 2),
             gen=np.array([a[3] for a in arch], np.int64), fit=np.array([a[4] for a in arch], float))
    return P, fit, A, history, stats
