"""Round 6 baseline: CEM-tune the scripted attacker against the scripted part of the TRAINING pool.

    ../bin/gpu_run ../.venv/bin/python r6_tune_attacker.py --minutes 35 --out results/r6_tuned_attacker.json

Search space: r6_opponents.ATT_BOUNDS (standoff, off-centre aim, alignment tolerance, push-through distance,
detour margin, steering gain, reverse allowed). Each generation: `pop` candidates x `m` matches each, every
match against an opponent drawn from the training mix restricted to its scripted kinds (same relative
weights as r6_train.DEFAULT_MIX; the self-play kinds are learner snapshots and do not exist for a scripted
attacker), with the same per-episode randomization (speed, reaction delay, action noise). Fitness = goal rate
minus conceded+own-goal rate. Fresh starts every generation (seeds 3001+). The held-out opponents are never
touched. Budget: same wall-clock cap as one PPO seed; env steps are counted and reported.
"""
import argparse, json, os, time
import numpy as np
from env import T_MAX, OUT_GOAL, OUT_CONCEDED, OUT_OWN_GOAL
import r6_opponents as ro
from r6_common import R6Arena, KINDS, pool_actions
from r6_train import DEFAULT_MIX

NAMES = list(ro.ATT_DEFAULT)
LO = np.array([ro.ATT_BOUNDS[k][0] for k in NAMES])
HI = np.array([ro.ATT_BOUNDS[k][1] for k in NAMES])


def scripted_mix():
    m = np.array([DEFAULT_MIX.get(k, 0.0) if not k.startswith("self") else 0.0 for k in KINDS])
    return m / m.sum()


def evaluate(params, m, seed):
    """params: (pop, d) -> fitness (pop,), goal (pop,), conc (pop,), env steps used."""
    pop = len(params)
    n = pop * m
    env = R6Arena(n, seed=seed, mix=scripted_mix())
    env.curriculum = 1.0
    env.reset(np.ones(n, dtype=bool))
    cand = np.repeat(np.arange(pop), m)
    P = {k: params[cand, i] for i, k in enumerate(NAMES)}
    orng = np.random.default_rng(seed + 5)
    live = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=np.int64)
    used = 0
    for _ in range(T_MAX):
        a0 = ro.attacker_param(env, P)
        a1 = pool_actions(env, [], None, orng)
        _, d, tr, oc, _ = env.step(a0, a1)
        used += int(live.sum())
        fin = (d | tr) & live
        out[fin] = oc[fin]
        live &= ~fin
        env.bvel[fin] = 0
        if not live.any():
            break
    g = (out == OUT_GOAL).reshape(pop, m).mean(1)
    c = ((out == OUT_CONCEDED) | (out == OUT_OWN_GOAL)).reshape(pop, m).mean(1)
    return g - c, g, c, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--m", type=int, default=96)
    ap.add_argument("--elite", type=float, default=0.25)
    ap.add_argument("--max-gens", type=int, default=400)
    ap.add_argument("--final-m", type=int, default=2000)
    ap.add_argument("--out", default="results/r6_tuned_attacker.json")
    args = ap.parse_args()
    rng = np.random.default_rng(3000)
    d = len(NAMES)
    x0 = np.array([ro.ATT_DEFAULT[k] for k in NAMES])
    mu = (x0 - LO) / (HI - LO)                     # search in the unit cube, start at the hand-written attacker
    sd = np.full(d, 0.3)
    t0 = time.time()
    steps = 0
    hist = []
    gen = 0
    while gen < args.max_gens and time.time() - t0 < args.minutes * 60 - 120:
        z = np.clip(mu + sd * rng.standard_normal((args.pop, d)), 0, 1)
        z[0] = mu                                   # always re-evaluate the current mean
        f, g, c, used = evaluate(LO + z * (HI - LO), args.m, 3001 + gen)
        steps += used
        el = z[np.argsort(-f)[: max(2, int(args.elite * args.pop))]]
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sd = np.maximum(0.7 * el.std(0) + 0.3 * sd, 0.02)
        hist.append(dict(gen=gen, mean_fit=float(f[0]), mean_goal=float(g[0]), best_fit=float(f.max()),
                         steps=steps, wall=round(time.time() - t0, 1)))
        if gen % 5 == 0:
            print(json.dumps(hist[-1]), flush=True)
        gen += 1
    best = {k: float(v) for k, v in zip(NAMES, LO + mu * (HI - LO))}
    # final check on the tuning pool (fresh seed), default vs tuned
    fd, gd, cd, _ = evaluate(np.array([x0]), args.final_m, 3999)
    ft, gt, ct, _ = evaluate(np.array([LO + mu * (HI - LO)]), args.final_m, 3999)
    res = dict(best=best, default=dict(zip(NAMES, map(float, x0))), gens=gen, env_steps=steps,
               wall_s=time.time() - t0, pool_check=dict(default=dict(goal=float(gd[0]), conc_own=float(cd[0])),
                                                         tuned=dict(goal=float(gt[0]), conc_own=float(ct[0]))),
               history=hist)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("best", best, "pool check", res["pool_check"], "steps", steps, flush=True)


if __name__ == "__main__":
    main()
