"""Reviewer check (not part of the builder pipeline).

1. Baseline fairness: give longest-queue-first and max-pressure (and the guard variant) a minimum
   green of M decisions (10*M s), or a switching hysteresis of h vehicles; tune M / h on the same training demands and score used for the other
   baselines (never the test seeds), then evaluate on the same held-out seeds as evaluate.py.
2. Censoring: the builder's mean wait counts only vehicles that crossed a stop line. Recompute it
   with every vehicle still queued at the end of the hour counted at its wait so far.

Usage: ../.venv/bin/python review_check.py  ->  runs/review_check.json
"""
import json, time
import numpy as np

import sim
from sim import run_episode, demand, K, DT, T_STEPS, CAP
import baselines as bl
from evaluate import pattern_rates, score, TUNE_SEED, TEST_SEED, PATTERNS


def with_min_green(base, m_dec):
    """Hold the current phase until it has been green for m_dec decisions (incl. the all-red)."""
    def ctrl(env):
        p = base(env)
        hold = env.since_switch < m_dec * K
        return np.where(hold, env.phase, p)
    return ctrl


def with_hysteresis(use_downstream, h):
    """LQF / max-pressure that switches only when the other phase's score beats the current one by > h."""
    def ctrl(env):
        ns, ew = bl._phase_scores(env, use_downstream)
        cur = np.where(env.phase == 1, ew, ns); oth = np.where(env.phase == 1, ns, ew)
        return np.where(oth > cur + h, 1 - env.phase, env.phase)
    return ctrl


def censored_mean_wait(env):
    """Mean wait per crossing, counting vehicles still at a stop line at the end with wait so far."""
    B, L = env.cnt.shape
    extra_w = np.zeros(B); extra_n = np.zeros(B)
    for b in range(B):
        for l in range(L):
            n = env.cnt[b, l]
            if n == 0:
                continue
            idx = (env.head[b, l] + np.arange(n)) % CAP
            rt = env.buf[b, l, idx]
            w = (env.t - rt[rt <= env.t]) * DT
            extra_w[b] += w.sum(); extra_n[b] += len(w)
    dep = env.dep.sum(1)
    return (env.wsum.sum(1) + extra_w) / np.maximum(dep + extra_n, 1), extra_n


def main():
    t0 = time.time()
    rng = np.random.default_rng(TUNE_SEED)
    dem = np.stack([demand("train", rng) for _ in range(8)])
    tuned = {}
    bases = {"longest_queue_first": bl.longest_queue_first, "max_pressure": bl.max_pressure,
             "max_pressure + 30s guard": bl.max_pressure_guard(30)}
    tune_log = {}
    for name, base in bases.items():
        sc = {}
        for m in [1, 2, 3, 4, 5, 6, 8]:
            sc[m] = score(run_episode(with_min_green(base, m), dem, TUNE_SEED).metrics())
        best = min(sc, key=sc.get)
        tuned[name] = (best, with_min_green(base, best))
        tune_log[name] = dict(scores=sc, best_min_green_decisions=best)
        print(name, "min-green tuning", {k: round(v, 2) for k, v in sc.items()}, "->", best, flush=True)

    for name, ud in [("longest_queue_first", False), ("max_pressure", True)]:
        sc = {}
        for h in [0, 1, 2, 3, 4, 6, 8, 12]:
            sc[h] = score(run_episode(with_hysteresis(ud, h), dem, TUNE_SEED).metrics())
        best = min(sc, key=sc.get)
        tune_log[name + " hysteresis"] = dict(scores=sc, best_h=best)
        print(name, "hysteresis tuning", {k: round(v, 2) for k, v in sc.items()}, "->", best, flush=True)
        tuned[name + " hyst"] = (best, with_hysteresis(ud, best))

    ctrls = {}
    plan_s = [(4, 2), (4, 2), (4, 2), (4, 2)]                 # from runs/eval_results.json
    plan_m = [(6, 3), (7, 2), (6, 3), (7, 2)]
    ctrls["fixed_time (tuned)"] = bl.fixed_time(plan_s)
    ctrls["fixed_time (tuned for mean wait)"] = bl.fixed_time(plan_m)
    for name, base in bases.items():
        ctrls[name] = base
        if tuned[name][0] > 1:      # M = 1 never holds (since_switch is already K at the next decision)
            ctrls[f"{name} + min green {tuned[name][0] * 10}s"] = tuned[name][1]
        if name + " hyst" in tuned:
            ctrls[f"{name} + hysteresis {tuned[name + ' hyst'][0]} veh"] = tuned[name + " hyst"][1]
    from policy import load_policy, greedy_controller
    model, _ = load_policy("checkpoints/ppo_best")
    ctrls["PPO shared policy"] = greedy_controller(model)

    out = dict(tuning=tune_log, patterns={})
    for pat in PATTERNS:
        R = pattern_rates(pat, 20)
        out["patterns"][pat] = {}
        for name, c in ctrls.items():
            env = run_episode(c, R, TEST_SEED + PATTERNS.index(pat), record=True)
            m = env.metrics()
            cm, n_wait = censored_mean_wait(env)
            r = {k: float(np.mean(m[k])) for k in ["throughput", "mean_wait", "p95_wait", "max_wait",
                                                    "quiet_max_wait", "worst_approach_mean_wait",
                                                    "in_network_end", "switches_per_int"]}
            r["mean_wait_incl_unfinished"] = float(cm.mean())
            r["unfinished_waiting_end"] = float(n_wait.mean())
            out["patterns"][pat][name] = r
            print(pat, name, {k: round(v, 1) for k, v in r.items()}, flush=True)
    out["seconds"] = time.time() - t0
    with open("runs/review_check.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
