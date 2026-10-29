"""Tune the baselines on training-distribution demand, then evaluate every controller on held-out
arrival seeds (steady, rush hour, and a lopsided 'hell hour'). Writes runs/eval_results.json,
runs/eval_table.md and runs/demo_traces.npz (one recorded rush-hour episode per controller).

Usage: ../.venv/bin/python evaluate.py [--ckpt checkpoints/ppo_best] [--episodes 20] [--quick]
"""
import argparse, itertools, json, os, time
import numpy as np

import sim
from sim import Grid, demand, run_episode, N_INT
import baselines as bl

TEST_SEED = 10_000          # arrival seed for held-out evaluation (training uses its own RNG stream)
TUNE_SEED = 777             # demand + arrival seed for baseline tuning (training distribution)
PATTERNS = ["steady", "rush", "hell"]


def pattern_rates(name, n):
    if name == "hell":
        # Rush hour at the top of the training range, lopsided towards eastbound traffic.
        R = demand("steady").copy()
        art = sim.LANE_PHASE == 1
        t = np.arange(sim.T_STEPS)
        b = 1.0 + 1.1 * np.exp(-0.5 * ((t - 900) / 300.0) ** 2)
        R[:, art] *= b[:, None]
        elm = (sim.LANE_PHASE == 0) & (sim.LANE_INT % 2 == 0)
        R[:, elm] *= (1.0 + 0.35 * (b - 1.0))[:, None]
        for l in sim.EDGE_LANES:
            if sim.LANE_PHASE[l] == 1:
                R[:, l] *= 1.3 if l % 4 == 2 else 0.7
    else:
        R = demand(name)
    return np.repeat(R[None], n, 0)


def score(m):
    """Selection score shared by the PPO checkpoint and the tuned baselines (seconds)."""
    return float(m["mean_wait"].mean() + 0.1 * m["max_wait"].mean())


def tune_fixed_time(n_dem=8, chunk=512, quick=False):
    rng = np.random.default_rng(TUNE_SEED)
    dem = np.stack([demand("train", rng) for _ in range(n_dem)])
    ews = range(1, 9) if not quick else range(2, 5)
    nss = range(1, 7) if not quick else range(1, 3)
    plans = list(itertools.product(ews, nss, ews, nss))      # (ew col0, ns col0, ew col1, ns col1)
    jobs = [(p, d) for p in plans for d in range(n_dem)]
    mw = np.zeros(len(plans)); mx_ = np.zeros(len(plans))
    for k in range(0, len(jobs), chunk):
        js = jobs[k:k + chunk]
        P = np.array([j[0] for j in js]); D = np.array([j[1] for j in js])
        ew = np.stack([P[:, 0], P[:, 2], P[:, 0], P[:, 2]], 1)   # I0,I2 use col-0 plan; I1,I3 col-1
        ns = np.stack([P[:, 1], P[:, 3], P[:, 1], P[:, 3]], 1)
        env = run_episode(bl.fixed_time_batch(ew, ns), dem[D], TUNE_SEED + k)
        m = env.metrics()
        idx = (k + np.arange(len(js))) // n_dem        # jobs are plan-major, n_dem demands per plan
        np.add.at(mw, idx, m["mean_wait"] / n_dem)
        np.add.at(mx_, idx, m["max_wait"] / n_dem)
    sc = mw + 0.1 * mx_
    best_score = plans[int(np.argmin(sc))]
    best_mean = plans[int(np.argmin(mw))]
    to_plan = lambda p: [(p[0], p[1]), (p[2], p[3]), (p[0], p[1]), (p[2], p[3])]
    info = dict(n_plans=len(plans), n_demands=n_dem,
                best_by_score=dict(plan=best_score, train_mean_wait=float(mw[plans.index(best_score)]),
                                   train_max_wait=float(mx_[plans.index(best_score)])),
                best_by_mean_wait=dict(plan=best_mean, train_mean_wait=float(mw[plans.index(best_mean)]),
                                       train_max_wait=float(mx_[plans.index(best_mean)])))
    return to_plan(best_score), to_plan(best_mean), info


def tune_guard(n_dem=8):
    rng = np.random.default_rng(TUNE_SEED)
    dem = np.stack([demand("train", rng) for _ in range(n_dem)])
    res = {}
    for thr in [30, 45, 60, 90, 120, 180]:
        m = run_episode(bl.max_pressure_guard(thr), dem, TUNE_SEED).metrics()
        res[thr] = score(m)
    best = min(res, key=res.get)
    return best, res


def summarize(m):
    keys = ["throughput", "mean_wait", "p95_wait", "max_wait", "quiet_mean_wait", "quiet_max_wait",
            "quiet_stuck_end", "worst_approach_mean_wait", "jain_fairness", "spillback_blocked_steps",
            "link_full_frac", "in_network_end", "rejected", "switches_per_int"]
    out = {}
    for k in keys:
        v = np.asarray(m[k], float)
        out[k] = dict(mean=float(v.mean()), std=float(v.std()), worst=float(v.max() if k != "jain_fairness" else v.min()))
    out["lane_mean_wait"] = np.asarray(m["lane_mean_wait"]).mean(0).round(2).tolist()
    out["lane_max_wait"] = np.asarray(m["lane_max_wait"]).max(0).round(1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo_best")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny tuning grid (smoke test)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    plan_s, plan_m, ft_info = tune_fixed_time(n_dem=2 if a.quick else 8, quick=a.quick)
    thr, guard_info = tune_guard(n_dem=2 if a.quick else 8)
    print(f"fixed-time tuned in {time.time() - t0:.0f}s: {ft_info}; guard threshold {thr}s {guard_info}", flush=True)

    ctrls = {
        "fixed_time (tuned)": bl.fixed_time(plan_s),
        "fixed_time (tuned for mean wait)": bl.fixed_time(plan_m),
        "longest_queue_first": bl.longest_queue_first,
        "max_pressure": bl.max_pressure,
        f"max_pressure + {thr}s guard": bl.max_pressure_guard(thr),
    }
    if os.path.exists(a.ckpt + ".npz"):
        from policy import load_policy, greedy_controller
        model, meta = load_policy(a.ckpt)
        ctrls["PPO shared policy"] = greedy_controller(model)
        ppo_meta = dict(ckpt=a.ckpt, step=meta["step"], val=meta.get("val"))
    else:
        print(f"no checkpoint at {a.ckpt}, skipping PPO")
        ppo_meta = None

    results = dict(tuning=dict(fixed_time=ft_info, guard=dict(best=thr, scores=guard_info)),
                   ppo=ppo_meta, episodes=a.episodes, test_seed=TEST_SEED, patterns={})
    traces = {}
    for pat in PATTERNS:
        R = pattern_rates(pat, a.episodes)
        results["patterns"][pat] = {}
        for name, c in ctrls.items():
            env = run_episode(c, R, TEST_SEED + PATTERNS.index(pat), record=True)
            results["patterns"][pat][name] = summarize(env.metrics())
            if pat == "rush":
                traces[name] = dict(q=np.array(env.trace_q)[:, 0], c=np.array(env.trace_c)[:, 0],
                                    p=np.array(env.trace_p)[:, 0], w=np.array(env.trace_w)[:, 0].astype(np.float32))
            print(pat, name, {k: round(v["mean"], 2) for k, v in results["patterns"][pat][name].items()
                              if isinstance(v, dict)}, flush=True)
    results["eval_seconds"] = time.time() - t0
    with open(os.path.join(a.out, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    names = list(ctrls)
    np.savez_compressed(os.path.join(a.out, "demo_traces.npz"), names=np.array(names),
                        **{f"{k}_{i}": traces[n][k] for i, n in enumerate(names) for k in ("q", "c", "p", "w")})
    # markdown table
    cols = [("throughput", "veh/h"), ("mean_wait", "mean wait s"), ("p95_wait", "p95 s"), ("max_wait", "max wait s"),
            ("quiet_mean_wait", "quiet mean s"), ("quiet_max_wait", "quiet max s"),
            ("worst_approach_mean_wait", "worst approach mean s"), ("jain_fairness", "Jain"),
            ("spillback_blocked_steps", "spillback blocked steps"), ("switches_per_int", "switches/int")]
    lines = []
    for pat in PATTERNS:
        lines.append(f"\n### {pat} ({a.episodes} held-out arrival seeds, mean ± std across episodes)\n")
        lines.append("| controller | " + " | ".join(c[1] for c in cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for n in names:
            r = results["patterns"][pat][n]
            cells = []
            for k, _ in cols:
                fmt = "{:.3f}" if k == "jain_fairness" else ("{:.0f}" if k in ("throughput", "spillback_blocked_steps") else "{:.1f}")
                cells.append(fmt.format(r[k]["mean"]) + " ± " + fmt.format(r[k]["std"]))
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "eval_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"eval done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
