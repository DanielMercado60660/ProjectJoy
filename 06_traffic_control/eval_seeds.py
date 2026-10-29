"""Follow-up (round 3): evaluate every PPO training seed on the same held-out seeds as evaluate.py,
against the reviewer-tuned rules (LQF + 20 s min green, max-pressure + 2-veh hysteresis), the
two tuned fixed-time plans, and the builder's free-switching LQF / max-pressure / guard (the best
rule in steady traffic is free-switching LQF). Tuned values are taken as-is from runs/eval_results.json and
runs/review_check.json (they were tuned on training demand only; nothing is re-tuned here).

Usage: ../.venv/bin/python eval_seeds.py  ->  runs/seeds/eval_seeds.json, runs/seeds/eval_seeds.md
"""
import json, os, time
import numpy as np

from sim import run_episode
import baselines as bl
from evaluate import pattern_rates, TEST_SEED, PATTERNS
from review_check import with_min_green, with_hysteresis, censored_mean_wait
from policy import load_policy, greedy_controller

EPISODES = 20
KEYS = ["throughput", "mean_wait", "p95_wait", "max_wait", "quiet_max_wait", "worst_approach_mean_wait",
        "in_network_end", "switches_per_int"]

PPO = {  # name -> checkpoint; seed 0 = the original runs
    "PPO b=0.5 s0": "checkpoints/ppo_best",
    "PPO b=0.5 s1": "checkpoints/seeds/ppo_s1_best",
    "PPO b=0.5 s2": "checkpoints/seeds/ppo_s2_best",
    "PPO b=0.5 s3": "checkpoints/seeds/ppo_s3_best",
    "PPO b=0.5 s4": "checkpoints/seeds/ppo_s4_best",
    "PPO b=0.5 s5": "checkpoints/seeds/ppo_s5_best",
    "PPO b=0 s0": "checkpoints/ablation_beta0_best",
    "PPO b=0 s1": "checkpoints/seeds/beta0_s1_best",
    "PPO b=0 s2": "checkpoints/seeds/beta0_s2_best",
    "PPO b=0 s3": "checkpoints/seeds/beta0_s3_best",
}


def main():
    t0 = time.time()
    ev = json.load(open("runs/eval_results.json"))["tuning"]["fixed_time"]
    to_plan = lambda p: [(p[0], p[1]), (p[2], p[3]), (p[0], p[1]), (p[2], p[3])]
    rc = json.load(open("runs/review_check.json"))["tuning"]
    m_lqf = rc["longest_queue_first"]["best_min_green_decisions"]
    h_mp = rc["max_pressure hysteresis"]["best_h"]
    ctrls = {
        "fixed_time (tuned)": bl.fixed_time(to_plan(ev["best_by_score"]["plan"])),
        "fixed_time (tuned for mean wait)": bl.fixed_time(to_plan(ev["best_by_mean_wait"]["plan"])),
        f"LQF + min green {m_lqf * 10}s": with_min_green(bl.longest_queue_first, m_lqf),
        f"max_pressure + hysteresis {h_mp} veh": with_hysteresis(True, h_mp),
        "longest_queue_first (no min green)": bl.longest_queue_first,
        "max_pressure (no min green)": bl.max_pressure,
        "max_pressure + 30s guard": bl.max_pressure_guard(30),
    }
    rules = list(ctrls)
    meta = {}
    for name, ck in PPO.items():
        if os.path.exists(ck + ".npz"):
            model, m = load_policy(ck)
            ctrls[name] = greedy_controller(model)
            meta[name] = dict(ckpt=ck, step=m["step"], val=m.get("val"), val_score=m.get("val_score"))
    out = dict(episodes=EPISODES, test_seed=TEST_SEED, rules=rules, ppo=meta, patterns={})
    for pat in PATTERNS:
        R = pattern_rates(pat, EPISODES)
        out["patterns"][pat] = {}
        for name, c in ctrls.items():
            env = run_episode(c, R, TEST_SEED + PATTERNS.index(pat), record=True)
            mt = env.metrics()
            r = {k: np.asarray(mt[k], float).tolist() for k in KEYS}
            r["mean_wait_incl_unfinished"] = censored_mean_wait(env)[0].tolist()
            out["patterns"][pat][name] = r
            print(pat, name, {k: round(float(np.mean(v)), 1) for k, v in r.items() if isinstance(v, list)}, flush=True)
    out["seconds"] = time.time() - t0
    os.makedirs("runs/seeds", exist_ok=True)
    with open("runs/seeds/eval_seeds.json", "w") as f:
        json.dump(out, f)
    write_table(out)
    print(f"done in {out['seconds']:.0f}s")


def write_table(out):
    cols = [("mean_wait", "mean wait", 1), ("max_wait", "worst wait", 0), ("quiet_max_wait", "Quiet worst", 0),
            ("throughput", "veh/h", 0), ("in_network_end", "cars left", 0), ("mean_wait_incl_unfinished", "mean incl. unfinished", 1)]
    L = []
    for pat, rows in out["patterns"].items():
        L.append(f"\n### {pat}: per controller / training seed (mean ± std over {out['episodes']} held-out episodes)\n")
        L.append("| controller | " + " | ".join(c[1] for c in cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for n, r in rows.items():
            L.append(f"| {n} | " + " | ".join(f"{np.mean(r[k]):.{d}f} ± {np.std(r[k]):.{d}f}" for k, _, d in cols) + " |")
        for fam in ["PPO b=0.5", "PPO b=0"]:
            seeds = [n for n in rows if n.startswith(fam + " s")]
            if len(seeds) < 2:
                continue
            cells = []
            for k, _, d in cols:
                per = np.array([np.mean(rows[n][k]) for n in seeds])
                cells.append(f"**{per.mean():.{d}f}** ± {per.std(ddof=1):.{d}f} [{per.min():.{d}f}–{per.max():.{d}f}]")
            L.append(f"| **{fam}, {len(seeds)} seeds** (mean ± sd across seeds [range]) | " + " | ".join(cells) + " |")
        # paired: each PPO seed vs the best rule on mean wait
        best = min(out["rules"], key=lambda n: np.mean(rows[n]["mean_wait"]))
        L.append(f"\nBest rule on mean wait: **{best}** ({np.mean(rows[best]['mean_wait']):.1f} s). Paired per episode:\n")
        for n in rows:
            if n.startswith("PPO"):
                d = np.array(rows[n]["mean_wait"]) - np.array(rows[best]["mean_wait"])
                L.append(f"- {n}: Δ mean wait {d.mean():+.1f} s (sd {d.std(ddof=1):.1f}), PPO lower in {int((d < 0).sum())}/{len(d)} episodes")
    with open("runs/seeds/eval_seeds.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
