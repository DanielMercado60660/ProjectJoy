"""Collect the held-out evaluation JSONs into markdown tables (results/summary.md)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def pct(x):
    return f"{100 * x[0]:.1f}% [{100 * x[1]:.1f}, {100 * x[2]:.1f}]"


def load(f):
    p = os.path.join(R, f + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


# (label, file, method key)
ROWS = [
    ("open-loop CPG, tuning round 1", "eval_run1_full", "cpg_baseline"),
    ("open-loop CPG, tuning round 2", "eval_cpg2_full", "cpg_baseline"),
    ("CPG + foothold planner, round 1", "eval_run1_full", "cpg_planner_baseline"),
    ("CPG + foothold planner, round 2", "eval_planner2_full", "cpg_planner_baseline"),
    ("CPG + foothold planner, round 3", "eval_planner3_full", "cpg_planner_baseline"),
    ("CPG + foothold planner, round 4", "eval_planner4_full", "cpg_planner_baseline"),
    ("residual PPO, blind (no stone obs)", "eval_blind_full", "residual_ppo"),
    ("residual PPO (full obs)", "eval_run1_full", "residual_ppo"),
]


def main_table(suffix="full", rows=ROWS, push=True):
    out = ["| method | completion | lava | flip | progress | J/m (completed) | m/s (completed) |"
           + (" completion with 40 N push | fail ≤2 s after push |" if push else ""),
           "|---|---|---|---|---|---|---|" + ("---|---|" if push else "")]
    for label, f, k in rows:
        d = load(f.replace("full", suffix))
        if d is None or k not in d:
            continue
        c, p = d[k]["clean"], d[k]["push"]
        line = (f"| {label} | {pct(c['success'])} | {100 * c['lava'][0]:.1f}% | {100 * c['flip'][0]:.1f}% | "
                f"{c['frac'][0]:.2f} | {c['energy_per_m_completed'][0]:.0f} | {c['speed_completed_mps'][0]:.2f} |")
        if push:
            line += (f" {pct(p['success'])} | {pct(p['fail_within_2s_of_push'])} (n={p['n_reached_push']}) |" if p else " not run | not run |")
        out.append(line)
    return "\n".join(out)


def main():
    lines = ["## Full gap range 0.05–0.20 m (500 held-out layouts)", main_table("full"), "",
             "## Easier: gaps 0.05–0.125 m (500 held-out layouts)", main_table("gap0125", push=False), ""]
    # flat push sweep
    lines.append("## Push sweep on a flat walkway (200 held-out walkways, lateral force 0.1 s)")
    lines.append("| method | 40 N | 80 N | 120 N |")
    lines.append("|---|---|---|---|")
    sets = {
        "open-loop CPG r1": [("eval_run1_flat_push", "cpg_baseline"), ("eval_flat_push80", "cpg_baseline"), ("eval_flat_push120", "cpg_baseline")],
        "open-loop CPG r2": [(None, None), ("eval_cpg2_flat_push80", "cpg_baseline"), (None, None)],
        "planner r1": [("eval_run1_flat_push", "cpg_planner_baseline"), ("eval_flat_push80", "cpg_planner_baseline"), ("eval_flat_push120", "cpg_planner_baseline")],
        "planner r4": [("eval_planner4_flat_push40", "cpg_planner_baseline"), ("eval_planner4_flat_push80", "cpg_planner_baseline"), ("eval_planner4_flat_push120", "cpg_planner_baseline")],
        "residual PPO": [("eval_run1_flat_push", "residual_ppo"), ("eval_flat_push80", "residual_ppo"), ("eval_flat_push120", "residual_ppo")],
    }
    for name, lst in sets.items():
        cells = []
        for f, k in lst:
            d = load(f) if f else None
            cells.append(pct(d[k]["push"]["success"]) if d and k in d else "not run")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    # paired comparisons vs PPO
    lines.append("## Paired comparisons (same layouts)")
    for f in ["eval_run1_full", "eval_planner3_full", "eval_planner4_full", "eval_run1_gap0125", "eval_planner4_gap0125"]:
        d = load(f)
        if not d:
            continue
        for key, v in d.items():
            if key.startswith("paired_success_diff"):
                base = key.replace("paired_success_diff_ppo_minus_", "")
                disc = d[f"discordant_vs_{base}"]
                both = d.get(f"both_completed_vs_{base}", {})
                e = both.get("diff_energy_per_m_ppo_minus_base")
                lines.append(f"- `{f}` PPO − {base}: completion {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], "
                             f"PPO-only {disc['ppo_only']}, base-only {disc['base_only']}"
                             + (f"; both completed n={both['n']}: J/m PPO {both['ppo_energy_per_m'][0]:.0f} vs base "
                                f"{both['base_energy_per_m'][0]:.0f} (diff {e[0]:+.0f} [{e[1]:+.0f}, {e[2]:+.0f}]), "
                                f"speed {both['ppo_speed'][0]:.2f} vs {both['base_speed'][0]:.2f} m/s" if e and both['n'] else ""))
    open(os.path.join(R, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
