"""Seed tables for the grip-randomized runs: results/dr_seeds/eval.json (+ selection.json)
-> results/dr_seeds/seeds_table.md. Seed x grip completion and mean lap time under both
checkpoint-selection rules, with mean and range over seeds.

    ../.venv/bin/python dr_seeds_table.py
"""
import json

import numpy as np

D = "results/dr_seeds"
GRIPS = [1.0, 0.8, 0.7, 0.6, 0.5]
# seed label -> controller name, per family and selection rule. Seed 0 under the grip-1.0 rule is the
# original checkpoint; under the randomized-grip rule it comes from the bit-identical seed-0 rerun (r0).
FAMILIES = {
    "PPO + grip rand.": {"g1": [("0", "ppo_dr"), ("1", "ppo_dr_s1"), ("2", "ppo_dr_s2")],
                         "rg": [("0", "ppo_dr_r0_rg"), ("1", "ppo_dr_s1_rg"), ("2", "ppo_dr_s2_rg")]},
    "ES + grip rand.": {"g1": [("0", "es_dr"), ("1", "es_dr_s1"), ("2", "es_dr_s2")],
                        "rg": [("0", "es_dr_r0_rg"), ("1", "es_dr_s1_rg"), ("2", "es_dr_s2_rg")]},
}
RULE = {"g1": "selected on grip-1.0 validation (original rule)",
        "rg": "selected on randomized-grip validation (μ ~ U(0.5, 1.1))"}


def main():
    rows = json.load(open(f"{D}/eval.json"))
    have = {r["controller"] for r in rows}
    try:
        sel = json.load(open(f"{D}/selection.json"))
    except FileNotFoundError:
        sel = {"runs": {}}

    def cell(c, split, g):
        r = [x for x in rows if x["controller"] == c and x["split"] == split and x["grip"] == g and x["mode"] == "strict"]
        return r[0] if r else None

    L = ["# Grip-randomized runs: seeds x selection rule (strict rules, 1 lap)\n",
         "Held-out: 100 unseen tracks; hard: 50 narrower/twistier OOD tracks. Cells: completion % (mean lap time "
         "of finished laps). Mean/range are over the seeds listed.\n"]
    for fam, rules in FAMILIES.items():
        for rule, seeds in rules.items():
            seeds = [(s, c) for s, c in seeds if c in have]
            if not seeds:
                continue
            L.append(f"\n## {fam}, {RULE[rule]}\n")
            L.append("| seed | ckpt (env steps) | " + " | ".join(f"held-out {g}" for g in GRIPS) + " | "
                     + " | ".join(f"hard {g}" for g in GRIPS) + " |")
            L.append("|---" * (2 + 2 * len(GRIPS)) + "|")
            acc = {k: [] for k in ("h", "hl", "d")}
            for s, c in seeds:
                tag = c[:-3] if c.endswith("_rg") else c
                info = sel["runs"].get(tag, {})
                b = info.get("best_rg" if rule == "rg" else "best_g1")
                steps = f"{b['env_steps'] / 1e6:.1f} M" if b else ("4.6 M" if c == "ppo_dr" else
                                                                   "304 M" if c == "es_dr" else "?")
                h = [cell(c, "heldout", g) for g in GRIPS]
                d = [cell(c, "hard", g) for g in GRIPS]
                acc["h"].append([x["completion"] * 100 for x in h])
                acc["hl"].append([x["mean_lap_time"] for x in h])
                acc["d"].append([x["completion"] * 100 for x in d])
                L.append(f"| {s} (`{c}`) | {steps} | "
                         + " | ".join(f"{x['completion']:.0%} ({x['mean_lap_time']:.1f} s)" for x in h) + " | "
                         + " | ".join(f"{x['completion']:.0%}" for x in d) + " |")
            h, hl, d = (np.array(acc[k], float) for k in ("h", "hl", "d"))
            L.append(f"| **mean** | | " + " | ".join(f"{a:.0f}% ({b:.1f} s)" for a, b in zip(h.mean(0), np.nanmean(hl, 0)))
                     + " | " + " | ".join(f"{a:.0f}%" for a in d.mean(0)) + " |")
            L.append(f"| **range** | | " + " | ".join(f"{a:.0f}–{b:.0f}%" for a, b in zip(h.min(0), h.max(0)))
                     + " | " + " | ".join(f"{a:.0f}–{b:.0f}%" for a, b in zip(d.min(0), d.max(0))) + " |")
    if "ray_dr" in have:
        L.append("\n## Reference: ray follower tuned with grip rand. (deterministic, one tuning run)\n")
        L.append("| | " + " | ".join(f"held-out {g}" for g in GRIPS) + " | " + " | ".join(f"hard {g}" for g in GRIPS) + " |")
        L.append("|---" * (1 + 2 * len(GRIPS)) + "|")
        L.append("| ray_dr | " + " | ".join(f"{cell('ray_dr', 'heldout', g)['completion']:.0%} "
                                          f"({cell('ray_dr', 'heldout', g)['mean_lap_time']:.1f} s)" for g in GRIPS)
                 + " | " + " | ".join(f"{cell('ray_dr', 'hard', g)['completion']:.0%}" for g in GRIPS) + " |")
    if sel["runs"]:
        L.append("\n## Checkpoint selection (validation tracks 5000–5127 only)\n")
        L.append("val_g1 = trainers' grip-1.0 validation score; val_rg = randomized-grip validation score "
                 "(mean 60 s fitness, m). " + json.dumps(sel.get("protocol", {})) + "\n")
        L.append("| run | snapshots | grip-1.0 rule: step (env steps), val_g1 / val_rg | randomized rule: step (env steps), "
                 "val_g1 / val_rg | g1 argmax = trainer's best file |")
        L.append("|---|---|---|---|---|")
        for tag, r in sel["runs"].items():
            b1, br = r["best_g1"], r["best_rg"]
            L.append(f"| {tag} | {r['n_snapshots']} | {r['step_key']} {b1['step']} ({b1['env_steps'] / 1e6:.1f} M), "
                     f"{b1['val_g1']:.0f} / {b1['val_rg']:.0f} | {r['step_key']} {br['step']} ({br['env_steps'] / 1e6:.1f} M), "
                     f"{br['val_g1']:.0f} / {br['val_rg']:.0f} | {r['g1_argmax_matches_trainer_best']} |")
        for name, e in sel.get("extra", {}).items():
            L.append(f"| {name} (original file) | – | val_g1 {e['val_g1']:.0f} / val_rg {e['val_rg']:.0f} | | |")
    open(f"{D}/seeds_table.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
