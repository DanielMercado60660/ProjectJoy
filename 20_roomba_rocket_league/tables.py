"""Print markdown tables from eval json: ../.venv/bin/python tables.py results/eval.json"""
import json, sys
t = json.load(open(sys.argv[1]))
cols = [("goal", "goal"), ("conceded", "conc."), ("own_goal", "own goal"), ("timeout", "timeout"),
        ("first_touch_learner", "1st touch"), ("touch_share", "touch share"), ("possession", "closer to ball"), ("intercepts_per_match", "regains/match"),
        ("time_to_goal_s", "t→goal s"), ("pin_frac", "pin frac"), ("pin_matches", "pin ≥2s")]
print("| policy | opponent | starts | " + " | ".join(c[1] for c in cols) + " |")
print("|---|---|---|" + "---|" * len(cols))
for k, m in t.items():
    p, o, s = k.split("|")
    if p == "random" and len(sys.argv) < 3:
        continue
    vals = []
    for c, _ in cols:
        v = m[c]
        vals.append("–" if v is None else (f"{v:.2f}" if c in ("intercepts_per_match", "time_to_goal_s") else f"{v:.3f}"))
    print(f"| {p} | {o} | {s} | " + " | ".join(vals) + " |")
