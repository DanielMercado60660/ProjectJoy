"""Print markdown tables from results/eval.json (pasted into RESULTS.md).

  ../.venv/bin/python tables.py [results/eval.json]
"""
import json
import sys

import numpy as np

E = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/eval.json"))
C = E["cells"]
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731


def main_table(theta, title):
    rows = [c for c in C if c["theta"] == theta and c["fric"] is None and c["mass"] is None]
    if not rows:
        return
    print(f"\n**{title}** (n={rows[0]['n']} episodes per method)\n")
    print("| method | success | success within budget | ever reached | drop | final err median | "
          "peak force median / p90 / max (N) | episodes over 10 N |")
    print("|---|---|---|---|---|---|---|---|")
    for c in rows:
        print(f"| {c['method']} | {pct(c['success'])} ± {pct(c['success_se'])} | {pct(c['success_within_budget'])} | "
              f"{pct(c['ever_success'])} | {pct(c['drop'])} | {np.degrees(c['final_err_med']):.1f}° "
              f"(start {np.degrees(c['err0_med']):.1f}°) | {c['peak_force_med']:.1f} / {c['peak_force_p90']:.1f} / "
              f"{c['peak_force_max']:.0f} | {pct(c['over_budget'])} |")


def rob_table():
    rows = [c for c in C if c["fric"] is not None]
    if not rows:
        return
    methods = list(dict.fromkeys(c["method"] for c in rows))
    conds = list(dict.fromkeys((c["fric"], c["mass"]) for c in rows))
    print(f"\n**Robustness** (θ∈[0.15,0.4] rad, n={rows[0]['n']} per cell; training range was friction and "
          "mass scale 0.7-1.3). Cell: success / success within budget / drop\n")
    print("| friction, mass | " + " | ".join(methods) + " |")
    print("|---|" + "---|" * len(methods))
    for fr, ms in conds:
        cells = []
        for m in methods:
            c = next(c for c in rows if c["method"] == m and c["fric"] == fr and c["mass"] == ms)
            cells.append(f"{pct(c['success'])} / {pct(c['success_within_budget'])} / {pct(c['drop'])}")
        print(f"| {fr}, {ms} | " + " | ".join(cells) + " |")


main_table([0.15, 0.4], "Main: held-out, θ∈[0.15, 0.4] rad (8.6-23°), training friction/mass distribution")
main_table([0.0, 0.4], "Including near-zero targets, θ∈[0, 0.4] rad")
main_table([0.4, 0.8], "Wider targets, θ∈[0.4, 0.8] rad (23-46°)")
main_table([0.8, float(np.pi / 2)], "Widest targets, θ∈[0.8, π/2] rad (46-90°)")
rob_table()
print(f"\nPPO checkpoint: {E['ckpt']}, iter {E['ppo_meta_state']['iter']}, env steps "
      f"{E['ppo_meta_state']['env_steps']}, final policy std {E['ppo_final_std']:.3f} (eval uses the mean action)")
