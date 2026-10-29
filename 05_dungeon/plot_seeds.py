"""Per-seed curriculum vs control over rounds 1-3 (from results/eval_seeds.json).

  ../.venv/bin/python plot_seeds.py     # -> media/curriculum_rounds_seeds.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ["", "_r11", "_r12", "_r13", "_r14", "_r15"]
SETS = [("designer_v1_s1", "designer vs v1 (v1_s1)"), ("designer_v1_md12_s7", "designer vs v1, exit ≥ 12"),
        ("designer_ctl1_s2", "ctl1 limit-cycle trap"), ("spiral", "spiral (held-out family)")]
ARMS = [("cur", "curriculum", "#2a7fd4"), ("ctl", "control", "#d4762a")]

res = json.load(open(os.path.join(HERE, "results", "eval_seeds.json")))["results"]
fig, axes = plt.subplots(1, len(SETS), figsize=(4 * len(SETS), 3.6), sharex=True)
for ax, (key, title) in zip(axes, SETS):
    for arm, label, col in ARMS:
        Y = np.array([[res[key][f"solver_{arm}{r}{s}"]["success"] for r in (1, 2, 3)] for s in SEEDS])
        for i, y in enumerate(Y):
            ax.plot([1, 2, 3], y, color=col, alpha=0.35, lw=1, marker="o", ms=3,
                    ls="--" if i == 0 else "-")
        ax.plot([1, 2, 3], Y.mean(0), color=col, lw=3, label=f"{label} (mean of 6)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("round")
    ax.set_ylim(0.85 if key == "spiral" else 0, 1.0)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("success rate")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Curriculum vs control, 6 training-seed chains per arm (thin lines; dashed = seed 0)", fontsize=11)
fig.tight_layout()
out = os.path.join(HERE, "media", "curriculum_rounds_seeds.png")
fig.savefig(out, dpi=110)
print(out)
