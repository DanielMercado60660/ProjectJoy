"""Static turn-2 audit figure from saved test rows; no simulation or MLX."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/r5/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env_r5 import GROUPS, BUDGETS


def main():
    data = {seed: json.loads(Path("results/r5", name).read_text()) for seed, name in
            ((51, "test.json"), (52, "test_seed52.json"), (53, "test_seed53.json"))}
    series = [("Event = uncertainty", data[51]["methods"]["event"], "#2166ac", "o"),
              ("Every step", data[51]["methods"]["every"], "#666666", "s")]
    series += [(f"Learned {s}", data[s]["methods"]["learned"], c, m)
               for s, c, m in ((51, "#e66101", "^"), (52, "#7b3294", "x"), (53, "#018571", "D"))]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey="row")
    for col, group in enumerate(GROUPS):
        for name, result, color, marker in series:
            rows = sorted([r for r in result["rows"] if r["group"] == group], key=lambda r: r["budget"])
            assert [r["budget"] for r in rows] == list(BUDGETS)
            for row, key, scale in ((0, "completion", 100), (1, "collisions", 1)):
                axes[row, col].plot(range(4), [r[key]*scale for r in rows], label=name,
                                    color=color, marker=marker, linewidth=1.6, markersize=5)
        axes[0, col].set_title(group.capitalize())
        axes[0, col].set_ylim(-4, 104)
        axes[1, col].set_ylim(-4, 104)
        axes[1, col].set_xticks(range(4), BUDGETS)
        axes[1, col].set_xlabel("Pulse budget")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Completion (%)")
    axes[1, 0].set_ylabel("Collisions / episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(.5, 1.0), frameon=False)
    fig.suptitle("Bat School R5: 128 held-out caves per group, reused across four budgets", y=1.035, fontsize=13)
    fig.text(.5, -.005, "Low collisions alone do not imply useful navigation: seed 52 waits silently to timeout in every episode.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .02, 1, .94))
    out = "media/r5_turn2_budget_sweep.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
