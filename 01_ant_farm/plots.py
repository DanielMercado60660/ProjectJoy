"""Charts from results/eval.json and logs/*.jsonl -> media/*.png

  ../.venv/bin/python plots.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dd"
COLORS = {"compass": "#2a78d6", "ppo1": "#eb6834", "classic": "#1baf7a", "ppo_nocompass": "#eda100",
          "es1": "#e87ba4", "random": "#9a988f", "ppo1_seed1": "#e34948", "ppo2": "#6250d6"}
NAMES = {"compass": "hand\n+compass", "classic": "hand\nclassic", "ppo1": "PPO",
         "ppo_nocompass": "PPO\nno compass", "es1": "ES", "random": "random", "ppo1_seed1": "PPO\nseed 1",
         "ppo2": "PPO +3x\nstarve pen."}
ORDER = ["random", "classic", "compass", "es1", "ppo_nocompass", "ppo1", "ppo1_seed1", "ppo2"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
os.makedirs("media", exist_ok=True)
R = {}
for f in ["results/eval.json", "results/eval_B.json", "results/eval_C.json", "results/eval_D.json"]:  # jobA and jobB write separate files
    if os.path.exists(f):
        for k, v in json.load(open(f)).items():
            R.setdefault(k, {}).update(v)


def bars(ax, metric, sset, title, fmt="{:.0f}"):
    eff = R.get("efficiency", {})
    pols = [p for p in ORDER if f"{sset}/{p}" in eff]
    vals = [eff[f"{sset}/{p}"][metric] for p in pols]
    x = np.arange(len(pols))
    ax.bar(x, vals, width=0.62, color=[COLORS[p] for p in pols], edgecolor=SURF, linewidth=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [NAMES[p] for p in pols], fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)


if R.get("efficiency"):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4))
    for row, sset in enumerate(["indist", "heldout"]):
        tag = "unseen worlds, training layout family" if sset == "indist" else "held-out family: farther food, long walls"
        bars(axs[row, 0], "delivered_mean", sset, f"Food delivered per colony (1000 steps)\n{tag}")
        if sset == "indist":
            axs[row, 0].axhline(1040, color=MUTED, lw=1, ls=":")
            axs[row, 0].text(-0.4, 1050, "all the food in the world (1040)", color=MUTED, fontsize=8.5)
        bars(axs[row, 1], "food_per_energy", sset, f"Food delivered per unit energy\n{tag}", "{:.2f}")
        bars(axs[row, 2], "ants_alive_frac", sset, f"Fraction of ants alive at the end\n{tag}", "{:.2f}")
    fig.tight_layout()
    fig.savefig("media/efficiency.png", dpi=130)
    plt.close(fig)

if R.get("recovery"):
    from evaluate import BLOCK_AT
    fig, ax = plt.subplots(figsize=(9, 4.2))
    k = np.ones(50) / 50
    for p in ORDER:
        if p not in R["recovery"]:
            continue
        c = np.convolve(np.array(R["recovery"][p]["curve"]), k, mode="valid")
        ax.plot(np.arange(len(c)) + 49, c, color=COLORS[p], lw=2, label=NAMES[p].replace("\n", " "))
    ax.axvline(BLOCK_AT, color=MUTED, lw=1, ls="--")
    ax.text(BLOCK_AT + 8, ax.get_ylim()[1] * 0.95, "wall dropped across\nthe busiest trail", color=MUTED, va="top", fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("food delivered per step (50-step mean, 32 worlds)")
    ax.set_title("Route-block recovery", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig("media/recovery.png", dpi=130)
    plt.close(fig)

runs = [r for r in ["ppo1", "ppo1_seed1", "ppo2", "ppo_nocompass"] if os.path.exists(f"logs/{r}.jsonl")]
if runs:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for r in runs:
        L = [json.loads(l) for l in open(f"logs/{r}.jsonl")]
        ax.plot([l["min"] for l in L], [l["deliv"] for l in L], color=COLORS[r], lw=1.2, alpha=0.45)
        E = [l for l in L if "eval_deliv" in l]
        ax.plot([l["min"] for l in E], [l["eval_deliv"] for l in E], color=COLORS[r], lw=2, marker="o", ms=4,
                label=NAMES[r].replace("\n", " ") + " (validation worlds 900-915)")
    eff = R.get("efficiency", {})
    for p in ["compass", "classic"]:
        if f"indist/{p}" in eff:
            v = eff[f"indist/{p}"]["delivered_mean"]
            ax.axhline(v, color=COLORS[p], lw=1.2, ls="--")
            ax.text(0.2, v + 8, NAMES[p].replace("\n", " ") + f" ({v:.0f}, seeds 0-31)", color=MUTED, fontsize=8.5)
    ax.axhline(1040, color=MUTED, lw=1, ls=":")
    ax.text(0.2, 1050, "all the food in the world (1040)", color=MUTED, fontsize=8.5)
    ax.set_xlabel("wall-clock minutes of training")
    ax.set_ylabel("food delivered per colony per episode")
    ax.set_title("PPO training (thin line: training worlds; dots: fixed eval worlds)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig("media/training.png", dpi=130)
    plt.close(fig)

if os.path.exists("logs/es1.jsonl"):
    L = [json.loads(l) for l in open("logs/es1.jsonl")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot([l["gen"] for l in L], [l["deliv_mean"] for l in L], color=COLORS["es1"], lw=1.5, label="population mean")
    ax.plot([l["gen"] for l in L], [l["deliv_max"] for l in L], color=COLORS["ppo1"], lw=1, alpha=.7, label="best member")
    ax.set_xlabel("generation"); ax.set_ylabel("food delivered per world (600 steps)")
    ax.set_title("Evolution strategies training", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("media/es_training.png", dpi=130)
    plt.close(fig)
print("plots written")
