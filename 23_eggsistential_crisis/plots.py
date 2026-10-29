"""Static figures for RESULTS.md from logs/*.jsonl and results/eval.json.

  ../.venv/bin/python plots.py
"""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# fixed method -> color (reference categorical palette, slot order)
COL = {"zero": "#2a78d6", "posture": "#eb6834", "random": "#1baf7a", "scripted gait (force": "#e87ba4",
       "scripted": "#eda100", "PPO before": "#008300", "PPO": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def color(name):
    return next(v for k, v in COL.items() if name.startswith(k))


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def training_curves():
    recs = []
    for f in sorted(glob.glob("logs/run*.jsonl")):
        recs += [json.loads(l) for l in open(f) if '"iter"' in l and '"succ"' in l]
    if not recs:
        return
    recs = [r for r in recs if not (r["succ"] == 0 and r["ep_return"] == 0)]  # empty episode window
    recs.sort(key=lambda r: r["env_steps"])
    x = np.array([r["env_steps"] for r in recs]) / 1e6
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, key, lab in zip(axs, ["succ", "over_budget", "std"],
                            ["train success (last 256 eps)", "train eps over 10 N budget", "policy std"]):
        ax.plot(x, [r[key] for r in recs], color=COL["PPO"], lw=1.5)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("env steps (M)", fontsize=8, color=MUTED)
        style(ax)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    th = [r["theta_max"] for r in recs]
    ch = [i for i in range(1, len(th)) if th[i] != th[i - 1]]
    for i in ch:
        for ax in axs:
            ax.axvline(x[i], color=MUTED, lw=0.6, ls=":")
    if ch:
        axs[0].text(x[ch[-1]], 0.05, f"  curriculum: θmax {th[0]:.1f} → {th[-1]:.2f} rad", fontsize=7, color=MUTED)
    for f in sorted(glob.glob("logs/run*.jsonl"))[1:]:
        r0 = next(json.loads(l) for l in open(f) if '"succ"' in l)
        for ax in axs:
            ax.axvline(r0["env_steps"] / 1e6, color="#e34948", lw=0.8, ls="--")
        axs[1].text(r0["env_steps"] / 1e6, 0.05, f" {f[5:-6]}: stronger force penalty", fontsize=7,
                    color=MUTED, rotation=90, va="bottom", ha="right", transform=axs[1].get_xaxis_transform())
    fig.tight_layout()
    fig.savefig("media/training_curves.png", dpi=130)


def eval_figs():
    E = json.load(open("results/eval.json"))
    cells = E["cells"]
    main = [c for c in cells if c["theta"] == [0.15, 0.4] and c["fric"] is None]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for c in main:
        y = np.degrees(c["err_curve_mean"])
        axs[0].plot(np.arange(1, len(y) + 1) * 0.04, y, color=color(c["method"]), lw=1.6, label=c["method"])
    axs[0].axhline(np.degrees(0.1), color=MUTED, lw=0.8, ls="--")
    axs[0].text(0.1, np.degrees(0.1) + 0.3, "success threshold 5.7°", fontsize=7, color=MUTED)
    axs[0].set_title("mean axis error over the episode (held-out, θ∈[0.15,0.4])", fontsize=9, loc="left", color=INK)
    axs[0].set_xlabel("time (s)", fontsize=8, color=MUTED)
    axs[0].set_ylabel("deg", fontsize=8, color=MUTED)
    axs[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    style(axs[0])
    for k, c in enumerate(main):
        pf = np.array(c["peak_forces"])
        axs[1].boxplot(pf, positions=[k], widths=0.5, showfliers=True, patch_artist=True,
                       boxprops=dict(facecolor=color(c["method"]), alpha=0.6, color=MUTED),
                       medianprops=dict(color=INK), whiskerprops=dict(color=MUTED),
                       capprops=dict(color=MUTED), flierprops=dict(markersize=3, markeredgecolor=MUTED))
    axs[1].axhline(E["force_budget_N"], color="#e34948", lw=1, ls="--")
    axs[1].text(-0.4, E["force_budget_N"] * 1.08, "budget 10 N", fontsize=7, color=MUTED)
    axs[1].set_yscale("log")
    short = {"zero (hold still)": "zero", "posture (tuned hold)": "posture", "scripted gait (tuned)": "gait",
             "scripted gait (force-aware retune)": "gait\nforce-aware", "PPO (asym. critic, touch)": "PPO\nfinal",
             "PPO before force fine-tune": "PPO\nrun1 only"}
    axs[1].set_xticks(range(len(main)), [short.get(c["method"], c["method"].split(" (")[0]) for c in main], fontsize=7)
    axs[1].set_title("peak contact force per episode (per-substep, N)", fontsize=9, loc="left", color=INK)
    style(axs[1])
    fig.tight_layout()
    fig.savefig("media/eval_error_force.png", dpi=130)

    # robustness: success (top row) and success within the 10 N budget (bottom row)
    rob = [c for c in cells if c["fric"] is not None or c["mass"] is not None]
    methods = list(dict.fromkeys(c["method"] for c in rob))
    fig, axs = plt.subplots(2, 2, figsize=(11, 5.6), sharey=True)
    for row, metric, ylab in [(0, "success", "success"), (1, "success_within_budget", "success within 10 N")]:
        for col, key, vals, lab in [(0, "fric", [0.3, 0.5, 1.0, 2.0], "friction scale (train range 0.7-1.3)"),
                                    (1, "mass", [0.4, 2.0, 3.0], "egg mass scale (train range 0.7-1.3)")]:
            ax = axs[row, col]
            w = 0.8 / len(methods)
            for j, m in enumerate(methods):
                ys = []
                for v in vals:
                    c = next(c for c in rob if c["method"] == m and c[key] == v
                             and (c["mass"] == 1.0 if key == "fric" else c["fric"] == 1.0))
                    ys.append(c[metric])
                ax.bar(np.arange(len(vals)) + (j - (len(methods) - 1) / 2) * w, ys, w * 0.9, color=color(m),
                       label=m)
            ax.set_xticks(range(len(vals)), [str(v) for v in vals], fontsize=8)
            if row == 1:
                ax.set_xlabel(lab, fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel(ylab + " (held-out)", fontsize=8, color=MUTED)
            style(ax)
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig("media/robustness.png", dpi=130)


if __name__ == "__main__":
    training_curves()
    try:
        eval_figs()
    except FileNotFoundError:
        print("no results/eval.json yet")
