"""Learning curves: validation score (60 s drives on 128 unseen training-distribution
tracks; m of progress, -100 per crash) vs environment steps and vs wall-clock.

    ../.venv/bin/python plot_curves.py   -> media/learning_curves.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [("es", "ES", "#2a78d6"), ("ppo", "PPO", "#eb6834"),
        ("es_dr", "ES + grip rand.", "#1baf7a"), ("ppo_dr", "PPO + grip rand.", "#eda100")]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(tag):
    p = f"logs/{tag}_history.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(line) for line in open(p)]
    return [r for r in rows if "val" in r]


def ray_val(n_val=128):
    """Same validation score for the tuned ray follower (dashed reference line)."""
    import numpy as np

    from racing.controllers import get_controller
    from racing.rollout import episodes, score
    from racing.track import make_tracks, pack
    bank = pack(make_tracks("train", n_val, start=5000))
    starts = np.random.default_rng(12345).random(n_val)
    prog, cr = episodes(get_controller("ray"), bank, np.arange(n_val), starts, 600)
    return float(score(prog, cr).mean())


def main(out="media/learning_curves.png"):
    rv = ray_val()
    print(f"ray follower validation score: {rv:.1f}")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    for ax, xkey, xlabel, xs in ((axs[0], "env_steps", "environment steps (millions, log scale)", 1e-6),
                                 (axs[1], "minutes", "wall-clock minutes (4 CPU threads, shared machine)", 1)):
        ax.set_facecolor(SURF)
        for tag, label, color in RUNS:
            rows = load(tag)
            if not rows:
                continue
            x = [r[xkey] * xs for r in rows]
            y = [r["val"] for r in rows]
            ax.plot(x, y, color=color, lw=2, label=label)
            ax.annotate(label, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        ax.axhline(rv, color=INK2, lw=1.2, ls="--")
        ax.annotate("ray follower (tuned)", (0.30, rv), xycoords=("axes fraction", "data"), xytext=(0, -12),
                    textcoords="offset points", color=INK2, fontsize=8)
        ax.set_xlabel(xlabel, color=INK2)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    axs[0].set_xscale("log")
    axs[0].set_ylabel("validation score (m in 60 s, -100 per crash)", color=INK2)
    axs[0].legend(frameon=False, labelcolor=INK2, loc="lower right")
    fig.suptitle("Learning curves on unseen validation tracks", color=INK, x=0.07, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
