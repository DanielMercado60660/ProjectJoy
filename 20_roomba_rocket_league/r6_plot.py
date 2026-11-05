"""Round 6 training curves: per-opponent-kind training goal rate and the reverse-driving fraction, per seed.
    ../.venv/bin/python r6_plot.py logs/r6_s0.jsonl logs/r6_s1.jsonl logs/r6_s2.jsonl --out media/r6_training_curves.png"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KINDS = ["goalie", "goalie_r", "keeper", "chaser", "marker", "self_arch", "self_cur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(args.logs) + 1, figsize=(5 * (len(args.logs) + 1), 4), sharey=True)
    for ax, f in zip(axes, args.logs):
        L = {}
        for l in open(f):
            r = json.loads(l)
            L[r["update"]] = r                       # last line per update wins (resumes)
        R = [L[k] for k in sorted(L)]
        x = [r["steps"] / 1e6 for r in R]
        for k in KINDS:
            ax.plot(x, [r["per_kind"].get(k, [None])[0] for r in R], label=k, lw=1)
        ax.set_title(f.split("/")[-1].replace(".jsonl", ""))
        ax.set_xlabel("env steps (M)")
        axes[-1].plot(x, [r["rev"] for r in R], label=f.split("/")[-1].replace(".jsonl", ""))
    axes[0].set_ylabel("training goal rate (last 3000 episodes)")
    axes[0].legend(fontsize=7)
    axes[-1].set_title("fraction of steps driving backwards")
    axes[-1].set_xlabel("env steps (M)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
