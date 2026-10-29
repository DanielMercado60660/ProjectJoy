"""Plots for RESULTS.md: PPO training curves and robustness (falls per loop / tracking error per condition).

  ../.venv/bin/python make_plots.py --eval results/eval_main.json --ckpts checkpoints/ppo_pd.pkl checkpoints/ppo_push.pkl
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/eval_main.json")
    ap.add_argument("--ckpts", nargs="*", default=["checkpoints/ppo_pd.pkl"])
    ap.add_argument("--baseline_score", type=float, default=None)
    args = ap.parse_args()
    # training curves: the checkpoints chain (each run continues the previous), so plot the longest
    # history once and mark where each earlier run ended
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    cks = [(p, pickle.load(open(p, "rb"))) for p in args.ckpts if os.path.exists(p)]
    if cks:
        h = max(cks, key=lambda c: len(c[1]["history"]))[1]["history"]
        steps = np.array([x["env_steps"] for x in h]) / 1e6
        rew = np.array([x["mean_rew"] for x in h])
        k = 25
        sm = lambda v: np.convolve(v, np.ones(k) / k, "valid")
        ax[0].plot(steps, rew, lw=0.3, color="0.75")
        ax[0].plot(steps[k - 1:], sm(rew), lw=1.2, label="mean per-step reward (stochastic, 25-update avg)")
        ff = np.array([np.nan if x["fall_frac"] is None else x["fall_frac"] for x in h])
        ffs = np.array([np.nanmean(ff[max(0, i - 50):i + 1]) for i in range(len(ff))])
        ax[1].plot(steps, ffs, lw=1.2)
        for p, c in cks:
            e = c["env_steps"] / 1e6
            for a in ax:
                a.axvline(e, color="k", lw=0.6, ls=":")
            ax[1].text(e, 0.02, os.path.basename(p).replace(".pkl", ""), rotation=90, fontsize=7, ha="right")
    if args.baseline_score:
        ax[0].axhline(args.baseline_score, color="k", ls="--", lw=0.8, label="tuned baseline (train seeds, det.)")
    ax[0].set_xlabel("env steps (M)"); ax[0].set_ylabel("reward")
    ax[0].set_ylim(0.85, 1.0); ax[0].legend(fontsize=7, loc="lower left")
    ax[1].set_xlabel("env steps (M)"); ax[1].set_ylabel("fraction of episodes ending in a fall (50-upd avg)")
    ax[0].set_title("run 1: nominal | run 2: pushes U(0,60) N | run 3: U(0,100) N, residual 0.25 rad", fontsize=8)
    plt.tight_layout(); plt.savefig("media/training_curves.png", dpi=110); plt.close()
    # robustness
    if os.path.exists(args.eval):
        E = json.load(open(args.eval))
        T = E["table"]
        names = list(T)
        conds = list(next(iter(T.values())))
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            x = np.arange(len(conds)) + i * w
            ax[0].bar(x, [T[n][c]["falls_per_loop"] for c in conds], w, label=n)
            ax[1].bar(x, [T[n][c]["foot_pos_err_cm"] for c in conds], w, label=n)
        ns, nl = len(E["seeds"]), E["loops_per_seed"]
        for a, lab in zip(ax, [f"falls per clip loop ({ns} seeds x {nl} loops)", "mean foot position error (cm)"]):
            a.set_xticks(np.arange(len(conds)) + 0.4 - w / 2)
            a.set_xticklabels(conds, rotation=30, fontsize=8)
            a.set_ylabel(lab)
        ax[0].legend(fontsize=7)
        plt.tight_layout(); plt.savefig("media/robustness.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
