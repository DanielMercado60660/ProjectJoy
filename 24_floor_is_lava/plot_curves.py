"""Training curves from logs/main_train.csv and logs/main_level.csv -> media/training_curves.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main(run="main", out="media/training_curves.png"):
    os.chdir(HERE)
    d = np.genfromtxt(f"logs/{run}_train.csv", delimiter=",", names=True)
    lv = np.genfromtxt(f"logs/{run}_level.csv", delimiter=",")
    # resumed runs repeat iterations after the last checkpoint: keep the last row per iteration
    _, keep = np.unique(d["iter"][::-1], return_index=True)
    d = d[::-1][keep]
    _, keepl = np.unique(lv[::-1, 0], return_index=True)
    lv = lv[::-1][keepl]
    s = d["env_steps"] / 1e6
    def sm(x, k=15):   # moving average with edge padding (no artificial drop at the ends)
        x = np.nan_to_num(x, nan=np.nanmean(x))
        xp = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
        return np.convolve(xp, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(s, sm(d["success"]), label="completion (all training episodes)", color="#2a6fdb")
    ax[0].plot(s, sm(d["lava"]), label="lava touch", color="#e8590c")
    full = lv[:, 3]
    ok = ~np.isnan(full)
    ax[0].plot(lv[ok, 1] / 1e6, sm(full[ok], 25), label="completion, full-difficulty episodes", color="#2b8a3e")
    ax[0].set_xlabel("env steps (M)"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7); ax[0].set_title("training episodes")
    ax[1].plot(lv[:, 1] / 1e6, lv[:, 2], color="#5f3dc4")
    ax[1].set_xlabel("env steps (M)"); ax[1].set_ylabel("gap curriculum level"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("curriculum (1 = full 0.05-0.20 m gaps)")
    ax[2].plot(s, sm(d["approx_kl"]), label="approx KL", color="#c92a2a")
    ax[2].plot(s, sm(d["clip_frac"]), label="clip fraction", color="#868e96")
    ax[2].plot(s, d["std"], label="policy std", color="#1098ad")
    ax[2].set_xlabel("env steps (M)"); ax[2].legend(fontsize=7); ax[2].set_title("PPO diagnostics")
    fig.tight_layout()
    os.makedirs("media", exist_ok=True)
    fig.suptitle(f"run: {run}", fontsize=9, x=0.01, ha="left")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    main()
    if os.path.exists(os.path.join(HERE, "logs/blind_train.csv")):
        main("blind", "media/training_curves_blind.png")
