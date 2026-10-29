"""Figure for the non-identifiable world: the (k, m) posterior ridge, where least squares lands,
and how one spring-scale weighing collapses the ridge.

    ../.venv/bin/python plot_world3.py      (needs results/world3.npy from `deeper.py world3`)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

R = np.load("results/world3.npy", allow_pickle=True).item()
K, M = np.exp(R["logk"]), np.exp(R["logm"])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, key, title in [(ax[0], "post_pushes", "After 6 random pushes: a ridge"),
                      (ax[1], "post_scale", "…plus one spring-scale weighing")]:
    W = R[key]
    a.contourf(M, K, W / W.max(), levels=[0.01, 0.1, 0.3, 0.6, 1.0], cmap="Blues")
    a.plot(M, R["ratio_stats"]["mean"] * M, color=INK2, ls=":", lw=1, label=f"k/m = {R['ratio_stats']['mean']:.3f}")
    a.scatter([R["m_true"]], [R["k_true"]], marker="*", s=180, color=ORANGE, edgecolor=INK, zorder=4, label="hidden truth")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(M[0], M[-1]); a.set_ylim(K[0], K[-1])
    a.set_xlabel("block mass m (kg)"); a.set_ylabel("pusher gain k")
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); a.set_yticks([0.5, 0.75, 1, 1.5, 2])
    a.set_title(title, loc="left")
F = R["lsq_fits"]
ax[0].scatter(np.exp(F[:, 1]), np.exp(F[:, 0]), s=22, color=AQUA, edgecolor=INK, lw=0.5, zorder=3,
              label="least squares, 20 random starts")
ax[0].legend(fontsize=8, frameon=False, loc="lower right")
ax[1].legend(fontsize=8, frameon=False, loc="lower right")
# m marginals
logm = R["logm"]
prior = np.ones_like(logm) / len(logm)
mp = R["post_pushes"].sum(0); ms = R["post_scale"].sum(0)
ax[2].plot(M, prior / prior.max(), color=INK2, ls="--", lw=1.5, label="prior (log-uniform)")
ax[2].plot(M, mp / mp.max(), color=BLUE, lw=2, label="after 6 pushes")
ax[2].plot(M, ms / ms.max(), color=ORANGE, lw=2, label="after pushes + weighing")
ax[2].axvline(R["m_true"], color=INK, ls=":", lw=1)
ax[2].set_xscale("log"); ax[2].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[2].set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); ax[2].set_xlabel("block mass m (kg)"); ax[2].set_yticks([])
ax[2].set_title("Marginal of m: pushes alone leave it\nprior-shaped (cut only by the k range)", loc="left")
ax[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("media/world3_ridge.png", dpi=130)
print("saved media/world3_ridge.png")
