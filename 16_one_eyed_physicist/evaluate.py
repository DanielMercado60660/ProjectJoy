"""Held-out evaluation of the network vs the classical tracker baseline.

  ../.venv/bin/python evaluate.py --ckpt checkpoints/main [--n_main 2000 --n_hard 1000]
Writes results.json, media/calibration.png, media/scatter.png.
"""
import argparse, json, os, subprocess, time
import numpy as np
from scipy.stats import norm

import scene, baseline
from model import load, predict

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/main")
ap.add_argument("--n_main", type=int, default=2000)
ap.add_argument("--n_hard", type=int, default=1000)
ap.add_argument("--out", default="results.json")
ap.add_argument("--media", default="media")
args = ap.parse_args()
os.makedirs(args.media, exist_ok=True)

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
HORIZON = np.arange(1, 31) / scene.FPS + scene.T / scene.FPS     # 1 s of future after the clip


def resim_err(p, e_hat):
    """Mean |y_pred - y_true| (cm) over the 1 s after the clip, re-simulating from the TRUE initial state
    with the estimated restitution (isolates the effect of the restitution error)."""
    t = np.broadcast_to(HORIZON, (len(e_hat), len(HORIZON))).copy()
    y_true = scene.heights(p, t)
    q = dict(p); q["e"] = np.clip(e_hat, 0.0, 0.999)
    y_hat = scene.heights(q, t)
    return np.abs(y_hat - y_true).mean(1) * 100


def metrics(p, mu, sd):
    err = mu - p["e"]
    z = np.abs(err) / sd
    cover = np.array([np.mean(z < norm.ppf(0.5 + l / 2)) for l in LEVELS])
    pit = norm.cdf((p["e"] - mu) / sd)
    rs = resim_err(p, mu)
    out = dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
               median_ae=float(np.median(np.abs(err))), p95_ae=float(np.percentile(np.abs(err), 95)),
               bias=float(err.mean()),
               nll=float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + err ** 2 / (2 * sd ** 2))),
               coverage={f"{l:g}": float(c) for l, c in zip(LEVELS, cover)},
               calib_mae=float(np.abs(cover - LEVELS).mean()),
               mean_sd=float(sd.mean()), sd_err_corr=float(np.corrcoef(sd, np.abs(err))[0, 1]),
               resim_cm_mean=float(rs.mean()), resim_cm_median=float(np.median(rs)),
               resim_cm_p95=float(np.percentile(rs, 95)))
    bins = {}
    for lo, hi in [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)]:
        m = (p["e"] >= lo) & (p["e"] < hi)
        bins[f"e {lo}-{hi}"] = dict(n=int(m.sum()), mae=float(np.abs(err[m]).mean()))
    nq = np.quantile(p["noise"], [0, 1 / 3, 2 / 3, 1])
    for lo, hi in zip(nq[:-1], nq[1:]):
        m = (p["noise"] >= lo) & (p["noise"] <= hi)
        bins[f"noise {lo:.2f}-{hi:.2f}"] = dict(n=int(m.sum()), mae=float(np.abs(err[m]).mean()))
    for lo, hi in [(0.0, 0.27), (0.27, 0.53), (0.53, 0.81)]:
        m = (p["expo"] >= lo) & (p["expo"] < hi)
        bins[f"exposure {lo}-{hi}"] = dict(n=int(m.sum()), mae=float(np.abs(err[m]).mean()))
    out["bins"] = bins
    return out, cover, pit, err


model, cfg = load(args.ckpt)
sets = {"main (in-distribution, noise 0-0.12)": scene.make_set(1001, args.n_main),
        "hard (noise 0.25-0.40, outside training range)": scene.make_set(2002, args.n_hard, noise_range=(0.25, 0.40))}
load_avg = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
res = dict(ckpt=args.ckpt, ckpt_step=cfg.get("step"), sets={}, uptime=load_avg)
# post-hoc sigma rescaling fitted on the VALIDATION set (seed 7, same 512 clips train.py validates on)
Xval, pval = scene.make_set(7, 512)
mv_n, sv_n = predict(model, Xval); mv_b, sv_b, _ = baseline.estimate_many(Xval)
k_net = float(np.sqrt(np.mean(((mv_n - pval["e"]) / sv_n) ** 2)))
k_cl = float(np.sqrt(np.mean(((mv_b - pval["e"]) / sv_b) ** 2)))
res["val_sigma_scale"] = dict(network=k_net, classical=k_cl)
plot = {}
for name, (X, p) in sets.items():
    t0 = time.time(); mu_n, sd_n = predict(model, X); t_net = (time.time() - t0) / len(X)
    t0 = time.time(); mu_b, sd_b, ok_b = baseline.estimate_many(X); t_base = (time.time() - t0) / len(X)
    r = {}
    r["network"], cn, pn, en = metrics(p, mu_n, sd_n)
    r["classical"], cb, pb, eb = metrics(p, mu_b, sd_b)
    r["classical"]["fit_fail_frac"] = float(1 - ok_b.mean())
    r["classical_fitted_only"], *_ = metrics({k: v[ok_b] for k, v in p.items()}, mu_b[ok_b], sd_b[ok_b])
    r["network_val_rescaled"], *_ = metrics(p, mu_n, sd_n * k_net)
    r["classical_val_rescaled"], *_ = metrics(p, mu_b, sd_b * k_cl)
    pri = np.full(len(X), baseline.PRIOR_MU)
    r["prior_mean"], *_ = metrics(p, pri, np.full(len(X), baseline.PRIOR_SD))
    r["n"] = len(X)
    r["sec_per_clip_under_load"] = dict(network_batched=t_net, classical=t_base)
    res["sets"][name] = r
    plot[name] = (p, mu_n, sd_n, mu_b, sd_b, cn, cb, pn, pb)
    print(name, json.dumps({k: {m: r[k][m] for m in ("mae", "median_ae", "calib_mae", "nll", "resim_cm_mean")}
                            for k in ("network", "classical", "prior_mean")}), flush=True)
json.dump(res, open(args.out, "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C_NET, C_CL = "#2a6fdb", "#e07b39"
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
for j, (name, (p, mu_n, sd_n, mu_b, sd_b, cn, cb, pn, pb)) in enumerate(plot.items()):
    ax = axs[0, j]
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect calibration")
    ax.plot(LEVELS, cn, "o-", color=C_NET, label="network (Gaussian NLL head)")
    ax.plot(LEVELS, cb, "s-", color=C_CL, label="classical fit (bootstrap)")
    ax.set_xlabel("nominal central-interval level"); ax.set_ylabel("observed coverage")
    ax.set_title(name, fontsize=10); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    ax = axs[1, j]
    bins = np.linspace(0, 1, 11)
    ax.hist(pn, bins, histtype="step", lw=2, color=C_NET, density=True, label="network")
    ax.hist(pb, bins, histtype="step", lw=2, color=C_CL, density=True, label="classical")
    ax.axhline(1, color="#999", ls="--", lw=1)
    ax.set_xlabel("PIT = Phi((e_true - mu)/sigma)  (flat = calibrated)"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("Restitution uncertainty calibration on held-out synthetic clips")
fig.tight_layout(); fig.savefig(f"{args.media}/calibration.png", dpi=120)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
for j, (name, (p, mu_n, sd_n, mu_b, sd_b, *_)) in enumerate(plot.items()):
    ax = axs[j]
    ax.plot([0.2, 0.95], [0.2, 0.95], color="#999", lw=1, ls="--")
    ax.scatter(p["e"], mu_b, s=5, alpha=0.5, color=C_CL, label="classical")
    ax.scatter(p["e"], mu_n, s=5, alpha=0.5, color=C_NET, label="network")
    ax.set_xlabel("true restitution e"); ax.set_ylabel("estimated e"); ax.set_title(name, fontsize=10)
    ax.set_ylim(0.0, 1.1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{args.media}/scatter.png", dpi=120)
print("wrote", args.out)
