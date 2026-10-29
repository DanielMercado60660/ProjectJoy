"""Turn results/*.npz into summary numbers (results/summary.json) and figures (media/*.png).

    ../.venv/bin/python analyze.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from inference import GRID, GridPosterior
from physics import DESIGNS, F_GRID, T_GRID

# reference palette (dataviz skill): categorical slots 1-3, text + surface tokens
C = {"active": "#2a78d6", "random": "#eb6834", "fixed": "#1baf7a", "net": "#6250d6"}
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"active": "Active (max EIG)", "random": "Random probes", "fixed": "Fixed protocol (8 N, 0.3 s)"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2})
STRATS = ["active", "random", "fixed"]
TARGET = 0.01          # target accuracy on mu_k


def pit_exact(D, strat):
    """Posterior CDF evaluated at the true mu (PIT), recomputed from the saved sequences."""
    des, Y, Cc, mu = D[f"{strat}_designs"], D[f"{strat}_y"], D[f"{strat}_cens"], D["mu"]
    nw, steps = des.shape[:2]
    pit = np.zeros((nw, steps))
    for i in range(nw):
        p = GridPosterior()
        for j in range(steps):
            p.update(des[i, j, 0], des[i, j, 1], Y[i, j], Cc[i, j])
            c = np.cumsum(p.w)
            pit[i, j] = np.interp(mu[i], GRID, c)
    return pit


def coverage_curve(pit, levels):
    return np.array([np.mean(np.abs(pit - 0.5) <= lv / 2) for lv in levels])


def first_hit(cond):
    """First (1-based) step where cond is true and stays true afterwards; inf if never."""
    steps = cond.shape[1]
    out = np.full(cond.shape[0], np.inf)
    for i, row in enumerate(cond):
        ok = np.flip(np.cumprod(np.flip(row)))   # true from here to the end
        idx = np.nonzero(ok)[0]
        if len(idx):
            out[i] = idx[0] + 1
    return out


def main():
    os.makedirs("media", exist_ok=True)
    D = np.load("results/active.npz")
    mu = D["mu"]
    nw, steps = D["active_mean"].shape
    n = np.arange(1, steps + 1)
    S = {"worlds": int(nw), "steps": int(steps), "target_abs_err": TARGET}

    pits = {s: pit_exact(D, s) for s in STRATS}
    S["per_strategy"] = {}
    for s in STRATS:
        err = np.abs(D[f"{s}_mean"] - mu[:, None])
        err_ls = np.abs(D[f"{s}_lsq"] - mu[:, None])
        se = D[f"{s}_lsq_se"]
        pit_ls = np.where(np.isfinite(se), norm.cdf((mu[:, None] - D[f"{s}_lsq"]) / np.where(np.isfinite(se), se, 1)), 0.5)
        # infinite se => interval = whole prior range => always covers
        cov = lambda lo, hi: float(np.mean((mu[:, None] >= lo) & (mu[:, None] <= hi)))
        lo50, hi50 = np.where(np.isfinite(se), D[f"{s}_lsq"] - 0.6745 * se, -np.inf), np.where(np.isfinite(se), D[f"{s}_lsq"] + 0.6745 * se, np.inf)
        lo90, hi90 = np.where(np.isfinite(se), D[f"{s}_lsq"] - 1.645 * se, -np.inf), np.where(np.isfinite(se), D[f"{s}_lsq"] + 1.645 * se, np.inf)
        hit_true = first_hit(err <= TARGET)
        hit_true_ls = first_hit(err_ls <= TARGET)
        hit_self = first_hit((D[f"{s}_hi90"] - D[f"{s}_lo90"]) <= 2 * TARGET)
        S["per_strategy"][s] = {
            "rmse_mu_bayes_by_n": np.sqrt(np.mean(err ** 2, 0)).round(5).tolist(),
            "rmse_mu_lsq_by_n": np.sqrt(np.mean(err_ls ** 2, 0)).round(5).tolist(),
            "median_abs_err_bayes_by_n": np.median(err, 0).round(5).tolist(),
            "pred_rmse_m_bayes_by_n": D[f"{s}_pred_bayes"].mean(0).round(5).tolist(),
            "pred_rmse_m_lsq_by_n": D[f"{s}_pred_lsq"].mean(0).round(5).tolist(),
            "coverage_bayes_50": cov(D[f"{s}_lo50"], D[f"{s}_hi50"]),
            "coverage_bayes_90": cov(D[f"{s}_lo90"], D[f"{s}_hi90"]),
            "coverage_lsq_50": cov(lo50, hi50),
            "coverage_lsq_90": cov(lo90, hi90),
            "coverage_bayes_90_by_n": np.mean((mu[:, None] >= D[f"{s}_lo90"]) & (mu[:, None] <= D[f"{s}_hi90"]), 0).round(3).tolist(),
            "coverage_lsq_90_by_n": np.mean((mu[:, None] >= lo90) & (mu[:, None] <= hi90), 0).round(3).tolist(),
            "pushes_to_target_bayes_median": float(np.median(hit_true)),
            "pushes_to_target_bayes_mean_reached": float(np.mean(hit_true[np.isfinite(hit_true)])),
            "frac_reached_target_bayes": float(np.mean(np.isfinite(hit_true))),
            "pushes_to_target_lsq_median": float(np.median(hit_true_ls)),
            "frac_reached_target_lsq": float(np.mean(np.isfinite(hit_true_ls))),
            "pushes_until_self_reported_90CI_halfwidth_le_target_median": float(np.median(hit_self)),
            "frac_self_reported_reached": float(np.mean(np.isfinite(hit_self))),
            "abs_err_when_self_reported_done_p90": float(np.percentile(
                [err[i, int(h) - 1] for i, h in enumerate(hit_self) if np.isfinite(h)], 90)) if np.isfinite(hit_self).any() else None,
            "median_choose_time_s": float(np.median(D[f"{s}_t_choose"])),
        }
        S["per_strategy"][s]["_hits"] = hit_true
        S["per_strategy"][s]["_pit_ls"] = pit_ls
    # ------------------------------------------------ figure 1: accuracy + prediction vs pushes
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for s in STRATS:
        P = S["per_strategy"][s]
        ax[0].plot(n, P["rmse_mu_bayes_by_n"], color=C[s], label=LABEL[s] + " · Bayes")
        ax[0].plot(n, P["rmse_mu_lsq_by_n"], color=C[s], ls="--", lw=1.5, label=LABEL[s] + " · least squares")
        ax[1].plot(n, np.array(P["pred_rmse_m_bayes_by_n"]) * 100, color=C[s], label=LABEL[s] + " · Bayes")
        ax[1].plot(n, np.array(P["pred_rmse_m_lsq_by_n"]) * 100, color=C[s], ls="--", lw=1.5, label=LABEL[s] + " · least squares")
        h = P["_hits"]
        xs = np.arange(1, steps + 2)
        ax[2].step(xs, [np.mean(h <= x) for x in xs], where="post", color=C[s], label=LABEL[s])
    ax[0].set_yscale("log"); ax[0].set_xlabel("pushes"); ax[0].set_ylabel("RMSE of μ_k estimate")
    ax[0].axhline(TARGET, color=INK2, lw=1, ls=":"); ax[0].set_title("How well is friction known?", loc="left")
    ax[1].set_yscale("log"); ax[1].set_xlabel("pushes"); ax[1].set_ylabel("trajectory RMSE on 30 new pushes (cm)")
    ax[1].set_title("Predicting new interventions", loc="left")
    ax[2].set_xlabel("pushes"); ax[2].set_ylabel(f"fraction of worlds with |error| ≤ {TARGET} from here on")
    ax[2].set_title("Pushes needed to reach target accuracy", loc="left"); ax[2].set_ylim(0, 1.02); ax[2].set_xlim(1, steps)
    ax[0].legend(fontsize=7.5, frameon=False); ax[2].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("media/active_vs_random.png", dpi=130); plt.close(fig)

    # ------------------------------------------------ figure 2: calibration (reliability) diagram
    levels = np.linspace(0.05, 0.95, 19)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    for s in STRATS:
        ax[0].plot(levels, coverage_curve(pits[s], levels), color=C[s], marker="o", ms=4, label=LABEL[s] + " · Bayes")
        ax[0].plot(levels, coverage_curve(S["per_strategy"][s]["_pit_ls"], levels), color=C[s], ls="--", lw=1.5,
                   label=LABEL[s] + " · least-squares ± z·SE")
    ax[0].set_xlabel("nominal central-interval level"); ax[0].set_ylabel("empirical coverage of true μ_k")
    ax[0].set_title(f"Calibration over {nw} hidden worlds × {steps} pushes", loc="left"); ax[0].legend(fontsize=7, frameon=False)
    sbi_path = "results/sbi_eval.json"
    if os.path.exists("results/sbi_pit.npz"):
        Q = np.load("results/sbi_pit.npz")
        ax[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
        ax[1].plot(levels, coverage_curve(Q["pit_exact"], levels), color=C["active"], marker="o", ms=4, label="exact grid posterior")
        ax[1].plot(levels, coverage_curve(Q["pit_net"], levels), color=C["net"], marker="s", ms=4, label="amortized MLX network")
        ax[1].set_title(f"Fresh random-design datasets (n={len(Q['pit_net'])})", loc="left")
        ax[1].set_xlabel("nominal central-interval level"); ax[1].set_ylabel("empirical coverage"); ax[1].legend(fontsize=8, frameon=False)
    else:
        ax[1].axis("off")
    fig.tight_layout(); fig.savefig("media/calibration.png", dpi=130); plt.close(fig)
    S["pit_coverage_bayes_all"] = {s: dict(zip(["50", "90"], coverage_curve(pits[s], [0.5, 0.9]).round(4).tolist())) for s in STRATS}

    # ------------------------------------------------ figure 3: what does the active detective choose?
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    des = D["active_designs"]
    first = des[:, 0]; later = des[:, 1:].reshape(-1, 2); mu_rep = np.repeat(mu, steps - 1)
    ax[0].scatter(mu_rep, later[:, 0] + np.random.default_rng(0).uniform(-.2, .2, len(later)), s=6, alpha=.25, color=C["active"], label="pushes 2-15")
    ax[0].scatter(mu, first[:, 0], s=12, color=INK, marker="x", label="push 1")
    ax[0].plot(GRID, 1.2 * GRID * 9.81, color=INK2, lw=1, ls=":", label="break-free force μ_s·m·g")
    ax[0].set_xlabel("hidden μ_k"); ax[0].set_ylabel("chosen force F (N)"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("Active detective adapts push force to the friction", loc="left")
    T_c = later[:, 1]
    ax[1].scatter(mu_rep, T_c + np.random.default_rng(1).uniform(-.02, .02, len(T_c)), s=6, alpha=.25, color=C["active"])
    ax[1].set_xlabel("hidden μ_k"); ax[1].set_ylabel("chosen push duration T (s)")
    ax[1].set_title("…and push duration", loc="left")
    fig.tight_layout(); fig.savefig("media/active_choices.png", dpi=130); plt.close(fig)

    for s in STRATS:
        S["per_strategy"][s].pop("_hits"); S["per_strategy"][s].pop("_pit_ls")
    if os.path.exists(sbi_path):
        S["sbi"] = json.load(open(sbi_path))
    if os.path.exists("results/world2.npz"):
        S["world2"] = world2_summary()
    def clean(o):   # strict JSON: non-finite numbers -> null ("never reached")
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    json.dump(clean(S), open("results/summary.json", "w"), indent=1, allow_nan=False)
    print(json.dumps({s: {k: v for k, v in S["per_strategy"][s].items() if not k.endswith("_by_n")} for s in STRATS}, indent=1))
    for s in STRATS:
        P = S["per_strategy"][s]
        print(s, "rmse mu @1,2,4,8,15:", [P["rmse_mu_bayes_by_n"][i] for i in (0, 1, 3, 7, 14)],
              "lsq:", [P["rmse_mu_lsq_by_n"][i] for i in (0, 1, 3, 7, 14)])
        print(s, "pred cm @1,2,4,8,15:", [round(P["pred_rmse_m_bayes_by_n"][i] * 100, 2) for i in (0, 1, 3, 7, 14)],
              "lsq:", [round(P["pred_rmse_m_lsq_by_n"][i] * 100, 2) for i in (0, 1, 3, 7, 14)])
    if "world2" in S:
        print("world2", json.dumps(S["world2"], indent=1))


def world2_summary():
    W = np.load("results/world2.npz")
    out = {"worlds": int(len(W["mu_k"]))}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = W["active_muk_mean"].shape[1]
    n = np.arange(1, steps + 1)
    for s in ["active", "random"]:
        r = {}
        for p, truth in [("muk", W["mu_k"]), ("mus", W["mu_s"])]:
            e = np.abs(W[f"{s}_{p}_mean"] - truth[:, None])
            r[f"rmse_{p}_by_n"] = np.sqrt(np.mean(e ** 2, 0)).round(4).tolist()
            for lv in (50, 90):
                r[f"cov{lv}_{p}"] = float(np.mean((truth[:, None] >= W[f"{s}_{p}_lo{lv}"]) & (truth[:, None] <= W[f"{s}_{p}_hi{lv}"])))
            r[f"median_ci90_width_{p}_final"] = float(np.median(W[f"{s}_{p}_hi90"][:, -1] - W[f"{s}_{p}_lo90"][:, -1]))
            r[f"pushes_to_{p}_err_le_0.02_median"] = float(np.median(first_hit(e <= 0.02)))
        r["frac_pushes_no_motion"] = float(1 - W[f"{s}_moved"].mean())
        out[s] = r
        ax[0].plot(n, r["rmse_muk_by_n"], color=C[s], label=LABEL[s])
        ax[1].plot(n, r["rmse_mus_by_n"], color=C[s], label=LABEL[s])
    for a_, t in zip(ax[:2], ["kinetic friction μ_k", "static friction μ_s"]):
        a_.set_yscale("log"); a_.set_xlabel("pushes"); a_.set_ylabel(f"RMSE of {t.split()[-1]}"); a_.set_title(t, loc="left")
        a_.legend(fontsize=8, frameon=False)
    floor = 0.25 / 9.81 / np.sqrt(12)   # F menu step 0.25 N -> mu_s only bracketed to 0.0255
    ax[1].axhline(floor, color=INK2, ls=":", lw=1)
    ax[1].text(1, floor * 1.08, "floor set by 0.25 N force steps", fontsize=8, color=INK2)
    # how close to the break-free threshold does each strategy push?
    for s in ["active", "random"]:
        F = W[f"{s}_design"][..., 0]
        thr = W["mu_s"][:, None] * 9.81
        ax[2].hist((F / thr).ravel(), bins=np.linspace(0, 4, 41), color=C[s], alpha=0.55, label=LABEL[s])
    ax[2].axvline(1, color=INK2, ls=":", lw=1)
    ax[2].set_xlabel("push force / break-free force"); ax[2].set_ylabel("pushes"); ax[2].legend(fontsize=8, frameon=False)
    ax[2].set_title("Active probes bracket the threshold", loc="left")
    fig.tight_layout(); fig.savefig("media/world2_two_frictions.png", dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    main()
