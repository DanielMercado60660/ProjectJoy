"""Cross-review checks (2026-10-22). Re-renders the held-out sets, re-runs both estimators, and writes
results_review.json: reproduction of the headline numbers, paired-bootstrap CIs on the network-vs-classical
differences, classical outliers, a physically clipped classical estimate, and the sigma-floor activity.

  ../bin/gpu_run ../.venv/bin/python review_check.py
"""
import json, subprocess, time
import numpy as np

import scene, baseline
from model import load, predict

# record when the quarter-pixel floor binds (baseline.estimate calls max(s, 0.25/SCALE))
FLOOR = 0.25 / scene.SCALE
floor_hits = []
def _max(a, b, *r):
    if b == FLOOR and not r:
        floor_hits.append(a < b)
    return max(a, b, *r)
baseline.max = _max

HORIZON = np.arange(1, 31) / scene.FPS + scene.T / scene.FPS
def resim(p, e_hat):
    t = np.broadcast_to(HORIZON, (len(e_hat), len(HORIZON))).copy()
    q = dict(p); q["e"] = np.clip(e_hat, 0.0, 0.999)
    return np.abs(scene.heights(q, t) - scene.heights(p, t)).mean(1) * 100

def boot(fn, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); N = len(a)
    d = [fn(a[i]) - fn(b[i]) for i in (rng.integers(0, N, N) for _ in range(n))]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

mae = lambda x: np.abs(x).mean(); med = lambda x: np.median(np.abs(x)); rmse = lambda x: np.sqrt((x ** 2).mean())
model, cfg = load("checkpoints/main")
out = dict(uptime=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), sets={})

Xv, pv = scene.make_set(7, 512)
floor_hits.clear(); baseline.estimate_many(Xv)
out["sigma_floor_active_frac_val_seed7"] = float(np.mean(floor_hits)); out["val_fitted_n"] = len(floor_hits)

for name, (seed, n, kw) in {"main": (1001, 2000, {}), "hard": (2002, 1000, dict(noise_range=(0.25, 0.40)))}.items():
    X, p = scene.make_set(seed, n, **kw)
    t0 = time.time(); mn, sn = predict(model, X); tn = (time.time() - t0) / n
    t0 = time.time(); mb, sb, ok = baseline.estimate_many(X); tb = (time.time() - t0) / n
    en, eb = mn - p["e"], mb - p["e"]
    ebc = np.clip(mb, 0.0, 1.0) - p["e"]                  # physical range only
    rn, rb = resim(p, mn), resim(p, mb)
    r = dict(net_mae=mae(en), cl_mae=mae(eb), net_median=med(en), cl_median=med(eb), net_rmse=rmse(en), cl_rmse=rmse(eb),
             net_resim_mean=rn.mean(), cl_resim_mean=rb.mean(), fail_frac=1 - ok.mean(),
             ci95_mae_net_minus_cl=boot(mae, en, eb), ci95_median_net_minus_cl=boot(med, en, eb),
             ci95_rmse_net_minus_cl=boot(rmse, en, eb),
             ci95_resim_mean_net_minus_cl=boot(np.mean, rn, rb),
             net_wins_frac=float(np.mean(np.abs(en) < np.abs(eb))),
             cl_err_abs_max=float(np.abs(eb).max()), net_err_abs_max=float(np.abs(en).max()),
             cl_est_range=[float(mb.min()), float(mb.max())], net_est_range=[float(mn.min()), float(mn.max())],
             cl_n_abs_err_gt_0p5=int((np.abs(eb) > 0.5).sum()),
             cl_clip01_mae=mae(ebc), cl_clip01_rmse=rmse(ebc),
             cl_clip01_fitted_mae=mae(ebc[ok]), cl_clip01_fitted_rmse=rmse(ebc[ok]),
             cl_fitted_mae=mae(eb[ok]), net_on_cl_fitted_mae=mae(en[ok]),
             cl_fitted_cov95=float(np.mean(np.abs(eb[ok]) < 1.96 * sb[ok])),
             net_mean_sd=float(sn.mean()), sec_per_clip_under_load=dict(network_batched=tn, classical=tb))
    # forecast error by true-e bin
    r["resim_mean_by_e"] = {f"{lo}-{hi}": dict(net=float(rn[m].mean()), cl=float(rb[m].mean()))
                            for lo, hi in [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)]
                            for m in [(p["e"] >= lo) & (p["e"] < hi)]}
    out["sets"][name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}
    print(name, json.dumps(out["sets"][name]), flush=True)
json.dump(out, open("results_review.json", "w"), indent=1)
print("wrote results_review.json")
