"""Does the tracker know a hidden ball must be *behind the wall*?  For fully hidden frames:
fraction of predicted means inside the wall's x-range, median sigma_x / sigma_y, and the fraction
of the predicted 2-sigma mass in x that lies inside the wall.  Writes results/shape_analysis.json."""
import json

import numpy as np
from scipy.stats import norm

import kalman
import sim
from evaluate import run_model
from train import load_model

out = {}
for variant, split in [("base", "test_long"), ("hard", "hard_test_long")]:
    d = sim.load_split(split)
    idx = np.arange(1000)
    frames = sim.render_split(d, idx)
    kc = json.load(open(f"results/kalman_tuning_{variant}.json"))["best"]["bounce=True"]
    preds = {"gru": run_model(load_model(f"checkpoints/{variant}/final")[0], frames),
             "kalman_bounce": kalman.track_split(d, idx, frames=frames, q=kc["q"], r=kc["r"], bounce=True)}
    vis, pres, occ = d["vis"][idx], d["present"][idx], d["occ"][idx]
    hid = (vis == 0) & pres[:, None]
    n, t, k = np.nonzero(hid)
    gx = d["pos"][idx][n, t, k, 0]
    # wall containing the true ball
    walls = occ[n]                                            # (M,2,2)
    inside_w = (gx[:, None] >= walls[..., 0]) & (gx[:, None] <= walls[..., 1]) & (walls[..., 0] >= 0)
    w = walls[np.arange(len(n)), inside_w.argmax(1)]          # (M,2)
    out[split] = {}
    for name, (mean, cov) in preds.items():
        mx_, sx, sy = mean[n, t, k, 0], np.sqrt(cov[n, t, k, 0, 0]), np.sqrt(cov[n, t, k, 1, 1])
        lo, hi = w[:, 0] + sim.R, w[:, 1] - sim.R             # centre positions that are fully hidden
        mass = norm.cdf((hi - mx_) / sx) - norm.cdf((lo - mx_) / sx)
        out[split][name] = dict(n=int(len(n)), mean_inside_wall=float(((mx_ >= lo) & (mx_ <= hi)).mean()),
                                median_sx_over_sy=float(np.median(sx / sy)),
                                mean_x_mass_inside_wall=float(mass.mean()))
        print(split, name, out[split][name])
json.dump(out, open("results/shape_analysis.json", "w"), indent=1)
