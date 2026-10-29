"""Robust comparison of the main (25-frame unroll) and ablation (one-step loss) checkpoints:
median open-loop error and fraction of diverged rollouts by horizon (mean is useless once
rollouts blow up to inf). Same 6,000 held-out rollouts as eval.py."""
import json, os
import numpy as np
from model import load, imagine

d = np.load("data/test.npz")
S = d["states"]
H = 50
init = np.concatenate([S[:, t] for t in (0, 25, 50)])
true = np.concatenate([S[:, t:t + H + 1] for t in (0, 25, 50)])
out = {}
for name, ck in [("multistep_unroll25", "checkpoints/dynamics"), ("onestep_unroll1", "checkpoints/ablation_unroll1")]:
    m, cfg = load(ck)
    with np.errstate(all="ignore"):
        p = imagine(m, init, H).astype(np.float64)
        e = np.linalg.norm((p - true).reshape(len(init), H + 1, -1, 4)[..., :2], axis=-1).mean(-1)
    e = np.nan_to_num(e, nan=np.inf)
    out[name] = dict(steps=cfg["step"], minutes=cfg["train_minutes"],
                     median={h: float(np.median(e[:, h])) for h in (1, 5, 10, 20, 25, 30, 40, 50)},
                     frac_diverged_err_gt_1={h: float((e[:, h] > 1).mean()) for h in (10, 20, 25, 30, 40, 50)})
    print(name, json.dumps(out[name]))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/ablation_summary.json", "w"), indent=2)
