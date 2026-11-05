"""R8 Gate C, reported (post hoc): 95% confidence intervals for the bake-off's primary linear lamp accuracy and
plate AUC, by a cluster bootstrap over the 30 held-out configs (frames of one room are not independent).
Refits the same deterministic linear probes (rep_probes.linear_probe) to get per-frame predictions.

    ../.venv/bin/python r8_rep/rep_probe_ci.py      -> results/r8/probes/ci.json
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rep_probes as R  # noqa: E402


def scores(F, y, tr, steps=800):
    """rep_probes.linear_probe, returning the per-item score (logit difference)."""
    mx.random.seed(0)
    F = R._std(F.astype(np.float32), tr)
    lin = nn.Linear(F.shape[1], 2)
    vg = nn.value_and_grad(lin, lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean() + 1e-3 * (m.weight ** 2).sum())
    opt = optim.Adam(learning_rate=1e-2)
    Xt, Yt = mx.array(F[tr]), mx.array(y[tr].astype(np.int32))
    for _ in range(steps):
        _, g = vg(lin, Xt, Yt)
        opt.update(lin, g)
        mx.eval(lin.parameters())
    o = np.array(lin(mx.array(F)))
    return o[:, 1] - o[:, 0]


def main(B=1000):
    it = R.load_items()
    rng = np.random.default_rng(0)
    out = {"method": "cluster bootstrap over held-out configs (1,000 resamples, rng 0); point = the bake-off number"}
    arms = [a for a in R.ORDER if a != "cnn64" and (R.FEAT / f"{a}.npz").exists()]
    for arm in arms:
        z = np.load(R.FEAT / f"{arm}.npz")
        rec = {}
        for t, name, key in ((0, "lamp", "glob"), (1, "plate", "glob")):
            s = it["task"] == t
            fr, y, cfg = it["frame"][s], (it["y"][s] > 0.5), it["cfg"][s]
            tr = cfg < 270
            sc = scores(z[key][fr], y.astype(np.float32), tr)
            te_cfg = np.unique(cfg[~tr])
            vals = []
            for _ in range(B):
                pick = rng.choice(te_cfg, len(te_cfg), replace=True)
                idx = np.concatenate([np.nonzero((~tr) & (cfg == c))[0] for c in pick])
                vals.append(((sc[idx] > 0) == y[idx]).mean() if t == 0 else R.auc(sc[idx], y[idx]))
            point = ((sc[~tr] > 0) == y[~tr]).mean() if t == 0 else R.auc(sc[~tr], y[~tr])
            rec[name + ("_acc" if t == 0 else "_auc")] = [round(float(point), 4), round(float(np.percentile(vals, 2.5)), 4),
                                                         round(float(np.percentile(vals, 97.5)), 4)]
        out[arm] = rec
        print(arm, json.dumps(rec), flush=True)
    (R.RES / "ci.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
