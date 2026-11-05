"""EXPLORATORY, not pre-registered (written after the protocol, run after zeroshot.py): breakdowns of the primary
condition's bearing errors by whether the plate is inside the camera's horizontal view, and of the distance head's
output by remaining time. Reads runs/r8_nav/preds_*.npz; writes results/r8/nav/posthoc.json. Never gating.

    ../.venv/bin/python r8_nav/posthoc.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from r8_nav import zeroshot as Z  # noqa: E402

OUT = HERE / "results" / "r8" / "nav" / "posthoc.json"
HALF_HFOV = {"normal": np.radians(22.5), "wide": np.radians(50.0)}


def load(cond):
    z = np.load(Z.PRED_DIR / f"preds_{cond.replace('|', '_')}.npz")
    out = {}
    for k in z.files:
        i, name = k.split("_", 1)
        out.setdefault(int(i), {})[name] = z[k]
    return out


def main():
    _, eps = Z.load_episodes()
    for e in eps:
        e["_truth"] = Z.truth(e)
    eps = [e for e in eps if len(e["_truth"]["q"]) >= 5]
    res = {"note": "exploratory, post hoc; the pre-registered numbers are in zeroshot.json"}
    for cam in ("normal", "wide"):
        for model in ("vint", "nomad"):
            row = {}
            conds = {"G1": f"{model}|{cam}|G1|0.25s", "wrong": f"{model}|{cam}|wrong|0.25s"}
            if model == "nomad":
                conds["masked"] = f"nomad|{cam}|masked|0.25s"
            preds = {g: load(c) for g, c in conds.items() if (Z.PRED_DIR / f"preds_{c.replace('|', '_')}.npz").exists()}
            if "G1" not in preds:
                continue
            for split in ("in_view", "out_of_view"):
                errs = {g: [] for g in preds}
                errs["straight"] = []
                for ei, e in enumerate(eps):
                    tr = e["_truth"]
                    q = tr["q"]
                    far = tr["dist"][q] >= Z.NEAR_M
                    bp = tr["bearing_plate"][q]
                    inv = np.abs(bp) < HALF_HFOV[cam]
                    sel = far & (inv if split == "in_view" else ~inv)
                    for g, p in preds.items():
                        wp = p[ei]["wp"]
                        th = np.arctan2(wp[:, 0, 1], wp[:, 0, 0])
                        errs[g].append(Z.deg(th[sel] - bp[sel]))
                    errs["straight"].append(Z.deg(bp[sel]))
                row[split] = {g: {"median_err_deg": float(np.median(np.concatenate(v))) if sum(map(len, v)) else None,
                                  "n_frames": int(sum(map(len, v)))} for g, v in errs.items()}
            # the A1 trained-horizon slice (last 5 s, body >= 0.15 m from the plate): models vs straight ahead
            hz = {g: [] for g in preds}
            hz["straight"] = []
            for ei, e in enumerate(eps):
                tr = e["_truth"]
                q = tr["q"]
                sl = (tr["remaining"][q] <= Z.HORIZON_S) & (tr["dist"][q] >= Z.NEAR_SLICE_M)
                bp = tr["bearing_plate"][q]
                for g, p in preds.items():
                    wp = p[ei]["wp"]
                    hz[g].append(Z.deg(np.arctan2(wp[:, 0, 1], wp[:, 0, 0])[sl] - bp[sl]))
                hz["straight"].append(Z.deg(bp[sl]))
            row["last5s_slice"] = {g: {"median_err_deg": float(np.median(np.concatenate(v))) if sum(map(len, v)) else None,
                                       "n_frames": int(sum(map(len, v)))} for g, v in hz.items()}
            # distance head by remaining-time bins (real vs wrong goal)
            bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40)]
            dist = {}
            for g in ("G1", "wrong"):
                if g not in preds:
                    continue
                d = {f"{a}-{b}s": [] for a, b in bins}
                for ei, e in enumerate(eps):
                    rem = e["_truth"]["remaining"][e["_truth"]["q"]]
                    for a, b in bins:
                        m = (rem > a) & (rem <= b)
                        d[f"{a}-{b}s"].extend(preds[g][ei]["dist"][m].tolist())
                dist[g] = {k: (round(float(np.median(v)), 2) if v else None) for k, v in d.items()}
            row["distance_median_by_remaining_time"] = dist
            res[f"{model}|{cam}"] = row
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
