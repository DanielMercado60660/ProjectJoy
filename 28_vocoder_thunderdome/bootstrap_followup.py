"""Follow-up (round 3), added in review: paired bootstrap over the 24 held-out clips for the GPE differences quoted in
RESULTS.md "Follow-up (round 3)" (the builder computed them inline; this makes them traceable).
Usage: ../.venv/bin/python bootstrap_followup.py  -> results/followup_val3k/bootstrap_gpe.json
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = {"gl64": ("results/metrics.json", "gl64"), "main": ("results/metrics.json", "neural"),
       "best": ("results/followup_val3k/best/metrics.json", "neural"),
       "last": ("results/followup_val3k/last/metrics.json", "neural"),
       "ablation3k": ("results_3k/metrics.json", "neural")}


def main(n_boot=5000, seed=0, key="gpe_pct"):
    pc = {}
    for n, (f, m) in SRC.items():
        j = json.load(open(os.path.join(ROOT, f)))
        pc[n] = {d["id"]: d[key] for d in j["per_clip"] if d["method"] == m}
    ids = sorted(pc["gl64"]); assert all(set(v) == set(ids) for v in pc.values())
    out = {"n_clips": len(ids), "n_boot": n_boot, "seed": seed, "metric": key, "unit": "clip (paired)", "pairs": {}}
    for a, b in [("best", "gl64"), ("best", "main"), ("last", "gl64"), ("last", "main"), ("last", "ablation3k")]:
        d = np.array([pc[a][i] - pc[b][i] for i in ids])
        idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        out["pairs"][f"{a}-{b}"] = {"mean_diff": round(float(d.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                                    "a_lower_on": int((d < 0).sum())}
        print(f"{a}-{b}: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  {a} lower on {(d < 0).sum()}/{len(d)}")
    return out


def schedule_and_selection(n_boot=5000, seed=0):
    """Same paired bootstrap for the schedule contrast (last vs ablation3k) and the selection contrast (best vs last)
    on several metrics. Clip-level CIs only: they say nothing about seed-to-seed variance (one seed per run)."""
    keys = ["gpe_pct", "vde_pct", "f0_rmse_cents", "periodicity_rmse", "mel_l1", "mrstft"]
    res = {}
    for k in keys:
        pc = {}
        for n, (f, m) in SRC.items():
            j = json.load(open(os.path.join(ROOT, f)))
            pc[n] = {d["id"]: d[k] for d in j["per_clip"] if d["method"] == m}
        ids = sorted(pc["gl64"])
        for a, b in [("ablation3k", "last"), ("best", "last")]:
            d = np.array([pc[a][i] - pc[b][i] for i in ids])
            idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            res.setdefault(f"{a}-{b}", {})[k] = {"mean_diff": round(float(d.mean()), 4),
                                                 "ci95": [round(float(lo), 4), round(float(hi), 4)]}
            print(f"{a}-{b} {k}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return res


if __name__ == "__main__":
    out = main()
    out["other_metrics"] = schedule_and_selection()
    json.dump(out, open(os.path.join(ROOT, "results", "followup_val3k", "bootstrap_gpe.json"), "w"), indent=1)
