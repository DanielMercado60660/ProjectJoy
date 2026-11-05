"""Round 6 tables: per-set means with bootstrap CIs and paired clip-bootstrap differences vs reference systems.

  ../.venv/bin/python r6_stats.py --sets test27,test28 --systems gt,gl64,old,recon,gan --ref old --out results_r6/tables_test.md
Reads results_r6/mos/<set>/<system>.json (r6_utmos.py) and results_r6/metrics/<set>/<system>.json (r6_eval.py).
Bootstrap: 5,000 resamples of clips (seed 0); paired differences resample the same clips for both systems.
Clips are the unit (one speaker for LJ sets), so the CIs cover clip sampling only, not training-seed variance.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results_r6")
METRICS = [("utmos", "UTMOS ↑", 2), ("gpe_pct", "GPE % ↓", 2), ("f0_rmse_cents", "f0 RMSE ¢ ↓", 1), ("vde_pct", "VDE % ↓", 2),
           ("mel_l1", "mel L1 ↓", 3), ("mrstft", "MR-STFT ↓", 3)]


def load(set_, sys_):
    d = {}
    p = os.path.join(R, "mos", set_, f"{sys_}.json")
    if os.path.exists(p):
        for r in json.load(open(p))["clips"]:
            d.setdefault(r["id"], {})["utmos"] = r["mos"]
    p = os.path.join(R, "metrics", set_, f"{sys_}.json")
    if os.path.exists(p):
        for r in json.load(open(p))["per_clip"]:
            d.setdefault(r["id"], {}).update({k: v for k, v in r.items() if k != "id"})
    return d


def boot(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(set_, systems, refs, labels):
    data = {s: load(set_, s) for s in systems}
    ids = sorted(set.intersection(*[set(d) for d in data.values() if d]))
    lines = [f"### {set_} ({len(ids)} clips)", "", "| system | " + " | ".join(h for _, h, _ in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    out = {"set": set_, "n": len(ids), "systems": {}, "paired": {}}
    for s in systems:
        cells, rec = [], {}
        for k, _, dp in METRICS:
            if not all(k in data[s].get(i, {}) for i in ids):
                cells.append("–"); continue
            v = np.array([data[s][i][k] for i in ids])
            m, lo, hi = boot(v)
            rec[k] = [m, lo, hi]
            cells.append(f"{m:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]" if k == "utmos" else f"{m:.{dp}f}")
        out["systems"][s] = rec
        lines.append(f"| {labels.get(s, s)} | " + " | ".join(cells) + " |")
    for ref in refs:
        lines += ["", f"Paired differences vs **{labels.get(ref, ref)}** (system − reference, mean [95% CI]; per-clip wins in parentheses):", "",
                  "| system | " + " | ".join(h for _, h, _ in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for s in systems:
            if s == ref:
                continue
            cells = []
            for k, _, dp in METRICS:
                if not all(k in data[s].get(i, {}) and k in data[ref].get(i, {}) for i in ids):
                    cells.append("–"); continue
                d = np.array([data[s][i][k] - data[ref][i][k] for i in ids])
                m, lo, hi = boot(d)
                better = int(np.sum(d > 0)) if k == "utmos" else int(np.sum(d < 0))
                out["paired"][f"{s}-{ref}:{k}"] = [m, lo, hi, better]
                cells.append(f"{m:+.{dp}f} [{lo:+.{dp}f}, {hi:+.{dp}f}] ({better}/{len(ids)})")
            lines.append(f"| {labels.get(s, s)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--ref", default="old", help="comma list of reference systems for paired tables")
    ap.add_argument("--labels", default="", help="sys=label,sys=label")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = dict(kv.split("=", 1) for kv in a.labels.split(",") if kv)
    md, js = [], []
    for s in a.sets.split(","):
        t, o = table(s, a.systems.split(","), a.ref.split(","), labels)
        md.append(t); js.append(o)
    open(a.out, "w").write("\n".join(md))
    json.dump(js, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
