"""Score the two reviewer baseline variants (chosen on val only) on the test splits with the same
metrics as evaluate.py. Writes review/baseline_variant_test.json.
  minarea150      : shipped outlier baseline (sigma 2) with min component area 150 (review/baseline_variants.json)
  erode3_s3       : outlier, sigma 3, mask eroded 3 px, min area 30 (review/baseline_erode.json best)"""
import json, pickle, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "review")
import baseline_variants as V, baseline_erode as E
from metrics import summarize, stat_scale
tr = pickle.load(open("data/train_meta.pkl", "rb")); scale = stat_scale([tr])
tr_d = [float(np.sqrt(((np.array(c["stat_dist_raw"]) / scale) ** 2).mean())) for m in tr for c in m["creatures"].values()]
edges = np.quantile(tr_d, [0.25, 0.5, 0.75]).tolist()
variants = {"minarea150": lambda X: V.run(X, sigma=2.0, min_area=150, multiscale=False),
            "erode3_s3": lambda X: E.run(X, sigma=3.0, erode=3, min_area=30)}
out = {}
for split in ("test_id", "test_ood"):
    d = np.load(f"data/{split}.npz"); X, Y = d["X"], d["Y"]
    metas = pickle.load(open(f"data/{split}_meta.pkl", "rb"))
    for name, f in variants.items():
        s, recs, _ = summarize(f(X), Y, metas, scale, edges)
        o = {k: s[k] for k in ("scene_iou", "pixel_iou", "det_rate", "det_rate_isolated", "det_rate_coverage", "fp_per_scene")}
        o["by_delta_det_iso"] = {k: v["det_rate_isolated"] for k, v in s["by_delta"].items()}
        o["by_stat_quartile_det_iso"] = {k: v["det_rate_isolated"] for k, v in s["by_stat_quartile"].items()}
        out[f"{split}/{name}"] = o
        print(split, name, {k: round(v, 3) for k, v in o.items() if not isinstance(v, dict)})
        print("   delta", {k: round(v, 3) for k, v in o["by_delta_det_iso"].items()})
json.dump(out, open("review/baseline_variant_test.json", "w"), indent=1)
