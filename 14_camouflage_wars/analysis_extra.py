"""Extra slices: near-perfect camouflage (δ < 0.25 sub-bins, and δ < 0.1 by skin family), false
positives by background family, and the detrended view of the flat-looking ripples scene (943)."""
import collections
import json, pickle
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import model as M, baseline as B
from metrics import creature_records
from train import predict

net = M.load("checkpoints/unet")
choice = json.load(open("baseline_choice.json"))["best"]
out = {}
for split in ("test_id", "test_ood"):
    d = np.load(f"data/{split}.npz"); X, Y = d["X"], d["Y"]
    metas = pickle.load(open(f"data/{split}_meta.pkl", "rb"))
    for name, P in (("unet", predict(net, X) > 0.5), ("baseline", B.run(X, choice["method"], choice["sigma"]))):
        recs, fps = creature_records(P, Y, metas)
        o = {}
        for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.25)):
            sel = [r for r in recs if r["kind"] == "same" and lo <= r["delta"] < hi and r["isolated"]]
            o[f"det_iso_delta_{lo}-{hi}"] = (len(sel), float(np.mean([r["iou"] >= 0.5 for r in sel])))
        fams = collections.defaultdict(list)
        for r in recs:
            if r["kind"] == "same" and r["delta"] < 0.1 and r["isolated"]:
                fams[r["family"]].append(r["iou"] >= 0.5)
        o["det_iso_delta_lt_0.1_by_skin_family"] = {f: (len(v), float(np.mean(v))) for f, v in sorted(fams.items())}
        fam = np.array([m["bg_family"] for m in metas])
        o["fp_per_scene_by_bg_family"] = {f: float(fps[fam == f].mean()) for f in sorted(set(fam))}
        out[f"{split}/{name}"] = o
# Scene 943 of test_ood looks flat; blur + remove the illumination plane to show the faint ripple
# pattern and the creature's phase break (media/flat_ripples_scene943_detrended.png).
x = np.load("data/test_ood.npz")["X"][943].astype(np.float32) / 255
yy, xx = np.mgrid[0:96, 0:96]
s_ = gaussian_filter(x, (3, 3, 0))
A = np.stack([np.ones(96 * 96), xx.ravel(), yy.ravel()], 1)
r = np.stack([s_[..., c].ravel() - A @ np.linalg.lstsq(A, s_[..., c].ravel(), rcond=None)[0] for c in range(3)], 1)
r = r.reshape(96, 96, 3)
r = (r - r.min((0, 1))) / (np.ptp(r, (0, 1)) + 1e-9)
Image.fromarray((r * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST).save(
    "media/flat_ripples_scene943_detrended.png")
json.dump(out, open("results/extra.json", "w"), indent=1)
print(json.dumps(out, indent=1))
