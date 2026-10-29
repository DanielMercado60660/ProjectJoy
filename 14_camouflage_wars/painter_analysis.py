"""What did the painter pick? Simple image statistics of the creature region, random skin (draw 0)
vs the skin picked against U-Net0, on the held-out painter run:
  skin_std      : mean per-channel std of pixels inside the creature (texture contrast of the skin)
  bg_std        : same for a 6-px ring of background around it
  mean_colour_gap : |mean RGB inside - mean RGB of the ring| (L2, 0-1 units): a crude "visible to a
                  colour-blob detector" proxy
Writes results/painter/analysis.json."""
import json, sys
import numpy as np
from scipy.ndimage import binary_dilation

res = sys.argv[1] if len(sys.argv) > 1 else "results/painter"
sc, im = np.load(f"{res}/scores.npz"), np.load(f"{res}/images.npz")
M, delta = sc["mask"], sc["delta"]
rows = np.arange(len(delta))


def stats(X):
    out = []
    for x, m in zip(X.astype(np.float32) / 255, M):
        ring = binary_dilation(m, iterations=6) & ~binary_dilation(m, iterations=1)
        out.append((x[m].std(0).mean(), x[ring].std(0).mean(), np.linalg.norm(x[m].mean(0) - x[ring].mean(0))))
    return np.array(out)


r, p = stats(im["random0"]), stats(im["painted_by_unet0"])
out = {}
for bn, sel in [("all", np.ones(len(delta), bool))] + [(f"delta={d}", delta == d) for d in sorted(set(delta))]:
    o = {}
    for j, k in enumerate(("skin_std", "bg_std", "mean_colour_gap")):
        o[f"{k}/random0"], o[f"{k}/painted_by_unet0"] = float(r[sel, j].mean()), float(p[sel, j].mean())
    o["frac_painted_skin_flatter_than_random0"] = float((p[sel, 0] < r[sel, 0]).mean())
    o["frac_painted_skin_std_below_half_bg"] = float((p[sel, 0] < 0.5 * p[sel, 1]).mean())
    o["frac_random0_skin_std_below_half_bg"] = float((r[sel, 0] < 0.5 * r[sel, 1]).mean())
    out[bn] = o
for k in ("base", "base_er", "unet1", "unet0"):
    if k in sc:
        # among painted (vs U-Net0) creatures that U-Net0 misses (IoU<0.5): how often does detector k find them?
        miss = sc["unet0"][rows, im["pick_unet0"]] < 0.5
        out[f"painted_missed_by_unet0/n"] = int(miss.sum())
        out[f"painted_missed_by_unet0/{k}_det_rate"] = float((sc[k][rows, im["pick_unet0"]][miss] >= 0.5).mean())
json.dump(out, open(f"{res}/analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# Where does the U-Net IoU drop come from: missed creature pixels or false positives elsewhere?
# (The background is pixel-identical across candidates, so any change outside the creature is a
# non-local effect of the skin: receptive field / GroupNorm statistics.)
import mlx.core as mx
import model as UM
far = np.stack([~binary_dilation(m, iterations=3) for m in sc["mask"]])
for ck_name, ck in (("unet0", "checkpoints/unet"), ("unet1", "checkpoints/unet_seed1")):
    net = UM.load(ck)
    for key in ("random0", "painted_by_unet0"):
        X = im[key]
        P = np.concatenate([np.array(mx.sigmoid(net(mx.array(X[i:i + 200].astype(np.float32) / 255)))) > 0.5
                            for i in range(0, len(X), 200)])
        recall = (P & sc["mask"]).sum((1, 2)) / sc["mask"].sum((1, 2))
        fp = (P & far).sum((1, 2))
        out[f"nonlocal/{ck_name}/{key}/recall_inside_creature"] = float(recall.mean())
        out[f"nonlocal/{ck_name}/{key}/fp_pixels_beyond_3px"] = float(fp.mean())
        out[f"nonlocal/{ck_name}/{key}/frac_scenes_fp_ge_20px"] = float((fp >= 20).mean())
json.dump(out, open(f"{res}/analysis.json", "w"), indent=1)
print({k: round(v, 3) for k, v in out.items() if k.startswith("nonlocal")})
