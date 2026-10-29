"""Visual of the painter's best camouflage: for each delta, the held-out scene where the painter
(minimising the frozen seed-0 U-Net) caused the largest IoU drop. Columns: random skin (draw 0) with
seed-0 U-Net mask, painted skin with seed-0 mask, painted skin with seed-1 mask, painted skin with
the reviewer baseline mask. Green = ground-truth outline, red fill = prediction.
Writes media/painter_best.png (and media/painter_grid.png: 8 random scenes, random vs painted)."""
import argparse, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
from scipy.ndimage import binary_erosion
sys.path.insert(0, "review")
import model as M
import baseline_erode as E

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results/painter")
ap.add_argument("--out", default="media")
ap.add_argument("--ckpt0", default="checkpoints/unet")
ap.add_argument("--ckpt1", default="checkpoints/unet_seed1")
ap.add_argument("--tag", default="painter")
a = ap.parse_args()
sc, im = np.load(f"{a.res}/scores.npz"), np.load(f"{a.res}/images.npz")
n0, n1 = M.load(a.ckpt0), M.load(a.ckpt1)
pred = lambda net, x: np.array(mx.sigmoid(net(mx.array(x[None].astype(np.float32) / 255))))[0] > 0.5
mask, delta, fam = sc["mask"], sc["delta"], sc["family"]
rows = np.arange(len(delta))
u0 = sc["unet0"]
drop = u0[:, 0] - u0[rows, im["pick_unet0"]]


def show(ax, x, p, g, title):
    o = x.astype(np.float32) / 255
    o = np.where(p[..., None], 0.55 * o + 0.45 * np.array([1, 0.1, 0.1]), o)
    edge = g & ~binary_erosion(g)
    o[edge] = [0.1, 1, 0.2]
    ax.imshow(o); ax.set_title(title, fontsize=7); ax.axis("off")


def iou(p, g):
    return (p & g).sum() / max((p | g).sum(), 1)


ds = sorted(set(delta.tolist()))
fig, axs = plt.subplots(len(ds), 5, figsize=(10, 2.2 * len(ds)))
for r, d in enumerate(ds):
    idx = np.where(delta == d)[0]
    i = idx[np.argmax(drop[idx])]
    xr, xp, g = im["random0"][i], im["painted_by_unet0"][i], mask[i]
    axs[r, 0].imshow(xp); axs[r, 0].set_title(f"painted ({fam[i]}, δ={d})", fontsize=7); axs[r, 0].axis("off")
    pr = pred(n0, xr); show(axs[r, 1], xr, pr, g, f"random skin · U-Net0 IoU {iou(pr, g):.2f}")
    pp = pred(n0, xp); show(axs[r, 2], xp, pp, g, f"painted · U-Net0 (attacked) {iou(pp, g):.2f}")
    p1 = pred(n1, xp); show(axs[r, 3], xp, p1, g, f"painted · U-Net1 (transfer) {iou(p1, g):.2f}")
    pb = E.seg(xp.astype(np.float32) / 255, sigma=3.0, erode=3, min_area=30)
    show(axs[r, 4], xp, pb, g, f"painted · eroded baseline {iou(pb, g):.2f}")
fig.suptitle("Painter (best of K skins at fixed δ, minimising frozen U-Net0) — held-out families; "
             "green = true outline, red = predicted", fontsize=8)
plt.tight_layout(); plt.savefig(f"{a.out}/{a.tag}_best.png", dpi=130); plt.close()

rng = np.random.default_rng(0)
sel = rng.choice(len(delta), 8, replace=False)
fig, axs = plt.subplots(2, 8, figsize=(16, 4.4))
for c, i in enumerate(sel):
    g = mask[i]
    for r, key in enumerate(("random0", "painted_by_unet0")):
        x = im[key][i]; p = pred(n0, x)
        show(axs[r, c], x, p, g, f"{'random' if r == 0 else 'painted'} δ={delta[i]} IoU0 {iou(p, g):.2f}")
fig.suptitle("Random scenes: top = random skin draw, bottom = painter's pick (vs U-Net0)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{a.out}/{a.tag}_grid.png", dpi=110); plt.close()
print("wrote", f"{a.out}/{a.tag}_best.png", f"{a.out}/{a.tag}_grid.png")
