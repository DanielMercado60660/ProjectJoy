"""Reviewer check 2: the sigma-2 feature pooling leaves a 2-3 px halo around every baseline blob
(red rings in media/ood_hardest.png). Sweep an erosion of the outlier mask before cleaning, plus
min_area, on the seen-family val split only (first 300 scenes, as tune_baseline.py)."""
import json, sys
import numpy as np
from scipy.ndimage import binary_erosion
sys.path.insert(0, ".")
import baseline as B
from metrics import scene_iou


def seg(img, sigma=2.0, erode=0, min_area=30):
    F = B.features(img, sigma).reshape(-1, 10)
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-3
    s = np.log1p(np.sqrt((((F - med) / mad) ** 2).mean(1)))
    m = (s > B._otsu(s)).reshape(img.shape[:2])
    if erode:
        m = binary_erosion(m, iterations=erode)
    return B.clean(m, min_area)


def run(X, **kw):
    return np.stack([seg(x.astype(np.float32) / 255, **kw) for x in X])


if __name__ == "__main__":
    d = np.load("data/val.npz"); X, Y = d["X"][:300], d["Y"][:300]
    grid = {}
    for sigma in (1.5, 2.0, 3.0):
        for er in (0, 1, 2, 3):
            for ma in (30, 150):
                k = f"s{sigma}_e{er}_a{ma}"
                grid[k] = float(scene_iou(run(X, sigma=sigma, erode=er, min_area=ma), Y).mean())
                print(k, round(grid[k], 4), flush=True)
    best = max(grid, key=grid.get)
    json.dump(dict(val_first300=grid, best=best), open("review/baseline_erode.json", "w"), indent=1)
    print("best", best, grid[best])
