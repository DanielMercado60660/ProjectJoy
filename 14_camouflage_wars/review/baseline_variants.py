"""Reviewer check: stronger classical baselines, tuned on the seen-family val split only.
Variants: outlier score with (a) min component area swept (creatures are >= ~180 px), and
(b) multi-scale features (sigma s and 2s concatenated). Writes review/baseline_variants.json."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
import baseline as B
from metrics import scene_iou


def seg_outlier_ms(img, sigma, min_area, multiscale):
    sig = (sigma, 2 * sigma) if multiscale else (sigma,)
    F = np.concatenate([B.features(img, s) for s in sig], -1).reshape(-1, 10 * len(sig))
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-3
    s = np.log1p(np.sqrt((((F - med) / mad) ** 2).mean(1)))
    m = (s > B._otsu(s)).reshape(img.shape[:2])
    return B.clean(m, min_area)


def run(X, **kw):
    return np.stack([seg_outlier_ms(x.astype(np.float32) / 255, **kw) for x in X])


if __name__ == "__main__":
    d = np.load("data/val.npz"); X, Y = d["X"][:300], d["Y"][:300]
    grid = {}
    for ms in (False, True):
        for sigma in (1.5, 2.0, 3.0):
            for ma in (30, 80, 150):
                t = time.time()
                grid[f"ms{int(ms)}_s{sigma}_a{ma}"] = float(scene_iou(run(X, sigma=sigma, min_area=ma, multiscale=ms), Y).mean())
                print(f"ms{int(ms)}_s{sigma}_a{ma}", round(grid[f"ms{int(ms)}_s{sigma}_a{ma}"], 4), f"{time.time()-t:.1f}s", flush=True)
    best = max(grid, key=grid.get)
    json.dump(dict(val_first300=grid, best=best), open("review/baseline_variants.json", "w"), indent=1)
    print("best", best, grid[best])
