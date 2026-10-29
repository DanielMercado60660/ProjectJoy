"""Pick the classical baseline variant and window size on the seen-family val split only."""
import json, pickle, time
import numpy as np
import baseline as B
from metrics import scene_iou

d = np.load("data/val.npz")
X, Y = d["X"][:300], d["Y"][:300]
res = {}
for method in ("outlier", "kmeans"):
    for sigma in (1.0, 1.5, 2.0, 3.0, 4.5, 6.0):
        t = time.time()
        P = B.run(X, method, sigma)
        res[f"{method}_s{sigma}"] = float(scene_iou(P, Y).mean())
        print(method, sigma, round(res[f"{method}_s{sigma}"], 4), f"{(time.time()-t)/len(X)*1000:.1f} ms/scene", flush=True)
best = max(res, key=res.get)
m, s = best.split("_s")
json.dump(dict(grid=res, best=dict(method=m, sigma=float(s))), open("baseline_choice.json", "w"), indent=1)
print("best", best)
