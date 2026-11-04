"""R6 model selection on a larger in-distribution set: 200 DR scenes (seeds 120000+, never used for
training/tuning/eval). Rule fixed before running (STATUS.md): the final classical version is the one
with the lowest clipped error (the tuning objective); the CNN is scored too for reference.
  ../bin/gpu_run ../.venv/bin/python select_r6.py -> results/r6_tracker_selection.json
"""
import json
from multiprocessing import Pool
import numpy as np
import scene_r6 as R
import track_classical as TC
import track_classical_r6 as C
import train_tracker_r6 as T6

VERSIONS = ["v1", "v2", "v3", "v4"]
PRMS = {v: json.load(open(f"checkpoints/r6_tracker_classical_{v}.json")) for v in VERSIONS}


def run(x):
    pre = C.precompute(x)
    return {v: C.track_scene(pre, p) for v, p in PRMS.items()}


if __name__ == "__main__":
    X, GT, VIS = R.make_set_dr(200, 120_000)
    with Pool(4) as p:
        out = p.map(run, list(X))
    res = {f"classical_{v}": TC.metrics(np.stack([o[v] for o in out]), GT, VIS) for v in VERSIONS}
    for nm in ["nf1", "nf3"]:
        m = T6.load(f"checkpoints/r6_tracker_cnn_{nm}")
        res[f"cnn_{nm}_raw"] = TC.metrics(np.stack([T6.predict(m, x)[0] for x in X]), GT, VIS)
    best = min(VERSIONS, key=lambda v: res[f"classical_{v}"]["clipped_err"])
    res["selected_classical"] = best
    json.dump(res, open("results/r6_tracker_selection.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v)
