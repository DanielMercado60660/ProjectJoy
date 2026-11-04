"""Round-2 selection (rule fixed in STATUS.md 2026-11-04 16:31 before any round-2 run): for each
tracker, R1 vs R2 by lowest clipped error on 200 new DR-v2 scenes (seeds 130000+, never used
elsewhere). The winner is the round-2 official tracker regardless of held-out results.
  ../bin/gpu_run ../.venv/bin/python select_r6_round2.py -> results/r6_tracker_selection_r2.json
"""
import json
from multiprocessing import Pool
import numpy as np
import scene_r6 as R
import track_classical as TC
import track_classical_r6 as C
import train_tracker_r6 as T6

PRMS = {"r1": json.load(open("checkpoints/r6_tracker_classical.json")),
        "r2": json.load(open("checkpoints/r6_tracker_classical_r2.json"))}


def run(x):
    pre = C.precompute(x)
    return {v: C.track_scene(pre, p) for v, p in PRMS.items()}


if __name__ == "__main__":
    X, GT, VIS = R.make_set_dr(200, R.DR_SEED["dr2sel"], fn=R.render_dr2)
    with Pool(4) as p:
        out = p.map(run, list(X))
    res = {f"classical_{v}": TC.metrics(np.stack([o[v] for o in out]), GT, VIS) for v in PRMS}
    for v, path in [("r1", "checkpoints/r6_tracker_cnn_nf1"), ("r2", "checkpoints/r6_tracker_cnn_r2")]:
        m = T6.load(path)
        res[f"cnn_{v}_raw"] = TC.metrics(np.stack([T6.predict(m, x)[0] for x in X]), GT, VIS)
    res["selected_classical"] = min(PRMS, key=lambda v: res[f"classical_{v}"]["clipped_err"])
    res["selected_cnn"] = min(["r1", "r2"], key=lambda v: res[f"cnn_{v}_raw"]["clipped_err"])
    json.dump(res, open("results/r6_tracker_selection_r2.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v)
