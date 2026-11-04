"""Post-freeze robustness check: is the learned tracker's held-out result a lucky training seed?
Evaluates the frozen CNN (seed 0) and a replicate trained identically with --seed 1, raw argmax,
on every evaluation set (same cached frames as eval_tracker_r6.py). Not used for any selection.
  ../.venv/bin/python replicate_r6.py -> results/r6_tracker_seed_replicate.json
"""
import json
import numpy as np
import scene_r6 as R
import train_tracker_r6 as T6
import eval_tracker_r6 as E

if __name__ == "__main__":
    nets = {"seed0": T6.load("checkpoints/r6_tracker_cnn_nf1"), "seed1": T6.load("checkpoints/r6_tracker_cnn_nf1_seed1")}
    res = {}
    for cond, (X, GT, VIS) in E.sets(True):
        res[cond] = {k: E.boot_ci(np.linalg.norm(np.stack([T6.predict(m, x)[0] for x in X]) - GT, axis=-1), VIS)
                     for k, m in nets.items()}
        print(cond, {k: f"{v['lost_rate']*100:.1f}% {v['mean_err_px']:.2f}px" for k, v in res[cond].items()}, flush=True)
    json.dump(res, open("results/r6_tracker_seed_replicate.json", "w"), indent=1)
