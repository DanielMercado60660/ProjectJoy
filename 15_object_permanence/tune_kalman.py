"""Grid-search the Kalman baseline's process noise q and measurement noise r on the *val* split
(never test).  Selection criterion: mean NLL over all present ball-frames, i.e. the same
objective the network is trained on.  Writes results/kalman_tuning_<variant>.json.

  OMP_NUM_THREADS=1 ../.venv/bin/python tune_kalman.py --variant base
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

import kalman
import metrics
import sim

QS = [0.01, 0.02, 0.035, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0]
RS = [0.07, 0.1, 0.14, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]
_G = {}


def _init(split, n):
    d = sim.load_split(split)
    idx = np.arange(n)
    _G.update(d=d, idx=idx, frames=sim.render_split(d, idx))


def _run(args):
    q, r, bounce = args
    d, idx = _G["d"], _G["idx"]
    m, c = kalman.track_split(d, idx, frames=_G["frames"], q=q, r=r, bounce=bounce)
    res = metrics.evaluate(m, c, d, idx)
    return dict(q=q, r=r, bounce=bounce, nll_all=res["all"]["nll"], nll_hidden=res["hidden"]["nll"],
                err_hidden=res["hidden"]["err"], cov2_hidden=res["hidden"]["cov2"],
                err_reappear=res["reappear"]["err"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "hard"])
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    split = "val" if args.variant == "base" else "hard_val"
    grid = list(itertools.product(QS, RS, [False, True]))
    with Pool(4, initializer=_init, initargs=(split, args.n)) as p:
        rows = p.map(_run, grid)
    best = {}
    for b in [False, True]:
        rs = [x for x in rows if x["bounce"] == b]
        best[f"bounce={b}"] = min(rs, key=lambda x: x["nll_all"])
        best[f"bounce={b} (hidden-nll-optimal)"] = min(rs, key=lambda x: x["nll_hidden"])
    os.makedirs("results", exist_ok=True)
    with open(f"results/kalman_tuning_{args.variant}.json", "w") as f:
        json.dump(dict(split=split, n=args.n, best=best, grid=rows), f, indent=1)
    for k, v in best.items():
        print(k, v)


if __name__ == "__main__":
    main()
