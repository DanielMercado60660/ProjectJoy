"""Reviewer follow-up to cpg.py: the original CEM search ended on its box edges (f = 4.0 Hz, the upper
frequency bound, for both tasks; several amplitudes at 1.0) and the bwd elite curve was still rising
at iteration 40. This reruns the same CEM with a wider box and more search, still on TRAINING reset
seeds only, and scores the pick on the held-out eval seeds. cpg.json is left untouched.

  box:    f in [0.5, 8] Hz (control Nyquist is 10 Hz at dt = 0.05), A in [0, 2] (clipping to [-1, 1]
          allows squarer, bang-bang-like waves), b in [-1, 1], phases [-pi, pi]
  search: CEM 60 iterations x population 48 x 1 episode, 2 restarts per task (CEM seeds 0 and 1),
          4 worker processes (2 tasks x 2 restarts)
  pick:   per task, the restart whose candidate scores best on 5 further training seeds (950..954)

  ../bin/gpu_run ../.venv/bin/python cpg_wide.py
"""
import json
import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import cpg  # noqa: E402
from common import EVAL_SEEDS, EP_LEN, TASKS  # noqa: E402

cpg.LO = np.array([0.5] + [0.0] * 6 + [-np.pi] * 6 + [-1.0] * 6)
cpg.HI = np.array([8.0] + [2.0] * 6 + [np.pi] * 6 + [1.0] * 6)
ITERS, POP, RESTARTS = 60, 48, 2
HERE = os.path.dirname(os.path.abspath(__file__))


def job(args):
    task, r = args
    th, curve = cpg.cem(task, ITERS, POP, seed=r)
    pick_score = float(np.mean(cpg.rollout(np.stack([th] * 5), task, list(range(950, 955)))))
    return task, r, th.tolist(), curve, pick_score


def main():
    t0 = time.time()
    jobs = [(t, r) for t in TASKS for r in range(RESTARTS)]
    with mp.get_context("spawn").Pool(min(4, len(jobs))) as pool:
        res = pool.map(job, jobs)
    out = {"tuning": f"CEM {ITERS} iters x pop {POP} x 1 episode x {RESTARTS} restarts on training seeds "
                     f"({ITERS * POP * EP_LEN * RESTARTS} env steps per task); f in [0.5,8] Hz, A in [0,2], "
                     f"phi in [-pi,pi], b in [-1,1]; restart picked on training seeds 950-954",
           "restarts": []}
    for task in TASKS:
        mine = [x for x in res if x[0] == task]
        for _, r, th, curve, sc in mine:
            out["restarts"].append(dict(task=task, cem_seed=r, theta=th, elite_curve=curve, train_pick_score=sc))
        _, r, th, curve, sc = max(mine, key=lambda x: x[4])
        th = np.array(th)
        ev = {tk: cpg.rollout(np.stack([th] * len(EVAL_SEEDS)), tk, EVAL_SEEDS).tolist() for tk in TASKS}
        out[task] = dict(theta=th.tolist(), cem_seed=r, train_pick_score=sc, elite_curve=curve, eval=ev)
        print(f"[cpg_wide] {task}: restart {r} (train {sc:.0f}), f = {th[0]:.2f} Hz, eval own "
              f"{np.mean(ev[task]):.0f} ± {np.std(ev[task]):.0f}", flush=True)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "results", "cpg_wide.json"), "w"), indent=1)
    print(f"[cpg_wide] done in {out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
