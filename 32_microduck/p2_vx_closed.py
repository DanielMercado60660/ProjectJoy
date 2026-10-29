"""Phase 2: forward commands the policy actually sends in closed loop (set_lamp dev episodes).

    python p2_vx_closed.py <weights> [...]     # -> results/p2/vx_closed.jsonl (one line per weights)
Same 40 episodes as p2_select (instances/arrangements/appearances/distractor-dev 0000-0009, k=0). Logs every
command's vx and reports, split by press-needed / no-press: mean vx, share of ticks with vx >= 0.2 (the body's
gait start, results/m0/body_frozen.json) and share with 0 < vx < 0.2 (commands that do not start a gait).
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_eval import load_set  # noqa: E402
from p2_select import SETS  # noqa: E402


def one(args):
    cfg, w = args
    from p2_learner import P2Controller
    from room import World
    from tasks import make_episode, run_episode
    pc = P2Controller(w)
    vx = []

    def fac(iface, instruction):
        g = pc.factory(iface, instruction)
        cmd = next(g)
        while True:
            vx.append(cmd.vx)
            obs = yield cmd
            cmd = g.send(obs)

    ep = make_episode(cfg, "set_lamp", 0)
    res, _, _ = run_episode(cfg, ep, fac, world=World(cfg))
    return {"config": cfg["id"] if isinstance(cfg, dict) and "id" in cfg else None,
            "press_needed": bool(res["needed_presses"]), "success": bool(res["success"]), "vx": vx}


def main():
    for w in sys.argv[1:]:
        jobs = [(c, w) for s in SETS for c in load_set(s, 10)]
        with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
            rows = pool.map(one, jobs, chunksize=1)
        res = {"weights": w, "episodes": len(rows), "successes": int(sum(r["success"] for r in rows))}
        for name, pn in (("press_needed", True), ("no_press", False)):
            v = np.concatenate([np.array(r["vx"]) for r in rows if r["press_needed"] == pn])
            res[name] = {"episodes": sum(r["press_needed"] == pn for r in rows), "ticks": int(len(v)),
                         "mean_vx": round(float(v.mean()), 3), "share_ge_0.2": round(float((v >= 0.2).mean()), 3),
                         "share_between_0_and_0.2": round(float(((v > 0) & (v < 0.2)).mean()), 3)}
        print(res)
        with open(HERE / "results" / "p2" / "vx_closed.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
