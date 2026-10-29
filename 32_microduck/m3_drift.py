"""M3: odometry drift inside real bot episodes, with and without landmark correction (truth scores
only). Runs go_and_return k=0 on the given training configs; records the distance between the bot's
pose and the true pose (start frame) every tick. Writes results/m3/drift_<tag>.json."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import interface  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, make_episode, run_episode  # noqa: E402


def run(args):
    cfg, gain = args
    ep = make_episode(cfg, "go_and_return", 0)
    w = World(cfg)
    fac, bot = controller("go_and_return", BUDGET_TICKS["go_and_return"], {"anchor_gain": gain})
    st = cfg["layout"]["start"]
    x0, y0 = st["pos"]
    c, s = np.cos(-st["yaw"]), np.sin(-st["yaw"])
    errs = []
    orig = interface.Interface.tick

    def tick(self, cmd):
        o = orig(self, cmd)
        p = w.duck.pos()[:2]
        t = np.array([c * (p[0] - x0) - s * (p[1] - y0), s * (p[0] - x0) + c * (p[1] - y0)])
        errs.append(float(np.linalg.norm(np.array(bot.pose()[:2]) - t)))
        return o
    interface.Interface.tick = tick
    try:
        res, _, _ = run_episode(cfg, ep, fac, world=w)
    finally:
        interface.Interface.tick = orig
    return {"config": cfg["id"], "gain": gain, "success": res["success"], "err_end": errs[-1],
            "err_max": max(errs), "err_mean": float(np.mean(errs)), "ticks": len(errs)}


if __name__ == "__main__":
    tag = sys.argv[1]
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][:20]
    with Pool(4) as pool:
        rows = pool.map(run, [(c, g) for g in (0.0, 0.15) for c in cfgs])
    summ = {}
    for g in (0.0, 0.15):
        rs = [r for r in rows if r["gain"] == g]
        summ[str(g)] = {k: round(float(np.median([r[k] for r in rs])), 3) for k in ("err_end", "err_max", "err_mean")}
        summ[str(g)]["success"] = int(sum(r["success"] for r in rs))
    (HERE / "results" / "m3" / f"drift_{tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))
