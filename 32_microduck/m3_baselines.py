"""M3: the find_ball no-memory baselines (perfect recognition, true pose and layout, no use of where the
ball went; oracle.no_memory_search) on the same episodes as the M3 evaluation: train 60 and every dev
family, k=0. Writes results/m3/find_ball_baselines.json."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m3_eval import load_set  # noqa: E402
from oracle import baseline_for  # noqa: E402
from room import World  # noqa: E402
from tasks import make_episode, run_episode  # noqa: E402


def one(args):
    cfg, setname, which = args
    ep = make_episode(cfg, "find_ball", 0)
    w = World(cfg)
    res, _, _ = run_episode(cfg, ep, baseline_for(w, which), world=w)
    return {"config": cfg["id"], "set": setname, "who": which, "success": bool(res["success"]),
            "sim_time": res["sim_time"], "hider": ep["ball_launch"]["hider"]}


if __name__ == "__main__":
    sets = [("train", 60), ("instances-dev", 50), ("arrangements-dev", 50), ("appearances-dev", 50), ("distractor-dev", 40)]
    jobs = [(c, s, w) for s, n in sets for c in load_set(s, n) for w in ("all_hiders", "occluder_only")]
    with Pool(4) as pool:
        rows = pool.map(one, jobs, chunksize=2)
    summ = {}
    for s, _ in sets:
        for w in ("all_hiders", "occluder_only"):
            rs = [r for r in rows if r["set"] == s and r["who"] == w]
            ok = [r for r in rs if r["success"]]
            summ[f"{w}/{s}"] = {"n": len(rs), "success": len(ok),
                                "within_20s": int(sum(r["sim_time"] <= 20 for r in ok)),
                                "median_s_success": float(np.median([r["sim_time"] for r in ok])) if ok else None}
    (HERE / "results" / "m3" / "find_ball_baselines.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    for k, v in summ.items():
        print(k, v)
