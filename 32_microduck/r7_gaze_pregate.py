"""Round 7 Stage 4A pre-gate: the chicken-head reflex, 20 matched set-lamp bot episodes gaze on vs off.

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_gaze_pregate.py   # ~10 min, ~0.1 GB
        -> episodes/r7_gaze/{on,off}/set_lamp/*.npz, results/r7/gaze_pregate.json

Rooms: the first 20 press-needed train configs (set_lamp, k=6: a fresh episode index, never recorded before).
Metrics (gate from the plan): per-tick 64 px grey frame change while walking (bot vx > 0.15) down >= 20% with
the reflex on (off: median ~15 grey levels in M3.1), plate-sighting share on the approach (lamp check -> first
press; the recorded memory's button track seen this tick) >= 0.65 (off ~0.70), falls <= 1 of 20. The reflex is
`gaze.py` (params={'gaze': True}); bot_pose is recorded (new field) in these episodes.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2_learner import prep_frame  # noqa: E402
from tasks import make_episode  # noqa: E402

OUT = HERE / "episodes" / "r7_gaze"
K = 6


def job(args):
    from record import record_episode
    cfg, gaze = args
    r = record_episode(cfg, "train", "set_lamp", K, params={"gaze": True} if gaze else None,
                       out_dir=OUT / ("on" if gaze else "off"), cold_renderer=True, bot_pose=True)
    z = np.load(r["path"], allow_pickle=False)
    H = json.loads(str(z["header"]))
    frames = z["frames"]
    grey = np.stack([prep_frame(f, 64).astype(np.float32).mean(-1) for f in frames])
    walking = z["cmd_bot"][:, 0] > 0.15
    change = np.abs(np.diff(grey, axis=0)).mean((1, 2))                     # per tick, grey levels
    walk_change = change[walking[1:] & np.isfinite(change)]
    mem = json.loads(str(z["memory"]))
    checks = [e["tick"] for e in H["events"] if e["event"] == "lamp_check"]
    presses = [e["tick"] for e in H["events"] if e["event"] == "press"]
    share = None
    if checks and presses and presses[0] > checks[0]:
        seen = [any(t[0] == "button" and t[4] == k for t in mem[k]["tracks"]) for k in range(checks[0], presses[0] + 1)]
        share = float(np.mean(seen))
    return {"config": cfg["id"], "gaze": gaze, "success": r["success"], "ticks": r["ticks"], "falls": H["result"]["falls"],
            "walk_ticks": int(walking.sum()), "frame_change_walking_median": float(np.median(walk_change)) if len(walk_change) else None,
            "approach_sighting_share": share, "presses": H["result"]["presses"], "needed": H["result"]["needed_presses"]}


def main():
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"]
    rooms = [c for c in cfgs if (make_episode(c, "set_lamp", K)["instruction"]["target"] == "on") != bool(c["layout"]["lamp_initial_on"])][:20]
    jobs = [(c, g) for c in rooms for g in (False, True)]
    with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    rep = {}
    for g in (False, True):
        rs = [r for r in rows if r["gaze"] == g]
        fc = [r["frame_change_walking_median"] for r in rs if r["frame_change_walking_median"] is not None]
        sh = [r["approach_sighting_share"] for r in rs if r["approach_sighting_share"] is not None]
        rep["on" if g else "off"] = {"episodes": len(rs), "successes": int(sum(r["success"] for r in rs)),
                                     "falls": int(sum(r["falls"] for r in rs)),
                                     "frame_change_walking_median": float(np.median(fc)) if fc else None,
                                     "approach_sighting_share_median": float(np.median(sh)) if sh else None, "rows": rs}
    off, on = rep["off"], rep["on"]
    rep["gate"] = {"frame_change_down_ge_20pct": bool(on["frame_change_walking_median"] is not None and off["frame_change_walking_median"]
                                                     and on["frame_change_walking_median"] <= 0.8 * off["frame_change_walking_median"]),
                   "sighting_share_ge_0_65": bool(on["approach_sighting_share_median"] is not None and on["approach_sighting_share_median"] >= 0.65),
                   "falls_le_1": on["falls"] <= 1}
    rep["gate"]["pass"] = all(rep["gate"].values())
    (HERE / "results" / "r7" / "gaze_pregate.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
