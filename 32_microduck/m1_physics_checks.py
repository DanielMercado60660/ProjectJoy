"""M1 physics checks, run before any split is drawn. Writes results/m1/physics_checks.json.

1. Room vs M0 scene. The room adds a ball and a sprung button to the same constraint solve, so
   trajectories can't stay bit-identical (round-off starts at ~1e-16 and grows, as in any walking
   gait). Instead: the divergence trace, plus tracking and falls on M0's development command
   sequence in both, which should agree.
2. Button plate. Can velstand walk onto a raised plate and press it? Plate heights 0 (flush: a 1 mm
   travel, top 1 mm above floor), 3, 5 and 8 mm; 8 approach headings x 3 lateral offsets; the duck
   walks a straight line (true-pose go-to, oracle) to a point 0.5 m past the plate centre.
   Reports press rate, multi-press rate (one walk-over counted as 2+ presses: must be 0 with the
   1 s re-arm in room.py), falls and stalls (didn't get past the plate within 30 s).
3. Sessions across processes: 300 steps in one process vs 150 steps + session file + a fresh
   Python process that resumes and runs 150 more. Final qpos/qvel/ctrl must match bit for bit.
"""
import hashlib
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs import hand_config  # noqa: E402
from duck import CTRL_DT, BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY, FallCounter, goto_command  # noqa: E402
from m0_tune import DEV_SEQUENCE, SETTLE  # noqa: E402
from room import World  # noqa: E402

OUT = HERE / "results" / "m1"
HEIGHTS_MM = (1, 3, 5, 8)
BUTTON_XY = np.array([0.0, 0.0])


def run_dev_sequence(make):
    """M0's development sequence; returns mean achieved per segment and falls."""
    body, step = make()
    fc, rows = FallCounter(), []
    for cmd, dur in DEV_SEQUENCE:
        body.set_command(*cmd)
        vs = []
        for k in range(int(round(dur / CTRL_DT))):
            step()
            fc.update(body.tilt())
            if (k + 1) * CTRL_DT > SETTLE:
                vs.append(body.planar_velocity())
        rows.append({"cmd": list(cmd), "mean": np.mean(vs, 0).round(4).tolist()})
    return {"falls": fc.falls, "segments": rows}


def room_vs_scene():
    def m0():
        d = Duck(BodyConfig(**FROZEN_BODY))
        return d, d.step

    def room():
        w = World(hand_config(empty=True, button_pos=(1.2, -1.2)))
        return w.duck, w.step

    # divergence trace on a fixed 8 s command sequence
    w = World(hand_config(empty=True, button_pos=(1.2, -1.2)))
    d0 = Duck(BodyConfig(**FROZEN_BODY))
    trace = []
    seq = [(0.3, 0, 0)] * 150 + [(0, 0, 0.6)] * 100 + [(0.2, 0, -0.5)] * 150
    for k, c in enumerate(seq):
        w.duck.set_command(*c)
        w.step()
        d0.set_command(*c)
        d0.step()
        if k in (0, 9, 49, 99, 199, 299, 399):
            trace.append({"step": k + 1, "max_abs_qpos_diff": float(np.abs(
                w.duck.data.qpos[:21] - d0.data.qpos[:21]).max())})
    a, b = run_dev_sequence(m0), run_dev_sequence(room)
    diffs = [np.abs(np.array(x["mean"]) - np.array(y["mean"])).max() for x, y in zip(a["segments"], b["segments"])]
    return {"divergence_trace": trace, "m0_scene": a, "room": b,
            "max_segment_mean_diff": float(max(diffs)), "falls": [a["falls"], b["falls"]]}


def plate_trial(args):
    h_mm, heading_deg, lateral = args
    cfg = hand_config(button_height=h_mm / 1000, button_pos=tuple(BUTTON_XY), empty=True)
    th = np.radians(heading_deg)
    fwd, left = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    start = BUTTON_XY - 0.6 * fwd + lateral * left
    goal = BUTTON_XY + 0.5 * fwd + lateral * left
    cfg["layout"]["start"] = {"pos": start.tolist(), "yaw": float(th)}
    w = World(cfg)
    fc, reached, max_depth = FallCounter(), False, 0.0
    for _ in range(int(30 / CTRL_DT)):
        if np.linalg.norm(w.duck.pos()[:2] - goal) < 0.10:
            reached = True
            break
        w.duck.set_command(*goto_command(w.duck, goal))
        w.step()
        fc.update(w.duck.tilt())
        max_depth = max(max_depth, w.button_depth())
    return {"height_mm": h_mm, "heading_deg": heading_deg, "lateral_m": lateral, "presses": w.press_count,
            "pressed": w.press_count > 0, "falls": fc.falls, "passed": reached,
            "max_depth_mm": round(1000 * max_depth, 3), "lamp_on": w.lamp_on}


SESSION_SCRIPT = [(0.3, 0, 0)] * 120 + [(0, 0, 0.6)] * 60 + [(0.25, 0, -0.4)] * 120


def _digest(world):
    d = world.duck.data
    return hashlib.sha256(np.concatenate([d.qpos, d.qvel, d.ctrl]).tobytes()).hexdigest()


def session_child(path):
    """Run in a fresh process: resume the session and finish the script."""
    import session
    w, mem = session.start(path)
    for c in SESSION_SCRIPT[w.step_count:]:
        w.duck.set_command(*c)
        w.step()
    print(json.dumps({"digest": _digest(w), "memory": mem, "steps": w.step_count}))


def session_across_processes():
    import session
    cfg = json.loads((HERE / "splits" / "train.json").read_text())["configs"][3]
    w = World(cfg)
    for c in SESSION_SCRIPT:
        w.duck.set_command(*c)
        w.step()
    full = _digest(w)
    path = OUT / "session_check.json"
    w2, _ = session.new(path, cfg)
    for c in SESSION_SCRIPT[:150]:
        w2.duck.set_command(*c)
        w2.step()
    session.save(path, w2, memory={"note": "half way"})
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--session-child", str(path)],
                       capture_output=True, text=True, check=True)
    child = json.loads(r.stdout.strip().splitlines()[-1])
    path.unlink()
    return {"config": cfg["id"], "continuous_digest": full, "resumed_digest": child["digest"],
            "steps": child["steps"], "memory_round_trip": child["memory"] == {"note": "half way"},
            "bit_exact": child["digest"] == full}


def speed():
    """Build time per config and control steps per second (one process, light load)."""
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][:10]
    t = time.perf_counter()
    worlds = [World(c) for c in cfgs]
    build = (time.perf_counter() - t) / len(cfgs)
    w = worlds[0]
    w.duck.set_command(0.3, 0, 0)
    t = time.perf_counter()
    for _ in range(1000):
        w.step()
    sps = 1000 / (time.perf_counter() - t)
    import os
    return {"build_s_per_config": round(build, 4), "control_steps_per_s": round(sps, 1),
            "sim_seconds_per_wall_second": round(sps * 0.02, 1), "load_1min": round(os.getloadavg()[0], 2)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"room_vs_m0_scene": room_vs_scene(), "session_across_processes": session_across_processes(),
           "speed": speed()}
    trials = [(h, hd, lat) for h in HEIGHTS_MM for hd in range(0, 360, 45) for lat in (-0.06, 0.0, 0.06)]
    with Pool(4) as pool:
        res = pool.map(plate_trial, trials)
    table = []
    for h in HEIGHTS_MM:
        rs = [r for r in res if r["height_mm"] == h]
        table.append({"height_mm": h, "n": len(rs), "press_rate": float(np.mean([r["pressed"] for r in rs])),
                      "multi_press_rate": float(np.mean([r["presses"] > 1 for r in rs])),
                      "falls": int(sum(r["falls"] for r in rs)),
                      "passed_rate": float(np.mean([r["passed"] for r in rs]))})
    rep["plate"] = {"table": table, "trials": res}
    rep["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physics_checks.json").write_text(json.dumps(rep, indent=1))
    r = rep["room_vs_m0_scene"]
    print("divergence:", r["divergence_trace"])
    print("dev sequence: falls", r["falls"], "max segment mean diff", round(r["max_segment_mean_diff"], 4))
    for row in table:
        print(row)
    print("session across processes:", rep["session_across_processes"])
    print("speed:", rep["speed"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--session-child":
        session_child(sys.argv[2])
    else:
        main()
