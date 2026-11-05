"""Round 8 Gate B, optional: a labelled 224 px frame set per stream from the measurement's bot episodes, for the next
question ("does a steady view help a world model / encoder?"). -> data/r8_stabcam/frames/*.npz (gitignored) + README.md

    ./r8_job.sh stabcam-frames env OMP_NUM_THREADS=1 ../.venv/bin/python r8_stabcam_frames.py --procs 4   # ~5-10 min

The 10 protocol bot episodes (set_lamp, k=8), reflex off and on, re-run deterministically; every 3rd tick-end frame:
  off: head, trunk, oracle (ORACLE), eis      on: reflex, eis_reflex
All 224 px, 45 deg; eis/eis_reflex are warped from a 140 deg render at 1472 px (the same centre sharpness as 224 px).
Labels are SIMULATOR TRUTH (labels only, never inputs): lamp state; per stream, from that stream's own effective view
(eis: R_true * R_est^T * R_target), whether the lamp / the button plate centre is inside the 45 deg view with a clear
line of sight, and their bearing (deg, + = left of the view axis) and floor distance (m) from the camera; the executed
command; the bot's own (sensor-only) estimate is not stored.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402

PX = 224
EVERY = 3                       # tick-end frames kept: every 3rd tick
WIDE_PX_224 = 1472              # focal 268 px ~ the 224 px / 45 deg output's 270
OFFSCREEN = 1536
TAN = np.tan(np.radians(S.FOVY / 2))
OUT = S.DATA / "frames"


def targets(world):
    m = world.model
    lamp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lamp")
    plate = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "button_plate")
    return {"lamp": lamp, "plate": plate}


def view_labels(m, rd, pos, R, gid):
    """In view (centre inside the 45 deg square view + clear ray, robot geoms ignored), bearing (deg), floor distance."""
    p = rd.geom_xpos[gid].copy()
    v = R.T @ (p - pos)
    in_fov = v[2] < 0 and abs(v[0] / -v[2]) < TAN and abs(v[1] / -v[2]) < TAN
    hit = np.array([-1], np.int32)
    d = p - pos
    frac = mujoco.mj_ray(m, rd, pos, d, np.array([1, 1, 0, 0, 0, 0], np.uint8), 1, -1, hit)
    clear = frac < 0 or hit[0] == gid or frac >= 0.995
    f = -R[:, 2]
    bearing = np.degrees(S.wrap(np.arctan2(d[1], d[0]) - np.arctan2(f[1], f[0])))
    return bool(in_fov and clear), float(bearing), float(np.hypot(d[0], d[1]))


def job(args):
    cid, k, mode = args
    path = OUT / f"bot_{cid}_k{k}_{mode}.npz"
    if path.exists():
        return str(path)
    cfg = {c["id"]: c for c in S.TRAIN}[cid]
    store = {"tick": [], "t": [], "cmd": [], "lamp_on": []}
    names = ("head", "trunk", "oracle", "eis") if mode == "off" else ("reflex", "eis_reflex")
    for n in names:
        for f in ("frames", "lamp_in_view", "lamp_bearing", "lamp_dist", "plate_in_view", "plate_bearing", "plate_dist", "R", "pos"):
            store[f"{n}/{f}"] = []
    store["valid"] = []
    tg = {}

    def hook(pr, x):
        w = pr.w
        if not x["tick_end"]:
            return
        tick = w.step_count // 5 - 1
        if tick % EVERY:
            return
        rig, m, rd = pr.rig, w.model, pr.rig.rd
        if not tg:
            tg.update(targets(w))
        wide = rig.render_pose(x["hp"], x["Rh"], WIDE_PX_224, wide=True)
        eis, valid = S.warp(wide, x["Rhat"], x["R_tgt"], n=PX)
        views = {"head" if mode == "off" else "reflex": (rig.render_head(PX), x["hp"], x["Rh"]),
                 "eis" if mode == "off" else "eis_reflex": (eis, x["hp"], x["R_eff"])}
        if mode == "off":
            views["trunk"] = (rig.render_trunk(PX),) + rig.trunk_pose()
            views["oracle"] = (rig.render_pose(x["hp"], x["R_or"], PX), x["hp"], x["R_or"])
        store["tick"].append(tick)
        store["t"].append(float(w.duck.data.time))
        store["cmd"].append(np.asarray(w.duck.command[:7], np.float32))
        store["lamp_on"].append(bool(w.lamp_on))
        store["valid"].append(float(valid.mean()))
        for n, (img, pos, R) in views.items():
            store[f"{n}/frames"].append(img)
            store[f"{n}/R"].append(np.asarray(R, np.float32))
            store[f"{n}/pos"].append(np.asarray(pos, np.float32))
            for obj in ("lamp", "plate"):
                iv, b, dist = view_labels(m, rd, pos, R, tg[obj])
                store[f"{n}/{obj}_in_view"].append(iv)
                store[f"{n}/{obj}_bearing"].append(b)
                store[f"{n}/{obj}_dist"].append(dist)

    t0 = time.time()
    res = S.run_bot(cfg, k, mode, "none_but_rig", hooks=[hook], offscreen=OFFSCREEN)
    meta = dict(res["meta"], every=EVERY, px=PX, streams=list(names), wall_s=round(time.time() - t0, 1),
                note="labels are simulator truth (labels only)")
    arrs = {"meta": np.array(json.dumps(meta))}
    for key, v in store.items():
        arrs[key] = np.array(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)
    print(time.strftime("%H:%M:%S"), "done", path.name, len(store["tick"]), "frames", meta["wall_s"], "s", flush=True)
    return str(path)


README = """# Round 8 Gate B frame set (r8_stabcam_frames.py)

One `.npz` per bot episode and reflex mode: `frames/bot_<config>_k8_<off|on>.npz` (the 10 protocol set_lamp episodes,
`results/r8/stabcam/protocol.json`). Every 3rd 10 Hz tick-end frame of the episode. Load with
`np.load(path)`; keys:

| key | shape | meaning |
|---|---|---|
| `meta` | JSON string | config, k, mode, success, ticks, streams, ... |
| `tick`, `t` | (N,) | 10 Hz tick index (frame at the end of that tick), sim time (s) |
| `cmd` | (N, 7) float32 | executed command during that tick: vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll |
| `lamp_on` | (N,) bool | lamp state (truth) |
| `valid` | (N,) | eis output pixels inside the wide render (fraction) |
| `<stream>/frames` | (N, 224, 224, 3) uint8 | the stream's view, 45 deg, upright |
| `<stream>/R`, `<stream>/pos` | (N, 3, 3), (N, 3) | the view's effective orientation (columns right, up, back) and position, world frame (truth) |
| `<stream>/{lamp,plate}_in_view` | (N,) bool | object centre inside the 45 deg view with a clear line of sight (truth) |
| `<stream>/{lamp,plate}_bearing` | (N,) deg | horizontal angle from the view axis, + = left (truth) |
| `<stream>/{lamp,plate}_dist` | (N,) m | floor distance from the camera (truth) |

Streams: `off` files hold `head` (as now), `trunk` (sim-only trunk camera), `oracle` (ORACLE: simulator-truth
smoothed orientation at the head position), `eis` (sensor-only electronic stabilisation); `on` files hold `reflex`
(the gaze.py reflex on, a different physics run) and `eis_reflex`. Within a file all streams share the same ticks,
so they are paired frame for frame. Everything under labels is SIMULATOR TRUTH: labels only, never policy inputs.
Caveat: MuJoCo's headlight turns with each rendering camera, so head/eis frames carry a brightness flicker with
the head's pitch that a real camera would not (protocol.json, secondary_no_headlight).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    proto = json.loads((S.OUT / "protocol.json").read_text())
    jobs = [(c, proto["episodes"]["bot"]["k"], mode) for c in proto["episodes"]["bot"]["configs"] for mode in ("off", "on")]
    with Pool(a.procs, maxtasksperchild=2) as pool:
        list(pool.imap_unordered(job, jobs, chunksize=1))
    (S.DATA / "README.md").write_text(README)
    print("frame set done")


if __name__ == "__main__":
    main()
