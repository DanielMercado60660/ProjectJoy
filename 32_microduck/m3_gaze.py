"""Chicken head (adaptive gaze stabilisation, gaze.py): does it steady the camera? Truth only scores.
The duck walks through the frozen interface with Gaze offsets added to its head commands; the bot's
own camera model (encoder FK + IMU tilt + its odometry heading) feeds the adaptation. Camera wobble =
std of camera yaw/pitch around a 1 s moving average (50 Hz, true camera). Conditions: vx 0.2/0.3/0.4
straight, vx 0.3 steering +-0.4, turning in place, and a stand-still check after walking (the head
must stop moving when the gait stops). Calibration of the per-axis phase compensation was done on
train-0000..0003 and train-0010..0015 (sweeps in the M3.1 notes); this script reports the frozen
setting on train-0020..0025, rooms not used to choose it. Writes results/m3/gaze.json."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bodymodel import BodyModel, rot_z, tilt_rotation  # noqa: E402
from gaze import GAZE_PARAMS, Gaze  # noqa: E402
from interface import Command, Interface  # noqa: E402
from odometry import Odometry  # noqa: E402
from room import World  # noqa: E402

TRAIN = json.loads((HERE / "splits" / "train.json").read_text())["configs"]
CONDS = [(0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.3, 0.4), (0.3, -0.4), (0.0, 0.55)]


def detrended_std(x, k=50):
    x = np.unwrap(x)
    trend = np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")[:len(x)]
    return float(np.degrees((x - trend).std()))


def run(args):
    cfg, vx, wz, enabled = args
    w = World(cfg)
    d = w.duck.data
    cam = w.duck.cam_id
    rows = []
    orig = w.step

    def step():
        r = orig()
        look = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
        rows.append([np.arctan2(look[1], look[0]), np.arcsin(look[2])])
        return r
    w.step = step
    it = Interface(w)
    bm, od, gz = BodyModel(), Odometry(), Gaze()
    gz.enabled = enabled
    obs = it.observe()
    head_after = []
    for k in range(130):
        walking = k < 100
        oy, op = gz.offsets(obs.sim_time)
        c = Command(vx=vx if walking else 0.0, wz=wz if walking else 0.0, head=(0.0, op, oy, 0.0))
        obs = it.tick(c)
        od.update(obs)
        _, _, cm = bm.fk(obs.body["joint_pos"])
        gz.update(obs, rot_z(od.psi) @ tilt_rotation(obs.body["gravity"]) @ cm)
        if k >= 115:
            head_after.append(c.head[1:3])
    R = np.array(rows[150:500])
    return {"yaw": detrended_std(R[:, 0]), "pitch": detrended_std(R[:, 1]),
            "standing_head_cmd_max": float(np.abs(np.array(head_after)).max())}


def main():
    cfgs = TRAIN[20:26]
    rep = {"params": GAZE_PARAMS, "configs": [c["id"] for c in cfgs], "conditions": {}}
    with Pool(4) as pool:
        for vx, wz in CONDS:
            off = pool.map(run, [(c, vx, wz, False) for c in cfgs])
            on = pool.map(run, [(c, vx, wz, True) for c in cfgs])
            rep["conditions"][f"vx{vx}_wz{wz}"] = {
                "off_yaw": np.mean([r["yaw"] for r in off]), "on_yaw": np.mean([r["yaw"] for r in on]),
                "off_pitch": np.mean([r["pitch"] for r in off]), "on_pitch": np.mean([r["pitch"] for r in on]),
                "on_standing_head_cmd_max": max(r["standing_head_cmd_max"] for r in on)}
    walk = [v for k, v in rep["conditions"].items() if not k.startswith("vx0.0")]
    rep["walking_mean"] = {k: float(np.mean([v[k] for v in walk])) for k in ("off_yaw", "on_yaw", "off_pitch", "on_pitch")}
    (HERE / "results" / "m3" / "gaze.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["conditions"].items():
        print(k, {kk: round(vv, 2) for kk, vv in v.items()})
    print("walking mean", {k: round(v, 2) for k, v in rep["walking_mean"].items()})


if __name__ == "__main__":
    main()
