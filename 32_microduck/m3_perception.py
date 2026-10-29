"""M3 stage 2: score the bot's perception against truth (truth only scores; the bot never sees it).

Frames from train, appearances-dev, distractor-dev and arrangements-dev configs: the duck is placed at
random free spots and headings, head level or pitched down, stands 8 ticks, and the frame is captured.
Detections use the bot's own camera model with the TRUE trunk pose, to measure perception alone.
Truth per object: in the 45 deg view with a clear line of sight to its centre (pads, button, mat at
the surface, ball centre, lamp centre). A detection matches a visible object of its class within
0.30 m (floor objects) or 8 deg of bearing (lamp). Reports recall, precision and position error per
class and family. Writes results/m3/perception.json.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bodymodel import BodyModel  # noqa: E402
from configs import _point_free, footprints  # noqa: E402
from interface import Command, Interface  # noqa: E402
from perception import CameraPose, detect  # noqa: E402
from room import World  # noqa: E402

OUT = HERE / "results" / "m3"
HALF_FOV = np.radians(22.5)


def visible(world, target, geom_ok):
    d = world.duck.data
    c = d.cam_xpos[world.duck.cam_id]
    R = d.cam_xmat[world.duck.cam_id].reshape(3, 3)
    v = R.T @ (target - c)
    if not (v[2] < 0 and abs(v[0] / -v[2]) < np.tan(HALF_FOV) and abs(v[1] / -v[2]) < np.tan(HALF_FOV)):
        return False
    gid = np.array([-1], np.int32)
    dist = mujoco.mj_ray(world.model, d, c, target - c, np.array([1, 1, 0, 0, 0, 0], np.uint8), 1, -1, gid)
    return bool(dist < 0 or gid[0] in geom_ok or dist >= 1 - 0.004 / np.linalg.norm(target - c))


def truth_objects(world):
    lay = world.cfg["layout"]
    m = world.model
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    objs = [(f"pad_{p['color']}", np.array([*p["pos"], 0.003]), {gid(f"pad_{p['color']}")}) for p in lay["pads"]]
    objs.append(("button", np.array([*lay["button"]["pos"], 0.0055]), {gid("button_plate")}))
    objs.append(("mat", np.array([*lay["home"]["pos"], 0.003]), {gid("home_mat")}))
    objs.append(("ball", world.ball_pos(), {world.ball_geom}))
    lamp_xyz = m.geom_pos[world.lamp_geom].copy()
    objs.append(("lamp_on" if world.lamp_on else "lamp_off", lamp_xyz, {world.lamp_geom}))
    return [(cls, p, visible(world, p, ok)) for cls, p, ok in objs]


def frames_for(args):
    cfg, n_views, seed = args
    rng = np.random.default_rng(seed)
    w = World(cfg)
    X, Y, obst = footprints(cfg["layout"], pad=0.2)
    bm = BodyModel()
    rows = []
    tries = 0
    while len(rows) < n_views and tries < 200:
        tries += 1
        p = rng.uniform(-1.3, 1.3, 2)
        if not _point_free(X, Y, obst, p):
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        pitch = float(rng.choice([0.0, 0.3]))
        w.reset(start={"pos": p.round(3).tolist(), "yaw": round(yaw, 4)})
        if rng.random() < 0.5:                 # half the views with the lamp toggled on
            w.lamp_on = not w.lamp_on
            w._apply_visuals()
        it = Interface(w)
        for _ in range(8):
            obs = it.tick(Command(head=(0.0, pitch, 0.0, 0.0)))
        frame = obs.frame()
        cam = CameraPose(bm, obs, (float(w.duck.pos()[0]), float(w.duck.pos()[1]), w.duck.yaw()))
        dets = detect(frame, cam)
        truth = truth_objects(w)
        rows.append({"config": cfg["id"], "family": cfg["id"].rsplit("-", 1)[0],
                     "dets": dets, "truth": [(c, p.tolist(), v) for c, p, v in truth],
                     "cam_xy": cam.pos[:2].tolist(), "cam_yaw": w.duck.yaw()})
    return rows


def score(rows):
    out = {}
    classes = ["pad_red", "pad_yellow", "pad_green", "pad_blue", "button", "ball", "mat", "lamp_on", "lamp_off"]
    fams = sorted({r["family"] for r in rows})
    for fam in fams + ["all"]:
        rs = [r for r in rows if fam == "all" or r["family"] == fam]
        out[fam] = {}
        for cls in classes:
            tp = fn = fp = partial = 0
            errs = []
            for r in rs:
                vis = [(c, np.array(p)) for c, p, v in r["truth"] if c == cls and v]
                dets = [d for d in r["dets"] if d["cls"] == cls]
                used = set()
                for c, p in vis:
                    best, bi = 9e9, None
                    for k, d in enumerate(dets):
                        if k in used:
                            continue
                        if cls in ("lamp_on", "lamp_off"):
                            b = np.arctan2(p[1] - r["cam_xy"][1], p[0] - r["cam_xy"][0])
                            e = abs((d["bearing"] - b + np.pi) % (2 * np.pi) - np.pi)
                            ok = e < np.radians(8)
                        else:
                            e = float(np.linalg.norm(np.array(d["xy"]) - p[:2]))
                            ok = e < 0.30
                        if ok and e < best:
                            best, bi = e, k
                    if bi is None:
                        fn += 1
                    else:
                        tp += 1
                        used.add(bi)
                        errs.append(best)
                # unmatched detections within 0.3 m of a same-class object whose centre isn't visible
                # (cut by the frame edge or partly hidden) are real objects, counted as "partial"
                allp = [np.array(p) for c, p, v in r["truth"] if c == cls]
                for k, d in enumerate(dets):
                    if k in used:
                        continue
                    if cls not in ("lamp_on", "lamp_off") and any(np.linalg.norm(np.array(d["xy"]) - q[:2]) < 0.3 for q in allp):
                        partial += 1
                    else:
                        fp += 1
            out[fam][cls] = {"visible": tp + fn, "recall": tp / max(tp + fn, 1),
                             "detections": tp + fp + partial, "partial": partial, "false": fp,
                             "precision": (tp + partial) / max(tp + fp + partial, 1),
                             "err_median": float(np.median(errs)) if errs else None,
                             "err_p90": float(np.percentile(errs, 90)) if errs else None}
    return out


def main(tag="v0"):
    t0 = time.time()
    S = HERE / "splits"
    train = json.loads((S / "train.json").read_text())["configs"][:20]
    app = json.loads((S / "appearances.json").read_text())["dev"][:20]
    dist = json.loads((S / "distractor.json").read_text())["dev"][:20]
    arr = json.loads((S / "arrangements.json").read_text())["dev"][:10]
    jobs = [(c, 8, 7000 + i) for i, c in enumerate(train + app + dist + arr)]
    with Pool(4) as pool:
        rows = [r for rs in pool.map(frames_for, jobs) for r in rs]
    rep = {"tag": tag, "frames": len(rows), "score": score(rows), "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"perception_{tag}.json").write_text(json.dumps({**rep, "rows": rows}, indent=0, default=float))
    for fam, d in rep["score"].items():
        print(fam, " | ".join(f"{c}: R{v['recall']:.2f} P{v['precision']:.2f} F{v['false']} e{(v['err_median'] or 0):.2f} n{v['visible']}"
                              for c, v in d.items()))
    print("frames", rep["frames"], "wall", rep["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v0")
