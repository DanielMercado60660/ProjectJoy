"""Reviewer checks for M1 (REVIEW_M1.md). Deterministic, CPU, about 1 minute with 4 processes.

  1. Break bit-exact save/restore. A 500-step walk-over of the button (with a push), plus a 300-step
     walk/turn/head/push script on a training and a distractor config. Each is saved at many steps:
     right after reset, mid-push, at the press, mid re-arm count. It is restored three ways: into a
     fresh world that was dirtied first, through a session JSON file, and into the same world after
     running on. Every later step must match the uninterrupted run bit for bit. Compared each step:
     qpos, qvel, ctrl, sensordata, xquat, the observation, room flags, flicker colour, geom_rgba and
     mat_emission.
  2. Every appearance and distractor sibling (330) against its training source: 150 steps, identical
     qpos/qvel.
  3. Button edge: the duck starts 0.00-0.19 m off the plate centre and stands, turns at 0.6 rad/s or
     turns at 0.3 rad/s for 20 s. Count presses.
  4. Duck footprint: the largest horizontal distance of any collision-geom AABB corner from the trunk,
     standing and while walking/turning. Reachability of every distinct layout at 0.18/0.20/0.25 m
     clearance (validate() uses 0.15 m).
  5. Colours as rendered by the overhead camera: every pad/button/ball shade of the appearance
     dev/test configs, classified 1-NN in CIELAB against the rendered shades of 150 training configs.
     Reports errors and the worst margin per category.
Writes results/m1/review_checks.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs as C  # noqa: E402
import session  # noqa: E402
from configs import hand_config  # noqa: E402
from m0_gate import FallCounter, goto_command  # noqa: E402
from room import World  # noqa: E402

OUT = HERE / "results" / "m1" / "review_checks.json"
load = lambda n: json.loads((HERE / "splits" / n).read_text())  # noqa: E731


# ------------------------------------------------------------------------- 1. save/restore
def _snap(w, obs):
    d = w.duck.data
    return np.concatenate([d.qpos, d.qvel, d.ctrl, d.sensordata, d.xquat.ravel(), obs,
                           [w.lamp_on, w.press_count, w.button_down, w.release_steps, w.step_count],
                           w.flicker_rgb, w.model.geom_rgba.ravel(), w.model.mat_emission.ravel()])


def _button_plan(w, k):
    w.duck.set_command(*goto_command(w.duck, np.array([1.1, 0.0]))) if k < 400 else w.duck.set_command(0, 0, 0.5)
    w.duck.push_force = np.array([1.5, -1.0, 0.0]) if 150 <= k < 160 else np.zeros(3)


def _script_plan(w, k):
    cmds = [(0.3, 0, 0)] * 100 + [(0, 0, 0.6)] * 60 + [(0.25, 0, -0.4)] * 100 + [(0, 0, 0)] * 40
    w.duck.set_command(*cmds[k % len(cmds)], head=(0.1, 0.2, 0.3 * np.sin(k / 10), 0))
    w.duck.push_force = np.array([0.0, 2.0, 0.0]) if 60 <= k < 64 else np.zeros(3)


PLANS = {"button": _button_plan, "script": _script_plan}


def _drive(w, k, plan):
    PLANS[plan](w, k)
    obs, _ = w.step()
    return _snap(w, obs)


def save_restore_case(args):
    name, cfg, plan, n, saves, via = args
    tmp = OUT.parent / f".review_session_{name}_{via}.json"   # deleted below
    ref_w = World(cfg)
    ref = [_drive(ref_w, k, plan) for k in range(n)]
    failures = []
    for k0 in saves:
        w = World(cfg)
        for k in range(k0):
            _drive(w, k, plan)
        if via == "fresh_dirty_world":
            st = w.get_state()
            w2 = World(cfg)
            for k in range(7):
                _drive(w2, k, "script")
            w2.set_state(st)
        elif via == "session_file":
            session.save(tmp, w, {"k": k0})
            w2, _ = session.start(tmp)
        else:   # same world, run on, then restore
            st = w.get_state()
            for k in range(k0, k0 + 13):
                _drive(w, k, "script")
            w.set_state(st)
            w2 = w
        for k in range(k0, n):
            if not np.array_equal(_drive(w2, k, plan), ref[k], equal_nan=True):
                failures.append({"saved_at": k0, "first_mismatch_step": k})
                break
    tmp.unlink(missing_ok=True)
    return {"case": name, "via": via, "saves": saves, "failures": failures}


def save_restore_cases():
    hc = hand_config(button_pos=(0.6, 0.0), empty=True)
    w, states = World(hc), []
    for k in range(500):
        _drive(w, k, "button")
        states.append((k + 1, w.button_down, w.release_steps))
    down = [s for s, b, r in states if b]
    rearm = [s for s, b, r in states if b and r > 0]
    saves_b = sorted({0, 1, 2, 3, 151, 155, 160, down[0], down[0] + 1, rearm[0], rearm[len(rearm) // 2], down[-1]})
    train, dist = load("train.json")["configs"], load("distractor.json")["test"]
    saves_s = [0, 1, 4, 5, 6, 61, 63, 64, 99, 100, 101, 162, 299]
    cases = [("button_walkover", hc, "button", 500, saves_b, v) for v in ("fresh_dirty_world", "session_file", "same_world")]
    cases += [(n, c, "script", 300, saves_s, v) for n, c in (("train-0005", train[5]), (dist[0]["id"], dist[0]))
              for v in ("fresh_dirty_world", "session_file")]
    return cases, {"press_step": down[0], "rearm_step": down[-1] + 1}


# ------------------------------------------------------------------------- 2. siblings
def sibling_case(args):
    src, sib = args
    cmds = [(0.3, 0, 0)] * 60 + [(0, 0, 0.7)] * 40 + [(0.25, 0, -0.5)] * 50

    def run(cfg):
        w, out = World(cfg), []
        for c in cmds:
            w.duck.set_command(*c)
            w.step()
            out.append(np.concatenate([w.duck.data.qpos, w.duck.data.qvel]))
        return np.array(out)
    return sib["id"], bool(np.array_equal(run(src), run(sib)))


# ------------------------------------------------------------------------- 3. button edge
def edge_case(args):
    dy, yaw0, mode = args
    w = World(hand_config(button_pos=(0.6, 0.0), empty=True, start=((0.6, dy), yaw0)))
    fc, times = FallCounter(), []
    wz = {"stand": 0.0, "turn": 0.6, "turn_slow": 0.3}[mode]
    for k in range(1000):
        w.duck.set_command(0, 0, wz if k > 50 else 0.0)
        pc = w.press_count
        w.step()
        fc.update(w.duck.tilt())
        if w.press_count != pc:
            times.append(round((k + 1) * 0.02, 2))
    return {"dy": dy, "yaw0": round(yaw0, 4), "mode": mode, "presses": w.press_count, "press_times_s": times,
            "falls": fc.falls}


# ------------------------------------------------------------------------- 4. footprint, clearance
def footprint():
    w = World(hand_config(empty=True, button_pos=(1.2, -1.2)))
    m, d, root = w.model, w.duck.data, w.duck.trunk_id
    gs = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b not in (0, root):
            b = int(m.body_parentid[b])
        if b == root and (m.geom_contype[g] or m.geom_conaffinity[g]):
            gs.append(g)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    def radius():
        pts = [d.geom_xpos[g] + (d.geom_xmat[g].reshape(3, 3) @ (m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:]).T).T
               for g in gs]
        return float(np.linalg.norm(np.concatenate(pts)[:, :2] - d.xpos[root][:2], axis=1).max())
    standing = radius()
    moving = []
    for k in range(300):
        w.duck.set_command(*((0.3, 0, 0) if k < 150 else (0, 0, 0.8)))
        w.step()
        moving.append(radius())
    return {"standing_m": round(standing, 3), "walking_turning_max_m": round(max(moving), 3)}


def clearance_case(args):
    lay, clearance = args
    X, Y, obst = C.footprints(lay, pad=clearance)
    seen, idx = C._reachable(X, Y, ~obst, lay["start"]["pos"])
    o = lay["occluder"]
    n = np.array([-np.sin(o["yaw"]), np.cos(o["yaw"])])
    pts = [lay["home"]["pos"], lay["button"]["pos"]] + [p["pos"] for p in lay["pads"]]
    pts += [list(np.array(o["pos"]) + s * 0.30 * n) for s in (-1, 1)]
    for q in pts:
        r, c = idx(q)
        if not (0 <= r < seen.shape[0] and 0 <= c < seen.shape[1]) or not seen[r, c]:
            return False
    return True


# ------------------------------------------------------------------------- 5. rendered colours
def _lab(rgb):
    rgb = np.asarray(rgb, float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def rendered_shades(cfg, S=480):
    w = World(cfg)
    r = mujoco.Renderer(w.model, S, S)
    r.update_scene(w.duck.data, camera="top")
    img = r.render().astype(float) / 255
    r.close()
    ext = max(cfg["layout"]["room"]["size"]) + 0.3   # room.py's orthographic fovy

    def px(xy):
        c, row = int((xy[0] + ext / 2) / ext * S), int((ext / 2 - xy[1]) / ext * S)
        return np.median(img[row - 2:row + 3, c - 2:c + 3].reshape(-1, 3), 0).tolist()
    lay = cfg["layout"]
    out = [(p["color"], px(p["pos"])) for p in lay["pads"]]
    return out + [("button", px(lay["button"]["pos"])), ("ball", px(lay["ball"]["pos"]))]


def colours(pool):
    train = load("train.json")["configs"][:150]
    app = load("appearances.json")
    tr = [x for xs in pool.map(rendered_shades, train) for x in xs]
    TRm, TRc = np.array([_lab(v) for _, v in tr]), [c for c, _ in tr]
    res = {"train_configs": len(train),
           "train_rendered_yellow_min_max": [np.array([v for c, v in tr if c == "yellow"]).min(0).round(2).tolist(),
                                             np.array([v for c, v in tr if c == "yellow"]).max(0).round(2).tolist()]}
    for s in ("dev", "test"):
        wrong, margins, worst = 0, {}, {}
        for cfg, shades in zip(app[s], pool.map(rendered_shades, app[s])):
            for cat, v in shades:
                d = np.sqrt(((TRm - _lab(v)) ** 2).sum(-1))
                own = d[[k for k, c in enumerate(TRc) if c == cat]].min()
                k_oth = min((k for k, c in enumerate(TRc) if c != cat), key=lambda k: d[k])
                wrong += int(d[k_oth] < own)
                m = float(d[k_oth] - own)
                margins.setdefault(cat, []).append(m)
                if m <= min(margins[cat]):
                    worst[cat] = {"margin_dE": round(m, 1), "nearest_other": TRc[k_oth], "rgb": np.round(v, 2).tolist(),
                                  "config": cfg["id"]}
        res[s] = {"misclassified": wrong, "n": sum(len(v) for v in margins.values()), "worst": worst,
                  "under_5dE": {c: int(sum(np.array(v) < 5)) for c, v in margins.items()},
                  "n_per_category": {c: len(v) for c, v in margins.items()}}
    return res


def main():
    rep = {}
    with Pool(4) as pool:
        cases, info = save_restore_cases()
        rep["save_restore"] = {"button_steps": info, "cases": pool.map(save_restore_case, cases)}
        rep["save_restore"]["all_bit_exact"] = all(not c["failures"] for c in rep["save_restore"]["cases"])
        by_id = {c["id"]: c for c in load("train.json")["configs"]}
        sibs = [(by_id[c["source_train_id"]], c) for f in ("appearances", "distractor")
                for s in ("dev", "test") for c in load(f"{f}.json")[s]]
        res = pool.map(sibling_case, sibs)
        rep["siblings"] = {"n": len(res), "differ": [i for i, ok in res if not ok]}
        edge = [(dy, yaw, m) for m in ("stand", "turn", "turn_slow") for dy in (0.0, 0.10, 0.13, 0.16, 0.19)
                for yaw in (0.0, float(np.pi / 2))]
        er = pool.map(edge_case, edge)
        summ = {}
        for m in ("stand", "turn", "turn_slow"):
            centre = [r for r in er if r["mode"] == m and r["dy"] == 0.0]
            off = [r for r in er if r["mode"] == m and r["dy"] > 0]
            summ[m] = {"centre_max_presses": max(r["presses"] for r in centre),
                       "edge_trials": len(off), "edge_trials_with_2plus": sum(r["presses"] >= 2 for r in off),
                       "falls": sum(r["falls"] for r in er if r["mode"] == m)}
        rep["button_edge"] = {"summary": summ, "trials": er}
        rep["footprint"] = footprint()
        layouts = {}
        for f in ("train", "instances", "arrangements", "appearances", "distractor"):
            d = load(f"{f}.json")
            for c in (d["configs"] if f == "train" else d["dev"] + d["test"]):
                layouts.setdefault(json.dumps(c["layout"], sort_keys=True), c["layout"])
        lays = list(layouts.values())
        rep["clearance"] = {"distinct_layouts": len(lays)}
        for cl in (0.15, 0.18, 0.20, 0.25):
            ok = pool.map(clearance_case, [(lay, cl) for lay in lays])
            rep["clearance"][f"unreachable_at_{cl:.2f}"] = int(len(ok) - sum(ok))
        rep["rendered_colours"] = colours(pool)
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("button_edge",)}, indent=1)[:4000])
    print(json.dumps(rep["button_edge"]["summary"], indent=1))


if __name__ == "__main__":
    main()
