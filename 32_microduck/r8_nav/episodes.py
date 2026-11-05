"""Record servo / bot set_lamp episodes with 4 Hz navigation frames (normal + wide camera) and ORACLE truth.

    OMP_NUM_THREADS=1 ./r8_job.sh nav-rec ../.venv/bin/python r8_nav/episodes.py record [--n 40] [--procs 4]
        -> episodes/r8_nav/<expert>_<config>.npz + episodes/r8_nav/index.jsonl
    ../.venv/bin/python r8_nav/episodes.py one train-0000 servo      (a single episode, for a look)

Protocol: results/r8/nav/protocol.json. The expert runs exactly as in its own evaluation (shared modules unchanged);
nav frames are rendered inside its ticks at 50 Hz policy steps floor(12.5 k) (0.24 / 0.26 s, mean 0.25 s = the
models' 4 Hz training spacing) by wrapping this World instance's step(); rendering only reads MjData, so the episode
is bit-identical with or without it (checked in tests/test_r8_nav.py). Each nav frame is a square 160 px render,
centre-cropped to 4:3 (160 x 120), once with the duck's head camera (fovy 45) and once with fovy 100 ("wide"; the
fovy is restored before the expert's own frame). Truth per frame (trunk pose, camera pose and axis, head joints,
on-plate flag, lamp, presses) is ORACLE: it scores, it never feeds a model.
Goals: G1 = the nav frame of the first plate crossing whose body centre is closest to the plate centre;
G2 = the duck placed standing at the plate centre (fresh World, body yaw = the trunk yaw at G1, zero twist, head
(0, 0, 0, 0), 1.0 s settle).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from duck import DEFAULT_POSE, HEAD  # noqa: E402
from interface import Interface  # noqa: E402
from r8_nav.frames import FOVY, RENDER, crop43  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, Judge, make_episode, start_episode  # noqa: E402

OUT = HERE / "episodes" / "r8_nav"
EXPERTS = ("servo", "bot")
SETTLE_STEPS = 50          # G2: 1.0 s standing on the plate
AFTER_EXIT_STEPS = 50      # stop 1.0 s after the body centre leaves the plate


def grid_step(k):
    """Policy step of the k-th 4 Hz nav frame: floor(12.5 k)."""
    return (25 * k) // 2


def train_configs(n=40):
    return json.loads((HERE / "splits" / "train.json").read_text())["configs"][:n]


def press_episode(cfg):
    """The smallest k whose set_lamp episode needs a press."""
    for k in range(64):
        ep = make_episode(cfg, "set_lamp", k)
        if (ep["instruction"]["target"] == "on") != bool(cfg["layout"]["lamp_initial_on"]):
            return ep
    raise RuntimeError(cfg["id"])


def make_expert(expert, cfg, ep):
    if expert == "servo":
        from p2_servo_baseline import controller_factory
        return controller_factory("oracle")(cfg, ep).factory
    from refbot import controller
    fac, _ = controller("set_lamp", BUDGET_TICKS["set_lamp"])
    return fac


class NavCam:
    """Renders both nav cameras and logs ORACLE truth; attach() wraps the world's step()."""

    def __init__(self, world):
        self.w = world
        self.r = mujoco.Renderer(world.model, RENDER, RENDER)
        self.cam = world.duck.cam_id
        b = world.cfg["layout"]["button"]
        self.plate = np.array(b["pos"], float)
        self.half = b["size"] / 2
        self.rows = {k: [] for k in ("normal", "wide", "step", "time", "trunk_xy", "trunk_yaw", "cam_pos", "cam_axis",
                                     "head", "on_plate", "lamp", "presses")}
        self.k = 0

    def render(self, duck=None):
        d = (duck or self.w.duck)
        m = self.w.model
        f0 = float(m.cam_fovy[self.cam])
        out = {}
        try:
            for name, fov in FOVY.items():
                m.cam_fovy[self.cam] = fov
                out[name] = crop43(d.head_image(self.r))
        finally:
            m.cam_fovy[self.cam] = f0
        return out

    def capture(self):
        w, d = self.w, self.w.duck
        img = self.render()
        dd = d.data
        xy = d.pos()[:2]
        R = dd.cam_xmat[self.cam].reshape(3, 3)
        row = {"normal": img["normal"], "wide": img["wide"], "step": w.step_count, "time": float(dd.time),
               "trunk_xy": xy.copy(), "trunk_yaw": d.yaw(), "cam_pos": dd.cam_xpos[self.cam].copy(),
               "cam_axis": -R[:, 2].copy(), "head": (dd.qpos[d.jq][HEAD] - DEFAULT_POSE[HEAD]).copy(),
               "on_plate": bool((np.abs(xy - self.plate) <= self.half).all()), "lamp": bool(w.lamp_on),
               "presses": int(w.press_count)}
        for k, v in row.items():
            self.rows[k].append(v)
        self.k += 1

    def attach(self):
        orig = self.w.step

        def hooked():
            out = orig()
            if self.w.step_count == grid_step(self.k):
                self.capture()
            return out
        self.w.step = hooked
        self.capture()                     # k = 0: the start frame

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}

    def close(self):
        self.r.close()


def pick_g1(a, plate):
    """Index of the first crossing's nav frame closest to the plate centre (None if the body never entered)."""
    on = a["on_plate"]
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return None
    first = idx[0]
    end = first
    while end + 1 < len(on) and on[end + 1]:
        end += 1
    run = np.arange(first, end + 1)
    dist = np.hypot(*(a["trunk_xy"][run] - plate).T)
    return int(run[np.argmin(dist)])


def placed_goal(cfg, yaw):
    """G2: the duck standing at the plate centre, body yaw `yaw`, head (0,0,0,0), after 1.0 s."""
    w = World(cfg)
    w.reset(start={"pos": list(map(float, cfg["layout"]["button"]["pos"])), "yaw": float(yaw)})
    w.duck.set_command(0.0, 0.0, 0.0, head=(0.0, 0.0, 0.0, 0.0))
    for _ in range(SETTLE_STEPS):
        w.step()
    nc = NavCam(w)
    img = nc.render()
    xy = w.duck.pos()[:2]
    nc.close()
    return img, {"g2_trunk_xy": xy.round(4).tolist(), "g2_yaw": round(float(w.duck.yaw()), 4),
                 "g2_tilt": round(float(w.duck.tilt()), 4)}


def run_episode(cfg, expert, nav=True, max_ticks=None):
    """Run one press-needed episode; returns (NavCam or None, meta, world)."""
    ep = press_episode(cfg)
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    judge = Judge(w, ep)
    nc = NavCam(w) if nav else None
    if nc is not None:
        nc.attach()
    fac = make_expert(expert, cfg, ep)
    gen = fac(iface, ep["instruction"])
    plate = np.array(cfg["layout"]["button"]["pos"], float)
    half = cfg["layout"]["button"]["size"] / 2
    budget = ep["budget_ticks"] if max_ticks is None else max_ticks
    entry = exit_ = None
    stop = None
    t0 = time.time()
    try:
        cmd = next(gen)
        while True:
            if iface.done:
                stop = "done"
                break
            if iface.tick_count >= budget:
                stop = "budget"
                break
            obs = iface.tick(cmd)
            judge.update(iface)
            xy = w.duck.pos()[:2]
            inside = bool((np.abs(xy - plate) <= half).all())
            if inside and entry is None:
                entry = w.step_count
            if entry is not None and exit_ is None and not inside:
                exit_ = w.step_count
            if exit_ is not None and w.step_count >= exit_ + AFTER_EXIT_STEPS:
                stop = "exit+1s"
                break
            if iface.done:
                stop = "done"
                break
            cmd = gen.send(obs)
    except StopIteration:
        stop = stop or "generator_end"
    finally:
        if iface._renderer is not None:
            iface._renderer.close()
    meta = {"config_id": cfg["id"], "expert": expert, "k": ep["k"], "target": ep["instruction"]["target"],
            "lamp_initial": bool(cfg["layout"]["lamp_initial_on"]), "entry_step": entry, "exit_step": exit_,
            "stop": stop, "ticks": iface.tick_count, "presses": int(w.press_count), "falls": judge.falls,
            "plate_xy": plate.tolist(), "plate_half": half, "wall_s": round(time.time() - t0, 1)}
    return nc, meta, w


def record_one(args):
    cfg, expert = args
    out = OUT / f"{expert}_{cfg['id']}.npz"
    nc, meta, w = run_episode(cfg, expert)
    a = nc.arrays()
    nc.close()
    g1 = pick_g1(a, np.array(meta["plate_xy"]))
    meta["g1_index"] = g1
    extra = {}
    if g1 is not None:
        meta["g1_dist"] = round(float(np.hypot(*(a["trunk_xy"][g1] - meta["plate_xy"]))), 4)
        meta["g1_time"] = round(float(a["time"][g1]), 3)
        img, info = placed_goal(cfg, a["trunk_yaw"][g1])
        meta.update(info)
        extra = {"g2_normal": img["normal"], "g2_wide": img["wide"]}
    meta["n_frames"] = int(len(a["step"]))
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **a, **extra)
    return meta


def record(n=40, procs=4):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, e) for e in EXPERTS for c in train_configs(n)]
    t0 = time.time()
    metas = []
    with Pool(procs) as pool:
        for m in pool.imap_unordered(record_one, jobs):
            metas.append(m)
            print(f"[{time.time() - t0:6.0f}s] {m['expert']:5s} {m['config_id']} frames {m['n_frames']:4d} "
                  f"entry {m['entry_step']} g1 {m['g1_index']} stop {m['stop']} wall {m['wall_s']}s", flush=True)
    metas.sort(key=lambda m: (EXPERTS.index(m["expert"]), m["config_id"]))
    with open(OUT / "index.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"done {len(metas)} episodes in {time.time() - t0:.0f}s; without a goal frame: "
          f"{sum(m['g1_index'] is None for m in metas)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("record", "one"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.n, a.procs)
    else:
        cid, expert = a.rest
        cfg = next(c for c in train_configs(300) if c["id"] == cid)
        OUT.mkdir(parents=True, exist_ok=True)
        print(json.dumps(record_one((cfg, expert))))
