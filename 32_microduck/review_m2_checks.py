"""M2 review checks (REVIEW_M2.md). Writes results/m2/review_checks.json.

A. Trivial controllers against the judges, on the M2 feasibility configs (60 training + 50
   instances-dev), episodes k = 0..3: "done at once" on every task; for find_ball also "stand still
   for 59 s, then done" and "look away 0.3 s, look back, done" (the ball still rolling). These are
   the floors every later controller must be compared with (M6).
B. Task 3 watch time: with the duck standing still at the start, ticks in the first 8 s during which
   the judge counts the ball as in view.
C. Yaw from the IMU: integrate the observation's gyro (projected on gravity) over 60 s of mixed
   walking/turning, once from one sample per 10 Hz tick (the observation before this review) and
   once from the tick's 50 Hz samples (gyro_substeps / gravity_substeps, added in this review).
D. Task 3 setup: which candidate grid of tasks._find_ball_setup each config's setup came from (grid 1
   = the main grid, 2 = wider distances/offsets, 3 = oblique views), per split. Setup only: no
   controller runs on a test config. Also: where the ball comes to rest relative to the occluder
   (along its centre line, behind it) and how far from the start, over every valid setup; and one
   0.45 m/s roll on open floor.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interface import TICK_STEPS, Command, Interface  # noqa: E402
from room import World  # noqa: E402
from skills import wrap  # noqa: E402
from tasks import TASKS, ball_visibility, make_episode, run_episode, start_episode  # noqa: E402

SPLITS, OUT = HERE / "splits", HERE / "results" / "m2"
K = 4


def done_now(iface, instruction):
    yield Command(done=True)


def wait_then_done(iface, instruction, n=590):
    for _ in range(n):
        yield Command()
    yield Command(done=True)


def look_away_back(iface, instruction):
    for _ in range(3):
        yield Command(head=(0, 0, 1.1, 0))
    for _ in range(3):
        yield Command()
    yield Command(done=True)


def trivial(args):
    cfg, task = args
    w = World(cfg)
    rows = []
    for k in range(K):
        try:
            ep = make_episode(cfg, task, k)
        except RuntimeError:
            continue
        row = {"config": cfg["id"], "task": task, "k": k}
        r, _, _ = run_episode(cfg, ep, done_now, world=w)
        row["done_now"] = r["success"]
        if task == "set_lamp":
            row["needed_presses"] = r["needed_presses"]
        if task == "find_ball":
            row["done_now_old_judge"] = bool(r["declared_done"] and r["ball_visible_at_end"]
                                             and r["ball_distance"] <= 0.8 and r["stopped"])
            row["wait_then_done"] = run_episode(cfg, ep, wait_then_done, world=w)[0]["success"]
            row["look_away_back"] = run_episode(cfg, ep, look_away_back, world=w)[0]["success"]
            start_episode(w, ep)
            it = Interface(w)
            vis = []
            for _ in range(80):
                it.tick(Command())
                vis.append(ball_visibility(w)[0])
            row["visible_ticks_first_8s"] = int(sum(vis))
            row["visible_at_8s"] = bool(vis[-1])
        rows.append(row)
    return rows


def yaw_drift(i, seed, segs=30):
    cfg = json.loads((SPLITS / "train.json").read_text())["configs"][i]
    w = World(cfg)
    it = Interface(w)
    rng = np.random.default_rng(seed)
    yaw0 = prev = w.duck.yaw()
    e10 = e50 = yaw0
    true = 0.0
    err10, err50 = [], []
    for s in range(segs):
        c = [Command(vx=0.3, wz=float(rng.uniform(-0.6, 0.6))), Command(wz=0.55 * float(rng.choice([-1, 1]))),
             Command()][s % 3]
        for _ in range(20):
            o = it.tick(c)
            g, gr = o.body["gyro"], o.body["gravity"]
            e10 += float(g @ (-gr / np.linalg.norm(gr))) * 0.1
            for g, gr in zip(o.body["gyro_substeps"], o.body["gravity_substeps"]):
                e50 += float(g @ (-gr / np.linalg.norm(gr))) * 0.1 / TICK_STEPS
            y = w.duck.yaw()
            true += wrap(y - prev)
            prev = y
            err10.append(abs(e10 - yaw0 - true))
            err50.append(abs(e50 - yaw0 - true))
    return {"config": cfg["id"], "seed": seed, "sim_s": segs * 2,
            "tick_sample_10hz": {"end_rad": round(err10[-1], 3), "max_rad": round(max(err10), 3)},
            "substeps_50hz": {"end_rad": round(err50[-1], 3), "max_rad": round(max(err50), 3)}}


def setup_grid(cfg):
    try:
        ep = make_episode(cfg, "find_ball", 0)
    except RuntimeError:
        return {"config": cfg["id"], "grid": None}
    o = cfg["layout"]["occluder"]
    O = np.array(o["pos"])
    a = np.array([np.cos(o["yaw"]), np.sin(o["yaw"])])
    n = np.array([-np.sin(o["yaw"]), np.cos(o["yaw"])])
    v = np.array(ep["start"]["pos"]) - O
    dn = round(abs(float(v @ n)), 2)
    # grid 1 view distances 0.8-1.2 straight on; grid 2 0.6/0.7/1.3/1.4; grid 3 oblique (d cos phi)
    grid = 1 if dn in (0.8, 0.9, 1.0, 1.1, 1.2) else 2 if dn in (0.6, 0.7, 1.3, 1.4) else 3
    r = np.array(ep["ball_launch"]["rest"]) - O
    return {"config": cfg["id"], "grid": grid, "normal_distance": dn,
            "rest_along_abs_m": round(abs(float(r @ a)), 4),                    # from the centre line
            "rest_behind_m": round(float(-np.sign(v @ n) * (r @ n)), 4),         # on the far side: > 0
            "start_to_rest_m": round(float(np.linalg.norm(r - v)), 4)}


def roll_check():
    """The RESULTS claim 'launched at 0.45 m/s it stops after ~0.7 m in ~6 s' (train-0000, open floor)."""
    from tasks import _roll
    cfg = json.loads((SPLITS / "train.json").read_text())["configs"][0]
    w = World(cfg)
    P0 = np.array([0.0, -1.0])
    rest, steps = _roll(w, cfg["layout"]["start"], P0, [0.45, 0.0])
    return {"config": cfg["id"], "from": P0.tolist(), "speed": 0.45,
            "distance_m": round(float(np.linalg.norm(rest - P0)), 3), "time_s": round(steps * 0.02, 2)}


def main():
    train = json.loads((SPLITS / "train.json").read_text())["configs"]
    inst_dev = json.loads((SPLITS / "instances.json").read_text())["dev"]
    every = [("train", c) for c in train]
    for f in ("instances", "arrangements", "appearances", "distractor"):
        d = json.loads((SPLITS / f"{f}.json").read_text())
        every += [(f"{f}-dev", c) for c in d["dev"]] + [(f"{f}-test", c) for c in d["test"]]
    with Pool(4) as pool:
        rows = [r for rs in pool.map(trivial, [(c, t) for t in TASKS for c in train[:60] + inst_dev]) for r in rs]
        drift = pool.starmap(yaw_drift, [(4, 0), (9, 1), (20, 2), (33, 3)])
        grids = pool.map(setup_grid, [c for _, c in every])
    group_of = {c["id"]: g for g, c in every}
    out = {"trivial": {}, "watch": {}, "yaw_drift_60s": drift, "find_ball_setup_grid": {}}
    for task in TASKS:
        for grp in ("train", "instances-dev"):
            rs = [r for r in rows if r["task"] == task and r["config"].startswith(grp)]
            k0 = [r for r in rs if r["k"] == 0]
            t = {"episodes": len(rs), "done_now": int(sum(r["done_now"] for r in rs)),
                 "k0_episodes": len(k0), "k0_done_now": int(sum(r["done_now"] for r in k0))}
            if task == "set_lamp":
                t["no_press_needed"] = int(sum(r["needed_presses"] == 0 for r in rs))
                t["k0_no_press_needed"] = int(sum(r["needed_presses"] == 0 for r in k0))
            if task == "find_ball":
                for key in ("done_now_old_judge", "wait_then_done", "look_away_back"):
                    t[key] = int(sum(r[key] for r in rs))
                t["k0_done_now_old_judge"] = int(sum(r["done_now_old_judge"] for r in k0))
                vt = np.array([r["visible_ticks_first_8s"] for r in rs])
                out["watch"][grp] = {"episodes": len(rs), "visible_ticks_min": int(vt.min()),
                                     "visible_ticks_median": float(np.median(vt)), "visible_ticks_max": int(vt.max()),
                                     "episodes_under_6_ticks": [(r["config"], r["k"], r["visible_ticks_first_8s"])
                                                                for r in rs if r["visible_ticks_first_8s"] < 6],
                                     "visible_at_8s": int(sum(r["visible_at_8s"] for r in rs))}
            out["trivial"][f"{task}/{grp}"] = t
    for g in sorted(set(group_of.values())):
        gs = [x["grid"] for x in grids if group_of[x["config"]] == g]
        out["find_ball_setup_grid"][g] = {str(k): gs.count(k) for k in (1, 2, 3, None)}
    ok = [x for x in grids if x["grid"] is not None]
    out["find_ball_rest_point"] = {"setups": len(ok), **{
        k: {"min": min(x[k] for x in ok), "median": float(np.median([x[k] for x in ok])), "max": max(x[k] for x in ok)}
        for k in ("rest_along_abs_m", "rest_behind_m", "start_to_rest_m")}}
    out["ball_roll_045"] = roll_check()
    out["setup_rows"] = grids
    out["rows"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_checks.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "setup_rows")}, indent=1, default=float))


if __name__ == "__main__":
    main()
