"""M2 measurements on training and development configs only (no test split is touched).

A. Skill battery under the ORACLE pose hook (TruePose), 40 training configs:
   turn_to to 4 bearings; go_to_planned to every pad from the start; return_home; press_button from
   the start (true press count must be exactly 1, i.e. one toggle, no re-press); a full scan; look to
   three head poses (requested vs achieved from the encoders). Obstacle contacts counted throughout.
B. Feasibility: the scripted oracle (oracle.py: skills + ground truth) on each task, episode k=0,
   for 60 training configs and the 50 instances-dev configs. Lesson 4: a task the oracle can't solve
   must be fixed before any mind is compared.
C. Task 3 setup coverage over every config in every split (setup only: start pose and ball launch;
   no controller runs on test configs).
D. M1's 1 s standing smoke check re-run on all 970 configs, because M2 changed the ball's rolling
   friction (configs themselves are unchanged).
E. (find_ball setup v2) Hiding-spot spread over episodes k=0..3 of the training and instances-dev
   configs: which hider, left/right of the start view, rest point offsets, and how long the ball is
   in view after launch.
F. (find_ball setup v2) No-memory baselines vs the oracle on the same episodes (k=0): perfect ball
   recognition and true pose/layout, but no use of where the ball went.
Writes results/m2/battery.json.
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
from interface import Interface  # noqa: E402
from oracle import baseline_for, controller_for  # noqa: E402
from room import World  # noqa: E402
from skills import (Ctx, TruePose, go_to_planned, look, press_button, return_home, run, scan,  # noqa: E402
                    turn_to, wrap)
from make_splits import smoke_check  # noqa: E402
from tasks import TASKS, make_episode, run_episode  # noqa: E402

SPLITS, OUT = HERE / "splits", HERE / "results" / "m2"


def load(name, part):
    d = json.loads((SPLITS / name).read_text())
    return d["configs"] if part == "configs" else d[part]


class ContactMeter:
    def __init__(self, world):
        m = world.model
        self.world = world
        root = world.duck.trunk_id
        self.robot = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root}
        self.obst = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                     for n in ("wall_N", "wall_S", "wall_E", "wall_W", "partition", "occluder")} - {-1}

    def touching(self):
        d = self.world.duck.data
        return any((c.geom1 in self.robot and c.geom2 in self.obst) or (c.geom2 in self.robot and c.geom1 in self.obst)
                   for c in d.contact[:d.ncon])


def skill_battery(cfg):
    w = World(cfg)
    lay = cfg["layout"]
    out = {"config": cfg["id"]}
    rng = np.random.default_rng(int(cfg["hash"], 16) % 2 ** 32)

    def fresh():
        w.reset()
        it = Interface(w)
        meter = ContactMeter(w)
        orig = it.tick
        it.contact_ticks = 0

        def tick(c):
            o = orig(c)
            it.contact_ticks += meter.touching()
            return o
        it.tick = tick
        return it, Ctx(it, TruePose(w), lay)

    it, ctx = fresh()
    out["turn_to"] = []
    for _ in range(4):
        b = float(rng.uniform(-np.pi, np.pi))
        r = run(ctx, turn_to(ctx, b))
        out["turn_to"].append({"ok": r["ok"], "error": r["achieved"]["error"], "ticks": r["ticks"],
                               "attempts": r["attempts"]})
    out["go_to"] = []
    for pad in lay["pads"]:
        it, ctx = fresh()
        r = run(ctx, go_to_planned(ctx, pad["pos"], stop_radius=0.08))
        h = run(ctx, return_home(ctx, lay["home"]["pos"]))
        out["go_to"].append({"pad": pad["color"], "ok": r["ok"], "reason": r["reason"], "ticks": r["ticks"],
                             "distance": r["achieved"].get("distance"), "path_length": r["achieved"].get("path_length"),
                             "straight_line": float(np.linalg.norm(np.array(pad["pos"]) - lay["start"]["pos"])),
                             "home_ok": h["ok"], "home_ticks": h["ticks"], "contact_ticks": it.contact_ticks})
    it, ctx = fresh()
    lamp0, p0 = w.lamp_on, w.press_count
    r = run(ctx, press_button(ctx, lay["button"]["pos"]))
    for _ in range(20):          # stand still 2 s after: no late re-press
        ctx.obs = it.tick(ctx.cmd())
    out["press_button"] = {"ok": r["ok"], "reason": r["reason"], "ticks": r["ticks"],
                           "closest": r["achieved"].get("closest_to_centre"), "presses": w.press_count - p0,
                           "toggled": w.lamp_on != lamp0, "contact_ticks": it.contact_ticks}
    it, ctx = fresh()
    r = run(ctx, scan(ctx, full=True))
    out["scan"] = {"ok": r["ok"], "covered_deg": r["achieved"]["covered_deg"], "ticks": r["ticks"],
                   "contact_ticks": it.contact_ticks}
    out["look"] = []
    for pose in ((0.0, 0.4, 0.8, 0.0), (0.0, -0.3, -1.0, 0.1), (0.5, 0.0, 0.0, 0.0)):
        r = run(ctx, look(ctx, pose))
        out["look"].append({"requested": r["requested"]["head"], "achieved": r["achieved"]["head"],
                            "ticks": r["ticks"]})
    return out


def feasibility(args):
    cfg, task = args
    try:
        ep = make_episode(cfg, task, 0)
    except RuntimeError as e:
        return {"config_id": cfg["id"], "task": task, "setup_error": str(e)}
    w = World(cfg)
    res, _, _ = run_episode(cfg, ep, controller_for(w, task), world=w)
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in res.items()}


def setup_stats(args):
    cfg, k = args
    try:
        ep = make_episode(cfg, "find_ball", k)
    except RuntimeError:
        return None
    L = ep["ball_launch"]
    # rest point in the hider's frame and the watch time (ticks the ball is in view after launch)
    from tasks import _hiders, ball_visibility, start_episode
    h = next(x for x in _hiders(cfg["layout"]) if x["name"] == L["hider"])
    E = h["C"] + L["end"] * h["L"] * h["a"]
    rest = np.asarray(L["rest"])
    w = World(cfg)
    start_episode(w, ep)
    w.duck.set_command(0, 0, 0)
    visible = 0
    for t in range(60):
        for _ in range(5):
            w.step()
        visible += ball_visibility(w)[0]
    return {"config": cfg["id"], "k": k, "hider": L["hider"], "left": L["rest_left_of_start_view"],
            "back_from_end_m": float(-((rest - E) @ h["a"]) * L["end"]),
            "behind_m": float(abs((rest - h["C"]) @ h["n"])), "visible_ticks_first_6s": int(visible),
            "rest_xy": rest.tolist()}


def baseline_run(args):
    cfg, which = args
    try:
        ep = make_episode(cfg, "find_ball", 0)
    except RuntimeError:
        return None
    w = World(cfg)
    ctrl = controller_for(w, "find_ball") if which == "oracle" else baseline_for(w, which)
    res, _, _ = run_episode(cfg, ep, ctrl, world=w)
    return {"config": cfg["id"], "who": which, "success": res["success"], "sim_time": res["sim_time"],
            "hider": ep["ball_launch"]["hider"]}


def setup_only(cfg):
    try:
        ep = make_episode(cfg, "find_ball", 0)
        return {"config": cfg["id"], "ok": True, "rest_error": ep["ball_launch"]["rest_error"],
                "speed": ep["ball_launch"]["speed"]}
    except RuntimeError as e:
        return {"config": cfg["id"], "ok": False, "error": str(e)}


def summarise_skills(rows):
    tt = [x for r in rows for x in r["turn_to"]]
    gt = [x for r in rows for x in r["go_to"]]
    pb = [r["press_button"] for r in rows]
    sc = [r["scan"] for r in rows]
    lk = [x for r in rows for x in r["look"]]
    err = np.abs([x["error"] for x in tt])
    eff = [x["path_length"] / max(x["straight_line"], 1e-6) for x in gt if x["ok"] and x["path_length"]]
    look_err = np.array([np.array(x["achieved"]) - np.array(x["requested"]) for x in lk])
    return {
        "turn_to": {"n": len(tt), "ok_rate": float(np.mean([x["ok"] for x in tt])),
                    "abs_error_median_rad": float(np.median(err)), "abs_error_p90_rad": float(np.percentile(err, 90)),
                    "ticks_median": float(np.median([x["ticks"] for x in tt])),
                    "attempts_mean": float(np.mean([x["attempts"] for x in tt]))},
        "go_to_pad": {"n": len(gt), "ok_rate": float(np.mean([x["ok"] for x in gt])),
                      "reasons": {k: int(sum(x["reason"] == k for x in gt)) for k in {x["reason"] for x in gt}},
                      "final_distance_median_m": float(np.median([x["distance"] for x in gt if x["distance"] is not None])),
                      "ticks_median": float(np.median([x["ticks"] for x in gt])),
                      "path_over_straight_median": float(np.median(eff)) if eff else None,
                      "runs_with_obstacle_contact": int(sum(x["contact_ticks"] > 0 for x in gt))},
        "return_home": {"n": len(gt), "ok_rate": float(np.mean([x["home_ok"] for x in gt]))},
        "press_button": {"n": len(pb), "crossed_rate": float(np.mean([x["ok"] for x in pb])),
                         "exactly_one_press": int(sum(x["presses"] == 1 for x in pb)),
                         "presses_hist": {str(k): int(sum(x["presses"] == k for x in pb)) for k in sorted({x["presses"] for x in pb})},
                         "ticks_median": float(np.median([x["ticks"] for x in pb])),
                         "runs_with_obstacle_contact": int(sum(x["contact_ticks"] > 0 for x in pb))},
        "scan_full": {"n": len(sc), "ok_rate": float(np.mean([x["ok"] for x in sc])),
                      "covered_deg_min": float(min(x["covered_deg"] for x in sc)),
                      "ticks_median": float(np.median([x["ticks"] for x in sc]))},
        "look": {"n": len(lk), "mean_abs_error_per_joint_rad": np.abs(look_err).mean(0).round(3).tolist(),
                 "names": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]},
    }


def summarise_feas(rows):
    out = {}
    for task in TASKS:
        for group in ("train", "instances-dev"):
            rs = [r for r in rows if r["task"] == task and r.get("config_id", r.get("config", "")).startswith(group)]
            run_ = [r for r in rs if "setup_error" not in r]
            if not rs:
                continue
            fails = [{k: r.get(k) for k in ("config_id", "config", "ticks", "falls", "declared_done", "stopped", "on_mat",
                                            "presses", "lamp_final", "target", "ball_visible_at_end",
                                            "ball_distance", "reached_pad_tick") if k in r} for r in run_ if not r["success"]]
            out[f"{task}/{group}"] = {
                "n": len(rs), "setup_errors": len(rs) - len(run_),
                "success": int(sum(r["success"] for r in run_)),
                "falls": int(sum(r["falls"] for r in run_)),
                "episodes_with_obstacle_contact": int(sum(r["collision_ticks"] > 0 for r in run_)),
                "median_sim_s": float(np.median([r["sim_time"] for r in run_])) if run_ else None,
                "failures": fails[:10]}
    return out


def main():
    t0 = time.time()
    train = load("train.json", "configs")
    inst_dev = load("instances.json", "dev")
    every = list(train)
    for f in ("instances", "arrangements", "appearances", "distractor"):
        every += load(f"{f}.json", "dev") + load(f"{f}.json", "test")
    with Pool(4) as pool:
        skills = pool.map(skill_battery, train[:40])
        feas = pool.map(feasibility, [(c, t) for t in TASKS for c in train[:60] + inst_dev])
        setups = pool.map(setup_only, every)
        smokes = pool.map(smoke_check, every)
        stats = [r for r in pool.map(setup_stats, [(c, k) for c in train[:60] + inst_dev for k in range(4)]) if r]
        base = [r for r in pool.map(baseline_run, [(c, wch) for wch in ("occluder_only", "all_hiders", "oracle")
                                                   for c in train[:60] + inst_dev]) if r]
    rep = {"skills": summarise_skills(skills), "feasibility": summarise_feas(feas),
           "find_ball_setup": {"n": len(setups), "ok": int(sum(s["ok"] for s in setups)),
                               "failed": [s for s in setups if not s["ok"]],
                               "rest_error_max_m": float(max(s["rest_error"] for s in setups if s["ok"]))},
           "find_ball_v2_spread": {
               "episodes": len(stats),
               "hider": {h: int(sum(r["hider"] == h for r in stats)) for h in ("occluder", "partition")},
               "left_of_start_view": int(sum(r["left"] for r in stats)),
               "back_from_end_m": np.percentile([r["back_from_end_m"] for r in stats], [0, 50, 100]).round(3).tolist(),
               "behind_m": np.percentile([r["behind_m"] for r in stats], [0, 50, 100]).round(3).tolist(),
               "visible_ticks_first_6s": np.percentile([r["visible_ticks_first_6s"] for r in stats], [0, 10, 50, 100]).tolist()},
           "find_ball_v2_baselines": {
               who: {"n": int(sum(r["who"] == who for r in base)),
                     "success": int(sum(r["success"] for r in base if r["who"] == who)),
                     "success_by_hider": {h: [int(sum(r["success"] for r in base if r["who"] == who and r["hider"] == h)),
                                              int(sum(1 for r in base if r["who"] == who and r["hider"] == h))]
                                          for h in ("occluder", "partition")},
                     "median_sim_s_successes": float(np.median([r["sim_time"] for r in base if r["who"] == who and r["success"]]))
                     if any(r["success"] for r in base if r["who"] == who) else None}
               for who in ("occluder_only", "all_hiders", "oracle")},
           "find_ball_v2_rows": {"spread": stats, "baselines": base},
           "split_smoke_recheck": {"n": len(smokes), "all_ok": all(s["ok"] for s in smokes),
                                   "max_ball_moved_m": max(s["ball_moved_m"] for s in smokes),
                                   "max_tilt_deg": max(s["tilt_deg"] for s in smokes)},
           "skill_rows": skills, "feasibility_rows": feas, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "battery.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("skills", "find_ball_setup", "find_ball_v2_spread", "find_ball_v2_baselines",
                                          "split_smoke_recheck", "wall_s")}, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "failures"} for k, v in rep["feasibility"].items()}, indent=1))
    for k, v in rep["feasibility"].items():
        for f in v["failures"]:
            print("FAIL", k, f)


if __name__ == "__main__":
    main()
