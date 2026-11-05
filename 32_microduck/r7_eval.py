"""Round 7 closed-loop evaluation with the sensors-only instrument (design/R7_CAMPAIGN.md section 3).

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select  --run runs/r7_1b_s0 [...]
    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py confirm NAME=WEIGHTS [...]
    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py fresh   NAME=WEIGHTS|bot [...]

Rooms: select = the 40 selection rooms (dev 0000-0009 x 4 families), confirm = the 80 confirmation rooms
(instances-dev 0030-0049, others 0020-0039; the project's second look), fresh = the 96-room Round 7 reserved
slice (splits/r7_*.json; one look, Stage 5). set_lamp, k=0, the task budget.
Every episode runs the policy exactly as p2_select / p2_confirm do (same P2Controller, same judge; results are
bit-identical, tests/test_r7_eval.py) plus an evaluation-side BeliefTracker on the learner's own frames
(odometry with anchors off, camera pose, detector, memory). Truth is read only to score. Per episode, beside
the judge's row: minimum body-centre distance to the plate, within 0.5 m, the tracker's first confirmed lamp
sighting (the learner's "lamp check"), the first plate entry, press ticks, the odometry path over the
check-to-entry stretch, and the three intentionality fields of the protocol: intentional_reach (entry within K
ticks of the check with path / straight line <= R; K, R from protocol.json's calibration), turn_onset_toward
(the first standing turn run of >= 6 ticks after the check starts toward the shorter bearing to the remembered
plate), verified_done (a done after a press with a fresh lamp sighting in between).
--run <dir>: every scheduled checkpoint, then the unchanged p2_select pick rule (press-needed successes, lamp
AUC from p2_diag, no-press successes, earlier step) -> results/r7/select_<run>.json (same shape as p2_select's).
Writes results/r7/eval_<rooms>_<name>.json and appends a summary line to results/r7/evals.jsonl.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from belief import BeliefTracker, pick_tracks, wrap  # noqa: E402
from interface import Interface  # noqa: E402
from m6_eval import load_set  # noqa: E402
from p2_confirm import ROOMS as CONFIRM_ROOMS  # noqa: E402
from p2_confirm import stages  # noqa: E402
from p2_select import SETS, scheduled_checkpoints, summarise  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, Judge, make_episode, start_episode  # noqa: E402

OUT = HERE / "results" / "r7"
DEFAULT_K, DEFAULT_R = 400, 2.5


def rooms_for(which):
    if which == "select":
        return [(c, s) for s in SETS for c in load_set(s, 10)]
    if which == "confirm":
        return [(c, s) for s, a, b in CONFIRM_ROOMS for c in load_set(s, b)[a:b]]
    if which == "fresh":
        out = []
        for f in ("instances", "arrangements", "appearances", "distractor"):
            for c in json.loads((HERE / "splits" / f"r7_{f}.json").read_text())["configs"]:
                out.append((c, f"{f}-r7"))
        return out
    raise ValueError(which)


def thresholds():
    try:
        p = json.loads((OUT / "protocol.json").read_text())
        cal = p.get("intentionality_calibration", {})
        return int(cal.get("K_ticks", DEFAULT_K)), float(cal.get("R_path_ratio", DEFAULT_R))
    except FileNotFoundError:
        return DEFAULT_K, DEFAULT_R


def controller_for(who, weights, cfg=None, ep=None):
    """(factory, bot or None, P2Controller or None). weights: a path, 'bot', 'servo:<variant>' (p2_servo_baseline),
    or 'selector:<spec>' (p2_skillsel)."""
    if who == "bot" or weights == "bot":
        from refbot import controller
        fac, bot = controller("set_lamp", BUDGET_TICKS["set_lamp"])
        return fac, bot, None
    if str(weights).startswith("servo:"):
        from p2_servo_baseline import controller_factory
        return controller_factory(weights.split(":", 1)[1])(cfg, ep).factory, None, None
    if str(weights).startswith("selector:"):
        from p2_skillsel import selector_controller
        fac, bot = selector_controller(weights.split(":", 1)[1], BUDGET_TICKS["set_lamp"])
        return fac, bot, None
    from p2_learner import P2Controller
    gaze = False
    if str(weights).startswith("gaze:"):                 # Stage 4A test-time replay: the chicken-head reflex on
        gaze, weights = True, weights[5:]
    cfgp = Path(weights).parent / "p2_config.json"
    pc = P2Controller(weights, json.loads(cfgp.read_text()) if cfgp.exists() else None, gaze=gaze)
    return pc.factory, None, pc


def run_instrumented(cfg, ep, who, weights, K, R):
    """One episode with the judge and the sensors-only instrument; returns the row."""
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    judge = Judge(w, ep)
    fac, bot, pc = controller_for(who, weights, cfg, ep)
    gen = fac(iface, ep["instruction"])
    bpos = np.array(cfg["layout"]["button"]["pos"])
    half = cfg["layout"]["button"]["size"] / 2
    tr = None if bot is not None else BeliefTracker()
    t0 = time.time()
    # per tick t (index = tick): executed command, truth xy / yaw after the tick, tracker view after the tick
    cmd_log, xy, yaw, press_n, lamp_seen, btn_rel, pose = [], [], [], [], [], [], []

    def snap_tracker():
        if bot is not None:
            mem = bot.mem
            st = mem.lamp_state()
            tracks = [[t["cls"], t["xy"][0], t["xy"][1], t["n"], t["last"]] for t in mem.tracks if t["n"] >= bot.P["confirm"]]
            px, py, ps = bot.pose()
        else:
            st = tr.mem.lamp_state()
            tracks = [[t["cls"], t["xy"][0], t["xy"][1], t["n"], t["last"]] for t in tr.mem.tracks if t["n"] >= tr.P["confirm"]]
            px, py, ps = tr.odo.pose()
        lamp_seen.append(None if st is None else (st[0], int(st[1])))
        tk = pick_tracks({"tracks": tracks})
        if "button" in tk:
            bx, by, last, _ = tk["button"]
            btn_rel.append((float(wrap(np.arctan2(by - py, bx - px) - ps)), float(np.hypot(bx - px, by - py)), int(last)))
        else:
            btn_rel.append(None)
        pose.append((float(px), float(py), float(ps)))

    def snap_truth():
        p = w.duck.pos()
        xy.append((float(p[0]), float(p[1])))
        yaw.append(float(w.duck.yaw()))
        press_n.append(int(w.press_count))

    obs = iface.observe()
    snap_truth()
    snap_tracker()
    try:
        cmd = next(gen)
        while not iface.done and iface.tick_count < ep["budget_ticks"]:
            cmd_log.append((float(cmd.vx), float(cmd.wz), bool(cmd.done)))
            obs = iface.tick(cmd)
            judge.update(iface)
            snap_truth()
            if iface.done:
                snap_tracker()                   # state unchanged: the done tick's belief is the previous one
                break
            cmd = gen.send(obs)                  # the bot (or the vector policy's own tracker) senses here
            if tr is not None:
                tr.update(obs)
            snap_tracker()
    except StopIteration:
        pass
    finally:
        if iface._renderer is not None:
            iface._renderer.close()
    res = judge.result(iface)
    res = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.items()}
    res.update(who=who, wall_s=round(time.time() - t0, 2), hider=None, target=ep["instruction"].get("target"),
               press_needed=bool(res["needed_presses"]), load_1min=round(os.getloadavg()[0], 2))
    if pc is not None:
        lat = np.array(pc.latency)
        res.update(latency_ms_median=float(np.median(lat) * 1000), latency_ms_p99=float(np.percentile(lat, 99) * 1000),
                   tracker_ms_median=float(np.median(pc.tracker_latency) * 1000) if pc.tracker_latency else None)
    res.update(instrument(cmd_log, xy, yaw, press_n, lamp_seen, btn_rel, pose, bpos, half, K, R))
    return res


def instrument(cmd_log, xy, yaw, press_n, lamp_seen, btn_rel, pose, bpos, half, K, R):
    """The stage counts and intentionality fields from the per-tick logs (truth: xy, press_n; tracker: the rest)."""
    T = len(xy)
    xy = np.asarray(xy)
    d = np.hypot(xy[:, 0] - bpos[0], xy[:, 1] - bpos[1])
    inside = (np.abs(xy - bpos) <= half).all(1)
    entries = [t for t in range(1, T) if inside[t] and not inside[t - 1]]
    presses = [t for t in range(1, T) if press_n[t] > press_n[t - 1]]
    check = next((t for t, s in enumerate(lamp_seen) if s is not None), None)
    done_tick = next((t for t, c in enumerate(cmd_log) if c[2]), None)
    first_entry = entries[0] if entries else None
    out = {"min_dist_plate_m": float(d.min()), "within_0_5m": bool(d.min() <= 0.5), "first_entry_tick": first_entry,
           "entry_ticks": entries, "press_ticks": presses, "lamp_check_tick": check, "done_tick": done_tick,
           "lamp_belief_at_end": lamp_seen[-1][0] if lamp_seen[-1] else None}
    # intentional reach: entry within K ticks of the check, odometry path / straight line <= R over that stretch
    reach = None
    if first_entry is not None:
        reach = False
        if check is not None and check <= first_entry <= check + K:
            P = np.asarray(pose[check:first_entry + 1])[:, :2]
            path = float(np.sum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))) if len(P) > 1 else 0.0
            straight = float(np.hypot(*(P[-1] - P[0]))) if len(P) > 1 else 0.0
            ratio = path / max(straight, 0.05)
            out.update(reach_path_m=round(path, 3), reach_straight_m=round(straight, 3), reach_path_ratio=round(ratio, 3),
                       reach_delay_ticks=first_entry - check)
            reach = bool(ratio <= R)
    out["intentional_reach"] = reach
    # turn onset: the first standing turn run (>= 6 ticks, |wz| > 0.1, vx < 0.05) after the check
    onset = None
    if check is not None:
        t = check + 1
        while t < len(cmd_log):
            vx, wz, _ = cmd_log[t]
            if vx < 0.05 and abs(wz) > 0.1:
                run = 0
                sgn = np.sign(wz)
                while t + run < len(cmd_log) and cmd_log[t + run][0] < 0.05 and np.sign(cmd_log[t + run][1]) == sgn and abs(cmd_log[t + run][1]) > 0.1:
                    run += 1
                if run >= 6:
                    rel = btn_rel[t]
                    out["turn_onset_tick"] = t
                    onset = None if rel is None else bool(np.sign(rel[0]) == sgn)
                    out["turn_onset_bearing_deg"] = None if rel is None else round(float(np.degrees(rel[0])), 1)
                    break
                t += max(run, 1)
            else:
                t += 1
    out["turn_onset_toward"] = onset
    # verified done: a done after a press with a fresh lamp sighting between the press and the done
    verified = None
    if done_tick is not None and presses and presses[0] < done_tick:
        last_press = max(p for p in presses if p < done_tick)
        seen = lamp_seen[done_tick] if done_tick < len(lamp_seen) else lamp_seen[-1]
        verified = bool(seen is not None and seen[1] > last_press)
    out["verified_done"] = verified
    out["done_after_press"] = bool(done_tick is not None and presses and presses[0] < done_tick)
    return out


def _one(args):
    cfg, setname, who, weights, K, R = args
    ep = make_episode(cfg, "set_lamp", 0)
    row = run_instrumented(cfg, ep, who, weights, K, R)
    row["set"] = setname
    return row


def r7_summary(rows):
    pn = [r for r in rows if r["press_needed"]]
    npn = [r for r in rows if not r["press_needed"]]
    lamp_right = lambda r: (r["lamp_final"] is True) == (r["target"] == "on")   # noqa: E731
    reached = [r for r in pn if (r.get("plate_visits") or 0) > 0]
    pressed = [r for r in pn if (r.get("presses") or 0) > 0]
    dap = [r for r in pn if r["done_after_press"]]
    succ = [r for r in pn if r["success"]]
    fam = {}
    for s in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == s]
        p = [r for r in rs if r["press_needed"]]
        fam[s] = {"n": len(rs), "press_needed": [int(sum(r["success"] for r in p)), len(p)],
                  "no_press": [int(sum(r["success"] for r in rs if not r["press_needed"])), len(rs) - len(p)],
                  "reached": int(sum((r.get("plate_visits") or 0) > 0 for r in p)),
                  "pressed": int(sum((r.get("presses") or 0) > 0 for r in p))}
    return {"press_needed": {"n": len(pn), "success": len(succ), "reached_plate": len(reached),
                             "within_0_5m": int(sum(r["within_0_5m"] for r in pn)),
                             "pressed": len(pressed), "lamp_right_at_end": int(sum(lamp_right(r) for r in pn)),
                             "said_done": int(sum(bool(r["declared_done"]) for r in pn)),
                             "said_done_without_press": int(sum(r["declared_done"] and not (r.get("presses") or 0) for r in pn)),
                             "done_after_press": len(dap),
                             "done_after_press_lamp_right": int(sum(lamp_right(r) for r in dap)),
                             "intentional_reach": int(sum(bool(r["intentional_reach"]) for r in reached)),
                             "unverified_success": int(sum(not (r["intentional_reach"] and r["verified_done"]) for r in succ)),
                             "verified_done": int(sum(bool(r["verified_done"]) for r in dap)),
                             "turn_onset_toward": [int(sum(r["turn_onset_toward"] is True for r in pn)),
                                                   int(sum(r["turn_onset_toward"] is not None for r in pn))],
                             "lamp_check_seen": int(sum(r["lamp_check_tick"] is not None for r in pn)),
                             "median_done_tick": float(np.median([r["done_tick"] for r in pn if r["done_tick"] is not None])) if any(r["done_tick"] is not None for r in pn) else None},
            "no_press": {"n": len(npn), "success": int(sum(r["success"] for r in npn)),
                         "said_done": int(sum(bool(r["declared_done"]) for r in npn)),
                         "median_done_tick": float(np.median([r["done_tick"] for r in npn if r["done_tick"] is not None])) if any(r["done_tick"] is not None for r in npn) else None},
            "falls": int(sum(r["falls"] for r in rows)), "collision_episodes": int(sum(r["collision_ticks"] > 0 for r in rows)),
            "by_family": fam,
            "latency_ms_median": float(np.median([r["latency_ms_median"] for r in rows])) if "latency_ms_median" in rows[0] else None,
            "load_1min_median": float(np.median([r["load_1min"] for r in rows]))}


def evaluate(which, policies, procs=None, tag=None):
    """policies: {name: weights or 'bot'}. Returns {name: (summary, rows)} and writes the files."""
    K, R = thresholds()
    rooms = rooms_for(which)
    out = {}
    for name, w in policies.items():
        who = "bot" if w == "bot" else name
        jobs = [(c, s, who, w, K, R) for c, s in rooms]
        t0 = time.time()
        with Pool(int(procs or os.environ.get("P2_PROCS", "4"))) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
        for r in rows:
            r["who"] = name
        summ = {"policy": name, "weights": w, "rooms": which, "n": len(rows), "K_ticks": K, "R_path_ratio": R,
                **summarise(rows), "stages": stages(rows), "r7": r7_summary(rows), "wall_s": round(time.time() - t0, 1),
                "uptime": os.popen("uptime").read().strip(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        OUT.mkdir(parents=True, exist_ok=True)
        fname = f"eval_{which}_{tag or name}.json"
        (OUT / fname).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float))
        with open(OUT / "evals.jsonl", "a") as f:
            f.write(json.dumps(summ, default=float) + "\n")
        print(json.dumps({k: v for k, v in summ.items() if k != "r7"}), flush=True)
        print("r7", json.dumps(summ["r7"]["press_needed"]), flush=True)
        out[name] = (summ, rows)
    return out


def select_run(run_dir, procs=None):
    """Every scheduled checkpoint on the 40 selection rooms + the open-loop lamp diagnostic; the p2_select pick rule."""
    import p2_diag
    p = HERE / run_dir
    policies = scheduled_checkpoints(p)
    name = p.name
    res = evaluate("select", {f"{name}_{c}": w for c, w in policies.items()}, procs)
    table = {}
    for c, w in policies.items():
        summ, rows = res[f"{name}_{c}"]
        t = {k: summ[k] for k in ("n", "success", "press_needed", "no_press", "presses_made", "said_done_without_press_pn", "falls")}
        d = p2_diag.run(w)
        t.update(auc=d["auc"], early_done_mismatch=[d["early_done_mismatch"], d["mismatch"]], aux_acc=d.get("aux_acc"),
                 r7=summ["r7"]["press_needed"])
        table[c] = t
        print(name, c, json.dumps(t), flush=True)
    pick = sorted(table, key=lambda k: (-table[k]["press_needed"][0], -(table[k]["auc"] or 0), -table[k]["no_press"][0], k))[0]
    rows = [dict(r, who=c) for c in policies for r in res[f"{name}_{c}"][1]]
    (OUT / f"select_{name}.json").write_text(json.dumps(
        {"run": name, "table": table, "chosen": pick, "weights": policies[pick], "rows": rows}, indent=1, default=float))
    print(name, "chosen:", pick, flush=True)
    return pick, policies[pick], table


def main():
    which = sys.argv[1]
    if sys.argv[2] == "--run":
        for run_dir in sys.argv[3:]:
            select_run(run_dir)
        return
    pols = dict(a.split("=", 1) for a in sys.argv[2:])
    evaluate(which, pols)


if __name__ == "__main__":
    main()
