"""Round 7 open-loop instruments (design/R7_CAMPAIGN.md section 3; results/r7/protocol.json).

    ../.venv/bin/python r7_probes.py onset   NAME=WEIGHTS [...]   # turn-onset probe   -> results/r7/onset_<name>.json
    ../.venv/bin/python r7_probes.py bearing NAME=WEIGHTS [...]   # bearing read (belief heads) -> results/r7/bearing_<name>.json
    ../.venv/bin/python r7_probes.py calibrate                    # K, R for intentional reach -> protocol.json
    ../.venv/bin/python r7_probes.py ceiling                      # scripted-done ceiling on the 80 rooms (offline)

Turn-onset probe (the panel's blind_turn_t0 protocol): on the p2_dev press-needed demos whose first turn after
the lamp check is blind (the plate's last sighting precedes the check), the policy is teacher-forced on the
bot's frames / body / instruction (and, for belief-vector policies, the demo's own belief vectors) and its
turn head is read at the first turning tick t0 (the body still carries no rotation). Reported: share of demos
with p(bot's direction) > p(opposite) at t0 (baseline 0.48-0.74) and demos whose argmax is the bot's direction
(baseline 2-5 of 27). Gate: >= 0.85 and >= 15/27.
Bearing read (arms with belief heads): the plate bearing decoded from the belief head against the sensors-only
label (and against truth, reported) at the lamp check, in view and out of view. Gate: median at the check <= 30
deg and >= 60% within 30 deg; out of view <= 30 deg.
Calibration: K (ticks from the first confirmed lamp sighting to the plate entry) and R (odometry path /
straight line over that stretch) such that the frozen bot's own press-needed successes pass >= 90%.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from belief import CLASSES, replay_poses, wrap  # noqa: E402

OUT = HERE / "results" / "r7"
BUTTON = CLASSES.index("button")


def _cfgs():
    cfgs = {}
    for f in ("train", "instances", "arrangements", "appearances", "distractor"):
        d = json.loads((HERE / "splits" / f"{f}.json").read_text())
        for part in ("configs", "dev", "test"):
            for c in d.get(part, []):
                cfgs[c["id"]] = c
    return cfgs


def dev_demos():
    return [json.loads(l) for l in open(HERE / "episodes" / "p2_dev" / "index.jsonl")]


def _blind_turn_episodes():
    """p2_dev press-needed demos with a blind first turn after the lamp check (the panel's selection rule)."""
    from p2_learner import body_vec, instr_vec, prep_frame
    eps = []
    for r in dev_demos():
        z = np.load(HERE / r["path"], allow_pickle=False)
        H = json.loads(str(z["header"]))
        if int(H["result"]["needed_presses"]) == 0:
            continue
        mem = json.loads(str(z["memory"]))
        T = len(mem)
        last_seen = np.full(T, -1)
        for t in range(T):
            tr = [x for x in mem[t]["tracks"] if x[0] == "button"]
            if tr:
                last_seen[t] = max(x[4] for x in tr)
        checks = [e["tick"] for e in H["events"] if e["event"] == "lamp_check"]
        if not checks:
            continue
        tc = checks[0]
        cmd = z["cmd_exec"]
        bot_turn = (cmd[:, 0] < 0.05) & (np.abs(cmd[:, 2]) > 0.1)
        t0 = None
        for t in range(tc, min(tc + 120, T)):
            if bot_turn[t] and last_seen[t] >= 0 and last_seen[t] < tc:
                t0 = t
                break
        if t0 is None:
            continue
        eps.append({"path": r["path"], "cfg": H["config_id"], "t0": t0, "tc": tc, "cmd": cmd,
                    "fr": np.stack([prep_frame(f, 64) for f in z["frames"]]), "bd": np.stack([body_vec(b) for b in z["body"]]),
                    "ins": np.repeat(instr_vec(H["instruction"])[None], T, 0),
                    "run_len": int(np.argmin(bot_turn[t0:]) if not bot_turn[t0:].all() else T - t0)})
    return eps


def _forward(model, e, upto=None):
    import mlx.core as mx
    n = len(e["bd"]) if upto is None else upto
    frames = mx.array(e["fr"][None, :n].astype(np.float32) / 255.0) if getattr(model, "pixels", True) else None
    kw = {}
    if getattr(model, "vec_dim", 0):
        from p2_belief_labels import load_side
        kw["vec"] = mx.array(load_side(e["path"])["vec"][None, :n])
    return model(frames, mx.array(e["bd"][None, :n]), mx.array(e["ins"][None, :n]), **kw)


def turn_onset(policies):
    import mlx.core as mx
    from p2_diag import load_model
    mx.set_default_device(mx.cpu)
    eps = _blind_turn_episodes()
    report = {"episodes": len(eps), "baseline": "none at t0 17-25 of 27; p(bot) > p(opp) 0.48-0.74; argmax = bot 2-5 of 27",
              "gate": "share >= 0.85 and argmax-agree >= 15 of 27"}
    for name, w in policies.items():
        model = load_model(w)
        if getattr(model, "turn_head", "regress") != "discrete":
            report[name] = {"skipped": "no discrete turn head"}
            continue
        agree = none = opp = 0
        rows = []
        for e in eps:
            reg, done, aux, _ = _forward(model, e)
            probs = np.array(mx.softmax(aux["turn"], axis=-1)[0])
            t0 = e["t0"]
            bot = 1 if e["cmd"][t0, 2] > 0.1 else 2
            am = int(probs[t0].argmax())
            if am == 0:
                none += 1
            elif am == bot:
                agree += 1
            else:
                opp += 1
            rows.append({"cfg": e["cfg"], "t0": t0, "bot": bot, "p_bot": float(probs[t0, bot]), "p_opp": float(probs[t0, 3 - bot]),
                         "p_none": float(probs[t0, 0]), "argmax": am, "walk_logit": float(np.array(aux["walk"][0, t0])) if "walk" in aux else None})
        share = float(np.mean([r["p_bot"] > r["p_opp"] for r in rows])) if rows else None
        report[name] = {"weights": w, "n": len(rows), "agree": agree, "none": none, "opp": opp, "share_p_bot_gt_p_opp": share,
                        "pass": bool(rows and share >= 0.85 and agree >= 15), "rows": rows}
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "rows"}), flush=True)
        (OUT / f"onset_{name}.json").write_text(json.dumps(report[name], indent=1))
    return report


def bearing_read(policies):
    """Plate bearing decoded from the belief head vs the sensors-only label (and truth), per tick category."""
    import mlx.core as mx
    from p2_belief_labels import load_side
    from p2_diag import load_model
    from p2_learner import body_vec, instr_vec, prep_frame
    mx.set_default_device(mx.cpu)
    cfgs = _cfgs()
    report = {}
    for name, w in policies.items():
        model = load_model(w)
        if not getattr(model, "belief_heads", False):
            report[name] = {"skipped": "no belief heads"}
            continue
        cat = {"check": [], "in_view": [], "out_of_view": [], "check_truth": [], "in_view_truth": [], "out_of_view_truth": []}
        for r in dev_demos():
            z = np.load(HERE / r["path"], allow_pickle=False)
            H = json.loads(str(z["header"]))
            T = len(z["body"])
            side = load_side(r["path"])
            V, L = side["vec"], side["last_seen"]
            e = {"path": r["path"], "fr": np.stack([prep_frame(f, 64) for f in z["frames"]]),
                 "bd": np.stack([body_vec(b) for b in z["body"]]), "ins": np.repeat(instr_vec(H["instruction"])[None], T, 0)}
            _, _, aux, _ = _forward(model, e)
            pred = np.array(aux["belief"][0]).reshape(T, -1, 4)[:, BUTTON]
            pb = np.arctan2(pred[:, 0], pred[:, 1])
            lb = np.arctan2(V[:, BUTTON * 6], V[:, BUTTON * 6 + 1])
            present = V[:, BUTTON * 6 + 5] > 0.5
            inview = V[:, BUTTON * 6 + 3] > 0.5
            truth = z["truth"]
            bpos = np.array(cfgs[H["config_id"]]["layout"]["button"]["pos"])
            tb = wrap(np.arctan2(bpos[1] - truth[:, 1], bpos[0] - truth[:, 0]) - truth[:, 2])
            err_l = np.degrees(np.abs(wrap(pb - lb)))
            err_t = np.degrees(np.abs(wrap(pb - tb)))
            checks = [ev["tick"] for ev in H["events"] if ev["event"] == "lamp_check" and ev.get("state") in ("on", "off")]
            if checks and present[checks[0]]:
                cat["check"].append(float(err_l[checks[0]]))
                cat["check_truth"].append(float(err_t[checks[0]]))
            iv = present & inview
            ov = present & ~inview
            cat["in_view"] += err_l[iv].tolist()
            cat["in_view_truth"] += err_t[iv].tolist()
            cat["out_of_view"] += err_l[ov].tolist()
            cat["out_of_view_truth"] += err_t[ov].tolist()
        s = {k: {"n": len(v), "median_deg": float(np.median(v)) if v else None,
                 "within_30_share": float(np.mean(np.array(v) <= 30)) if v else None} for k, v in cat.items()}
        s["pass"] = bool(s["check"]["n"] and s["check"]["median_deg"] <= 30 and s["check"]["within_30_share"] >= 0.6
                         and s["out_of_view"]["median_deg"] is not None and s["out_of_view"]["median_deg"] <= 30)
        s["baseline"] = "ridge readout of R3 s1's GRU: check 57-72 deg (17-27% within 30), out of view 53-65, in view 13-17"
        report[name] = s
        print(name, json.dumps(s), flush=True)
        (OUT / f"bearing_{name}.json").write_text(json.dumps(s, indent=1))
    return report


def _bot_reach_stats(paths, cfgs):
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        H = json.loads(str(z["header"]))
        if not (int(H["result"]["needed_presses"]) and H["result"]["success"]):
            continue
        mem = json.loads(str(z["memory"]))
        truth = z["truth"]
        bpos = np.array(cfgs[H["config_id"]]["layout"]["button"]["pos"])
        half = cfgs[H["config_id"]]["layout"]["button"]["size"] / 2
        inside = (np.abs(truth[:, :2] - bpos) <= half).all(1)
        entries = [t for t in range(1, len(inside)) if inside[t] and not inside[t - 1]]
        check = next((t for t, m in enumerate(mem) if m.get("lamp") is not None), None)
        if not entries or check is None or entries[0] < check:
            rows.append({"cfg": H["config_id"], "entry": entries[0] if entries else None, "check": check, "usable": False})
            continue
        poses = replay_poses(z["body"], z["imu_sub"])
        P = poses[check:entries[0] + 1, :2]
        path = float(np.sum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))))
        straight = float(np.hypot(*(P[-1] - P[0])))
        rows.append({"cfg": H["config_id"], "entry": entries[0], "check": check, "delay": entries[0] - check,
                     "path": path, "straight": straight, "ratio": path / max(straight, 0.05), "usable": True})
    return rows


def calibrate(K0=400, R0=2.5):
    cfgs = _cfgs()
    rows = [json.loads(l) for l in open(HERE / "episodes" / "index.jsonl")]
    train = [HERE / r["path"] for r in rows if r["task"] == "set_lamp" and "error" not in r]
    dev = [HERE / r["path"] for r in dev_demos()]
    rep = {}
    for name, paths in (("train_demos", train), ("p2_dev", dev)):
        rs = _bot_reach_stats(paths, cfgs)
        us = [r for r in rs if r["usable"]]
        pass0 = float(np.mean([r["delay"] <= K0 and r["ratio"] <= R0 for r in us])) if us else None
        rep[name] = {"press_needed_successes": len(rs), "usable": len(us),
                     "delay_p90": float(np.percentile([r["delay"] for r in us], 90)) if us else None,
                     "delay_max": max(r["delay"] for r in us) if us else None,
                     "ratio_p90": float(np.percentile([r["ratio"] for r in us], 90)) if us else None,
                     "ratio_median": float(np.median([r["ratio"] for r in us])) if us else None,
                     "pass_share_at_defaults": pass0, "rows": rs}
        print(name, json.dumps({k: v for k, v in rep[name].items() if k != "rows"}), flush=True)
    allu = [r for n in rep for r in rep[n]["rows"] if r["usable"]]
    K, R = K0, R0
    share = float(np.mean([r["delay"] <= K and r["ratio"] <= R for r in allu]))
    if share < 0.9:                                   # widen, never tighten: K to the pooled 90th percentile if that is
        K = max(K0, int(np.ceil(np.percentile([r["delay"] for r in allu], 90) / 10) * 10))   # above the default,
        while share < 0.9 and R < 10:                 # then R in 0.1 steps until the bot's own successes pass >= 90%
            R = round(R + 0.1, 1)
            share = float(np.mean([r["delay"] <= K and r["ratio"] <= R for r in allu]))
    cal = {"K_ticks": K, "R_path_ratio": R, "bot_pass_share": share, "pooled_successes": len(allu),
           "defaults": [K0, R0], "written": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    rep["calibration"] = cal
    (OUT / "calibration.json").write_text(json.dumps(rep, indent=1))
    pp = OUT / "protocol.json"
    P = json.loads(pp.read_text())
    P["intentionality_calibration"] = cal
    pp.write_text(json.dumps(P, indent=1))
    print("calibration", json.dumps(cal), flush=True)
    return cal


def ceiling():
    """The scripted-done counterfactual on the 80 rooms: press-needed rows with a press, the lamp right at the end and
    no done (from the stored confirmation rows; nothing is re-run)."""
    out = {}
    for name in ("p2_hr_s0", "p2_hr_s1", "p2_r3_s0", "p2_r3_s1", "p2_e_s0", "p2_e_s1", "p2_h_s0", "p2_h_s1"):
        rows = json.loads((HERE / "results" / "p2" / f"confirm_{name}.json").read_text())
        pn = [r for r in rows if r["press_needed"]]
        lamp_right = lambda r: (r["lamp_final"] is True) == (r["target"] == "on")   # noqa: E731
        unreported = [r["config_id"] for r in pn if (r.get("presses") or 0) > 0 and lamp_right(r) and not r["declared_done"]]
        succ = int(sum(r["success"] for r in pn))
        out[name] = {"press_needed_successes": succ, "lamp_right_press_no_done": len(unreported),
                     "scripted_done_counterfactual": succ + len(unreported), "rooms": unreported}
    (OUT / "done_ceiling.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rooms"} for k, v in out.items()}, indent=1))
    return out


def main():
    cmd = sys.argv[1]
    pols = dict(a.split("=", 1) for a in sys.argv[2:])
    if cmd == "onset":
        turn_onset(pols)
    elif cmd == "bearing":
        bearing_read(pols)
    elif cmd == "calibrate":
        calibrate()
    elif cmd == "ceiling":
        ceiling()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
