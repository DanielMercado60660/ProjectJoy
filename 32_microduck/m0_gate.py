"""M0 locomotion gate for the velstand body in plain CPU MuJoCo.

Measures (brief M0 step 4-5):
  A. Pad reach: a 1.5 m pad in 8 directions x 2 start headings, driven by a simple
     go-to controller that reads the TRUE trunk pose (an oracle navigator: this tests the
     legs, not perception). Reach = trunk within 0.10 m of the pad centre within 60 s.
  B. 20 scripted runs of 60 s at the chosen command bounds (plus the same 20 at the full
     training bounds): falls per 100 s, steady-state vx / yaw-rate tracking vs the 25%
     target, head-pose tracking.
  C. A 1 s lateral push on the trunk (world-frame force through xfrc_applied) while standing
     and while walking: fall rate and time back to upright.
Writes results/m0/gate.json and prints a summary. Fall = trunk tilt > 45 deg (re-armed
when tilt < 20 deg). The body config is FROZEN_BODY below (chosen in m0_tune.py / the
from-rest test before this script was run).
"""
import json
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from duck import CTRL_DT, HEAD_CAPS, BodyConfig, Duck  # noqa: E402

OUT = HERE / "results" / "m0"
FROZEN_BODY = dict(scene="allcollisions", phys_delay=4, jv_lag=1)
FALL_TILT, REARM_TILT = np.radians(45), np.radians(20)
SETTLE_S = 1.5

# Command bounds. "chosen" = what later milestones may send; "training" = velstand's own ranges.
BOUNDS = {
    "chosen": {"vx": (0.2, 0.4), "vx_back": -0.3, "vy": 0.0, "wz": (0.5, 0.6)},
    "training": {"vx": (0.05, 0.4), "vx_back": -0.4, "vy": 0.3, "wz": (0.05, 1.0)},
}


class FallCounter:
    def __init__(self):
        self.falls, self.fallen, self.fallen_steps = 0, False, 0

    def update(self, tilt):
        if not self.fallen and tilt > FALL_TILT:
            self.falls += 1
            self.fallen = True
        elif self.fallen and tilt < REARM_TILT:
            self.fallen = False
        self.fallen_steps += int(self.fallen)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- A. pad reach
def goto_command(duck, target, vx=0.3, wz=0.6, align=0.3):
    p = duck.pos()
    dx, dy = target[0] - p[0], target[1] - p[1]
    e = wrap(np.arctan2(dy, dx) - duck.yaw())
    if abs(e) > align:
        return 0.0, 0.0, float(np.sign(e) * wz)
    return vx, 0.0, float(np.clip(1.5 * e, -wz, wz))


def pad_trial(args):
    body, direction_deg, start_yaw, seed = args
    duck = Duck(BodyConfig(**body), seed=seed)
    duck.reset(yaw=start_yaw)
    target = 1.5 * np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    fc, reached_t, min_d = FallCounter(), None, 9.0
    for k in range(int(60 / CTRL_DT)):
        d = float(np.linalg.norm(duck.pos()[:2] - target))
        min_d = min(min_d, d)
        if d < 0.10:
            reached_t = (k + 1) * CTRL_DT
            break
        duck.set_command(*goto_command(duck, target))
        duck.step()
        fc.update(duck.tilt())
    # stop and settle 2 s on the pad
    duck.set_command(0, 0, 0)
    for _ in range(100):
        duck.step()
        fc.update(duck.tilt())
    return {"direction_deg": direction_deg, "start_yaw": start_yaw, "reached": reached_t is not None,
            "time_s": reached_t, "min_dist": round(min_d, 4),
            "final_dist": round(float(np.linalg.norm(duck.pos()[:2] - target)), 4), "falls": fc.falls}


# -------------------------------------------------------- B. 60 s scripted runs
def make_sequence(rng, bounds, total_s=60.0):
    """Random segments of 4-8 s from the bounds; ~1/3 of segments also move the head."""
    segs, t = [], 0.0
    b = bounds
    kinds = ["stand", "fwd", "fwd", "back", "turn", "fwd_turn"] + (["lat"] if b["vy"] else [])
    while t < total_s - 1e-9:
        dur = float(min(rng.uniform(4, 8), total_s - t))
        kind = kinds[rng.integers(len(kinds))]
        vx = vy = wz = 0.0
        if kind == "fwd":
            vx = rng.uniform(*b["vx"])
        elif kind == "back":
            vx = b["vx_back"]
        elif kind == "lat":
            vy = rng.choice([-1, 1]) * rng.uniform(0.1, b["vy"])
        elif kind == "turn":
            wz = rng.choice([-1, 1]) * rng.uniform(*b["wz"])
        elif kind == "fwd_turn":
            vx = rng.uniform(*b["vx"])
            wz = rng.choice([-1, 1]) * rng.uniform(*b["wz"])
        head = [0.0, 0.0, 0.0, 0.0]
        if rng.random() < 1 / 3:
            head = (rng.uniform(-1, 1, 4) * HEAD_CAPS * 0.8).round(3).tolist()
        segs.append({"kind": kind, "cmd": [round(vx, 3), round(vy, 3), round(wz, 3)],
                     "head": head, "dur": round(dur, 3)})
        t += dur
    return segs


def scripted_run(args):
    body, run_id, bounds_name = args
    rng = np.random.default_rng(1000 + run_id)
    seq = make_sequence(rng, BOUNDS[bounds_name])
    duck = Duck(BodyConfig(**body), seed=run_id)
    duck.reset()
    fc, out = FallCounter(), []
    for seg in seq:
        duck.set_command(*seg["cmd"], head=seg["head"])
        vs, herr, fell_here = [], [], fc.falls
        for k in range(int(round(seg["dur"] / CTRL_DT))):
            duck.step()
            fc.update(duck.tilt())
            if (k + 1) * CTRL_DT > SETTLE_S and not fc.fallen:
                vs.append(duck.planar_velocity())
                herr.append(np.abs(duck.head_pose_error()))
        mean_v = np.mean(vs, 0).tolist() if vs else [None] * 3
        out.append({**seg, "achieved": mean_v,
                    "head_abs_err": np.mean(herr, 0).round(4).tolist() if herr else None,
                    "falls": fc.falls - fell_here})
    return {"run": run_id, "bounds": bounds_name, "sim_s": round(duck.time, 2),
            "falls": fc.falls, "fallen_s": fc.fallen_steps * CTRL_DT, "segments": out}


def tracking_summary(runs):
    rows = {"vx": [], "wz": []}
    head_err, head_err_cmd = [], []
    for r in runs:
        for s in r["segments"]:
            if s["achieved"][0] is None:
                continue
            for ax, key in ((0, "vx"), (2, "wz")):
                c = s["cmd"][ax]
                if abs(c) > 1e-6:
                    a = s["achieved"][ax]
                    rows[key].append({"cmd": c, "achieved": a, "rel_err": abs(a - c) / abs(c),
                                      "kind": s["kind"]})
            if s["head_abs_err"] is not None:
                (head_err_cmd if any(s["head"]) else head_err).append(s["head_abs_err"])
    summ = {}
    for key, rs in rows.items():
        if not rs:
            continue
        rel = np.array([x["rel_err"] for x in rs])
        ratio = np.array([x["achieved"] / x["cmd"] for x in rs])
        summ[key] = {"segments": len(rs), "median_rel_err": float(np.median(rel)),
                     "frac_within_25pct": float(np.mean(rel <= 0.25)),
                     "median_achieved_over_cmd": float(np.median(ratio)),
                     "ratio_p10_p90": np.percentile(ratio, [10, 90]).round(3).tolist()}
    # Per segment kind (reviewer addition): pure forward / backward / turn in place / forward+turn.
    for key, rs in rows.items():
        for kind in sorted({x["kind"] for x in rs}):
            ratio = np.array([x["achieved"] / x["cmd"] for x in rs if x["kind"] == kind])
            summ.setdefault(f"{key}_by_kind", {})[kind] = {
                "segments": len(ratio), "median_achieved_over_cmd": round(float(np.median(ratio)), 4),
                "ratio_p10_p90": np.percentile(ratio, [10, 90]).round(3).tolist(),
                "ratio_min": round(float(ratio.min()), 4),
                "frac_within_25pct": round(float(np.mean(np.abs(ratio - 1) <= 0.25)), 4)}
    for name, arr in (("head_still", head_err), ("head_commanded", head_err_cmd)):
        if arr:
            a = np.array(arr)
            summ[name] = {"segments": len(a), "mean_abs_err_rad_per_joint": a.mean(0).round(4).tolist(),
                          "names": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]}
    return summ


# --------------------------------------------------------------- C. push test
def push_trial(args):
    body, force, direction, walking, phase_steps = args
    duck = Duck(BodyConfig(**body), seed=int(phase_steps))
    duck.reset()
    duck.set_command(0.3 if walking else 0.0, 0, 0)
    for _ in range(100 + int(phase_steps)):  # 2 s warm-up + phase offset
        duck.step()
    fc = FallCounter()
    duck.push_force = np.array([0.0, direction * force, 0.0])
    for _ in range(int(1.0 / CTRL_DT)):
        duck.step()
        fc.update(duck.tilt())
    duck.push_force = np.zeros(3)
    upright_t, calm = None, 0
    for k in range(int(8.0 / CTRL_DT)):
        duck.step()
        fc.update(duck.tilt())
        calm = calm + 1 if duck.tilt() < np.radians(15) else 0
        if upright_t is None and calm >= 25:  # upright for 0.5 s
            upright_t = (k + 1 - 24) * CTRL_DT
    return {"force_N": force, "dir": direction, "walking": walking, "phase": int(phase_steps),
            "fell": fc.falls > 0, "upright_after_s": upright_t,
            "recovered_8s": upright_t is not None and not fc.fallen}


def main(body=FROZEN_BODY):
    """Robustness re-run on another body: python m0_gate.py '{"scene": "allcollisions"}'."""
    t0 = time.time()
    out_name = "gate.json" if body == FROZEN_BODY else f"gate_{BodyConfig(**body).label()}.json"
    report = {"body": body, "body_label": BodyConfig(**body).label(),
              "fall_tilt_deg": 45, "settle_s": SETTLE_S, "bounds": BOUNDS}
    with Pool(4) as pool:
        pads = pool.map(pad_trial, [(body, dd, sy, i) for i, (dd, sy) in enumerate(
            (dd, sy) for dd in range(0, 360, 45) for sy in (0.0, 2.0))])
        report["pad"] = {"trials": pads, "reached": sum(p["reached"] for p in pads),
                         "n": len(pads), "falls": sum(p["falls"] for p in pads),
                         "median_time_s": float(np.median([p["time_s"] for p in pads if p["reached"]]))
                         if any(p["reached"] for p in pads) else None}
        for bname in ("chosen", "training"):
            runs = pool.map(scripted_run, [(body, i, bname) for i in range(20)])
            sim_s = sum(r["sim_s"] for r in runs)
            falls = sum(r["falls"] for r in runs)
            report[f"runs_{bname}"] = {
                "n_runs": len(runs), "sim_s": sim_s, "falls": falls,
                "falls_per_100s": 100 * falls / sim_s, "fallen_s": sum(r["fallen_s"] for r in runs),
                "tracking": tracking_summary(runs), "runs": runs}
        push_args = [(body, f, dr, w, ph) for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
                     for dr in (-1, 1) for w in (False, True) for ph in (0, 7, 13)]
        pushes = pool.map(push_trial, push_args)
    ptab = []
    for f in sorted({p["force_N"] for p in pushes}):
        for w in (False, True):
            ps = [p for p in pushes if p["force_N"] == f and p["walking"] == w]
            ts = [p["upright_after_s"] for p in ps if p["upright_after_s"] is not None]
            tf = [p["upright_after_s"] for p in ps if p["upright_after_s"] is not None and p["fell"]]
            ptab.append({"force_N": f, "impulse_Ns": f * 1.0, "walking": w, "n": len(ps),
                         "fall_rate": float(np.mean([p["fell"] for p in ps])),
                         "recovered_8s": float(np.mean([p["recovered_8s"] for p in ps])),
                         "median_upright_after_s": float(np.median(ts)) if ts else None,
                         # reviewer addition: over the trials that fell only (the column above
                         # pools in trials that never left upright, which read 0.02 s)
                         "fell_median_upright_after_s": float(np.median(tf)) if tf else None,
                         "fell_max_upright_after_s": float(max(tf)) if tf else None})
    report["push"] = {"table": ptab, "trials": pushes}
    ch = report["runs_chosen"]
    report["gate"] = {
        "falls_per_100s_chosen": ch["falls_per_100s"],
        "pad_reached": f"{report['pad']['reached']}/{report['pad']['n']}",
        "pass": bool(ch["falls_per_100s"] < 1.0 and report["pad"]["reached"] == report["pad"]["n"]),
    }
    report["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / out_name).write_text(json.dumps(report, indent=1))

    print(f"body {report['body_label']}  wall {report['wall_s']} s")
    print(f"pad: {report['pad']['reached']}/{report['pad']['n']} reached, median "
          f"{report['pad']['median_time_s']} s, falls {report['pad']['falls']}")
    for bname in ("chosen", "training"):
        r = report[f"runs_{bname}"]
        print(f"runs[{bname}]: {r['sim_s']:.0f} s sim, falls {r['falls']} "
              f"({r['falls_per_100s']:.2f}/100 s)")
        for k, v in r["tracking"].items():
            print(f"   {k}: {json.dumps(v)}")
    for row in ptab:
        print(f"push {row['force_N']:.2f} N walking={row['walking']!s:5s} fall {row['fall_rate']:.2f} "
              f"recovered {row['recovered_8s']:.2f} upright after {row['median_upright_after_s']}")
    print("GATE:", report["gate"])


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else FROZEN_BODY)
