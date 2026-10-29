"""Reviewer spot checks for M0 (REVIEW_M0.md). Deterministic, CPU, about 6 s.

  1. Gait start at the frozen bounds from several standing states (not only straight after
     reset, which is what m0_from_rest.py's "rest" column measures).
  2. Gate segments that never started (|achieved| < 0.05 on the commanded axis), per body.
  3. Standing 2 N push: trunk drift and peak tilt (RESULTS' "spot check").
  4. Do the torque-limit and low-foot-friction sweep rows do anything? Peak |actuator force|
     on a gate-like walk, and the floor/foot friction and contact priority.
  5. Head error with no head command: standing vs walking segments of gate.json.
  6. Rows missing from the m0_tune.py sweep, on its development sequence: training's obs noise
     (RESULTS claimed it "made no difference") and a policy-rate action lag of 3 and 6 control
     steps (60/120 ms: what training's BAM delay_min/max_lag 3-6 would mean if those lags were
     control steps rather than 5 ms physics steps; the vendored tree does not say which).
  7. How deep a push "fall" is: max tilt and min trunk height in the walking push trials of
     m0_gate.py at 1 N and 2 N (same set-up and seeds), for the trials that fell.
Writes results/m0/review_checks.json.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from duck import CTRL_DT, BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY  # noqa: E402

BODIES = {"chosen": FROZEN_BODY, "plain": {"scene": "allcollisions"}}
STARTS = ("after_reset", "stand_3s", "walk_then_stand_3s", "stand_3s_head_yaw")
CMDS = [(0.2, 0, 0), (-0.3, 0, 0), (0, 0, 0.5), (0, 0, -0.5), (0, 0, 0.6), (0, 0, -0.6), (0.2, 0, 0.5)]


def prep(duck, start):
    duck.reset()
    if start == "stand_3s":
        duck.set_command(0, 0, 0)
        steps = 150
    elif start == "walk_then_stand_3s":
        duck.set_command(0.3, 0, 0)
        for _ in range(150):
            duck.step()
        duck.set_command(0, 0, 0)
        steps = 150
    elif start == "stand_3s_head_yaw":
        duck.set_command(0, 0, 0, head=(0, 0, 0.8, 0))
        steps = 150
    else:
        steps = 0
    for _ in range(steps):
        duck.step()


def start_check():
    out = {}
    for bname, body in BODIES.items():
        duck = Duck(BodyConfig(**body))
        rows = []
        for cmd in CMDS:
            ax = 2 if cmd[2] else 0
            row = {"cmd": cmd, "axis": "wz" if ax == 2 else "vx"}
            for start in STARTS:
                prep(duck, start)
                duck.set_command(*cmd)
                vs = []
                for k in range(300):
                    duck.step()
                    if k >= 75:
                        vs.append(duck.planar_velocity()[ax])
                row[start] = round(float(np.mean(vs)), 4)
            row["all_started"] = all(abs(row[s]) >= 0.05 for s in STARTS)
            rows.append(row)
            print(bname, cmd, {s: row[s] for s in STARTS})
        out[bname] = rows
    return out


def non_starts():
    out = {}
    for fn in ("gate.json", "gate_allcollisions.json"):
        g = json.loads((HERE / "results" / "m0" / fn).read_text())
        bad, n = [], 0
        for run in g["runs_chosen"]["runs"]:
            prev = None
            for s in run["segments"]:
                if s["achieved"][0] is not None and s["kind"] in ("back", "turn", "fwd", "fwd_turn"):
                    ax = 2 if s["kind"] == "turn" else 0
                    n += 1
                    if abs(s["achieved"][ax]) < 0.05:
                        bad.append({"run": run["run"], "kind": s["kind"], "cmd": s["cmd"],
                                    "achieved": round(s["achieved"][ax], 4),
                                    "prev": prev["kind"] if prev else "run_start"})
                prev = s
        out[fn] = {"moving_segments": n, "never_started": bad}
        print(fn, n, "moving segments,", len(bad), "never started")
    return out


def standing_push():
    duck = Duck(BodyConfig(**FROZEN_BODY))
    duck.reset()
    for _ in range(100):
        duck.step()
    p0, tilt = duck.pos()[:2].copy(), []
    duck.push_force = np.array([0.0, 2.0, 0.0])
    for _ in range(50):
        duck.step()
        tilt.append(duck.tilt())
    duck.push_force = np.zeros(3)
    for _ in range(100):
        duck.step()
        tilt.append(duck.tilt())
    r = {"force_N": 2.0, "drift_m": round(float(np.linalg.norm(duck.pos()[:2] - p0)), 4),
         "max_tilt_deg": round(float(np.degrees(max(tilt))), 2)}
    print("standing 2 N push", r)
    return r


def sweep_noops():
    duck = Duck(BodyConfig(**FROZEN_BODY))
    m = duck.model
    peak = 0.0
    for cmd in ((0.4, 0, 0), (0, 0, 0.6), (-0.3, 0, 0), (0.3, 0, 0.6)):
        duck.set_command(*cmd)
        for _ in range(250):
            duck.step()
            peak = max(peak, float(np.abs(duck.data.actuator_force).max()))
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    r = {"peak_abs_actuator_force_Nm": round(peak, 4),
         "floor_friction": float(m.geom_friction[floor, 0]), "floor_priority": int(m.geom_priority[floor]),
         "foot_friction": float(m.geom_friction[foot, 0]), "foot_priority": int(m.geom_priority[foot]),
         "note": "equal priority -> MuJoCo uses the max of the two frictions, so foot mu < 1.0 is a no-op"}
    print("sweep no-ops", r)
    return r


def head_still_split():
    g = json.loads((HERE / "results" / "m0" / "gate.json").read_text())
    st, wk = [], []
    for run in g["runs_chosen"]["runs"]:
        for s in run["segments"]:
            if s["head_abs_err"] is None or any(s["head"]):
                continue
            (st if s["kind"] == "stand" else wk).append(s["head_abs_err"])
    r = {"names": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
         "stand_segments": len(st), "stand_mean_abs_err": np.mean(st, 0).round(4).tolist(),
         "walk_segments": len(wk), "walk_mean_abs_err": np.mean(wk, 0).round(4).tolist()}
    print("head, no head command", r)
    return r


def extra_sweep_rows():
    from m0_tune import run_sequence
    cfgs = [{**FROZEN_BODY, "obs_noise": 1.0}, {"scene": "allcollisions", "obs_noise": 1.0},
            {"scene": "allcollisions", "delay_min": 3, "delay_max": 3},
            {"scene": "allcollisions", "delay_min": 6, "delay_max": 6}]
    out = []
    for c in cfgs:
        r = run_sequence(c)
        out.append({"label": r["label"], "falls": r["falls"], "mean_rel_err": round(r["mean_rel_err"], 4),
                    "segments": r["segments"]})
        print("extra sweep", r["label"], "falls", r["falls"], "rel err", round(r["mean_rel_err"], 3),
              " ".join(f"{s['mean'][int(np.argmax(np.abs(s['cmd'])))]:+.3f}" for s in r["segments"] if any(s["cmd"])))
    return out


def push_depth():
    out = []
    for force in (1.0, 2.0):
        for direction in (-1, 1):
            for phase in (0, 7, 13):
                duck = Duck(BodyConfig(**FROZEN_BODY), seed=phase)
                duck.reset()
                duck.set_command(0.3, 0, 0)
                for _ in range(100 + phase):
                    duck.step()
                z0, tilt, z = duck.pos()[2], [], []
                for k in range(50 + 400):
                    duck.push_force = np.array([0.0, direction * force, 0.0]) if k < 50 else np.zeros(3)
                    duck.step()
                    tilt.append(duck.tilt())
                    z.append(duck.pos()[2])
                if max(tilt) > np.radians(45):
                    out.append({"force_N": force, "dir": direction, "phase": phase,
                                "max_tilt_deg": round(float(np.degrees(max(tilt))), 1),
                                "z_start_m": round(float(z0), 4), "min_z_m": round(float(min(z)), 4),
                                "final_tilt_deg": round(float(np.degrees(tilt[-1])), 1)})
    print("push depth (fell trials):", [(r["force_N"], r["max_tilt_deg"], r["min_z_m"]) for r in out])
    return out


def main():
    rep = {"push_depth": push_depth(), "extra_sweep_rows": extra_sweep_rows(), "start_check": start_check(), "gate_non_starts": non_starts(),
           "standing_push_2N": standing_push(), "sweep_noops": sweep_noops(),
           "head_no_command": head_still_split()}
    (HERE / "results" / "m0" / "review_checks.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
