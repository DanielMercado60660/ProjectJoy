"""M0 development sweep: which body settings make velstand track best in plain MuJoCo?

The "reasonable fixes" the brief allows: actuator gain, torque (current) limit, action delay,
foot friction/contact, plus the two training-side details infer_policy.py leaves out (physics-step
actuator lag and the one-step joint-velocity lag). Each config walks a fixed development
command sequence (seed-free, deterministic) and we score steady-state tracking and falls.
This is tuning, so it uses its own command sequence, not the 20 gate runs.
Writes results/m0/tune.json.
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from duck import CTRL_DT, BodyConfig, Duck  # noqa: E402

DEV_SEQUENCE = [  # (vx, vy, wz), seconds
    ((0.0, 0.0, 0.0), 3), ((0.15, 0, 0), 6), ((0.3, 0, 0), 6), ((0.4, 0, 0), 6),
    ((0.0, 0, 0), 3), ((-0.2, 0, 0), 6), ((0, 0.15, 0), 6), ((0, -0.25, 0), 6),
    ((0, 0, 0.6), 6), ((0, 0, -1.0), 6), ((0.2, 0, 0.5), 6), ((0, 0, 0), 3),
]
SETTLE = 1.5  # seconds ignored at the start of each segment


def run_sequence(cfg_dict, seq=DEV_SEQUENCE, seed=0):
    duck = Duck(BodyConfig(**cfg_dict), seed=seed)
    rows, falls, fallen = [], 0, False
    for cmd, dur in seq:
        duck.set_command(*cmd)
        vs = []
        for k in range(int(round(dur / CTRL_DT))):
            duck.step()
            tilt = duck.tilt()
            if not fallen and tilt > np.radians(45):
                falls, fallen = falls + 1, True
            elif fallen and tilt < np.radians(20):
                fallen = False
            if (k + 1) * CTRL_DT > SETTLE:
                vs.append(duck.planar_velocity())
        rows.append({"cmd": cmd, "mean": np.mean(vs, 0).round(4).tolist()})
    # relative tracking error per moving axis
    errs = []
    for r in rows:
        for ax in range(3):
            c = r["cmd"][ax]
            if abs(c) > 1e-6:
                errs.append(abs(r["mean"][ax] - c) / abs(c))
    return {"cfg": cfg_dict, "label": BodyConfig(**cfg_dict).label(), "falls": falls,
            "mean_rel_err": float(np.mean(errs)), "segments": rows}


def grid():
    base = dict(scene="allcollisions")
    out = [base]
    for kp in (0.7, 0.85, 1.25, 1.5, 2.0):
        out.append({**base, "kp_scale": kp})
    for pd in (2, 3, 4, 6):
        out.append({**base, "phys_delay": pd})
    out.append({**base, "jv_lag": 1})
    out.append({**base, "delay_min": 1, "delay_max": 1})
    for mu in (0.7, 1.3, 2.0):
        out.append({**base, "foot_friction": mu})
    out.append({**base, "foot_solref": 0.04})
    for tau in (0.6, 1.5):
        out.append({**base, "torque_limit": tau})
    out.append({"scene": "groundcontact"})
    # training-like combos
    for kp, pd in itertools.product((1.0, 1.25, 1.5), (3, 4)):
        out.append({**base, "kp_scale": kp, "phys_delay": pd, "jv_lag": 1})
    return out


def main():
    t0 = time.time()
    cfgs = grid()
    with Pool(4) as pool:
        res = pool.map(run_sequence, cfgs)
    res.sort(key=lambda r: (r["falls"], r["mean_rel_err"]))
    for r in res:
        seg = " ".join(f"{r_['mean'][int(np.argmax(np.abs(r_['cmd'])))]:+.2f}" for r_ in r["segments"]
                       if any(r_["cmd"]))
        print(f"{r['label']:42s} falls={r['falls']} relerr={r['mean_rel_err']:.3f} | {seg}")
    out = HERE / "results" / "m0" / "tune.json"
    out.write_text(json.dumps({"sequence": DEV_SEQUENCE, "settle_s": SETTLE, "results": res,
                               "wall_s": time.time() - t0}, indent=1))
    print(f"wrote {out} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
