"""R8 gate A: M0's locomotion battery and start checks on the BAM M6 body (thin wrapper, no changes to
m0_gate.py / m0_from_rest.py or their results).

    ../.venv/bin/python r8_bam_gate.py battery <body>   # m0_gate.main(body), written to results/r8/bam/
    ../.venv/bin/python r8_bam_gate.py start <body>     # start thresholds -> results/r8/bam/start_<body>.json
    ../.venv/bin/python r8_bam_gate.py speed            # control steps/s, XML vs BAM -> speed.json
    ../.venv/bin/python r8_bam_gate.py summary          # side-by-side + verdict -> summary.json
    ../.venv/bin/python r8_bam_gate.py gate             # xml re-run + primary + their starts + speed + summary
    ../.venv/bin/python r8_bam_gate.py variants         # the sensitivity variants (use ./r8_job.sh for both)
    ../.venv/bin/python r8_bam_gate.py posthoc          # POST-HOC exploratory fixed-lag sweep (not pre-registered)

Bodies, metrics and thresholds are fixed in results/r8/bam/protocol.json (written before any run).
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
import m0_gate  # noqa: E402
from duck import BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY  # noqa: E402

OUT = HERE / "results" / "r8" / "bam"
M0 = HERE / "results" / "m0"
BAM = dict(scene="allcollisions", jv_lag=1, actuator="bam_m6")
BODIES = {
    "xml": dict(FROZEN_BODY),                                   # M0's frozen body, re-run (same code path)
    "bam": {**BAM, "bam": {}},                                  # PRIMARY (protocol.json)
    "bam_infer": {**BAM, "bam": {"semantics": "cpu", "lag_min": 0, "lag_max": 0}},  # infer_policy exactly
    "bam_cpu": {**BAM, "bam": {"semantics": "cpu"}},
    "bam_lag4": {**BAM, "bam": {"lag_min": 4, "lag_max": 4}},
    "bam_resample": {**BAM, "bam": {"lag_resample": True}},
    "bam_limiter": {**BAM, "bam": {"max_current": 1.75}},
    "bam_vin65": {**BAM, "bam": {"vin": 6.5, "force_vin": 8.2}},
    "bam_vin82": {**BAM, "bam": {"vin": 8.2, "force_vin": 8.2}},
    "bam_train": {**BAM, "bam": {"max_current": 1.75, "force_vin": 8.2, "lag_resample": True}},
}
# POST-HOC, EXPLORATORY (added after the gate verdict; NOT in protocol.json, never gating): a fixed actuator-lag
# sweep on the primary semantics, to see whether any reading of training's delay brings turn/drift into range.
POSTHOC = {f"posthoc_lag{k}": {**BAM, "bam": {"lag_min": k, "lag_max": k}} for k in (0, 1, 2, 3, 5, 6)}
STARTS = ("after_reset", "stand_3s", "walk_then_stand_3s", "stand_3s_head_yaw", "after_walk")
START_CMDS = [(0.1, 0, 0), (0.15, 0, 0), (0.2, 0, 0), (0.3, 0, 0), (0.4, 0, 0), (-0.2, 0, 0), (-0.3, 0, 0),
              (0, 0, 0.3), (0, 0, -0.3), (0, 0, 0.5), (0, 0, -0.5), (0, 0, 0.6), (0, 0, -0.6), (0, 0, 1.0),
              (0, 0.15, 0), (0, -0.15, 0), (0, 0.3, 0)]
STARTED = 0.05   # |steady achieved| on the commanded axis (m/s or rad/s): the M0 review's criterion


def gate_file(name):
    return OUT / f"gate_{name}.json"


# ----------------------------------------------------------------------------------- battery
def battery(name):
    body = {**BODIES, **POSTHOC}[name]
    m0_gate.OUT = OUT                                       # never results/m0
    t0 = time.time()
    m0_gate.main(body)
    written = OUT / ("gate.json" if body == FROZEN_BODY else f"gate_{BodyConfig(**body).label()}.json")
    written.replace(gate_file(name))
    print(f"[r8] battery {name} -> {gate_file(name).name} ({time.time() - t0:.0f} s wall)")


# ------------------------------------------------------------------------------- start checks
def prep(duck, start):
    """review_m0_checks.prep's four standing states + m0_from_rest's 'after 3 s walking at 0.3'."""
    duck.reset()
    steps, cmd, head = 0, (0, 0, 0), (0, 0, 0, 0)
    if start == "stand_3s":
        steps = 150
    elif start == "walk_then_stand_3s":
        duck.set_command(0.3, 0, 0)
        for _ in range(150):
            duck.step()
        steps = 150
    elif start == "stand_3s_head_yaw":
        steps, head = 150, (0, 0, 0.8, 0)
    elif start == "after_walk":
        steps, cmd = 150, (0.3, 0, 0)
    duck.set_command(*cmd, head=head)
    for _ in range(steps):
        duck.step()


def start_cmd(args):
    body, cmd = args
    ax = int(np.argmax(np.abs(cmd)))
    duck = Duck(BodyConfig(**body))
    row = {"cmd": list(cmd), "axis": "vx vy wz".split()[ax]}
    for start in STARTS:
        prep(duck, start)
        duck.set_command(*cmd)
        vs = []
        for k in range(300):                      # 6 s; steady state = mean over the last 4.5 s
            duck.step()
            if k >= 75:
                vs.append(duck.planar_velocity()[ax])
        row[start] = round(float(np.mean(vs)), 4)
    standing = STARTS[:4]
    row["started_from_standing"] = int(sum(abs(row[s]) >= STARTED for s in standing))
    row["started_after_walk"] = bool(abs(row["after_walk"]) >= STARTED)
    return row


def start(name):
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(start_cmd, [(BODIES[name], c) for c in START_CMDS])
    res = {"body": BODIES[name], "label": BodyConfig(**BODIES[name]).label(), "started_threshold": STARTED,
           "starts": list(STARTS), "rows": rows, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"start_{name}.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(name, r["cmd"], {s: r[s] for s in STARTS}, "standing starts", r["started_from_standing"], "/4")


# ------------------------------------------------------------------------------------- speed
def speed(reps=3, n=2000):
    out = {k: [] for k in ("xml", "bam")}
    for _ in range(reps):
        for name in ("xml", "bam"):
            duck = Duck(BodyConfig(**BODIES[name]), seed=0)
            duck.set_command(0.3, 0, 0)
            for _ in range(100):
                duck.step()
            t = time.perf_counter()
            for _ in range(n):
                duck.step()
            out[name].append(n / (time.perf_counter() - t))
    res = {"method": f"single process, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, plain scene, vx 0.3, "
                     f"{n} control steps after 100 warm-up, {reps} interleaved repeats, median",
           "steps_per_s": {k: round(float(np.median(v)), 1) for k, v in out.items()},
           "all": {k: [round(x, 1) for x in v] for k, v in out.items()},
           "bam_over_xml": round(float(np.median(out["bam"]) / np.median(out["xml"])), 3),
           "load_avg": [round(x, 2) for x in os.getloadavg()],
           "note": "the plan's 7,690 steps/s for the XML body was measured at load ~4.9"}
    (OUT / "speed.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------- summary
def q(a, p):
    return round(float(np.percentile(a, p)), 4) if len(a) else None


def metrics(g):
    """The protocol's metrics from one m0_gate report."""
    ch, tr = g["runs_chosen"], g["runs_training"]
    t = ch["tracking"]
    fwd = [s for r in ch["runs"] for s in r["segments"] if s["kind"] == "fwd" and s["achieved"][0] is not None]
    drift = np.array([s["achieved"][2] for s in fwd])
    lat = [s for r in tr["runs"] for s in r["segments"] if s["kind"] == "lat" and s["achieved"][0] is not None]
    lat_ratio = np.array([s["achieved"][1] / s["cmd"][1] for s in lat])
    bins = {}
    for lo, hi in ((0.2, 0.27), (0.27, 0.34), (0.34, 0.41)):
        sel = [s["achieved"][2] for s in fwd if lo <= s["cmd"][0] < hi]
        bins[f"vx {lo}-{hi}"] = {"n": len(sel), "median_wz": q(sel, 50)}
    row = lambda d: {"median": round(d["median_achieved_over_cmd"], 4), "p10_p90": d["ratio_p10_p90"],  # noqa: E731
                     "frac_within_25pct": d.get("frac_within_25pct"), "segments": d["segments"]}
    m = {
        "forward": row(t["vx_by_kind"]["fwd"]),
        "backward": row(t["vx_by_kind"]["back"]),
        "forward_while_turning": row(t["vx_by_kind"]["fwd_turn"]),
        "turn_all": row(t["wz"]),
        "turn_in_place": row(t["wz_by_kind"]["turn"]),
        "turn_while_walking": row(t["wz_by_kind"]["fwd_turn"]),
        "straight_yaw_drift": {"segments": len(drift), "median": q(drift, 50), "p10_p90": [q(drift, 10), q(drift, 90)],
                               "frac_abs_over_0.1": round(float(np.mean(np.abs(drift) > 0.1)), 4) if len(drift) else None,
                               "by_vx": bins},
        "lateral_training_bounds": {"segments": len(lat), "median_ratio": q(lat_ratio, 50),
                                    "p10_p90": [q(lat_ratio, 10), q(lat_ratio, 90)]},
        "falls": {"pad": g["pad"]["falls"], "runs_chosen": ch["falls"], "runs_training": tr["falls"],
                  "total": g["pad"]["falls"] + ch["falls"] + tr["falls"]},
        "pad": {"reached": f"{g['pad']['reached']}/{g['pad']['n']}", "median_time_s": g["pad"]["median_time_s"]},
        "push_fall_rate": {f"{r['force_N']}N_{'walk' if r['walking'] else 'stand'}": r["fall_rate"]
                           for r in g["push"]["table"]},
        "push_recovered_8s_min": min(r["recovered_8s"] for r in g["push"]["table"]),
        "head_still_err": t.get("head_still", {}).get("mean_abs_err_rad_per_joint"),
        "head_commanded_err": t.get("head_commanded", {}).get("mean_abs_err_rad_per_joint"),
        "wall_s": g.get("wall_s"),
    }
    m["gate"] = verdict(m)
    return m


def verdict(m):
    """protocol.json's gate: forward and turn medians within +-25%, |median straight drift| < 0.05, 0 falls."""
    c = {"forward_ratio_in_0.75_1.25": 0.75 <= m["forward"]["median"] <= 1.25,
         "turn_ratio_in_0.75_1.25": 0.75 <= m["turn_all"]["median"] <= 1.25,
         "abs_median_drift_below_0.05": abs(m["straight_yaw_drift"]["median"]) < 0.05,
         "zero_falls": m["falls"]["total"] == 0}
    return {**c, "pass": all(c.values())}


def summary():
    res = {"protocol": json.loads((OUT / "protocol.json").read_text())["gate"],
           "M0_original": metrics(json.loads((M0 / "gate.json").read_text()))}
    for name in BODIES:
        if gate_file(name).exists():
            res[name] = metrics(json.loads(gate_file(name).read_text()))
            res[name]["label"] = BodyConfig(**BODIES[name]).label()
            res[name]["body"] = BODIES[name]
        sf = OUT / f"start_{name}.json"
        if sf.exists() and name in res:
            res[name]["start"] = {f"{r['axis']} {r['cmd'][['vx', 'vy', 'wz'].index(r['axis'])]:+g}":
                                  {"standing": f"{r['started_from_standing']}/4",
                                   "achieved": [r[s] for s in STARTS]} for r in json.loads(sf.read_text())["rows"]}
    if (OUT / "speed.json").exists():
        res["speed"] = json.loads((OUT / "speed.json").read_text())
    post = {n: metrics(json.loads(gate_file(n).read_text())) for n in POSTHOC if gate_file(n).exists()}
    if post:
        res["posthoc_exploratory"] = {"note": "added after the verdict, not pre-registered, never gating; fixed lag "
                                      "(physics steps) on the primary semantics; lag 4 = variant bam_lag4", **post}
    res["primary_gate"] = res.get("bam", {}).get("gate")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    cols = [k for k in ("M0_original", *BODIES) if k in res]
    print(f"{'':28s}" + "".join(f"{c:>13s}" for c in cols))
    for key, f in (("forward", "median"), ("backward", "median"), ("turn_all", "median"),
                   ("turn_in_place", "median"), ("turn_while_walking", "median"), ("straight_yaw_drift", "median"),
                   ("lateral_training_bounds", "median_ratio")):
        print(f"{key + ' ' + f:28s}" + "".join(f"{res[c][key][f]:>13.3f}" for c in cols))
    print(f"{'falls (pad+runs)':28s}" + "".join(f"{res[c]['falls']['total']:>13d}" for c in cols))
    print(f"{'pad reached':28s}" + "".join(f"{res[c]['pad']['reached']:>13s}" for c in cols))
    print(f"{'GATE':28s}" + "".join(f"{('PASS' if res[c]['gate']['pass'] else 'fail'):>13s}" for c in cols))
    for n, m in post.items():
        print(f"{n:16s} fwd {m['forward']['median']:.3f} turn {m['turn_all']['median']:.3f} (in place "
              f"{m['turn_in_place']['median']:.3f}) drift {m['straight_yaw_drift']['median']:+.3f} falls "
              f"{m['falls']['total']} pads {m['pad']['reached']}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "battery":
        battery(sys.argv[2])
    elif mode == "start":
        start(sys.argv[2])
    elif mode == "speed":
        speed()
    elif mode == "summary":
        summary()
    elif mode == "posthoc":
        for name in POSTHOC:
            battery(name)
        summary()
    elif mode in ("gate", "variants", "all"):   # gate = XML re-run + primary; variants = the rest
        names = {"gate": ["xml", "bam"], "all": list(BODIES)}.get(mode, [n for n in BODIES if n not in ("xml", "bam")])
        for name in names:
            battery(name)
            if name in ("xml", "bam", "bam_infer", "bam_limiter", "bam_train"):
                start(name)
        if mode != "variants":
            speed()
        summary()
    else:
        sys.exit(__doc__)
