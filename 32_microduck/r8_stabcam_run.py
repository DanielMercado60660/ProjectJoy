"""Round 8 Gate B runner (r8_stabcam.py has the streams). Subcommands:

  dev        filter tuning on dev rooms (train-0000..0005 scripted, train-0010/0011 bot k=8), orientations only
             -> data/r8_stabcam/dev/*.npz, results/r8/stabcam/dev_filters.json
  scripted   measurement: train-0020..0025 x the six M3.1 segments x reflex off/on -> data/r8_stabcam/runs/
  nohl       secondary: the same segments with the camera headlight's diffuse term at 0 (10 Hz renders)
             -> data/r8_stabcam/runs_nohl/
  bot        measurement: 10 press-needed set_lamp bot episodes (protocol.json lists them) x reflex off/on
  analyze    metrics from the saved runs -> results/r8/stabcam/wobble.json (+ a printed table)

Long runs go through the R8 throttle, e.g.
  ./r8_job.sh stabcam-scripted env OMP_NUM_THREADS=1 ../.venv/bin/python r8_stabcam_run.py scripted --procs 4
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402

DEV_SCRIPTED = [f"train-{i:04d}" for i in range(0, 6)]
DEV_BOT_START = 6                 # dev bot rooms: the first 2 press-needed train configs from index 6 (k=8)
MEAS_SCRIPTED = [f"train-{i:04d}" for i in range(20, 26)]
BOT_K = 8
CFG = {c["id"]: c for c in S.TRAIN}
FILTER_CANDIDATES = {
    "ema_tau1.0": {"kind": "ema", "tau": 1.0},
    "ema_tau0.5": {"kind": "ema", "tau": 0.5},
    "ab_wn1": {"kind": "ab", "wn": 1.0},
    "ab_wn2": {"kind": "ab", "wn": 2.0},
    "ab_wn4": {"kind": "ab", "wn": 4.0},
    "notch_rampyaw": {"kind": "notch", "stride_hz": 1.86, "ramp_yaw": True, "ramp_pitch": False},
    "notch_rampboth": {"kind": "notch", "stride_hz": 1.86, "ramp_yaw": True, "ramp_pitch": True},
    "notch_noramp": {"kind": "notch", "stride_hz": 1.86, "ramp_yaw": False, "ramp_pitch": False},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_npz(path, res):
    """Flatten the probe result (nested dicts of arrays + meta) into one compressed npz."""
    flat = {"meta": np.array(json.dumps(res["meta"]))}
    for k, v in res.items():
        if k == "meta":
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}/{kk}"] = vv
        else:
            flat[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    out = {"meta": json.loads(str(z["meta"]))}
    for k in z.files:
        if k == "meta":
            continue
        if "/" in k:
            a, b = k.split("/", 1)
            out.setdefault(a, {})[b] = z[k]
        else:
            out[k] = z[k]
    return out


def press_needed_rooms(n=10, start=40, k=BOT_K):
    """The first n train configs from index `start` whose set_lamp episode k needs a press (r7_gaze_pregate's rule)."""
    from tasks import make_episode
    out = []
    for c in S.TRAIN[start:]:
        ep = make_episode(c, "set_lamp", k)
        if (ep["instruction"]["target"] == "on") != bool(c["layout"]["lamp_initial_on"]):
            out.append(c["id"])
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------------------------------------ jobs
def _dev_job(a):
    kind, cid, x, y = a
    path = S.DATA / "dev" / (f"{cid}_vx{x}_wz{y}.npz" if kind == "scripted" else f"bot_{cid}_k{x}.npz")
    if path.exists():
        return str(path)
    if kind == "scripted":
        r = S.run_scripted(CFG[cid], x, y, "off", "none", keep_mats=True)
    else:
        r = S.run_bot(CFG[cid], x, "off", "none", keep_mats=True)
    save_npz(path, r)
    return str(path)


def _meas_job(a):
    kind, cid, x, y, mode, variant = a
    sub = "runs" if variant == "main" else "runs_nohl"
    name = f"{cid}_vx{x}_wz{y}_{mode}.npz" if kind == "scripted" else f"bot_{cid}_k{x}_{mode}.npz"
    path = S.DATA / sub / name
    if path.exists():
        return str(path)
    t = time.time()
    if kind == "scripted":
        if variant == "main":
            r = S.run_scripted(CFG[cid], x, y, mode, "50" if mode == "off" else "10")
        else:
            r = S.run_scripted(CFG[cid], x, y, mode, "10", headlight_diffuse=0.0)
    else:
        r = S.run_bot(CFG[cid], x, mode, "50" if mode == "off" else "10")
    r["meta"]["wall_s"] = round(time.time() - t, 1)
    save_npz(path, r)
    log("done", name, r["meta"]["wall_s"], "s")
    return str(path)


def run_pool(fn, jobs, procs):
    with Pool(procs, maxtasksperchild=4) as pool:
        return list(pool.imap_unordered(fn, jobs, chunksize=1))


# ------------------------------------------------------------------------------------------------ dev tuning
def filter_streams(mats, filt):
    """Offline: the oracle and eis effective orientations for a filter, plus the eis target offsets (deg)."""
    Rh, Rhat = mats["Rh"], mats["Rhat"]
    fo, fe = S.TargetFilter(filt), S.TargetFilter(filt)
    A_or, A_e, offs, clamped = [], [], [], []
    for k in range(len(Rh)):
        a = S.angles(Rh[k])
        y, p = fo.step(a[0], a[1])
        A_or.append(S.angles(S.level_matrix(y, p)))
        b = S.angles(Rhat[k])
        y, p = fe.step(b[0], b[1])
        offs.append(S.axis_angle_deg(S.level_matrix(y, p), Rhat[k]))
        y, p, c = S.clamp_target(y, p, Rhat[k])
        clamped.append(c)
        A_e.append(S.angles(Rh[k] @ Rhat[k].T @ S.level_matrix(y, p)))
    return np.array(A_or), np.array(A_e), np.array(offs), np.array(clamped)


def wobble_window(A, lo=S.WINDOW[0], hi=S.WINDOW[1]):
    """M3.1's number exactly: m3_gaze.detrended_std on the 50 Hz window (deg), per axis."""
    from m3_gaze import detrended_std
    return [detrended_std(A[lo:hi, i]) for i in range(3)]


def wobble_mask(A, mask):
    res = np.stack([S.detrended(A[:, i]) for i in range(3)], 1)
    return [float(np.degrees(res[mask, i].std())) for i in range(3)] if mask.sum() > 10 else [None] * 3


def cmd_dev(args):
    dev_bot = [(c, BOT_K) for c in press_needed_rooms(2, start=DEV_BOT_START)]
    assert not set(c for c, _ in dev_bot) & set(MEAS_SCRIPTED), dev_bot
    jobs = [("scripted", c, vx, wz) for c in DEV_SCRIPTED for vx, wz in S.SCRIPTED] + \
           [("bot", c, k, None) for c, k in dev_bot]
    log("dev runs", len(jobs))
    paths = run_pool(_dev_job, jobs, args.procs)
    runs = [load_npz(Path(p)) for p in sorted(paths)]
    runs = [r for r in runs if r["meta"]["kind"] == "scripted" or (r["meta"]["config"], r["meta"]["k"]) in dev_bot]
    rep = {"configs_scripted": DEV_SCRIPTED, "bot": dev_bot, "candidates": {}}
    for name, filt in FILTER_CANDIDATES.items():
        per = {"scripted": {}, "bot": []}
        for r in runs:
            A_or, A_e, offs, clamped = filter_streams(r["mats"], filt)
            m = r["meta"]
            if m["kind"] == "scripted":
                key = f"vx{m['vx']}_wz{m['wz']}"
                lo, hi = S.WINDOW
                per["scripted"].setdefault(key, []).append(
                    {"oracle": wobble_window(A_or), "eis": wobble_window(A_e), "head": wobble_window(r["ang"]["head"]),
                     "off_p99": float(np.percentile(offs[lo:hi], 99)), "off_max": float(offs[lo:hi].max()),
                     "clamped": float(clamped[lo:hi].mean())})
            else:
                walk = r["walk"].astype(bool)
                per["bot"].append({"config": m["config"], "oracle": wobble_mask(A_or, walk), "eis": wobble_mask(A_e, walk),
                                   "head": wobble_mask(r["ang"]["head"], walk),
                                   "off_p99": float(np.percentile(offs[walk], 99)), "off_max": float(offs[walk].max()),
                                   "clamped": float(clamped[walk].mean()), "clamped_all": float(clamped.mean())})
        walk_keys = [k for k in per["scripted"] if not k.startswith("vx0.0")]
        summ = {}
        for s in ("oracle", "eis", "head"):
            summ[f"scripted_walk_{s}"] = np.mean([[np.mean([r[s][i] for r in per["scripted"][k]]) for i in range(3)] for k in walk_keys], 0).round(3).tolist()
            summ[f"bot_walk_{s}"] = np.mean([b[s] for b in per["bot"]], 0).round(3).tolist()
        summ["scripted_off_p99_max"] = max(r["off_p99"] for k in per["scripted"] for r in per["scripted"][k])
        summ["bot_off_p99_max"] = max(b["off_p99"] for b in per["bot"])
        summ["bot_clamped_walk"] = float(np.mean([b["clamped"] for b in per["bot"]]))
        e1, e2 = summ["scripted_walk_eis"], summ["bot_walk_eis"]
        summ["score"] = float(np.mean([e1[0] / 2.0, e1[1] / 1.0, e2[0] / 2.0, e2[1] / 1.0]))
        rep["candidates"][name] = {"filter": filt, "summary": summ, "per": per}
        log(name, json.dumps(summ))
    best = min(rep["candidates"], key=lambda n: rep["candidates"][n]["summary"]["score"])
    rep["selection_rule"] = ("lowest mean of eis walking wobble yaw/2.0 deg and pitch/1.0 deg over the dev scripted walking "
                             "average and the two dev bot episodes (sensor-only stream, the gate's units)")
    rep["selected"] = best
    S.OUT.mkdir(parents=True, exist_ok=True)
    (S.OUT / "dev_filters.json").write_text(json.dumps(rep, indent=1))
    log("selected", best)


def cmd_measure(args, kind, variant="main"):
    if kind == "scripted":
        jobs = [("scripted", c, vx, wz, mode, variant) for c in MEAS_SCRIPTED for vx, wz in S.SCRIPTED for mode in ("off", "on")]
    else:
        proto = json.loads((S.OUT / "protocol.json").read_text())
        jobs = [("bot", c, BOT_K, None, mode, "main") for c in proto["episodes"]["bot"]["configs"] for mode in ("off", "on")]
    # longest first (bot off runs render the most)
    jobs.sort(key=lambda j: (j[0] != "bot", j[4] != "off"))
    log(kind, variant, "jobs", len(jobs))
    run_pool(_meas_job, jobs, args.procs)
    log("all done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "scripted", "nohl", "bot", "analyze"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("P2_PROCS", "4")))
    args = ap.parse_args()
    if args.cmd == "dev":
        cmd_dev(args)
    elif args.cmd == "scripted":
        cmd_measure(args, "scripted")
    elif args.cmd == "nohl":
        cmd_measure(args, "scripted", "nohl")
    elif args.cmd == "bot":
        cmd_measure(args, "bot")
    elif args.cmd == "analyze":
        from r8_stabcam_analyze import analyze
        analyze()


if __name__ == "__main__":
    main()
