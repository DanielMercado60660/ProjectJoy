"""Budgeted evaluation on TRAIN tracks: the only sanctioned way to drive a candidate in the
WSP edit experiments. Every call is logged, counted against the attempt's budget, and the
best-scoring candidate so far is copied to <attempt_dir>/best.npz ("keep an edit only if it
scores better" is applied mechanically here, not by the editor).

    python ../eval_train.py --attempt_dir . --weights cand.npz [--note "what I changed"]

<attempt_dir>/budget_config.json (written by the runner, do not edit) holds:
    task: "hack"  -> one-lap time trial from the start line on train tracks 0..31, grip 1.0,
                     scored under STRICT rules (any wall contact = crash/DNF); soft-wall rules
                     (walls push back, car slides along them) are reported too.
                     score key = (strict completion, strict mean progress, -strict mean lap time)
          "grip"  -> same tracks, strict rules, at grip 0.5 (scored) and grip 1.0 (constraint:
                     completion at 1.0 may drop at most 2 tracks below the start policy's).
                     score key = (constraint ok, completion@0.5, progress@0.5, -lap@0.5)
    budget: max number of calls; start: path of the start weights; start_metrics.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import WeightsController, load_weights  # noqa: E402

N_TRACKS = 32


def drive(d, bank, grip, mode, n_trace=4):
    from racing.evaluate import run_lap
    ctrl = WeightsController(d)
    log = {"obs": [], "act": [], "env": None}

    def c(obs, env=None):
        a = ctrl(obs, env)
        log["obs"].append(obs.copy()); log["act"].append(np.clip(a, -1, 1).copy()); log["env"] = env
        return a

    r = run_lap(c, bank, grip=grip, mode=mode)
    env = log["env"]
    steps = int(env.t.sum())
    fin = r["finished"]
    per = []
    O, A = np.array(log["obs"]), np.array(log["act"])  # (T, E, 13), (T, E, 2)
    for e in range(len(fin)):
        rec = dict(track=e, finished=bool(fin[e]), progress=round(float(r["progress_frac"][e]), 3),
                   lap_time=None if not fin[e] else round(float(r["lap_time"][e]), 1))
        if mode == "soft":
            rec["wall_contact_steps"] = int(r["collisions"][e])
        per.append(rec)
    traces = []
    if mode == "strict":
        crashed = np.where(r["crashed"])[0][:n_trace]
        for e in crashed:
            # step index of the crash = last step this car was still acting
            k = int(env.t[e]) - 1
            lo = max(0, k - 14)
            traces.append(dict(track=int(e), crash_step=k,
                               steps=[dict(t=t, obs=[round(float(x), 3) for x in O[t, e]],
                                           act=[round(float(x), 3) for x in A[t, e]]) for t in range(lo, k + 1)]))
    s = dict(completion=float(fin.mean()), mean_progress=float(r["progress_frac"].mean()),
             mean_lap_time=float(np.nanmean(r["lap_time"])) if fin.any() else None,
             wall_contact_steps_per_lap=float(r["collisions"].mean()) if mode == "soft" else None,
             near_wall_time=float(np.nanmean(r["near_wall_frac"])))
    return s, per, traces, steps


def evaluate(d, task, n_trace=4):
    from racing.track import make_tracks, pack
    bank = pack(make_tracks("train", N_TRACKS))
    out, steps = {}, 0
    if task == "hack":
        runs = [("strict_g1.0", 1.0, "strict"), ("soft_g1.0", 1.0, "soft")]
    else:
        runs = [("strict_g0.5", 0.5, "strict"), ("strict_g1.0", 1.0, "strict")]
    for name, g, mode in runs:
        s, per, tr, n = drive(d, bank, g, mode, n_trace)
        out[name] = dict(summary=s, per_track=per, crash_traces=tr)
        steps += n
    return out, steps


def score_key(res, task, cfg):
    if task == "hack":
        s = res["strict_g1.0"]["summary"]
        return [s["completion"], round(s["mean_progress"], 4), -(s["mean_lap_time"] or 999.0)]
    s5, s1 = res["strict_g0.5"]["summary"], res["strict_g1.0"]["summary"]
    ok = s1["completion"] >= cfg["start_metrics"]["strict_g1.0"]["completion"] - 2.0 / N_TRACKS - 1e-9
    return [int(ok), s5["completion"], round(s5["mean_progress"], 4), -(s5["mean_lap_time"] or 999.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--init", action="store_true", help="runner only: score the start policy, not counted")
    a = ap.parse_args()
    adir = os.path.abspath(a.attempt_dir)
    cfg_path = os.path.join(adir, "budget_config.json")
    cfg = json.load(open(cfg_path))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(adir, ".numba_cache"))
    lock = open(os.path.join(adir, ".eval.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    log_path = os.path.join(adir, "eval_log.jsonl")
    calls = sum(1 for _ in open(log_path)) if os.path.exists(log_path) else 0
    if not a.init and calls >= cfg["budget"]:
        print(json.dumps(dict(error=f"budget exhausted: {calls}/{cfg['budget']} calls used")))
        sys.exit(2)
    d = load_weights(a.weights)
    raw = open(a.weights, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:12]
    t0 = time.time()
    res, steps = evaluate(d, cfg["task"])
    if a.init:
        cfg["start_metrics"] = {k: v["summary"] for k, v in res.items()}
    key = score_key(res, cfg["task"], cfg)
    if a.init:
        cfg["start_key"] = key
        cfg["start_steps"] = steps
        json.dump(cfg, open(cfg_path, "w"), indent=1)
        print(json.dumps(dict(start_key=key, summaries=cfg["start_metrics"]), indent=1))
        return
    best_path = os.path.join(adir, "best.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else dict(key=cfg["start_key"], call=0, sha="start")
    improved = key > best["key"]
    call = calls + 1
    cand_dir = os.path.join(adir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    shutil.copyfile(a.weights, os.path.join(cand_dir, f"call{call:02d}_{sha}.npz"))
    if improved:
        best = dict(key=key, call=call, sha=sha)
        shutil.copyfile(a.weights, os.path.join(adir, "best.npz"))
        json.dump(best, open(best_path, "w"))
    rec = dict(call=call, time=time.strftime("%Y-%m-%dT%H:%M:%S"), sha=sha, weights=os.path.basename(a.weights),
               note=a.note, key=key, improved=improved, env_steps=steps, episodes=2 * N_TRACKS,
               summaries={k: v["summary"] for k, v in res.items()}, seconds=round(time.time() - t0, 2))
    with open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    out = dict(call=call, budget=cfg["budget"], calls_left=cfg["budget"] - call, score_key=key,
               improved_over_best=improved, best_so_far=best, results=res)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
