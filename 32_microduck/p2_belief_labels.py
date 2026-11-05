"""Round 7 Stage 1A: per-episode belief side files (belief.py vectors) and the pre-registered label gate.

    OMP_NUM_THREADS=1 ../bin/gpu_run ../.venv/bin/python p2_belief_labels.py build --index episodes/index.jsonl \
        --index episodes/dagger_index.jsonl ... [--procs 4]       # -> episodes/cache/r7_belief_v1/<stem>_<key>.npz
    ../.venv/bin/python p2_belief_labels.py gate                   # -> results/r7/belief_labels_gate.json

Side files hold, for every ORIGINAL tick of the recording (first-done / NaN cleaning is applied later by
p2_train through ep['ticks']): vec (T, 58) float32, last_seen (T, 8) int64, pose (T, 3) float64, plus the
source path/size/mtime for validation. Frames and labels caches are untouched. Nothing here reads truth
except the gate, which scores label error against the simulator (evaluation only, never a training signal).
Gate (protocol.json, Stage 1A): plate-bearing label error on approach segments (lamp check -> first press)
median <= 6 deg and p90 <= 30 deg on train and dev demos; a confirmed button track on >= 45% of set-lamp ticks
in every source; >= 80% of check-to-press ticks learnable under the Hr sampler (chunk-aligned 128-tick
windows: the track's sighting lies inside the tick's chunk). Kill: approach p90 > 45 deg.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from belief import CLASSES, DIM, VERSION, replay_poses, vectors_from_recording, wrap  # noqa: E402

SIDE = HERE / "episodes" / "cache" / VERSION
BUTTON = CLASSES.index("button")


def side_path(rel):
    p = (HERE / rel).resolve()
    key = hashlib.sha256(str(p).encode()).hexdigest()[:16]
    return SIDE / f"{p.stem}_{key}.npz"


def build_from_arrays(header, memory, body, imu_sub, body_model=None):
    d = header.get("dagger") or {}
    restart = int(d["takeover_tick"]) if d.get("takeover_fresh") and d.get("takeover_tick") is not None else None
    poses = replay_poses(body, imu_sub, body_model, restart_at=restart)
    V, L = vectors_from_recording(memory, poses, restart_at=restart)
    return V, L, poses, restart


def load_side(rel, build=True, body_model=None):
    """The side file for a recording (built on demand). Returns dict(vec, last_seen, pose) or None."""
    p = (HERE / rel).resolve()
    stat = p.stat()
    sp = side_path(rel)
    if sp.exists():
        with np.load(sp, allow_pickle=False) as z:
            if (str(z["src"]) == str(p) and int(z["src_size"]) == stat.st_size
                    and int(z["src_mtime_ns"]) == stat.st_mtime_ns and str(z["version"]) == VERSION):
                return {k: z[k] for k in ("vec", "last_seen", "pose")}
    if not build:
        return None
    with np.load(p, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        memory = json.loads(str(z["memory"]))
        V, L, poses, restart = build_from_arrays(header, memory, z["body"], z["imu_sub"], body_model)
    SIDE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build_", dir=SIDE) as temp:
        work = Path(temp) / sp.name
        np.savez(work, vec=V, last_seen=L, pose=poses.astype(np.float64), src=np.array(str(p)),
                 src_size=np.array(stat.st_size), src_mtime_ns=np.array(stat.st_mtime_ns), version=np.array(VERSION),
                 restart_at=np.array(-1 if restart is None else restart))
        work.replace(sp)
    return {"vec": V, "last_seen": L, "pose": poses}


def _job(rel):
    try:
        d = load_side(rel)
        return {"path": rel, "ticks": int(len(d["vec"])), "button_share": float((d["vec"][:, BUTTON * 6 + 5] > 0.5).mean())}
    except Exception as e:  # noqa: BLE001
        return {"path": rel, "error": repr(e)[:200]}


def build_all(index_files, procs=4):
    rels, seen = [], set()
    for f in index_files:
        for line in open(HERE / f):
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or r.get("split", "train") != "train" or r["path"] in seen:
                continue
            seen.add(r["path"])
            rels.append(r["path"])
    t0 = time.time()
    with Pool(procs) as pool:
        out = pool.map(_job, rels, chunksize=8)
    errs = [o for o in out if "error" in o]
    rep = {"episodes": len(rels), "errors": len(errs), "first_errors": errs[:5],
           "ticks": int(sum(o.get("ticks", 0) for o in out)), "wall_s": round(time.time() - t0, 1),
           "indexes": list(index_files), "version": VERSION}
    print(json.dumps(rep), flush=True)
    return rep


# ----------------------------------------------------------------------------------------------- the gate
def _load_cfgs():
    cfgs = {}
    for f in ("train", "instances", "arrangements", "appearances", "distractor"):
        d = json.loads((HERE / "splits" / f"{f}.json").read_text())
        for part in ("configs", "dev", "test"):
            for c in d.get(part, []):
                cfgs[c["id"]] = c
    return cfgs


def score_episode(path, cfgs):
    """Truth only to score: plate-bearing label error per tick where a button track exists."""
    with np.load(path, allow_pickle=False) as z:
        H = json.loads(str(z["header"]))
        memory = json.loads(str(z["memory"]))
        truth = z["truth"]
        V, L, poses, _ = build_from_arrays(H, memory, z["body"], z["imu_sub"])
    T = len(memory)
    bpos = np.array(cfgs[H["config_id"]]["layout"]["button"]["pos"])
    rel_true = wrap(np.arctan2(bpos[1] - truth[:, 1], bpos[0] - truth[:, 0]) - truth[:, 2])
    present = V[:, BUTTON * 6 + 5] > 0.5
    rel_lbl = np.arctan2(V[:, BUTTON * 6], V[:, BUTTON * 6 + 1])
    err = np.degrees(np.abs(wrap(rel_lbl - rel_true)))
    # pose drift of the replay (start-aligned truth)
    x0, y0, yaw0 = truth[0, 0], truth[0, 1], truth[0, 2]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    tx = c * (truth[:, 0] - x0) - s * (truth[:, 1] - y0)
    ty = s * (truth[:, 0] - x0) + c * (truth[:, 1] - y0)
    pos_err = np.hypot(poses[:, 0] - tx, poses[:, 1] - ty)
    ev = H["events"]
    checks = [e["tick"] for e in ev if e["event"] == "lamp_check"]
    presses = [e["tick"] for e in ev if e["event"] == "press"]
    appr = None
    if checks and presses and presses[0] > checks[0]:
        sel = present & (np.arange(T) >= checks[0]) & (np.arange(T) <= presses[0])
        if sel.any():
            e = err[sel]
            appr = {"n": int(sel.sum()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90))}
    return {"config": H["config_id"], "ticks": T, "press_needed": int(H["result"]["needed_presses"]) > 0,
            "label_ticks": int(present.sum()), "err_median": float(np.median(err[present])) if present.any() else None,
            "err_p90": float(np.percentile(err[present], 90)) if present.any() else None,
            "pos_err_median": float(np.median(pos_err)), "pos_err_end": float(pos_err[-1]), "approach": appr,
            "check": checks[0] if checks else None, "press": presses[0] if presses else None,
            "last_seen_button": L[:, BUTTON].tolist()}


def learnable_hr(ticks, last_seen, W=128):
    """Share of ticks whose supporting sighting lies inside the tick's chunk (Hr: chunks aligned to the episode start)."""
    ok = [last_seen[t] >= (t // W) * W for t in ticks if last_seen[t] >= 0]
    return float(np.mean(ok)) if ok else None


def learnable_windows(T, ticks, last_seen, W=128, burn=32, p_start=0.2):
    """R3 window sampler: P(sighting inside the window | the tick is in the loss mask), as the panel's probe."""
    nS = max(1, T - W + 1)
    num = den = 0.0
    for t in ticks:
        ls = last_seen[t]
        if ls < 0:
            continue
        lo, hi = max(0, t - (W - 1)), min(t - burn, nS - 1)
        p_in_rand = max(0, hi - lo + 1) / nS                       # starts s with t in the loss mask
        p_learn_rand = max(0, min(hi, ls) - lo + 1) / nS if hi >= lo else 0.0   # of those, s <= last sighting
        p_in_0 = 1.0 if burn <= t < W else 0.0
        p_learn_0 = p_in_0 * (1.0 if ls >= 0 else 0.0)
        den += p_start * p_in_0 + (1 - p_start) * p_in_rand
        num += p_start * p_learn_0 + (1 - p_start) * p_learn_rand
    return float(num / den) if den > 0 else None


def gate(out=HERE / "results" / "r7" / "belief_labels_gate.json"):
    cfgs = _load_cfgs()
    rows = [json.loads(l) for l in open(HERE / "episodes" / "index.jsonl")]
    demo = [HERE / r["path"] for r in rows if r["task"] == "set_lamp" and "error" not in r][:40]
    dev = sorted((HERE / "episodes" / "p2_dev").rglob("*.npz"))[:30]
    rep = {"version": VERSION, "dim": DIM}
    for name, paths in (("train_demos_40", demo), ("p2_dev_30", dev)):
        res = [score_episode(p, cfgs) for p in paths]
        ap = [r["approach"] for r in res if r["approach"]]
        all_appr = np.concatenate([[a["median"]] for a in ap]) if ap else np.array([])
        # pooled approach-tick errors for the p90
        pooled = []
        for r, p in zip(res, paths):
            if r["approach"]:
                pooled.append((r["approach"]["median"], r["approach"]["p90"], r["approach"]["n"]))
        hr = [learnable_hr(range(r["check"], r["press"] + 1), np.array(r["last_seen_button"]))
              for r in res if r["check"] is not None and r["press"] is not None and r["press"] > r["check"]]
        wn = [learnable_windows(r["ticks"], range(r["check"], r["press"] + 1), np.array(r["last_seen_button"]))
              for r in res if r["check"] is not None and r["press"] is not None and r["press"] > r["check"]]
        hr = [h for h in hr if h is not None]
        wn = [w for w in wn if w is not None]
        rep[name] = {"episodes": len(res), "with_label_ticks": int(sum(r["label_ticks"] > 0 for r in res)),
                     "approach_segments": len(ap),
                     "approach_bearing_err_deg_median_of_medians": float(np.median(all_appr)) if len(all_appr) else None,
                     "approach_bearing_err_deg_median_of_p90s": float(np.median([a["p90"] for a in ap])) if ap else None,
                     "approach_bearing_err_deg_worst_p90": float(max(a["p90"] for a in ap)) if ap else None,
                     "approach_bearing_err_deg_pooled_p90": float(np.percentile(np.concatenate(
                         [np.full(a["n"], a["median"]) for a in ap]), 90)) if ap else None,
                     "all_ticks_bearing_err_deg_median": float(np.median([r["err_median"] for r in res if r["err_median"] is not None])),
                     "pose_err_m_end_median": float(np.median([r["pos_err_end"] for r in res])),
                     "pose_err_m_end_p90": float(np.percentile([r["pos_err_end"] for r in res], 90)),
                     "check_to_press_learnable_hr_mean": float(np.mean(hr)) if hr else None,
                     "check_to_press_learnable_r3_windows_mean": float(np.mean(wn)) if wn else None,
                     "rows": [{k: v for k, v in r.items() if k != "last_seen_button"} for r in res]}
        print(name, json.dumps({k: v for k, v in rep[name].items() if k != "rows"}), flush=True)
    # coverage per source: share of set-lamp ticks with a confirmed button track (20-episode sample each)
    src = {"index": "episodes/index.jsonl", "dagger": "episodes/dagger_index.jsonl", "dagger2": "episodes/dagger2_index.jsonl",
           "dagger3": "episodes/dagger3_index.jsonl", "recovery": "episodes/recovery_index.jsonl", "noisy": "episodes/noisy_index.jsonl"}
    rng = np.random.default_rng(0)
    cov = {}
    for name, f in src.items():
        sl = [json.loads(l) for l in open(HERE / f)]
        sl = [r for r in sl if "error" not in r and r["task"] == "set_lamp"]
        pick = rng.choice(len(sl), min(20, len(sl)), replace=False)
        shares = []
        for i in pick:
            d = load_side(sl[i]["path"])
            shares.append(float((d["vec"][:, BUTTON * 6 + 5] > 0.5).mean()))
        cov[name] = {"episodes": len(sl), "sampled": int(len(pick)), "button_track_share_of_ticks": float(np.mean(shares))}
        print(name, cov[name], flush=True)
    rep["coverage"] = cov
    g = {"approach_median_le_6_train": rep["train_demos_40"]["approach_bearing_err_deg_median_of_medians"] <= 6,
         "approach_median_le_6_dev": rep["p2_dev_30"]["approach_bearing_err_deg_median_of_medians"] <= 6,
         "approach_p90_le_30_train": rep["train_demos_40"]["approach_bearing_err_deg_median_of_p90s"] <= 30,
         "approach_p90_le_30_dev": rep["p2_dev_30"]["approach_bearing_err_deg_median_of_p90s"] <= 30,
         "coverage_ge_045_every_source": all(v["button_track_share_of_ticks"] >= 0.45 for v in cov.values()),
         "check_to_press_learnable_hr_ge_080_train": (rep["train_demos_40"]["check_to_press_learnable_hr_mean"] or 0) >= 0.8,
         "check_to_press_learnable_hr_ge_080_dev": (rep["p2_dev_30"]["check_to_press_learnable_hr_mean"] or 0) >= 0.8,
         "kill_approach_p90_gt_45": max(rep["train_demos_40"]["approach_bearing_err_deg_median_of_p90s"],
                                        rep["p2_dev_30"]["approach_bearing_err_deg_median_of_p90s"]) > 45}
    g["pass"] = all(v for k, v in g.items() if not k.startswith("kill")) and not g["kill_approach_p90_gt_45"]
    rep["gate"] = g
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("gate", json.dumps(g), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "gate"))
    ap.add_argument("--index", action="append", default=[])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("P2_PROCS", "4")))
    a = ap.parse_args()
    if a.cmd == "build":
        rep = build_all(a.index, a.procs)
        (HERE / "results" / "r7" / "belief_labels_build.json").write_text(json.dumps(rep, indent=1))
    else:
        gate()


if __name__ == "__main__":
    main()
