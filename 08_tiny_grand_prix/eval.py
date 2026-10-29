"""Evaluate controllers on one-lap time trials.

Splits: heldout (100 unseen tracks, same generator), hard (50 unseen narrower,
twistier tracks: out of distribution), train (first 100 training tracks, to show
the generalization gap). Grip levels: training grip is 1.0.

    ../.venv/bin/python eval.py --controllers es ppo ray pursuit
Writes results/eval.json, results/eval_per_track.npz and results/eval_table.md.
"""
import argparse
import json
import os
import time

import numpy as np

from racing.controllers import LABELS, get_controller
from racing.evaluate import run_lap, summarize
from racing.track import make_tracks, pack

SPLITS = {"heldout": 100, "hard": 50, "train": 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["es", "ppo", "es_dr", "ppo_dr", "ray", "ray_dr", "ray_default", "pursuit"])
    ap.add_argument("--grips", nargs="+", type=float, default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--modes", nargs="+", default=["strict", "respawn"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, per_track = [], {}
    for split in a.splits:
        bank = pack(make_tracks(split, SPLITS[split]))
        for name in a.controllers:
            ctrl = get_controller(name, bank)
            for grip in a.grips:
                for mode in a.modes:
                    t0 = time.time()
                    res = run_lap(ctrl, bank, grip=grip, mode=mode)
                    s = summarize(res)
                    s.update(controller=name, split=split, grip=grip, mode=mode)
                    rows.append(s)
                    key = f"{name}|{split}|{grip}|{mode}"
                    per_track[key + "|lap_time"] = res["lap_time"]
                    per_track[key + "|collisions"] = res["collisions"]
                    per_track[key + "|progress"] = res["progress_frac"]
                    per_track[key + "|lat_p95"] = res["lat_p95"]
                    print(f"{split:8s} {name:12s} grip={grip:.2f} {mode:7s} done={s['completion']:.2f} "
                          f"crash={s['crash_rate']:.2f} coll/lap={s['collisions_per_lap']:.2f} "
                          f"lap={s['mean_lap_time']:.1f}s ({time.time() - t0:.1f}s)", flush=True)
    json.dump(rows, open(f"{a.out}/eval.json", "w"), indent=1)
    np.savez(f"{a.out}/eval_per_track.npz", **per_track)
    write_table(rows, per_track, a, f"{a.out}/eval_table.md")


def write_table(rows, per_track, a, path):
    L = []
    for split in a.splits:
        for mode in a.modes:
            L.append(f"\n### {split} tracks, {mode} rules\n")
            L.append("| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | "
                     "lap time on tracks every controller finished |")
            L.append("|---|---|---|---|---|---|---|")
            for grip in a.grips:
                lts = {c: per_track[f"{c}|{split}|{grip}|{mode}|lap_time"] for c in a.controllers}
                common = np.all([np.isfinite(v) for v in lts.values()], 0)
                for c in a.controllers:
                    r = [x for x in rows if x["controller"] == c and x["split"] == split
                         and x["grip"] == grip and x["mode"] == mode][0]
                    cm = f"{np.mean(lts[c][common]):.1f} s (n={common.sum()})" if common.any() else "n/a"
                    L.append(f"| {LABELS.get(c, c)} | {grip} | {r['completion']:.0%} | {r['crash_rate']:.0%} | "
                             f"{r['collisions_per_lap']:.2f} | {r['mean_lap_time']:.1f} s | {cm} |")
    L.append("\n### Driving style / wall-hugging check (strict rules)\n")
    L.append("|lat| = distance from the centerline as a fraction of the room to the wall "
             "(0 = centerline, 1 = touching). near-wall = share of time with |lat| > 0.8. "
             "jitter = mean |change of steering command| per step (range of steer is [-1, 1]).\n")
    L.append("| split | controller | grip | mean abs lat | p95 abs lat | time near wall | steer jitter | "
             "crashes into inside (apex) wall |")
    L.append("|---|---|---|---|---|---|---|---|")
    for split in a.splits:
        for grip in [g for g in a.grips if g in (1.0, 0.8, 0.6)]:
            for c in a.controllers:
                r = [x for x in rows if x["controller"] == c and x["split"] == split
                     and x["grip"] == grip and x["mode"] == "strict"]
                if r:
                    r = r[0]
                    L.append(f"| {split} | {LABELS.get(c, c)} | {grip} | {r['lat_mean']:.2f} | {r['lat_p95']:.2f} | "
                             f"{r['near_wall_time']:.1%} | {r['steer_jitter']:.3f} | "
                             + (f"{r['inside_wall_share']:.0%} |" if r["crash_rate"] > 0 else "no crashes |"))
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
