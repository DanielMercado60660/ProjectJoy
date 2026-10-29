"""Write splits/*.json: one common training pool plus dev and test sets per named factor.

  train.json         300 configs: train layout family, train appearance family, no distractor
  instances.json     dev 50 / test 120: train families, fresh layout + instance + appearance draws
  arrangements.json  dev 50 / test 120: reserved layout families (dev / test rules), train appearance draws
  appearances.json   dev 50 / test 120: the physical world of a training config (same layout and
                     instance seeds) with a reserved appearance family: only the look changes
  distractor.json    dev 40 / test 120: a training config plus the flickering screen
Every config is fully resolved (no generator needed to read it), pinned to MuJoCo 3.16.0, validated
(geometry, reachability, family rules) and smoke-built (1 s standing: no fall, no press, ball still).
Seeds come from disjoint integer ranges per (file, split, stream), so no held-out seed is a training seed.
Writes splits/MANIFEST.json with sha256 per file. Run once, before any training; check_splits.py
re-verifies everything.
"""
import datetime as dt
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs import APPEARANCE_FAMILIES, ARRANGEMENT_FAMILIES, make_config  # noqa: E402
from room import MUJOCO_VERSION, World, config_hash  # noqa: E402

SPLITS = HERE / "splits"
N_TRAIN, N_DEV, N_TEST, N_DIST_DEV = 300, 50, 120, 40
# Seed ranges: stream base + 1,000,000 * slot; slots below never overlap.
SLOT = {("train", "train"): 0, ("instances", "dev"): 1, ("instances", "test"): 2,
        ("arrangements", "dev"): 3, ("arrangements", "test"): 4,
        ("appearances", "dev"): 5, ("appearances", "test"): 6,
        ("distractor", "dev"): 7, ("distractor", "test"): 8}
STREAM = {"layout": 10_000_000, "appearance": 20_000_000, "instance": 30_000_000}


def seeds_for(factor, split, i):
    s = SLOT[(factor, split)]
    return {k: base + 1_000_000 * s + i for k, base in STREAM.items()}


def smoke_check(cfg):
    """1 s of standing in the built world: no fall, no button press, ball still."""
    w = World(cfg)
    ball0 = w.ball_pos()
    for _ in range(50):
        w.step()
    tilt = float(np.degrees(w.duck.tilt()))
    moved = float(np.linalg.norm(w.ball_pos()[:2] - ball0[:2]))
    ok = tilt < 10 and w.press_count == 0 and moved < 0.01
    return {"id": cfg["id"], "ok": bool(ok), "tilt_deg": round(tilt, 2), "presses": w.press_count,
            "ball_moved_m": round(moved, 5), "button_depth_mm": round(1000 * w.button_depth(), 3)}


def build_all():
    train = [make_config(f"train-{i:04d}", "train", "train", seeds_for("train", "train", i)) for i in range(N_TRAIN)]
    out = {"train": {"train": train}}
    for factor in ("instances", "arrangements", "appearances", "distractor"):
        out[factor] = {}
        for split in ("dev", "test"):
            n = (N_DIST_DEV if factor == "distractor" else N_DEV) if split == "dev" else N_TEST
            cfgs = []
            for i in range(n):
                sd = seeds_for(factor, split, i)
                cid = f"{factor}-{split}-{i:04d}"
                if factor == "instances":
                    cfg = make_config(cid, "train", "train", sd)
                elif factor == "arrangements":
                    cfg = make_config(cid, split, "train", sd)
                else:
                    # the physical world of training config j (same layout and instance seeds)
                    src = train[i % N_TRAIN] if split == "test" else train[(N_TRAIN - 1 - i) % N_TRAIN]
                    sd = {"layout": src["seeds"]["layout"], "instance": src["seeds"]["instance"],
                          "appearance": sd["appearance"]}
                    if factor == "appearances":
                        cfg = make_config(cid, "train", split, sd)
                    else:
                        sd["appearance"] = src["seeds"]["appearance"]
                        cfg = make_config(cid, "train", "train", sd, distractor=True)
                    cfg["source_train_id"] = src["id"]
                    cfg["hash"] = config_hash(cfg)
                cfgs.append(cfg)
            out[factor][split] = cfgs
    return out


def main():
    t0 = time.time()
    data = build_all()
    every = [c for f in data.values() for s in f.values() for c in s]
    with Pool(4) as pool:
        checks = pool.map(smoke_check, every)
    bad = [c for c in checks if not c["ok"]]
    if bad:
        raise SystemExit(f"{len(bad)} configs failed the standing smoke check: {bad[:5]}")
    SPLITS.mkdir(exist_ok=True)
    files = {}
    common = {"mujoco_version": MUJOCO_VERSION,
              "arrangement_families": ARRANGEMENT_FAMILIES, "appearance_families": APPEARANCE_FAMILIES}
    (SPLITS / "train.json").write_text(json.dumps(
        {"factor": "train", "note": "the common training pool shared by every factor", **common,
         "configs": data["train"]["train"]}, indent=0))
    for factor in ("instances", "arrangements", "appearances", "distractor"):
        (SPLITS / f"{factor}.json").write_text(json.dumps(
            {"factor": factor, "train": "train.json", **common,
             "dev": data[factor]["dev"], "test": data[factor]["test"]}, indent=0))
    for p in sorted(SPLITS.glob("*.json")):
        if p.name != "MANIFEST.json":
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("configs.py", "room.py", "make_splits.py")}
    manifest = {
        "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "written before any training (brief M1); never tune on a 'test' set",
        "mujoco_version": MUJOCO_VERSION, "files_sha256": files, "code_sha256": code,
        "counts": {f: {s: len(v) for s, v in d.items()} for f, d in data.items()},
        "smoke_checks": {"n": len(checks), "all_ok": True,
                         "max_tilt_deg": max(c["tilt_deg"] for c in checks),
                         "max_ball_moved_m": max(c["ball_moved_m"] for c in checks),
                         "max_button_depth_mm": max(c["button_depth_mm"] for c in checks)},
        "layout_attempts_mean": float(np.mean([c["layout_attempts"] for c in every])),
        "wall_s": round(time.time() - t0, 1),
    }
    (SPLITS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
