"""Bounded runner: validation-only tuning, then resumable matched train/eval cells.
Run the WHOLE command through ../bin/gpu_run. No nested locks or concurrent jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
METHODS = ("lp", "rnd_ep", "disagree", "ngu", "frontier", "random")


def code_hashes():
    names = ["env.py", "agents.py", "train.py", "train_r5.py", "r5_core.py", "evaluate_r5.py", "classical_r5.py", "run_r5.py"]
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def child(args, log, deadline):
    remaining = deadline - time.monotonic()
    if remaining < 8:
        return False
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        proc = subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=max(1, remaining - 2))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            print(f"PARTIAL: deadline; last atomic checkpoint retained; see {log}", flush=True)
            return False
    if code:
        raise RuntimeError(f"Runner failed ({code}); inspect {log}; no cell is silently skipped")
    return True


def train_cell(tag, method, world, seed, steps, lag, selection, deadline):
    stem = f"{method}_{world}_s{seed}"
    ck = ROOT / "checkpoints" / f"r5_{tag}" / f"{stem}.pkl"
    logs = ROOT / "logs/r5" / tag / stem
    minutes = min(44.0, (deadline - time.monotonic() - 3) / 60)
    if minutes <= .05:
        return False, ck
    if method in ("frontier", "random"):
        args = ["classical_r5.py", "--selection", selection]
    else:
        args = ["train_r5.py", "--lp-lag", lag]
        if ck.exists():
            args += ["--resume"]
    args += ["--method", method, "--world", world, "--seed", seed, "--steps", steps,
             "--max-minutes", minutes, "--checkpoint", ck, "--log-dir", logs]
    ok = child(args, logs / "console.log", deadline)
    outcome = logs / "outcome.json"
    complete = ok and outcome.exists() and json.loads(outcome.read_text())["steps"] == steps
    return complete, ck


def exploration_score(result):
    e = result["exploration"]
    return e["rooms"] / 9 + e["interactions"] / 8 - .25 * e["tv_steps"] / result["explore_steps"]


def tune(tag, deadline):
    from evaluate_r5 import evaluate, save_eval
    from r5_core import VAL_SEEDS
    out = ROOT / "results/r5" / tag; out.mkdir(parents=True, exist_ok=True)
    selection = out / "selection.json"
    if selection.exists():
        print(f"Selection already frozen: {selection}"); return True
    records = []
    # Six classical settings; tune on discovery, never on surprise-task performance.
    for mechanism in (0., 6., 12.):
        for frontier in (0., 2.):
            cfg = dict(mechanism_bonus=mechanism, frontier_bonus=frontier); scores = []
            for world in ("plain", "tv"):
                path = out / f"val_frontier_m{mechanism:g}_f{frontier:g}_{world}.json"
                if path.exists():
                    result = json.loads(path.read_text())
                else:
                    result, _ = evaluate("frontier", VAL_SEEDS, world == "tv", frontier_cfg=cfg, deadline=deadline)
                    save_eval(path, result)
                scores.append(exploration_score(result))
            records.append(dict(method="frontier", cfg=cfg, score=sum(scores) / len(scores)))
    best = max(records, key=lambda r: r["score"])["cfg"]
    for lag in (4, 16):
        scores = []
        for world in ("plain", "tv"):
            pilot_tag = f"{tag}_lag{lag}"
            ok, ck = train_cell(pilot_tag, "lp", world, 519, 131072, lag, None, deadline)
            if not ok:
                return False
            path = out / f"val_lp_lag{lag}_{world}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                result, _ = evaluate("lp", VAL_SEEDS, world == "tv", ck, deadline=deadline)
                save_eval(path, result)
            scores.append(exploration_score(result))
        records.append(dict(method="lp", cfg=dict(lp_lag=lag), score=sum(scores) / len(scores)))
    selected_lp = max((r for r in records if r["method"] == "lp"), key=lambda r: r["score"])["cfg"]
    selection.write_text(json.dumps(dict(frontier=best, **selected_lp, records=records,
                                        selection_metric="rooms/9 + interactions/8 - 0.25*TV_fraction",
                                        validation_seeds=list(VAL_SEEDS), hashes=code_hashes()), indent=2))
    print(f"FROZEN_SELECTION {selection}", flush=True)
    return True


def battery(tag, seed, steps, episodes, selection, deadline):
    from evaluate_r5 import evaluate, save_eval
    from r5_core import TEST_SEEDS, OLD_SEEDS
    selected = json.loads(Path(selection).read_text())
    out = ROOT / "results/r5" / tag; out.mkdir(parents=True, exist_ok=True)
    # Freeze the actual configuration/code used; do not mix a changed recipe into a tag.
    manifest = dict(seed=seed, steps=steps, episodes=episodes, selection=selected, hashes=code_hashes(),
                    test_seeds=list(TEST_SEEDS[:episodes]), secondary_seeds=list(OLD_SEEDS))
    manifest_path = out / f"manifest_s{seed}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Manifest changed: use a new tag (or explicitly preserve and remove stale R5 artifacts)")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for world in ("plain", "tv"):
        for method in METHODS:
            name = f"{method}_{world}_s{seed}"
            path = out / f"{name}.json"
            secondary = out / f"{name}_old.json"
            ok, ck = train_cell(tag, method, world, seed, steps, selected["lp_lag"], selection, deadline)
            if not ok:
                print(f"PARTIAL {name}: resume same command; matched table withheld", flush=True); return False
            for split, seeds in ((path, TEST_SEEDS[:episodes]), (secondary, OLD_SEEDS)):
                if split.exists():
                    continue
                result, recording = evaluate(method, seeds, world == "tv", ck if method not in ("frontier", "random") else None,
                                             selected["frontier"], deadline=deadline, record=split == path)
                result.update(training_seed=seed, charged_pretrain_steps=steps,
                              pretrain_checkpoint=str(ck), split="fresh" if split == path else "old_secondary")
                save_eval(split, result, recording)
            print(f"COMPLETE_CELL {name}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("tune", "run")); p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, choices=(510, 511, 512), default=510)
    p.add_argument("--steps", type=int, default=1048576); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--selection"); p.add_argument("--max-minutes", type=float, required=True)
    a = p.parse_args()
    if not a.tag.replace("_", "").replace("-", "").isalnum():
        p.error("tag must contain only letters, digits, underscore or hyphen")
    if not 0 < a.max_minutes <= 45 or a.steps % 4096 or not 1 <= a.episodes <= 32:
        p.error("max-minutes in (0,45], steps a multiple of 4096, episodes 1..32")
    if a.stage == "run" and not a.selection:
        p.error("--selection required; finish validation tuning before testing")
    deadline = time.monotonic() + a.max_minutes * 60
    try:
        ok = tune(a.tag, deadline) if a.stage == "tune" else battery(a.tag, a.seed, a.steps, a.episodes, a.selection, deadline)
    except TimeoutError as e:
        print(f"PARTIAL {e}", flush=True); ok = False
    status = "COMPLETE" if ok else "PARTIAL_RESUMABLE"
    print(status, flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
