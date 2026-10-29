"""Resumable round-5 tuning/evaluation. All long commands require gpu_run.

Checkpoints are atomic JSON snapshots of completed worlds, sufficient statistics,
immutable configuration, source hash, seed/counter state and elapsed compute.
There are no trainable weights or optimizer state in this exact-inference study.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from r5_core import (AIM_OFFSETS, BUDGETS, FAMILIES, METHODS, PROTOCOLS, Guard,
                     PlannerConfig, TimeLimit, evaluate_world, tune_world)

ROOT = Path(__file__).resolve().parent
SCHEMA = 1


def atomic_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Round-5 outputs must stay inside this project")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hash():
    h = hashlib.sha256()
    for name in ("physics.py", "r5_core.py", "run_r5.py"):
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def load_state(path, config, resume):
    path = Path(path)
    if path.exists():
        if not resume:
            raise ValueError(f"Refusing to replace {path}; use --resume or a fresh path")
        state = read_json(path)
        if state["schema"] != SCHEMA or state["config"] != config:
            raise ValueError("Resume config/source/tuning mismatch; preserve this run and use a new path")
        return state
    if resume:
        raise ValueError(f"Cannot resume missing checkpoint {path}")
    return dict(schema=SCHEMA, config=config, next_world=0, status="partial",
                elapsed_seconds=0.0, rows=[], tuning_totals=[],
                rng_state=dict(algorithm="PCG64 / SeedSequence", root_seed=config["seed"],
                               addressing="[seed, world, stream, step]; stream 1 world, 2 camera, 3 planner"),
                weights=None, optimizer=None,
                non_neural_note="Weights/optimizer are not applicable; exact inference plus tuned protocols.")


def choose_tuning(totals):
    selected = {}
    for method in ("lsq", "tobit"):
        candidates = [x for x in totals if x["method"] == method]
        # Predeclared tie-break: more hits, lower absolute error, then enumeration.
        best = min(candidates, key=lambda x: (-x["successes"], x["abs_error_sum"]))
        selected[method] = {k: best[k] for k in ("protocol", "offset")}
    return selected


def add_tuning(totals, fresh):
    if not totals:
        return fresh
    for old, new in zip(totals, fresh, strict=True):
        assert all(old[k] == new[k] for k in ("method", "protocol", "offset"))
        for field in ("successes", "abs_error_sum", "trials"):
            old[field] += new[field]
    return totals


def run(args):
    started = time.monotonic()
    guard = Guard(args.max_minutes)
    if args.worlds < 1:
        raise ValueError("--worlds must be positive")
    planner = PlannerConfig(args.grid_size, args.particles, args.fantasies, args.design_stride)
    planner.validate()
    config = dict(mode=args.mode, seed=args.seed, worlds=args.worlds,
                  planner=vars(planner), source_sha256=source_hash(),
                  legacy_mu=args.legacy_mu if args.mode == "evaluate" else False,
                  budgets=list(BUDGETS), families=list(FAMILIES), methods=list(METHODS),
                  output=str(Path(args.out).resolve()))
    tuning = None
    if args.mode == "evaluate":
        tuning_file = read_json(args.tuning)
        if tuning_file["status"] != "complete":
            raise ValueError("Tuning must finish before any test run")
        if args.seed == tuning_file["seed"]:
            raise ValueError("Evaluation must use a different seed from tuning")
        if tuning_file["source_sha256"] != config["source_sha256"]:
            raise ValueError("Tuning was produced by different code; use fresh tuning or restore the code")
        tuning = tuning_file["selected"]
        config["tuning_sha256"] = hashlib.sha256(Path(args.tuning).read_bytes()).hexdigest()
        config["tuning"] = tuning
        config["tuning_path"] = str(Path(args.tuning).resolve())
    else:
        config["protocols"] = list(PROTOCOLS)
        config["aim_offsets"] = list(AIM_OFFSETS)
    if Path(args.out).exists() and not args.resume:
        raise ValueError(f"Refusing to replace output {args.out}")
    state = load_state(args.checkpoint, config, args.resume)
    elapsed_before = state["elapsed_seconds"]
    start_world = state["next_world"]
    if state["status"] != "complete":
        state["status"] = "partial"
    atomic_json(args.checkpoint, state)
    stopped_by_guard = False
    try:
        while state["next_world"] < args.worlds:
            guard.check()
            i = state["next_world"]
            if args.mode == "tune":
                fresh = tune_world(args.seed, i, args.grid_size, guard)
                state["tuning_totals"] = add_tuning(state["tuning_totals"], fresh)
            else:
                fresh = evaluate_world(args.seed, i, planner, tuning, guard, args.legacy_mu)
                state["rows"].extend(fresh)
            state["next_world"] += 1
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            atomic_json(args.checkpoint, state)
            print(f"{args.mode}: {state['next_world']}/{args.worlds} worlds; "
                  f"elapsed_total={state['elapsed_seconds']:.2f}s; checkpoint={args.checkpoint}", flush=True)
            if args.stop_after_worlds and state["next_world"] - start_world >= args.stop_after_worlds:
                break
    except TimeLimit:
        stopped_by_guard = True
        print("WALL GUARD: current incomplete world discarded; checkpoint is resumable", flush=True)
    complete = state["next_world"] == args.worlds
    state["status"] = "complete" if complete else "partial"
    state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
    atomic_json(args.checkpoint, state)
    if complete:
        if args.mode == "tune":
            artifact = dict(status="complete", seed=args.seed, worlds=args.worlds,
                            source_sha256=config["source_sha256"], selected=choose_tuning(state["tuning_totals"]),
                            candidates=state["tuning_totals"], grid_size=args.grid_size,
                            selection_metric="Mean success over budgets 1,2,4; tie by placement MAE",
                            tuning_family="matched only", checkpoint=str(Path(args.checkpoint).resolve()),
                            elapsed_seconds=state["elapsed_seconds"])
        else:
            artifact = summarize(state)
            artifact["checkpoint"] = str(Path(args.checkpoint).resolve())
        atomic_json(args.out, artifact)
        print(json.dumps(dict(status="complete", output=args.out,
                              selected=artifact.get("selected"), rows=len(state["rows"]))), flush=True)
    else:
        print(json.dumps(dict(status="partial", next_world=state["next_world"],
                              checkpoint=args.checkpoint)), flush=True)
    return 75 if stopped_by_guard else 0


def wilson(successes, n):
    z = 1.959963984540054
    p, den = successes / n, 1 + z * z / n
    middle = (p + z * z / (2 * n)) / den
    radius = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float(middle - radius), float(middle + radius)]


def paired_delta(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    return dict(n=len(d), difference=mean, paired_se=se,
                a_only_successes=int(np.sum(d == 1)), b_only_successes=int(np.sum(d == -1)),
                normal_95_interval=[max(-1.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)]
                if se is not None else None)


def summarize(state):
    if state["status"] != "complete":
        raise ValueError("Refusing headline summary of a partial run")
    rows = state["rows"]
    expected = state["config"]["worlds"] * len(FAMILIES) * (3 + len(METHODS) * len(BUDGETS))
    if len(rows) != expected:
        raise ValueError(f"Incomplete record matrix: {len(rows)} != {expected}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["method"], row["budget"])].append(row)
    groups = []
    for (family, method, budget), data in sorted(grouped.items()):
        if len(data) != state["config"]["worlds"] or {r["world"] for r in data} != set(range(state["config"]["worlds"])):
            raise ValueError("Missing/duplicated world within a metric group")
        successes = sum(r["success"] for r in data)
        n = len(data)
        out = dict(family=family, method=method, budget=budget, n=n,
                   successes=successes, success_rate=successes / n, success_wilson95=wilson(successes, n),
                   mean_attempts=float(np.mean([r["attempts"] for r in data])),
                   success_per_attempt=successes / sum(r["attempts"] for r in data),
                   placement_mae_cm=100 * float(np.mean([abs(r["placement_error"]) for r in data])),
                   placement_p90_cm=100 * float(np.quantile([abs(r["placement_error"]) for r in data], .9)),
                   mu_rmse=float(np.sqrt(np.mean([r["mu_error"] ** 2 for r in data]))),
                   mu_bias=float(np.mean([r["mu_error"] for r in data])),
                   mean_selection_seconds=float(np.mean([r["selection_seconds"] for r in data])))
        if "posterior_mean" in data[0]:
            for level in (50, 90):
                covered = sum(r[f"ci{level}"][0] <= r["mu_true"] <= r[f"ci{level}"][1] for r in data)
                out[f"posterior_coverage{level}"] = covered / n
                out[f"posterior_coverage{level}_wilson95"] = wilson(covered, n)
            out.update(posterior_mean_width90=float(np.mean([r["ci90"][1] - r["ci90"][0] for r in data])),
                       posterior_mu_rmse=float(np.sqrt(np.mean([(r["posterior_mean"] - r["mu_true"]) ** 2 for r in data]))),
                       posterior_mu_bias=float(np.mean([r["posterior_mean"] - r["mu_true"] for r in data])),
                       posterior_underresolved_fraction=float(np.mean([r["grid_step_over_sd"] > .5 for r in data])),
                       max_grid_step_over_sd=max(r["grid_step_over_sd"] for r in data),
                       success_brier=float(np.mean([(r["predicted_success"] - r["success"]) ** 2 for r in data])),
                       confident_failure_joint_rate=float(np.mean([r["predicted_success"] >= .9 and not r["success"] for r in data])),
                       confidence90_fraction=float(np.mean([r["predicted_success"] >= .9 for r in data])))
        groups.append(out)
    paired = []
    for family in FAMILIES:
        for budget in BUDGETS:
            for other in ("eig", "lsq", "tobit"):
                a = sorted(grouped[(family, "task_voi", budget)], key=lambda x: x["world"])
                b = sorted(grouped[(family, other, budget)], key=lambda x: x["world"])
                paired.append(dict(family=family, budget=budget, comparison=f"task_voi - {other}",
                                   **paired_delta([r["success"] for r in a], [r["success"] for r in b])))
    degradation = []
    for method, budget in sorted({(r["method"], r["budget"]) for r in rows}):
        a = sorted(grouped[("gain_shift", method, budget)], key=lambda x: x["world"])
        b = sorted(grouped[("matched", method, budget)], key=lambda x: x["world"])
        degradation.append(dict(method=method, budget=budget, comparison="gain_shift - matched",
                                **paired_delta([r["success"] for r in a], [r["success"] for r in b])))
    return dict(status="complete", config=state["config"], elapsed_seconds=state["elapsed_seconds"],
                groups=groups, paired_success_differences=paired, success_degradation=degradation,
                caveats=["No neural component; exact grid inference and approximate myopic planning.",
                         "LSQ/Tobit coverage belongs to a diagnostic Bayesian posterior, not a fitter confidence interval.",
                         "World is the sampling unit; budgets and methods are paired, never pooled as independent trials.",
                         "Paired normal intervals are approximate; n<2 gives no interval; inspect discordant counts at small n.",
                         "Selection time includes both scores for both planners to match planning compute; timing is not a speed contest.",
                         "gain_shift is held out of tuning. mu_oracle does not know gain; world_oracle is privileged."])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("tune", "evaluate"))
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--grid-size", type=int, default=6001)
    p.add_argument("--particles", type=int, default=64)
    p.add_argument("--fantasies", type=int, default=16)
    p.add_argument("--design-stride", type=int, default=1, help="Smoke only: subset probe menu")
    p.add_argument("--max-minutes", type=float, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tuning", help="Completed validation artifact (evaluate only)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-worlds", type=int, default=0, help="Checkpoint boundary pause for scheduling/resume checks")
    p.add_argument("--legacy-mu", action="store_true", help="Secondary R1 seed-2026 friction split; new tasks/noise")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.mode == "evaluate" and not args.tuning:
        raise SystemExit("evaluate requires --tuning")
    sys.exit(run(args))
