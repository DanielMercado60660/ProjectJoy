"""One follow-up: joint friction/gain inference on unchanged probe histories.

Tune prior width and probe owner on NEW validation worlds, then freeze for NEW
test worlds. Original baselines remain frozen; nuisance MLE receives the same
four validation choices as nuisance Bayes. No trained neural component.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from r5_core import BUDGETS, FAMILIES, Guard, PlannerConfig, TimeLimit, evaluate_world
from r5_gain import (GAIN_HALF_WIDTHS, NEW_METHODS, OWNERS, GainConfig,
                     corrected_history, followup_world)
from run_r5 import (ROOT, atomic_json, load_state, paired_delta, read_json,
                    source_hash as original_source_hash, summarize, wilson)

FORBIDDEN_SEEDS = {6202611, 6202612, 6202613}


def source_hash():
    h = hashlib.sha256(original_source_hash().encode())
    for name in ("r5_gain.py", "run_r5_turn2.py"):
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def validation_candidates(original_rows, cfg, guard):
    results = []
    for owner in OWNERS:
        for width in GAIN_HALF_WIDTHS:
            rows = []
            for family in FAMILIES:
                rows.extend(corrected_history([r for r in original_rows if r["family"] == family], owner, width, cfg, guard=guard))
            for method in ("gain_bayes", "gain_mle"):
                selected = [r for r in rows if r["method"] == method and r["budget"] > 0]
                results.append(dict(method=method, owner=owner, half_width=width,
                                    successes=sum(r["success"] for r in selected), trials=len(selected),
                                    abs_error_sum=sum(abs(r["placement_error"]) for r in selected),
                                    brier_sum=sum((r["predicted_success"] - r["success"]) ** 2 for r in selected)))
    return results


def add_candidates(total, new):
    if not total:
        return new
    for a, b in zip(total, new, strict=True):
        assert all(a[k] == b[k] for k in ("method", "owner", "half_width"))
        for key in ("successes", "trials", "abs_error_sum", "brier_sum"):
            a[key] += b[key]
    return total


def select_candidates(candidates):
    selected = {}
    for method in ("gain_bayes", "gain_mle"):
        choices = [c for c in candidates if c["method"] == method]
        best = min(choices, key=lambda c: (-c["successes"] / c["trials"], c["abs_error_sum"] / c["trials"]))
        selected[method] = {k: best[k] for k in ("owner", "half_width")}
    return selected


def new_group(data):
    row, n = data[0], len(data)
    hits = sum(r["success"] for r in data)
    mean = lambda key: float(np.mean([r[key] for r in data]))
    out = dict(family=row["family"], method=row["method"], budget=row["budget"], n=n,
               successes=hits, success_rate=hits / n, success_wilson95=wilson(hits, n),
               mean_attempts=mean("attempts"), success_per_attempt=hits / sum(r["attempts"] for r in data),
               placement_mae_cm=100 * float(np.mean([abs(r["placement_error"]) for r in data])),
               placement_p90_cm=100 * float(np.quantile([abs(r["placement_error"]) for r in data], .9)),
               mu_rmse=float(np.sqrt(np.mean([r["mu_error"] ** 2 for r in data]))), mu_bias=mean("mu_error"),
               posterior_mean_width90=float(np.mean([r["ci90"][1] - r["ci90"][0] for r in data])),
               posterior_underresolved_fraction=float(np.mean([r["grid_step_over_sd"] > .5 for r in data])),
               max_grid_step_over_sd=max(r["grid_step_over_sd"] for r in data),
               success_brier=float(np.mean([(r["predicted_success"] - r["success"]) ** 2 for r in data])),
               confidence90_fraction=float(np.mean([r["predicted_success"] >= .9 for r in data])),
               confident_failure_joint_rate=float(np.mean([r["predicted_success"] >= .9 and not r["success"] for r in data])),
               mean_selection_seconds=mean("selection_seconds"))
    for level in (50, 90):
        covered = sum(r[f"ci{level}"][0] <= r["mu_true"] <= r[f"ci{level}"][1] for r in data)
        out[f"posterior_coverage{level}"] = covered / n
        out[f"posterior_coverage{level}_wilson95"] = wilson(covered, n)
    if "gain_ci90" in row:
        out.update(gain_rmse=float(np.sqrt(np.mean([r["gain_error"] ** 2 for r in data]))),
                   gain_mean_width90=float(np.mean([r["gain_ci90"][1] - r["gain_ci90"][0] for r in data])),
                   gain_underresolved_fraction=float(np.mean([r["gain_grid_step_over_sd"] > .5 for r in data])),
                   mean_correction_seconds=mean("correction_seconds"),
                   mean_mu_conditional_underresolved_mass=mean("mu_conditional_underresolved_mass"),
                   mean_gain_conditional_underresolved_mass=mean("gain_conditional_underresolved_mass"),
                   max_joint_cell_mass=max(r["joint_max_cell_mass"] for r in data))
        for level in (50, 90):
            out[f"gain_coverage{level}"] = float(np.mean([r[f"gain_ci{level}"][0] <= r["gain_true"] <= r[f"gain_ci{level}"][1] for r in data]))
    return out


def summarize_followup(state):
    if state["status"] != "complete":
        raise ValueError("Partial follow-up cannot produce a headline summary")
    expected = state["config"]["worlds"] * 62
    if len(state["rows"]) != expected:
        raise ValueError(f"Incomplete follow-up matrix: {len(state['rows'])} != {expected}")
    original = dict(state, rows=[r for r in state["rows"] if r["method"] not in NEW_METHODS])
    out = summarize(original)
    grouped = defaultdict(list)
    for row in state["rows"]:
        grouped[row["family"], row["method"], row["budget"]].append(row)
    for key, rows in sorted(grouped.items()):
        if key[1] not in NEW_METHODS:
            continue
        if len(rows) != state["config"]["worlds"] or {r["world"] for r in rows} != set(range(state["config"]["worlds"])):
            raise ValueError("Missing or duplicated follow-up world")
        out["groups"].append(new_group(rows))
    comparisons = []
    for family in FAMILIES:
        for budget in BUDGETS:
            a = sorted(grouped[family, "gain_bayes", budget], key=lambda r: r["world"])
            for other in ("same_probes_unit_bayes", "gain_mle", "eig", "task_voi", "lsq", "tobit"):
                b = sorted(grouped[family, other, budget], key=lambda r: r["world"])
                comparisons.append(dict(family=family, budget=budget, comparison=f"gain_bayes - {other}",
                                        **paired_delta([r["success"] for r in a], [r["success"] for r in b])))
    out["gain_fix_paired_differences"] = comparisons
    out["caveats"].extend([
        "Turn 2 adds a physical gain assumption learned from the earlier failure; gain_shift is now within model support.",
        "This tests fresh worlds, not a structurally unseen mismatch family. The uniform gain prior differs from the test family's mixture.",
        "Original LSQ/Tobit protocols remain frozen; gain-MLE and gain-Bayes each received all four NEW validation candidates.",
        "same_probes_unit_bayes follows gain_bayes's selected probe owner, isolating the model change on identical observations.",
        "Gain MLE intervals are a diagnostic joint Bayesian posterior, not MLE confidence intervals.",
        "Conditional grid-resolution diagnostics can expose thin ridges hidden by well-resolved marginals."])
    return out


def run(args):
    guard = Guard(args.max_minutes)
    if args.seed in FORBIDDEN_SEEDS:
        raise ValueError("Turn 2 requires FRESH seeds, not any original R5 train/test seed")
    if args.worlds < 1:
        raise ValueError("Positive --worlds required")
    base_cfg = PlannerConfig(args.base_grid_size, args.base_particles, args.base_fantasies, args.base_stride)
    gain_cfg = GainConfig(args.mu_size, args.gain_size)
    base_cfg.validate()
    gain_cfg.validate()
    if args.resolution_check and args.mode != "evaluate":
        raise ValueError("--resolution-check is evaluation-only; freeze validation selections")
    baseline = read_json(args.baseline_tuning)
    if baseline["status"] != "complete" or baseline["source_sha256"] != original_source_hash():
        raise ValueError("Frozen original baseline tuning/source must match")
    config = dict(mode="turn2_" + args.mode, seed=args.seed, worlds=args.worlds, source_sha256=source_hash(),
                  original_source_sha256=original_source_hash(), planner=vars(base_cfg), gain_grid=vars(gain_cfg),
                  baseline_tuning=baseline["selected"],
                  baseline_tuning_sha256=hashlib.sha256(Path(args.baseline_tuning).read_bytes()).hexdigest(),
                  output=str(Path(args.out).resolve()), owners=list(OWNERS), half_widths=list(GAIN_HALF_WIDTHS),
                  resolution_check=args.resolution_check)
    selected = None
    if args.mode == "evaluate":
        tune = read_json(args.tuning)
        if tune["status"] != "complete" or tune["source_sha256"] != source_hash():
            raise ValueError("Complete same-source turn-2 tuning is required")
        if args.seed in (tune["seed"], baseline["seed"]):
            raise ValueError("Test seeds must differ from validation seeds")
        if tune["config"]["baseline_tuning_sha256"] != config["baseline_tuning_sha256"]:
            raise ValueError("Frozen baselines changed since validation")
        if tune["config"]["planner"] != config["planner"]:
            raise ValueError("Original planning settings must remain identical to validation")
        if tune["config"]["gain_grid"] != config["gain_grid"] and not args.resolution_check:
            raise ValueError("Changed gain grid requires explicit --resolution-check")
        if args.resolution_check and any(config["gain_grid"][k] < tune["config"]["gain_grid"][k] for k in ("mu_size", "gain_size")):
            raise ValueError("Resolution checks may only increase gain-model grid sizes")
        selected = tune["selected"]
        config["selected"] = selected
        config["tuning_sha256"] = hashlib.sha256(Path(args.tuning).read_bytes()).hexdigest()
    if Path(args.out).exists() and not args.resume:
        raise ValueError("Refusing to replace an existing follow-up output")
    state = load_state(args.checkpoint, config, args.resume)
    before, initial = state["elapsed_seconds"], state["next_world"]
    atomic_json(args.checkpoint, state)
    limited = False
    try:
        while state["next_world"] < args.worlds:
            guard.check()
            original_rows = evaluate_world(args.seed, state["next_world"], base_cfg, baseline["selected"], guard)
            if args.mode == "tune":
                candidates = validation_candidates(original_rows, gain_cfg, guard)
                state["tuning_totals"] = add_candidates(state["tuning_totals"], candidates)
            else:
                records = followup_world(original_rows, selected, gain_cfg, base_cfg.grid_size, guard)
                state["rows"].extend(records)
            state["next_world"] += 1
            state["elapsed_seconds"] = before + time.monotonic() - guard.started
            atomic_json(args.checkpoint, state)
            print(f"turn2 {args.mode}: {state['next_world']}/{args.worlds}; total={state['elapsed_seconds']:.2f}s", flush=True)
            if args.stop_after_worlds and state["next_world"] - initial >= args.stop_after_worlds:
                break
    except TimeLimit:
        limited = True
        print("WALL GUARD: incomplete world discarded, checkpoint resumable", flush=True)
    state["status"] = "complete" if state["next_world"] == args.worlds else "partial"
    state["elapsed_seconds"] = before + time.monotonic() - guard.started
    atomic_json(args.checkpoint, state)
    if state["status"] == "complete":
        if args.mode == "tune":
            artifact = dict(status="complete", source_sha256=source_hash(), config=config, seed=args.seed,
                            selected=select_candidates(state["tuning_totals"]), candidates=state["tuning_totals"],
                            selection_metric="Balanced matched/gain_shift success over budgets 1,2,4; ties by MAE then enumeration",
                            elapsed_seconds=state["elapsed_seconds"])
        else:
            artifact = summarize_followup(state)
        artifact["checkpoint"] = str(Path(args.checkpoint).resolve())
        atomic_json(args.out, artifact)
        print(json.dumps(dict(status="complete", out=args.out, rows=len(state["rows"]), selected=artifact.get("selected"))), flush=True)
    else:
        print(json.dumps(dict(status="partial", next_world=state["next_world"])), flush=True)
    return 75 if limited else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("tune", "evaluate"))
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--baseline-tuning", default="results/r5/tuning_main.json")
    p.add_argument("--tuning", help="Turn-2 validation artifact (evaluate only)")
    p.add_argument("--mu-size", type=int, default=2401)
    p.add_argument("--gain-size", type=int, default=641)
    p.add_argument("--base-grid-size", type=int, default=6001)
    p.add_argument("--base-particles", type=int, default=64)
    p.add_argument("--base-fantasies", type=int, default=16)
    p.add_argument("--base-stride", type=int, default=1)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-minutes", type=float, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-worlds", type=int, default=0)
    p.add_argument("--resolution-check", action="store_true", help="Explicit gain-grid-only refinement with frozen validation selection")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.mode == "evaluate" and not args.tuning:
        raise SystemExit("evaluate requires --tuning")
    sys.exit(run(args))
