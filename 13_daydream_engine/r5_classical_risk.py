"""Turn 3: validation-ranked soft-contact ensemble, frozen hybrid, fresh paired tasks.

NumPy throughout except the lazy frozen-hybrid loader used by `eval` on Claude's host.
This is parameter diversity within a misspecified family, not a calibrated posterior.
"""
import argparse
import itertools
from pathlib import Path
import time

import numpy as np
import sim
from experiment_r5 import model_loader
from r5_core import (Budget, BudgetExpired, SEEDS, TURN2_SEEDS, candidates, digest,
                     holm_adjust, launch_states, paired_test, physics_rollout,
                     planning_metrics, read_json, uncertainty_metrics, write_json)
from r5_planning import choose
from r5_soft_contact import soft_rollout, validate_parameters

METHODS = ("hybrid_mean", "hybrid_risk", "soft_contact", "soft_contact_5x",
           "classical_mean", "classical_risk")
FRESH_SEEDS = dict(scene=73001, action=73002, target=73101)


def top_validated(fit, count=5):
    """Rank only validated trials, breaking ties by original trial index; deduplicate."""
    if not fit["complete"] or fit["family"] != "finite_spring_damper_v1":
        raise ValueError("need a complete soft-contact fit")
    ranked = sorted(((i, r) for i, r in enumerate(fit["trials"])
                     if r.get("val_mean_error") is not None and np.isfinite(r["val_mean_error"])),
                    key=lambda ir: (ir[1]["val_mean_error"], ir[0]))
    members, seen = [], set()
    for index, row in ranked:
        validate_parameters(row["parameters"])
        key = tuple(sorted(row["parameters"].items()))
        if key in seen:
            continue
        seen.add(key)
        members.append(dict(trial_index=index, parameters=row["parameters"],
                            val_mean_error=row["val_mean_error"]))
        if len(members) == count:
            return members
    raise ValueError(f"need {count} distinct validated fits, found {len(members)}")


def soft_ensemble(init, horizon, members, budget):
    return np.stack([soft_rollout(init, horizon, m["parameters"], final=True, budget=budget)
                     for m in members])


def frozen_inputs(path):
    selection = read_json(path)
    if not selection["complete"] or selection["selected"]["members"] != 5:
        raise ValueError("need a complete frozen five-member selection")
    soft = selection["soft_contact"]
    if digest(soft["fit"]) != soft["fit_sha256"]:
        raise ValueError("frozen soft-contact fit changed")
    checkpoint = Path(selection["selected"]["checkpoint"])
    if digest(checkpoint / "model.safetensors") != selection["weights_sha256"]:
        raise ValueError("frozen final checkpoint changed")
    fit = read_json(soft["fit"])
    if fit["selected"]["parameters"] != soft["parameters"]:
        raise ValueError("frozen single-fit parameters differ")
    for key in ("train_sha256", "val_sha256"):
        if fit[key] != selection["selected"]["config"][key]:
            raise ValueError("classical/hybrid data mismatch")
    return selection, fit


def select_beta(metrics):
    # Exactly the existing hybrid tuning rule: success, mean distance, lower beta.
    return min(map(float, metrics), key=lambda b: (-metrics[str(b)]["real_success"],
                                                  metrics[str(b)]["real_distance"]["mean"], b))


def tune(a, budget):
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    frozen, fit = frozen_inputs(a.selection)
    members = top_validated(fit)
    seed, horizon = frozen["validation_seed"], fit["horizon"]
    betas = sorted(map(float, frozen["beta_rows"]))
    n = len(frozen["beta_rows"][str(betas[0])])
    if seed != 53003 or n < 1 or any(len(r) != n for r in frozen["beta_rows"].values()):
        raise ValueError("need the same seed-53003 validation tasks as frozen hybrid")
    if n == 64 and (a.shots != 256 or horizon != 40 or betas != [0., .25, .5, 1., 2.]):
        raise ValueError("turn-2 protocol is 64 tasks / 256 shots / h40 / five original betas")
    scenes = sim.random_init(n, np.random.default_rng(seed))
    targets = np.random.default_rng(seed + 1).uniform(.15, .85, (n, 2))
    rows = {str(b): [] for b in betas}
    for t, (scene, target) in enumerate(zip(scenes, targets)):
        budget.check()
        v = candidates(t, a.shots, seed + 2)
        # Replay the frozen decisions to verify scene/target/horizon identity. These
        # outcomes do not influence the classical score; they only audit the protocol.
        old = [frozen["beta_rows"][str(b)][t] for b in betas]
        for r in old:
            if r["task_id"] != t or not np.any(np.all(v == r["velocity"], axis=1)):
                raise ValueError("frozen validation task/candidate stream mismatch")
        replay = physics_rollout(launch_states(scene, np.array([r["velocity"] for r in old])),
                                 horizon, final=True, budget=budget)
        if not np.allclose(np.linalg.norm(replay[:, :2] - target, axis=1),
                           [r["real"] for r in old], rtol=0, atol=2e-6):
            raise ValueError("frozen validation scene/target/horizon replay mismatch")
        endpoints = soft_ensemble(launch_states(scene, v), horizon, members, budget)
        choices = [choose(v, endpoints, target, b) for b in betas]
        real = physics_rollout(launch_states(scene, np.array([r["velocity"] for r in choices])),
                               horizon, final=True, budget=budget)
        for b, row, state in zip(betas, choices, real):
            row.update(task_id=t, real=float(np.linalg.norm(state[:2] - target)))
            rows[str(b)].append(row)
    metrics = {b: planning_metrics(r) for b, r in rows.items()}
    beta = select_beta(metrics)
    write_json(a.out, dict(complete=True, members=members, beta=beta, beta_rows=rows,
                           beta_metrics=metrics, frozen_selection=str(Path(a.selection).resolve()),
                           frozen_selection_sha256=digest(a.selection), fit_sha256=digest(frozen["soft_contact"]["fit"]),
                           protocol=dict(tasks=n, shots=a.shots, horizon=horizon, scene_seed=seed,
                                         target_seed=seed + 1, action_seed=seed + 2, betas=betas),
                           frozen_validation_replay_matches=True, seconds=budget.elapsed,
                           interpretation="top-5 distinct validated fits; not a posterior"))
    print(f"CLASSICAL_TUNE_OK members=5 beta={beta} tasks={n} frozen_replay=True seconds={budget.elapsed:.1f}", flush=True)


def require_fresh(seed, action_seed, target_seed):
    used = {0, 1, 2, 52001, 52101, 53004, 53005} | set(SEEDS.values()) | set(TURN2_SEEDS.values())
    streams = {seed, action_seed, target_seed}
    if len(streams) != 3 or streams & used:
        raise ValueError("turn-3 scene/action/target streams must be distinct and fresh")


def task_choices(scene, target, task_id, predictor, frozen, classical, shots, action_seed, budget):
    """All choices fixed before truth is run. Shared nested shots; shared mean/risk forecasts."""
    horizon = classical["protocol"]["horizon"]
    n = max(shots)
    v = candidates(task_id, 5 * n, action_seed)
    initial = launch_states(scene, v)
    started = time.monotonic()
    single = soft_rollout(initial, horizon, frozen["soft_contact"]["parameters"], final=True, budget=budget)[None]
    single_seconds = time.monotonic() - started
    started = time.monotonic()
    ensemble = soft_ensemble(initial[:n], horizon, classical["members"], budget)
    classical_seconds = time.monotonic() - started
    started = time.monotonic()
    hybrid = predictor.rollout(initial[:n], horizon, final=True, budget=budget)
    hybrid_seconds = time.monotonic() - started
    if hybrid.shape != (5, n, 12) or ensemble.shape != (5, n, 12):
        raise ValueError("both ensembles must have five members")
    predictions = {"hybrid_mean": (hybrid, 0.), "hybrid_risk": (hybrid, frozen["beta"]),
                   "soft_contact": (single, 0.), "soft_contact_5x": (single, 0.),
                   "classical_mean": (ensemble, 0.), "classical_risk": (ensemble, classical["beta"])}
    rows = {}
    for method, (endpoints, beta) in predictions.items():
        for effort in shots:
            count = effort * (5 if method == "soft_contact_5x" else 1)
            row = choose(v[:count], endpoints[:, :count], target, beta)
            row.update(candidates=count, members=len(endpoints), member_frames=count * len(endpoints) * horizon,
                       beta=beta)
            rows[f"{method}/random/{effort}"] = row
    real = physics_rollout(launch_states(scene, np.array([r["velocity"] for r in rows.values()])),
                           horizon, final=True, budget=budget)
    for row, state in zip(rows.values(), real):
        row.update(real=float(np.linalg.norm(state[:2] - target)), real_position=state[:2].tolist())
    return dict(task_id=task_id, scene=scene.tolist(), target=target.tolist(), rows=rows,
                shared_prediction_seconds=dict(single=single_seconds, classical=classical_seconds, hybrid=hybrid_seconds))


def summarize(out, shots):
    tasks = [read_json(p) for p in sorted((out / "tasks").glob("*.json"))]
    if not tasks:
        return
    keys = [f"{m}/random/{n}" for m in METHODS for n in shots]
    if any(set(t["rows"]) != set(keys) for t in tasks):
        raise ValueError("incomplete paired task")
    raw = {k: [t["rows"][k] for t in tasks] for k in keys}
    metrics = {k: planning_metrics(rows) for k, rows in raw.items()}
    tests = {}
    for n in shots:
        for left, right in itertools.combinations(METHODS, 2):
            ka, kb = f"{left}/random/{n}", f"{right}/random/{n}"
            tests[f"{ka} vs {kb}"] = paired_test([r["real"] < sim.R for r in raw[ka]],
                                                 [r["real"] < sim.R for r in raw[kb]])
    uncertainty = {k: uncertainty_metrics(
        [np.nan if r["disagreement"] is None else r["disagreement"] for r in rows],
        [np.inf if r["predicted_position"] is None else np.linalg.norm(np.array(r["predicted_position"]) - r["real_position"])
         for r in rows]) for k, rows in raw.items() if k.startswith(("hybrid", "classical"))}
    write_json(out / "summary.json", dict(status=read_json(out / "status.json"), metrics=metrics,
        paired_tests=holm_adjust(tests), holm_family_size=len(tests), selected_uncertainty=uncertainty,
        task_ids=[t["task_id"] for t in tasks],
        primary_comparisons=[f"hybrid_risk/random/{n} vs classical_risk/random/{n}" for n in shots],
        units="success <0.06; fooled = dream <0.06 and real >=0.06; coverage 100%; 5x uses 5*label shots",
        inference="one Holm family over all same-effort method pairs; task-level paired bootstrap 95% CIs are unadjusted"))


def evaluate(a, budget):
    require_fresh(a.seed, a.action_seed, a.target_seed)
    frozen, fit = frozen_inputs(a.selection)
    classical = read_json(a.classical_selection)
    if (not classical["complete"] or classical["frozen_selection_sha256"] != digest(a.selection)
            or classical["fit_sha256"] != digest(frozen["soft_contact"]["fit"])
            or classical["members"] != top_validated(fit)):
        raise ValueError("classical selection/frozen inputs mismatch")
    out = Path(a.out)
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ("r5_classical_risk.py", "r5_core.py", "r5_planning.py", "r5_soft_contact.py", "r5_model.py", "experiment_r5.py", "sim.py")}
    spec = dict(turn=3, tasks=a.tasks, shots=a.shots, scene_seed=a.seed, action_seed=a.action_seed,
                target_seed=a.target_seed, selection_sha256=digest(a.selection),
                classical_selection_sha256=digest(a.classical_selection), code=code, chunk=a.chunk)
    manifest = out / "manifest.json"
    if manifest.exists() and read_json(manifest) != spec:
        raise ValueError("resume inputs/code changed; use a new output directory")
    if not manifest.exists():
        if out.exists() and any(out.iterdir()):
            raise ValueError("output directory is not empty and has no manifest")
        write_json(manifest, spec)
    scenes = sim.random_init(a.tasks, np.random.default_rng(a.seed))
    targets = np.random.default_rng(a.target_seed).uniform(.15, .85, (a.tasks, 2))
    predictor = None
    try:
        for t, (scene, target) in enumerate(zip(scenes, targets)):
            path = out / "tasks" / f"{t:05d}.json"
            if path.exists():
                saved = read_json(path)
                if saved["task_id"] != t or saved["scene"] != scene.tolist() or saved["target"] != target.tolist():
                    raise ValueError("saved task identity changed")
                continue
            budget.check()
            if predictor is None:
                predictor, _, _ = model_loader(frozen["selected"]["checkpoint"], a.chunk)
            started = time.monotonic()
            record = task_choices(scene, target, t, predictor, frozen, classical, a.shots, a.action_seed, budget)
            record["seconds"] = time.monotonic() - started
            write_json(path, record)
            print(f"CLASSICAL_TASK_OK id={t} seconds={record['seconds']:.2f}", flush=True)
    except BudgetExpired:
        print("CLASSICAL_WALL_GUARD: complete paired tasks retained; resume identical command", flush=True)
    done = len(list((out / "tasks").glob("*.json")))
    if done > a.tasks:
        raise ValueError("unexpected extra tasks in output directory")
    write_json(out / "status.json", dict(complete=done == a.tasks, completed_tasks=done,
                                         requested_tasks=a.tasks, invocation_seconds=budget.elapsed))
    summarize(out, a.shots)
    print(f"CLASSICAL_EVAL_DONE tasks={done}/{a.tasks} seconds={budget.elapsed:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tune")
    p.add_argument("--shots", type=int, default=256)
    p = sub.add_parser("eval")
    p.add_argument("--classical-selection", required=True)
    p.add_argument("--tasks", type=int, default=200)
    p.add_argument("--shots", type=int, nargs="+", default=[256, 4096])
    p.add_argument("--seed", type=int, default=FRESH_SEEDS["scene"])
    p.add_argument("--action-seed", type=int, default=FRESH_SEEDS["action"])
    p.add_argument("--target-seed", type=int, default=FRESH_SEEDS["target"])
    p.add_argument("--chunk", type=int, default=512)
    for p in sub.choices.values():
        p.add_argument("--selection", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--max-minutes", type=float, required=True)
    a = ap.parse_args()
    shots = a.shots if isinstance(a.shots, list) else [a.shots]
    if min(shots) < 1 or len(set(shots)) != len(shots) or getattr(a, "tasks", 1) < 1 or getattr(a, "chunk", 1) < 1:
        ap.error("counts must be positive and efforts unique")
    try:
        {"tune": tune, "eval": evaluate}[a.command](a, Budget(a.max_minutes))
    except BudgetExpired:
        raise SystemExit("PARTIAL: no completed classical selection saved; rerun within remaining budget")


if __name__ == "__main__":
    main()
