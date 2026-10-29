"""R5 preparation, strong physics fit, validation tuning and paired held-out evaluation.

Only tune/eval imports MLX, lazily. Every phase has a wall-clock guard. Test
results are resumable task records; a timeout never produces a half-paired row.
"""
import argparse
import itertools
from pathlib import Path
import time

import numpy as np
import sim
from r5_core import (Budget, BudgetExpired, SEEDS, SCALE, write_json, read_json, digest,
                     physics_rollout, parameter_candidates, sample_windows, pos_error,
                     forecast_metrics, uncertainty_metrics, planning_metrics,
                     paired_test, holm_adjust, candidates, launch_states)
from r5_planning import choose, score_endpoints, cem_search


def prepare(a, budget):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for name, count in (("train", a.scenes), ("val", a.val_scenes)):
        budget.check()
        states, flags = [], []
        rng = np.random.default_rng(SEEDS[name])
        # Small chunks keep cancellation and memory bounded.
        for start in range(0, count, 128):
            budget.check()
            s, f = sim.simulate(sim.random_init(min(128, count - start), rng), a.frames)
            states.append(s.astype(np.float32))
            flags.append(f)
        np.savez_compressed(out / f"{name}.npz", states=np.concatenate(states), flags=np.concatenate(flags), seed=SEEDS[name])
    write_json(out / "manifest.json", dict(seeds=SEEDS, train_scenes=a.scenes, val_scenes=a.val_scenes,
                                          frames=a.frames, seconds=budget.elapsed,
                                          train_sha256=digest(out / "train.npz"), val_sha256=digest(out / "val.npz")))
    print(f"PREPARE_OK seconds={budget.elapsed:.1f}", flush=True)


def fit(a, budget):
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    data = Path(a.data)
    tr, va = np.load(data / "train.npz"), np.load(data / "val.npz")
    h = min(a.horizon, tr["flags"].shape[1], va["flags"].shape[1])
    train = sample_windows(tr["states"], tr["flags"], a.samples, h, np.random.default_rng(51013))
    val = sample_windows(va["states"], va["flags"], min(a.samples, 512), h, np.random.default_rng(51012))
    proposals = parameter_candidates(n=a.candidates)
    rows, partial = [], False
    try:
        for i, p in enumerate(proposals):
            pred = physics_rollout(train[:, 0], h, p, budget=budget)
            loss = float(np.mean(((pred[:, 1:] - train[:, 1:]) / SCALE) ** 2))
            rows.append(dict(parameters=p, train_loss=loss))
        # Local refinement around best train fit, even if coarse grid already exact.
        centre = min(rows, key=lambda r: r["train_loss"])["parameters"]
        rng = np.random.default_rng(51014)
        for _ in range(a.refine):
            p = dict(restitution=float(np.clip(centre["restitution"] + rng.normal(0, .04), .5, 1.25)),
                     separation=float(np.clip(centre["separation"] + rng.normal(0, .08), .5, 1.5)))
            pred = physics_rollout(train[:, 0], h, p, budget=budget)
            rows.append(dict(parameters=p, train_loss=float(np.mean(((pred[:, 1:] - train[:, 1:]) / SCALE) ** 2))))
        for row in sorted(rows, key=lambda r: r["train_loss"])[:8]:
            pred = physics_rollout(val[:, 0], h, row["parameters"], budget=budget)
            row["val_loss"] = float(np.mean(((pred[:, 1:] - val[:, 1:]) / SCALE) ** 2))
    except BudgetExpired:
        partial = True
    evaluated = [r for r in rows if "val_loss" in r]
    chosen = min(evaluated, key=lambda r: r["val_loss"]) if evaluated else None
    write_json(a.out, dict(complete=not partial, selected=chosen, trials=rows, seconds=budget.elapsed,
                           train_sha256=digest(data / "train.npz"), val_sha256=digest(data / "val.npz"),
                           window_horizon=h, train_samples=len(train), val_samples=len(val),
                           ranges=dict(restitution=[.5, 1.25], separation=[.5, 1.5])))
    print(f"FIT_{'PARTIAL' if partial else 'OK'} trials={len(rows)} selected={chosen}", flush=True)
    if not chosen:
        raise RuntimeError("no validated fitted baseline; rerun fit to a fresh path with more budget")


def model_loader(path, chunk):
    from r5_model import load_checkpoint, Predictor, checkpoint_path
    source = checkpoint_path(path).resolve()
    model, meta = load_checkpoint(source)
    return Predictor(model, chunk), meta, source


def tune(a, budget):
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    data = Path(a.data)
    val = np.load(data / "val.npz")
    truth = val["states"][:a.tasks, :a.horizon + 1]
    if truth.shape[1] != a.horizon + 1:
        raise ValueError("validation horizon unavailable")
    fit_info = read_json(a.fit)
    if not fit_info["complete"] or fit_info["val_sha256"] != digest(data / "val.npz"):
        raise ValueError("fit incomplete or different validation data")
    models = []
    for path in a.checkpoints:
        predictor, meta, source = model_loader(path, a.chunk)
        if meta["config"]["val_sha256"] != digest(data / "val.npz"):
            raise ValueError("checkpoint validation data mismatch")
        imagined = predictor.rollout(truth[:, 0], a.horizon, budget=budget)
        error = pos_error(imagined.mean(axis=0), truth)[:, 1:]
        if not np.isfinite(error).all():
            raise FloatingPointError("nonfinite validation forecast")
        models.append(dict(checkpoint=str(source), step=meta["step"], mean_error=float(error.mean()),
                           members=predictor.model.members, config=meta["config"]))
    selected = min(models, key=lambda x: x["mean_error"])
    predictor, meta, source = model_loader(selected["checkpoint"], a.chunk)
    # Independent validation task seeds; no fresh headline task is used for tuning.
    scenes = sim.random_init(a.tasks, np.random.default_rng(SEEDS["tune"]))
    targets = np.random.default_rng(SEEDS["tune"] + 1).uniform(.15, .85, (a.tasks, 2))
    rows = {str(beta): [] for beta in a.betas}
    for t, (scene, target) in enumerate(zip(scenes, targets)):
        budget.check()
        v = candidates(t, a.shots, SEEDS["tune"] + 2)
        endpoints = predictor.rollout(launch_states(scene, v), a.horizon, final=True, budget=budget)
        choices = [choose(v, endpoints, target, beta) for beta in a.betas]
        real = physics_rollout(launch_states(scene, np.array([r["velocity"] for r in choices])),
                               a.horizon, final=True, budget=budget)
        for beta, row, outcome in zip(a.betas, choices, real):
            row["real"] = float(np.linalg.norm(outcome[:2] - target))
            row["task_id"] = t
            rows[str(beta)].append(row)
    metrics = {b: planning_metrics(r) for b, r in rows.items()}
    # Primary tuning criterion is realized validation success; tie break on distance,
    # then lower beta. No abstention/coverage tradeoff is available.
    beta = min(a.betas, key=lambda b: (-metrics[str(b)]["real_success"],
                                      metrics[str(b)]["real_distance"]["mean"], b))
    # Freeze an immutable checkpoint generation and fitted parameters, not a mutable latest pointer.
    write_json(a.out, dict(complete=True, models=models, selected=selected, beta=beta,
                           beta_metrics=metrics, beta_rows=rows, fit=str(Path(a.fit).resolve()),
                           fit_sha256=digest(a.fit), parameters=fit_info["selected"]["parameters"],
                           weights_sha256=digest(source / "model.safetensors"),
                           validation_seed=SEEDS["tune"], seconds=budget.elapsed,
                           primary_comparison="hybrid_risk vs hybrid_mean at random 256 shots"))
    print(f"TUNE_OK checkpoint={source} beta={beta} seconds={budget.elapsed:.1f}", flush=True)


def task_predictions(scene, target, task_id, predictor, params, a, beta, budget):
    """Return all method/effort choices; score reality only after every choice is fixed."""
    e = predictor.model.members
    methods = ("walls", "fitted", "hybrid_mean", "hybrid_risk")
    max_shots = max(a.shots)
    rows = {}

    def predict(method, velocity):
        x = launch_states(scene, velocity)
        if method.startswith("hybrid"):
            return predictor.rollout(x, a.horizon, final=True, budget=budget)
        return physics_rollout(x, a.horizon, params, walls=method == "walls", final=True, budget=budget)[None]

    # One shared ensemble forecast for random mean/risk. Baselines get an extra
    # member-frame-matched arm: E times as many random shots (separately labelled).
    learned_endpoints = None
    for method in methods:
        n = max_shots * (e if a.matched_cost and not method.startswith("hybrid") else 1)
        velocity = candidates(task_id, n, a.action_seed)
        started = time.monotonic()
        if method == "hybrid_risk":
            endpoints = learned_endpoints
        else:
            endpoints = predict(method, velocity)
            if method == "hybrid_mean":
                learned_endpoints = endpoints
        seconds = time.monotonic() - started
        penalty = beta if method == "hybrid_risk" else 0.0
        for count in a.shots:
            row = choose(velocity[:count], endpoints[:, :count], target, penalty)
            row.update(method=method, search="random", effort=count, candidates=count,
                       member_frames=count * a.horizon * (e if method.startswith("hybrid") else 1),
                       prediction_seconds_shared=seconds)
            rows[f"{method}/random/{count}"] = row
            if a.matched_cost and not method.startswith("hybrid"):
                ncost = e * count
                row = choose(velocity[:ncost], endpoints[:, :ncost], target)
                row.update(method=method, search="random_member_matched", effort=count, candidates=ncost,
                           member_frames=ncost * a.horizon, prediction_seconds_shared=seconds)
                rows[f"{method}/random_member_matched/{count}"] = row
        started = time.monotonic()
        cem = cem_search(lambda v: predict(method, v), target, task_id, a.cem_iters,
                         a.population, penalty, a.action_seed)
        for iterations, row in cem.items():
            count = int(iterations) * a.population
            row.update(method=method, search="cem", effort=int(iterations), candidates=count,
                       member_frames=count * a.horizon * (e if method.startswith("hybrid") else 1),
                       prediction_seconds_shared=time.monotonic() - started)
            rows[f"{method}/cem/{iterations}"] = row
    real = physics_rollout(launch_states(scene, np.array([r["velocity"] for r in rows.values()])),
                           a.horizon, final=True, budget=budget)
    for row, state in zip(rows.values(), real):
        row["real"] = float(np.linalg.norm(state[:2] - target))
        row["real_position"] = state[:2].tolist()
        row["task_id"] = task_id
    return rows


def evaluate(a, budget):
    selection = read_json(a.selection)
    source = Path(selection["selected"]["checkpoint"])
    if digest(source / "model.safetensors") != selection["weights_sha256"]:
        raise ValueError("selected checkpoint changed")
    predictor, meta, _ = model_loader(source, a.chunk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # A resume is permitted only for an identical experiment and identical code.
    code = {p.name: digest(p) for p in Path(__file__).parent.glob("*r5*.py")}
    spec = dict(selection_sha256=digest(a.selection), seed=a.seed, action_seed=a.action_seed,
                tasks=a.tasks, forecast_scenes=a.forecast_scenes, horizon=a.horizon,
                shots=a.shots, cem_iters=a.cem_iters, population=a.population,
                matched_cost=a.matched_cost, code=code, label=a.label,
                legacy_sha256=digest(a.legacy_data) if a.legacy_data else None)
    manifest = out / "manifest.json"
    if manifest.exists():
        if read_json(manifest) != spec:
            raise ValueError("evaluation resume specification/code changed; use a fresh output directory")
    else:
        write_json(manifest, spec)
    scene_rng = np.random.default_rng(a.seed)
    legacy = np.load(a.legacy_data)["states"] if a.legacy_data else None
    if legacy is not None and len(legacy) < max(a.tasks, a.forecast_scenes):
        raise ValueError("legacy split has too few scenes")
    scenes = sim.random_init(a.tasks, scene_rng) if legacy is None else legacy[:a.tasks, 0].astype(float)
    targets = np.random.default_rng(a.seed + 100).uniform(.15, .85, (a.tasks, 2))
    params, beta = selection["parameters"], selection["beta"]
    try:
        forecast_path = out / "forecast.json"
        if not forecast_path.exists():
            # One window per independent scene; no overlapping-window pseudoreplication.
            init = (sim.random_init(a.forecast_scenes, np.random.default_rng(a.seed + 200))
                    if legacy is None else legacy[:a.forecast_scenes, 0].astype(float))
            budget.check()
            truth, flags = sim.simulate(init, a.horizon)
            learned = predictor.rollout(init, a.horizon, budget=budget)
            preds = dict(hybrid_mean=learned.mean(axis=0),
                         walls=physics_rollout(init, a.horizon, walls=True, budget=budget),
                         fitted=physics_rollout(init, a.horizon, params, budget=budget))
            fm = {k: forecast_metrics(p, truth, flags) for k, p in preds.items()}
            # Independent, unselected trajectories: disagreement vs realized forecast error.
            spread = np.sqrt(((learned.reshape(predictor.model.members, len(init), a.horizon + 1, 3, 4)[..., :2]
                              - preds["hybrid_mean"].reshape(len(init), a.horizon + 1, 3, 4)[None, ..., :2]) ** 2)
                             .sum(axis=-1).mean(axis=(0, 3)))
            err = pos_error(preds["hybrid_mean"], truth)
            um = {str(h): uncertainty_metrics(spread[:, h], err[:, h])
                  for h in range(1, a.horizon + 1) if str(h) in fm["hybrid_mean"]}
            # Preserve raw numbers and labels for independent auditing/calibration plots.
            np.savez_compressed(out / "forecast_raw.npz", truth=truth, flags=flags,
                                hybrid=preds["hybrid_mean"], walls=preds["walls"], fitted=preds["fitted"],
                                disagreement=spread, error=err)
            write_json(forecast_path, dict(metrics=fm, uncertainty=um, n_scenes=len(init), seed=a.seed + 200))
        for task_id, (scene, target) in enumerate(zip(scenes, targets)):
            path = out / "tasks" / f"{task_id:05d}.json"
            if path.exists():
                continue
            budget.check()
            started = time.monotonic()
            rows = task_predictions(scene, target, task_id, predictor, params, a, beta, budget)
            write_json(path, dict(task_id=task_id, scene=scene.tolist(), target=target.tolist(), rows=rows,
                                  seconds=time.monotonic() - started))
            print(f"TASK_OK id={task_id} seconds={time.monotonic() - started:.2f}", flush=True)
    except BudgetExpired:
        print("EVAL_WALL_GUARD: complete tasks retained; rerun identical command to resume", flush=True)
    complete = len(list((out / "tasks").glob("*.json"))) if (out / "tasks").exists() else 0
    write_json(out / "status.json", dict(completed_tasks=complete, requested_tasks=a.tasks,
                                        complete=complete == a.tasks and (out / "forecast.json").exists(),
                                        invocation_seconds=budget.elapsed))
    summarize(out)
    print(f"EVAL_DONE tasks={complete}/{a.tasks} seconds={budget.elapsed:.1f}", flush=True)


def summarize(out):
    out = Path(out)
    tasks = [read_json(p) for p in sorted((out / "tasks").glob("*.json"))]
    if not tasks:
        return
    keys = list(tasks[0]["rows"])
    if any(set(t["rows"]) != set(keys) for t in tasks):
        raise ValueError("incomplete paired task")
    metrics = {k: planning_metrics([t["rows"][k] for t in tasks]) for k in keys}
    tests = {}
    groups = sorted(set(k.split("/", 1)[1] for k in keys))
    for group in groups:
        methods = [k for k in keys if k.split("/", 1)[1] == group]
        for left, right in itertools.combinations(methods, 2):
            tests[f"{left} vs {right}"] = paired_test(
                [t["rows"][left]["real"] < sim.R for t in tasks],
                [t["rows"][right]["real"] < sim.R for t in tasks])
    # Additional paired comparisons at equal member-frame budget.
    for k in keys:
        method, search, effort = k.split("/")
        if search == "random_member_matched":
            for learned in ("hybrid_mean", "hybrid_risk"):
                other = f"{learned}/random/{effort}"
                tests[f"{other} vs {k}"] = paired_test(
                    [t["rows"][other]["real"] < sim.R for t in tasks],
                    [t["rows"][k]["real"] < sim.R for t in tasks])
    raw = {k: [t["rows"][k] for t in tasks] for k in keys}
    selected_uncertainty = {k: uncertainty_metrics(
        [np.nan if r["disagreement"] is None else r["disagreement"] for r in rows],
        [np.inf if r["predicted_position"] is None else np.linalg.norm(np.array(r["predicted_position"]) - r["real_position"]) for r in rows])
        for k, rows in raw.items() if k.startswith("hybrid")}
    write_json(out / "summary.json", dict(status=read_json(out / "status.json"), metrics=metrics,
                                          paired_tests=holm_adjust(tests), selected_uncertainty=selected_uncertainty,
                                          task_ids=[t["task_id"] for t in tasks],
                                          units="box side 1; success distance < 0.06; fooled: dream <0.06 and real >=0.06",
                                          primary_comparison="hybrid_risk vs hybrid_mean, random/256; all tests Holm adjusted"))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", type=int, default=2000)
    p.add_argument("--val-scenes", type=int, default=256)
    p.add_argument("--frames", type=int, default=60)
    p = sub.add_parser("fit")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidates", type=int, default=80)
    p.add_argument("--refine", type=int, default=16)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=3)
    p = sub.add_parser("tune")
    p.add_argument("--data", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", type=int, default=24)
    p.add_argument("--shots", type=int, default=256)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--betas", type=float, nargs="+", default=[0., .25, .5, 1., 2.])
    p.add_argument("--chunk", type=int, default=512)
    p = sub.add_parser("eval")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", type=int, default=64)
    p.add_argument("--forecast-scenes", type=int, default=256)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--shots", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--cem-iters", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--matched-cost", action="store_true")
    p.add_argument("--seed", type=int, default=SEEDS["test"])
    p.add_argument("--action-seed", type=int, default=SEEDS["actions"])
    p.add_argument("--label", default="fresh_headline")
    p.add_argument("--legacy-data", default="", help="old data/test.npz; secondary scene split only")
    p.add_argument("--chunk", type=int, default=512)
    for p in sub.choices.values():
        p.add_argument("--max-minutes", type=float, required=True)
    return ap


def main():
    ap = parser()
    a = ap.parse_args()
    for key in ("tasks", "shots", "cem_iters", "population", "chunk", "horizon", "forecast_scenes",
                "scenes", "val_scenes", "frames", "samples", "candidates"):
        value = getattr(a, key, None)
        if value is not None and min(value if isinstance(value, list) else [value]) < 1:
            ap.error(f"{key} must be positive")
    if hasattr(a, "betas") and min(a.betas) < 0:
        ap.error("betas must be nonnegative")
    budget = Budget(a.max_minutes)
    try:
        {"prepare": prepare, "fit": fit, "tune": tune, "eval": evaluate}[a.command](a, budget)
    except BudgetExpired:
        raise SystemExit("PARTIAL: wall guard reached; no completed selection saved; rerun to a fresh output path")


if __name__ == "__main__":
    main()
