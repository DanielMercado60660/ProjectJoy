"""Read-only NumPy analysis of the completed R5 receipts; no engine imports.

Outputs are new derived artifacts, never replacements for run receipts. Frame
percentiles are pooled; outcome means weight episodes equally. Repeated engine
conditions are not independent seeds. Prediction uses leave-one-seed-out OLS.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

RUNS = {"nominal": "nominal_plan.json", "stress": "stress_plan.json", "old_secondary": "old_plan.json"}
STAGES = ("preprocess_ms", "queue_ms", "dispatch_ms", "compute_ms", "postprocess_ms",
          "injected_ms", "pickup_ms", "overhead_ms")
COUNTS = ("scheduled", "captured", "applied", "on_time", "deadline_misses", "dropped",
          "pending_at_end", "inflight_at_end", "sensor_ticks_lost")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, min=None, max=None, p50=None, p95=None, p99=None)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                **{f"p{p}": float(np.percentile(a, p)) for p in (50, 95, 99)})


def aggregate(es):
    actions = [a for e in es for a in e["actions"]]
    active = [a for a in actions if a["task_active"]]
    counts = {k: sum(e["metrics"][k] for e in es) for k in COUNTS}
    task = {k: stats([e["task"][k] for e in es if e["task"][k] is not None])
            for k in es[0]["task"] if k != "horizon_s"}
    return dict(episodes=len(es), seeds=sorted({e["job"]["seed"] for e in es}),
                receipts=[e["file"] for e in es], counts=counts, task=task,
                coverage=counts["applied"]/counts["scheduled"],
                miss_fraction=counts["deadline_misses"]/counts["scheduled"],
                age_ms=stats([a["age_ms"] for a in actions]),
                active_age_ms=stats([a["age_ms"] for a in active]),
                stages_ms={k: stats([a[k] for a in actions]) for k in STAGES},
                active_stages_ms={k: stats([a[k] for a in active]) for k in STAGES})


def audit_run(root, name, plan_name):
    plan_path = root / plan_name
    plan, status = read(plan_path), read(root / name / "run_status.json")
    assert status["complete"] and status["done"] == status["planned"] == len(plan["jobs"])
    paths = sorted((root / name).glob("episode_*.json"))
    assert len(paths) == len(plan["jobs"])
    hashes = {str(plan_path): sha(plan_path), str(root / name / "run_status.json"): sha(root / name / "run_status.json")}
    for path, expected in plan["sources"].items():
        assert sha(path) == expected, f"Runtime source drift: {path}"
    assert sha("checkpoints/r5_selected/manifest.json") == plan["checkpoint"]
    es, max_residual = [], 0.0
    for i, path in enumerate(paths):
        e = read(path)
        assert path.name == f"episode_{i:05d}.json" and e["job"] == plan["jobs"][i]
        assert e["plan_sha256"] == hashes[str(plan_path)]
        m, actions = e["metrics"], e["actions"]
        assert len(actions) == m["applied"]
        assert m["captured"] == sum(m[k] for k in ("applied", "dropped", "pending_at_end", "inflight_at_end"))
        assert m["scheduled"] == m["captured"]+m["sensor_ticks_lost"]
        assert m["deadline_misses"] == m["scheduled"]-sum(not a["deadline_missed"] for a in actions)
        assert (e["background_matmuls"] > 0) == (e["job"]["load"] == "loaded")
        for a in actions:
            residual = abs(sum(a[k] for k in STAGES)-a["age_ms"])
            max_residual = max(max_residual, residual)
            assert residual < 1e-6 and a["applied_s"] >= a["captured_s"]
        for k in ("p50", "p95", "p99"):
            assert abs(stats([a["age_ms"] for a in actions])[k]-m["age_ms"][k]) < 1e-8
        e["file"] = str(path)
        es.append(e)
        hashes[str(path)] = sha(path)
    envs = [e[k] for e in es for k in ("env_before", "env_after")]
    environment = dict(first=envs[0]["time"], last=envs[-1]["time"],
                       load1=stats([v["loadavg"][0] for v in envs]),
                       power=sorted({v["power"]["stdout"] for v in envs}),
                       thermal=sorted({v["thermal"]["stdout"] for v in envs}),
                       uptime_first=envs[0]["uptime"]["stdout"], uptime_last=envs[-1]["uptime"]["stdout"],
                       thread_configs=list({json.dumps(v["threads"], sort_keys=True): v["threads"] for v in envs}.values()),
                       versions=envs[0]["versions"],
                       loaded_matmuls=stats([e["background_matmuls"] for e in es if e["job"]["load"] == "loaded"]))
    return es, dict(episodes=len(es), distinct_seeds=sorted({e["job"]["seed"] for e in es}),
                    predicted_s=plan["predicted_s"], actual_s=status["wall_this_invocation_s"],
                    environment=environment, max_stage_sum_residual_ms=max_residual,
                    sources=hashes, runtime_hashes=plan["sources"], omitted=plan["omitted"])


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def demean(x, groups):
    x, groups = np.asarray(x, float), np.asarray(groups)
    return np.array([v-x[groups == g].mean() for v, g in zip(x, groups)])


def relationship(x, y, seeds):
    x, y, seeds = np.asarray(x, float), np.asarray(y, float), np.asarray(seeds)
    xc, yc = demean(x, seeds), demean(y, seeds)
    within_r = corr(xc, yc)
    result = dict(episodes=len(x), independent_seeds=len(set(seeds)), raw_pearson=corr(x, y),
                  seed_demeaned_pearson=within_r, within_seed_r2=None if within_r is None else within_r**2,
                  within_seed_slope_per_ms=float(xc@yc/(xc@xc)) if xc@xc > 1e-12 else None)
    if len(set(seeds)) >= 2 and np.var(y) > 1e-12:
        pred, base = np.zeros(len(y)), np.zeros(len(y))
        for seed in set(seeds):
            train = seeds != seed
            test = ~train
            design = np.column_stack([np.ones(train.sum()), x[train]])
            b = np.linalg.lstsq(design, y[train], rcond=None)[0]
            pred[test] = b[0]+b[1]*x[test]
            base[test] = y[train].mean()
        result["leave_one_seed_out"] = dict(
            r2=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()),
            rmse=float(np.sqrt(np.mean((y-pred)**2))),
            training_mean_baseline_rmse=float(np.sqrt(np.mean((y-base)**2))))
    return result


def prediction_table(es, outcome, active=False):
    features = {k: [] for k in ("observation_age_ms", "service_without_queue_ms", "compute_ms")}
    for e in es:
        aa = [a for a in e["actions"] if not active or a["task_active"]]
        features["observation_age_ms"].append(np.mean([a["age_ms"] for a in aa]))
        features["service_without_queue_ms"].append(np.mean([a["age_ms"]-a["queue_ms"] for a in aa]))
        features["compute_ms"].append(np.mean([a["compute_ms"] for a in aa]))
    return {k: relationship(x, [e["task"][outcome] for e in es], [e["job"]["seed"] for e in es])
            for k, x in features.items()}


def paired(es, axis, a, b, outcome):
    """b - a, exact matching on every other job field; summarize by seed first."""
    def key(e):
        return tuple(sorted((k, str(v)) for k, v in e["job"].items() if k != axis))
    left = {key(e): e for e in es if e["job"][axis] == a}
    right = {key(e): e for e in es if e["job"][axis] == b}
    assert left.keys() == right.keys()
    seed_deltas, pairs = defaultdict(list), []
    for key_, e in left.items():
        other = right[key_]
        delta = float(outcome(other)-outcome(e))
        seed_deltas[e["job"]["seed"]].append(delta)
        pairs.append(dict(seed=e["job"]["seed"], a=e["file"], b=other["file"], delta=delta))
    means = {str(s): float(np.mean(v)) for s, v in sorted(seed_deltas.items())}
    return dict(direction=f"{b} minus {a}", pairs=pairs, deltas=stats([p["delta"] for p in pairs]),
                negative_pairs=sum(p["delta"] < 0 for p in pairs),
                seed_mean_deltas=means, seed_mean_range=stats(list(means.values())))


def analyze(root):
    result = dict(method=dict(percentiles="Pooled applied actions, not averages of episode percentiles",
                             outcome="Equal episode means; per-seed pairing retains repeated-condition dependence",
                             stages="Host timing; stage MEANS add to mean age, percentiles do not add",
                             regression="Post-hoc one-predictor OLS with intercept; leave one entire seed out. "
                                        "R2 denominator is pooled outcome variance; negative R2 allowed. "
                                        "Seed-demeaned correlations are descriptive, not independent-frame inference."),
                  analysis_source=dict(path=__file__, sha256=sha(__file__)), audits={}, cells={},
                  engine_rollup={}, policy_rollup={}, load_effects={}, prediction={}, paired_policies={})
    all_es = {}
    for run, plan in RUNS.items():
        es, result["audits"][run] = audit_run(root, run, plan)
        all_es[run] = es
        cells = defaultdict(list)
        for e in es:
            j = e["job"]
            key = "/".join(map(str, (run, j["event"], j["engine"], j["policy"], j["load"], j["delay_ms"])))
            cells[key].append(e)
        result["cells"].update({k: aggregate(v) for k, v in sorted(cells.items())})
        for event in sorted({e["job"]["event"] for e in es}):
            for engine in sorted({e["job"]["engine"] for e in es}):
                aa = [e for e in es if e["job"]["event"] == event and e["job"]["engine"] == engine
                      and e["job"]["delay_ms"] == 0]
                key = f"{run}/{event}/{engine}"
                result["engine_rollup"][key] = aggregate(aa)
                result["load_effects"][key] = paired(
                    aa, "load", "ambient", "loaded",
                    lambda e: np.mean([a["age_ms"] for a in e["actions"] if a["task_active"]]))
                for load in ("ambient", "loaded"):
                    result["engine_rollup"][f"{key}/{load}"] = aggregate([e for e in aa if e["job"]["load"] == load])
        for delay in sorted({e["job"]["delay_ms"] for e in es}):
            tr = [e for e in es if e["job"]["event"] == "tracker" and e["job"]["delay_ms"] == delay]
            for pol in ("queue", "latest"):
                result["policy_rollup"][f"{run}/tracker/{delay}/{pol}"] = aggregate([e for e in tr if e["job"]["policy"] == pol])
            result["paired_policies"][f"{run}/tracker/{delay}"] = paired(
                tr, "policy", "queue", "latest", lambda e: e["task"]["pursuit_error_px"])
            if run != "old_secondary":
                for outcome in ("pursuit_error_px", "tracking_error_px"):
                    result["prediction"][f"{run}/tracker/{delay}/{outcome}"] = prediction_table(tr, outcome)
    tr = [e for e in all_es["stress"] if e["job"]["event"] == "tracker"]
    for outcome in ("pursuit_error_px", "tracking_error_px"):
        result["prediction"][f"stress/tracker/all/{outcome}"] = prediction_table(tr, outcome)
    cars = [e for e in all_es["nominal"] if e["job"]["event"] == "car"]
    result["prediction"]["nominal/car/finish_s"] = prediction_table(cars, "finish_s", active=True)
    result["car_success_prediction"] = "Not estimable: every nominal car finished (zero outcome variance)."
    result["car_by_seed"] = {str(s): aggregate([e for e in cars if e["job"]["seed"] == s])
                             for s in sorted({e["job"]["seed"] for e in cars})}
    result["round3_isolated_ms"] = {}
    for ev, filename in (("car", "e08_grand_prix"), ("tracker", "e15_object_permanence")):
        path = Path("results") / (filename+".json")
        timing = read(path)["batches"]["1"]["timing"]
        result["round3_isolated_ms"][ev] = dict(source=str(path), sha256=sha(path),
            medians={e: timing[e]["median_ms"] for e in ("numpy_cpu", "mlx_gpu", "coreml_cpu", "coreml_ane")})
    return result


def markdown(d):
    out = ["# Round 5 receipt analysis — turn 2", "", "Generated by `analyze_r5.py`; no MLX executed.", "",
           "Ages are pooled applied-action percentiles in ms. Outcome means weight episodes equally. "
           "Car full-horizon ages include post-finish time; active-only values are also in the JSON. "
           "All raw receipt links and SHA-256 hashes are retained. Core ML names denote requested units, "
           "not measured placement. Small seed counts preclude broad qualification.", "",
           "## Run audit", "", "| Run | Episodes / seeds | Predicted / actual min | 1-min load range | Power |",
           "|---|---:|---:|---:|---|"]
    for run, audit in d["audits"].items():
        env = audit["environment"]
        out.append(f"| {run} | {audit['episodes']} / {len(audit['distinct_seeds'])} | "
                   f"{audit['predicted_s']/60:.3f} / {audit['actual_s']/60:.3f} | "
                   f"{env['load1']['min']:.3f}–{env['load1']['max']:.3f} | "
                   f"{'AC' if all('AC Power' in p for p in env['power']) else 'CHECK JSON'} |")
    for run in RUNS:
        sections = sorted({(k.split('/')[1], k.split('/')[-1]) for k in d["cells"] if k.startswith(run+'/')})
        for event, delay in sections:
            rows = {k: v for k, v in d["cells"].items() if k.startswith(f"{run}/{event}/") and k.endswith('/'+delay)}
            out += ["", f"## {run} · {event} · injected {delay} ms", "",
                    "| Engine / policy / load | N | Age p50 / p95 / p99 ms | Miss / scheduled | Drops | Pending + in-flight | Coverage | Task means | Receipts |",
                    "|---|---:|---:|---:|---:|---:|---:|---|---|"]
            for k, r in rows.items():
                j = k.split('/')
                age = ' / '.join(f"{r['age_ms'][p]:.3f}" for p in ('p50','p95','p99'))
                c, t = r["counts"], r["task"]
                if event == "car":
                    task = f"finish {int(t['finished']['mean']*r['episodes'])}/{r['episodes']}; crash {int(t['crashed']['mean']*r['episodes'])}; lap {t['finish_s']['mean']:.4f} s"
                else:
                    hidden = t['hidden_tracking_error_px']['mean']
                    task = f"pursuit {t['pursuit_error_px']['mean']:.4f}; track {t['tracking_error_px']['mean']:.4f}; hidden {hidden:.4f} px; ≤5px {t['within_5px_fraction']['mean']:.1%}"
                refs = ', '.join(f"[{Path(p).stem.removeprefix('episode_')}]({run}/{Path(p).name})" for p in r['receipts'])
                out.append(f"| {' / '.join(j[2:5])} | {r['episodes']} | {age} | {c['deadline_misses']}/{c['scheduled']} | "
                           f"{c['dropped']} | {c['pending_at_end']} + {c['inflight_at_end']} | {r['coverage']:.2%} | {task} | {refs} |")
            out += ["", "Mean latency breakdown (ms; these columns sum to mean age). Preprocess includes sensor "
                    "rendering. Compute includes native dispatch hidden by the backend. See the outcome row for receipts.", "",
                    "| Engine / policy / load | Preprocess | Queue | Dispatch | Compute | Postprocess | Injected | Pickup | Residual | Mean age |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for k, r in rows.items():
                out.append('| '+' / '.join(k.split('/')[2:5])+' | '+
                           ' | '.join(f"{r['stages_ms'][s]['mean']:.3f}" for s in STAGES)+f" | {r['age_ms']['mean']:.3f} |")
    out += ["", "## Does latency predict outcome?", "", d["method"]["regression"], "",
            "Each row is episode-level, not a regression over dependent frame observations. "
            "Service = observation age minus queue wait. Cross-validation excludes every condition of the held-out seed. "
            "Negative predictive R² means worse than the pooled outcome mean; scene difficulty is not an input. "
            "The policy intervention also changes the GRU's processed history, so this is not a causal age-only estimate.", "",
            "| Run / event / delay / outcome | Predictor | Episodes / seeds | Within-seed r | Within-seed R² | Slope / ms | Held-out-seed R² | RMSE / training-mean RMSE |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, table in d["prediction"].items():
        for feature, r in table.items():
            cv = r['leave_one_seed_out']
            out.append(f"| {key} | {feature} | {r['episodes']} / {r['independent_seeds']} | "
                       f"{r['seed_demeaned_pearson']:.4f} | {r['within_seed_r2']:.4f} | "
                       f"{r['within_seed_slope_per_ms']:.5f} | {cv['r2']:.4f} | {cv['rmse']:.4f} / {cv['training_mean_baseline_rmse']:.4f} |")
    out += ["", "## Matched scheduling effect", "", "Latest minus FIFO pursuit error (negative is better). "
            "Pairing fixes seed, engine, load, delay and horizon. Independent replication is the seed count, not pair count.", "",
            "| Run / delay | Pairs improved | Mean delta px | Per-seed mean range px |", "|---|---:|---:|---:|"]
    for key, r in d["paired_policies"].items():
        out.append(f"| {key} | {r['negative_pairs']}/{len(r['pairs'])} | {r['deltas']['mean']:.5f} | "
                   f"{r['seed_mean_range']['min']:.4f} to {r['seed_mean_range']['max']:.4f} |")
    return '\n'.join(out)+'\n'


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out-prefix", default="results/r5/analysis_turn2")
    args = ap.parse_args()
    start = time.perf_counter()
    result = analyze(Path("results/r5"))
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    prefix.with_suffix('.md').write_text(markdown(result))
    print(f"ANALYSIS OK: {sum(a['episodes'] for a in result['audits'].values())} audited episodes, "
          f"{len(result['cells'])} cells; {time.perf_counter()-start:.3f} s; {prefix}.json/.md")
