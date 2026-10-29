"""Recompute round-5 receipts, provenance, paired comparisons and calibration. No MLX."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from r5_core import canonical_keys, mistakes, calibration
from r5_baselines import macro_nll, scores
from r5_data import exclusions, fingerprint

METHODS = ("fixed", "learned", "entropy", "uncertainty", "disagreement", "rule", "logistic")


def bootstrap_mean(x, seed=50242001):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    # Game batches are the clusters: allocation couples all simultaneous games.
    means = x[rng.integers(len(x), size=(5000, len(x)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def game_summary(records, mean_simulations=32):
    result = {}
    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        for r in rows:
            pairs = np.asarray(r["results"]).reshape(-1, 2)
            keep = (pairs != -2).all(1)
            actual = pairs[keep].ravel()
            assert int(keep.sum()) == r["completed_pairs"]
            for k, value in (("wins", 1), ("draws", 0), ("losses", -1)):
                assert int((actual == value).sum()) == r[k]
            for cost in r["costs"].values():
                assert cost["simulations"] == mean_simulations*cost["decisions"]
        total = {k: sum(r[k] for r in rows) for k in ("wins", "draws", "losses", "completed_pairs", "unfinished_games")}
        total["games"] = 2*total["completed_pairs"]
        total["score"] = (total["wins"]+.5*total["draws"])/total["games"] if total["games"] else None
        full = [r for r in rows if r["completed_pairs"] == len(r["starts"])]
        total["batch_bootstrap_95ci"] = bootstrap_mean([r["score"] for r in full]) if full else None
        total["ci_complete_batches_only"] = len(full)
        total["batches"] = len(rows)
        total["costs"] = {m: {k: sum(r["costs"][m][k] for r in rows)
                             for k in ("decisions", "simulations", "seconds")} for m in (method, "fixed")}
        denominator = total["costs"]["fixed"]["seconds"]
        total["time_ratio"] = total["costs"][method]["seconds"]/denominator if denominator else None
        result[method] = total
    by_batch = {}
    for r in records:
        by_batch.setdefault(r["paired_batch"], {})[r["method"]] = r
    for batch in by_batch.values():
        if len(batch) == 2:
            a, b = batch.values()
            assert a["starts"] == b["starts"], "Game arms do not share starts"
    return result


def analyze():
    forbidden, manifest = exclusions()
    out, datasets, error_maps = {}, {}, {}
    for run in ("main", "ac_large"):
        root = Path("results/r5")/run
        data = {s: dict(np.load(root/(s+".npz"))) for s in ("train", "val", "test")}
        datasets[run] = data
        frozen = json.loads((root/"frozen.json").read_text())
        assert frozen["exclusions"]["replays"] == manifest["replays"]
        sets = {s: set(map(int, d["key"])) for s, d in data.items()}
        for split, d in data.items():
            assert fingerprint(root/(split+".npz")) == frozen["sha256"][split]
            assert np.array_equal(canonical_keys(d["pos"], d["mask"]), d["key"])
            assert len(sets[split]) == len(d["key"]) and not (sets[split] & forbidden)
            for other in data:
                if split != other:
                    assert not (sets[split] & sets[other])
        report = json.loads((root/"evaluation.json").read_text())
        ck = Path(f"checkpoints/r5_{run}/selected")
        assert fingerprint(ck/"head.safetensors") == report["selected_sha256"]
        assert fingerprint(ck/"baselines.json") == report["baselines_sha256"]
        assert fingerprint(root/"test.npz") == report["test_sha256"]
        baselines = json.loads((ck/"baselines.json").read_text())
        for name in ("rule", "logistic"):
            p = scores(name, data["val"]["x"], baselines, None)
            score = macro_nll(data["val"]["y"], p, data["val"]["values"], data["val"]["group"])
            assert np.isclose(score, baselines[name]["score"])
        n = len(data["test"]["y"])
        moves, simcounts, receipts = {}, {}, {}
        for row in report["records"]:
            key = row["regime"]+"/"+row["method"]
            moves.setdefault(key, np.full(n, -1, int))
            simcounts.setdefault(key, np.zeros(n, int))
            ix = np.asarray(row["indices"])
            assert (moves[key][ix] == -1).all(), "Repeated scored positions"
            moves[key][ix] = row["moves"]
            simcounts[key][ix] = row["per_position_sims"]
            assert sum(row["per_position_sims"]) == row["simulations"]
            receipts.setdefault(key, []).append(row)
        errors = {k: mistakes(data["test"]["values"], mv) for k, mv in moves.items()}
        error_maps[run] = errors
        fixed = errors["sim/fixed"]
        table = {}
        for key, err in errors.items():
            r = report["summary"][key]
            assert (moves[key] >= 0).all() and int(err.sum()) == r["mistakes"]
            assert int(simcounts[key].sum()) == r["simulations"]
            assert np.isclose(sum(x["seconds"] for x in receipts[key]), r["seconds"])
            rescued = int(((data["test"]["y"] == 1) & (err == 0)).sum())
            introduced = int(((data["test"]["y"] == 0) & (err == 1)).sum())
            assert (rescued, introduced) == (r["rescued"], r["introduced"])
            better = int(((fixed == 1) & (err == 0)).sum())
            worse = int(((fixed == 0) & (err == 1)).sum())
            ratios = [x["seconds"]/x["fixed_seconds"] for x in receipts[key]]
            table[key] = dict(**{k:v for k,v in r.items() if k != "groups"},
                extra_errors_vs_fixed=int(err.sum()-fixed.sum()), rescued_vs_fixed=better,
                introduced_vs_fixed=worse, mcnemar_exact_p=binomtest(better, better+worse).pvalue if better+worse else 1.,
                avoided_per_1000_sims=1000*r["avoided_per_sim"],
                block_time_ratio_min=min(ratios), block_time_ratio_max=max(ratios),
                blocks_outside_10_percent=sum(abs(x-1)>.1 for x in ratios))
        train_p = float(data["train"]["y"].mean())
        constant = calibration(data["test"]["y"], np.full(n, train_p))
        logs = [json.loads(x) for x in Path(f"logs/r5/{run}_train.log").read_text().splitlines() if x.startswith('{')]
        eval_logs = [json.loads(x) for x in Path(f"logs/r5/{run}_eval.log").read_text().splitlines() if x.startswith('{')]
        training = [r for r in logs if r["event"] == "train"]
        rate = [r["boards_per_s"] for r in logs if r["event"] == "data_rate"]
        selected = json.loads((ck/"head.json").read_text())
        first_best = next(i+1 for i,r in enumerate(training) if r["best_full_validation_macro_nll"] == selected["validation_score"])
        out[run] = dict(rows=frozen["rows"], coverage=frozen["coverage"], table=table,
            calibration=report["calibration"], constant_train_prevalence=constant,
            games=game_summary(json.loads((root/"games.json").read_text())["records"]),
            selected={k:v for k,v in selected.items() if k != "config"},
            rule=baselines["rule"], logistic={k:v for k,v in baselines["logistic"].items() if k not in ("mean","std","weights")},
            all_logistic_converged=all(r["converged"] for r in baselines["trials"] if r["kind"] == "logistic"),
            training_complete=next(r for r in logs if r["event"] == "train_complete"),
            evaluation_complete=next(r for r in eval_logs if r["event"] == "evaluation_complete"),
            first_best_round=first_best, full_rounds_without_selection_improvement=len(training)-first_best,
            data_boards_per_s_median=float(np.median(rate)), train_steps_per_s_median=float(np.median([r["steps_per_s"] for r in training])),
            provenance=dict(dataset_hashes=True, selected_hashes=True, within_run_split_overlap=0,
                            retained_replay_and_old_split_overlap=0, replay_files=len(manifest["replays"])))
    cross = {a+"/"+b: len(set(datasets["main"][a]["key"]) & set(datasets["ac_large"][b]["key"]))
             for a in datasets["main"] for b in datasets["ac_large"]}
    out["cross_run_key_overlaps"] = cross
    ac_index = {int(key):i for i,key in enumerate(datasets["ac_large"]["test"]["key"])}
    shared = np.asarray([ac_index[int(key)] for key in datasets["main"]["test"]["key"]])
    out["shared_test_errors"] = {k:dict(n=len(shared), main=int(error_maps["main"][k].sum()),
                                       ac_large=int(error_maps["ac_large"][k][shared].sum()))
                                  for k in error_maps["main"]}
    starts = {}
    for run in ("main", "ac_large"):
        rows = json.loads(Path(f"results/r5/{run}/games.json").read_text())["records"]
        starts[run] = {r["paired_batch"]:r["starts"] for r in rows}
    common = set(starts["main"]) & set(starts["ac_large"])
    out["shared_game_start_batches"] = sum(starts["main"][i] == starts["ac_large"][i] for i in common)
    out["notes"] = ["Tests/validation are nested, not independent seed evidence. Training overlap is not an exact prefix.",
                    "McNemar p-values are exploratory and unadjusted; time-ratio failures are not matched comparisons.",
                    "Game CIs bootstrap paired-start batches, not individual games."]
    path = Path("results/r5/turn2_analysis.json")
    path.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(path)
    for run in ("main", "ac_large"):
        r = out[run]
        print(run, "data/s",r["data_boards_per_s_median"],"steps/s",r["train_steps_per_s_median"], "best round",r["first_best_round"])
        for method, g in r["games"].items():
            print("games",method,g["score"],g["batch_bootstrap_95ci"],"time ratio",g["time_ratio"])
    return out


if __name__ == "__main__":
    analyze()
