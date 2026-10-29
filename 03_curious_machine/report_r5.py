"""Only summarize complete, matched fresh-split cells; preserve all per-seed sources."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_r5 import METHODS


def report(directory, seeds, output, methods=METHODS):
    directory = Path(directory); records = {}; sources = []; budgets = set(); tests = set(); protocols = set()
    for world in ("plain", "tv"):
        for method in methods:
            rows = []
            for seed in seeds:
                path = directory / f"{method}_{world}_s{seed}.json"
                r = json.loads(path.read_text())  # missing cells are a hard failure
                assert r["split"] == "fresh" and r["method"] == method and r["world"] == world
                assert r["training_seed"] == seed
                if method not in ("frontier", "random"):
                    assert r["pretrain_steps"] == r["charged_pretrain_steps"]
                else:
                    pretrain = Path(r["pretrain_checkpoint"])
                    # Check the charged budget against the real saved classical run.
                    import pickle
                    state = pickle.loads(pretrain.read_bytes())
                    assert state["steps"] == r["charged_pretrain_steps"]
                assert len(r["tasks"]) == len(r["seeds"]) * 7
                budgets.add((r["charged_pretrain_steps"], r["explore_steps"], r["task_steps"], r["max_expansions"]))
                protocols.add((r.get('planner_type','replay'), r['policy_rng_seed'], r['transferred']))
                tests.add(tuple(r["seeds"])); rows.append(r); sources.append(str(path))
            records[(world, method)] = rows
    assert len(budgets) == len(tests) == len(protocols) == 1, "Unmatched budgets, test seeds or transfer protocol"
    lines = ["# Round 5 measured summary", "", f"Training seeds: {seeds}; budgets (pretrain, explore, task, expansions): {next(iter(budgets))}.",
             "Mean ± population SD over training seeds. One seed is descriptive only. Classical policies have no fitted training seed; repeated rows are not independent replicas.", "",
             "| world | method | rooms | mechanisms | TV steps | task success | capped task steps | plan coverage | empty replay success |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for (world, method), rows in records.items():
        fields = [("exploration", "rooms"), ("exploration", "interactions"), ("exploration", "tv_steps"),
                  ("transfer", "success_rate"), ("transfer", "capped_steps"), ("transfer", "plan_coverage"),
                  ("empty_replay", "success_rate")]
        nums = [np.array([r[a][b] for r in rows]) for a, b in fields]
        lines.append(f"| {world} | {method} | " + " | ".join(f"{v.mean():.3f} ± {v.std():.3f}" for v in nums) + " |")
        summary[f"{world}/{method}"] = {f"{a}/{b}": dict(mean=float(v.mean()), std=float(v.std()), per_seed=v.tolist())
                                       for (a, b), v in zip(fields, nums)}
    lines += ["", "Sources:", ""] + [f"- `{s}`" for s in sources]
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    output.with_suffix(".json").write_text(json.dumps(dict(seeds=seeds, sources=sources, summary=summary), indent=2))
    print("MATCHED_REPORT", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--directory", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--turn2", action="store_true")
    a = p.parse_args()
    report(a.directory, a.seeds, a.out, ("lpm", "rnd_ep", "disagree", "ngu", "frontier", "random") if a.turn2 else METHODS)
