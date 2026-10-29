"""Exploratory analysis of SAVED R5 runs; never a new held-out evaluation.

Cross probe histories, estimators and terminal controllers on the first N saved
worlds. Write new artifacts only. No probe or original result is changed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from physics import G, MU_HI, MU_LO, NT
from r5_core import Belief, FINAL_ACTIONS, Guard, TerminalTask, true_stop
from run_r5 import atomic_json, paired_delta, read_json, source_hash, summarize


def mean(x):
    return float(np.mean(x))


def reconstruct(row, grid_size):
    post = Belief(grid_size)
    for h in row["history"]:
        y = np.array([np.nan if v is None else v for v in h["y"]])
        post.update(np.asarray(h["action"]), y, np.asarray(h["cens"]))
    return post


def run(args):
    guard = Guard(args.max_minutes)
    if Path(args.out).exists():
        raise ValueError("Preserve diagnostic artifacts; choose a new --out")
    state = read_json(args.checkpoint)
    assert state["status"] == "complete" and state["config"]["source_sha256"] == source_hash()
    recomputed = summarize(state)
    stored = read_json(args.summary)
    assert stored["groups"] == recomputed["groups"]
    assert stored["paired_success_differences"] == recomputed["paired_success_differences"]
    full_rows = state["rows"]
    swap_rows = []
    for row in full_rows:
        if row["budget"] != 4 or row["method"] not in ("eig", "task_voi", "lsq", "tobit") or row["world"] >= args.worlds:
            continue
        guard.check()
        post = reconstruct(row, state["config"]["planner"]["grid_size"])
        task = TerminalTask(row["target"])
        estimates = {m: post.estimate(m) for m in ("eig", "lsq", "tobit")}
        policies = {"saved": np.asarray(row["action"]), "bayes": task.bayes_action(post)[0]}
        for name, mu_hat in estimates.items():
            policies[f"point_{name}"] = task.point_action(mu_hat)
            menu10 = FINAL_ACTIONS[FINAL_ACTIONS[:, 0] == 10]
            policies[f"force10_{name}"] = TerminalTask(row["target"], menu10).point_action(mu_hat)
        mu, gain = row["mu_true"], row["gain_true"]
        force = row["action"][0]
        # An exact effective coefficient for matching stop distance at THIS force.
        mu_effective = mu / (gain * gain + (1 - gain) * mu * G / force)
        w = post.weights
        out = dict(world=row["world"], family=row["family"], probe_policy=row["method"],
                   mu=mu, gain=gain, estimates=estimates,
                   post_mean_minus_mle=estimates["eig"] - estimates["tobit"],
                   lsq_minus_mle=estimates["lsq"] - estimates["tobit"],
                   boundary_mass=float(w[(post.grid < MU_LO + .005) | (post.grid > MU_HI - .005)].sum()),
                   underresolved=row["grid_step_over_sd"] > .5,
                   error_to_true=abs(row["mu_estimate"] - mu),
                   error_to_mu_over_gain=abs(row["mu_estimate"] - mu / gain),
                   error_to_stop_effective_mu=abs(row["mu_estimate"] - mu_effective),
                   final_force=force, last_probe_force=row["history"][-1]["action"][0],
                   last_probe_duration=row["history"][-1]["action"][1],
                   censored_fraction=mean([v for h in row["history"] for v in h["cens"]]),
                   controllers={})
        for name, action in policies.items():
            stop = float(true_stop(action, mu, gain))
            out["controllers"][name] = dict(action=action.tolist(), error_cm=100 * abs(stop - row["target"]),
                                             success=bool(abs(stop - row["target"]) <= .025))
        swap_rows.append(out)
    grouped = defaultdict(list)
    for r in swap_rows:
        grouped[r["family"], r["probe_policy"]].append(r)
    swaps = []
    fields = ("boundary_mass", "underresolved", "error_to_true", "error_to_mu_over_gain",
              "error_to_stop_effective_mu", "final_force", "last_probe_force", "last_probe_duration",
              "censored_fraction")
    for (family, policy), rows in sorted(grouped.items()):
        out = dict(family=family, probe_policy=policy, n=len(rows),
                   mean_abs_post_mean_minus_mle=mean([abs(r["post_mean_minus_mle"]) for r in rows]),
                   mean_abs_lsq_minus_mle=mean([abs(r["lsq_minus_mle"]) for r in rows]),
                   **{k: mean([r[k] for r in rows]) for k in fields})
        out["controllers"] = {name: dict(success_rate=mean([r["controllers"][name]["success"] for r in rows]),
                                         mae_cm=mean([r["controllers"][name]["error_cm"] for r in rows]))
                              for name in rows[0]["controllers"]}
        interior = [r for r in rows if r["boundary_mass"] < .01]
        out["interior_n"] = len(interior)
        out["interior_controllers"] = {name: mean([r["controllers"][name]["success"] for r in interior])
                                       for name in rows[0]["controllers"]}
        swaps.append(out)
    # These analyses use ALL saved worlds, without rerunning any simulator policy.
    policy_overlap, oracle_stats = [], []
    indexed = {(r["world"], r["family"], r["method"], r["budget"]): r for r in full_rows}
    for family in ("matched", "gain_shift"):
        for budget in (1, 2, 4):
            a = [r for r in full_rows if r["family"] == family and r["method"] == "eig" and r["budget"] == budget]
            b = [indexed[r["world"], family, "task_voi", budget] for r in a]
            policy_overlap.append(dict(family=family, budget=budget, n=len(a),
                                       last_probe_same_fraction=mean([x["history"][-1]["action"] == y["history"][-1]["action"] for x, y in zip(a, b)]),
                                       eig_predicted_success_mean=mean([r["predicted_success"] for r in a]),
                                       voi_predicted_success_mean=mean([r["predicted_success"] for r in b])))
        oracle = [r for r in full_rows if r["family"] == family and r["method"] == "mu_oracle"]
        oracle_stats.append(dict(family=family, n=len(oracle),
                                min_abs_placement_error_cm=100 * min(abs(r["placement_error"]) for r in oracle),
                                max_nominal_aim_error_cm=100 * max(abs(float(true_stop(np.array(r["action"]), r["mu_true"])) - r["target"]) for r in oracle)))
    boundary_stats = []
    for family in ("matched", "gain_shift"):
        for method in ("eig", "task_voi", "lsq", "tobit"):
            rows = [r for r in full_rows if r["family"] == family and r["method"] == method and r["budget"] == 4]
            under = [r for r in rows if r["grid_step_over_sd"] > .5]
            boundary_stats.append(dict(family=family, method=method, n=len(rows), underresolved=len(under),
                                       underresolved_near_edge=sum(min(r["posterior_mean"] - MU_LO, MU_HI - r["posterior_mean"]) < .005 for r in under)))
    ac = read_json(args.resolution)
    n_ac = ac["config"]["worlds"]
    ac_differences = []
    for g in ac["groups"]:
        rows = [r for r in full_rows if r["world"] < n_ac and r["family"] == g["family"] and r["method"] == g["method"] and r["budget"] == g["budget"]]
        ac_differences.append(dict(family=g["family"], method=g["method"], budget=g["budget"], n=n_ac,
                                   main_prefix_success=mean([r["success"] for r in rows]), ac_success=g["success_rate"],
                                   difference=g["success_rate"] - mean([r["success"] for r in rows])))
    artifact = dict(status="complete", exploratory=True, source_checkpoint=args.checkpoint,
                    source_summary=args.summary, source_resolution=args.resolution,
                    source_sha256=source_hash(), swap_worlds=args.worlds, swap_budget=4,
                    main_summary_recomputed_exactly=True, swaps=swaps, raw_swaps=swap_rows,
                    policy_overlap=policy_overlap, oracle_stats=oracle_stats,
                    boundary_diagnostics=boundary_stats, ac_same_prefix_differences=ac_differences,
                    elapsed_seconds=time.monotonic() - guard.started)
    atomic_json(args.out, artifact)
    print(json.dumps({k: v for k, v in artifact.items() if k not in ("raw_swaps", "swaps", "ac_same_prefix_differences")}, indent=2))
    for row in swaps:
        print(row["family"], row["probe_policy"], {k: round(v["success_rate"], 4) for k, v in row["controllers"].items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/r5_main.json")
    p.add_argument("--summary", default="results/r5/main_summary.json")
    p.add_argument("--resolution", default="results/r5/ac_resolution_summary.json")
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--max-minutes", type=float, required=True)
    run(p.parse_args())
