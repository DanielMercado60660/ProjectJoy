"""Evaluate every solver on every level set; write results/eval.json and results/eval_tables.md.

  ../.venv/bin/python evaluate.py            # needs checkpoints/ and results/designer_*.npz
  ../.venv/bin/python evaluate.py --quick    # 100 levels per set, 1 rollout (smoke)
"""
import argparse, json, os
import numpy as np

from dungeon.data import test_set, exit_in_view_probe
from dungeon.mazes import ALL_FAMILIES, TRAIN_FAMILIES, levels_stats
from dungeon.policy import load_policy
from dungeon.rollout import run_policy
from dungeon.scripted import ALGOS, run_batch, oracle_lengths, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVERS = ["solver_v1", "solver_ctl1", "solver_cur1"]


def designer_set(name, n_arch=0, seed=0):
    p = os.path.join(HERE, "results", f"designer_{name}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    L = dict(walls=z["walls"], start=z["start"], goal=z["goal"])
    if n_arch and len(z["arch_walls"]):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z["arch_walls"]), min(n_arch, len(z["arch_walls"])), replace=False)
        L = {k: np.concatenate([L[k], z["arch_" + k][idx]]) for k in L}
    return L


def eval_on(levels, solver, K, seed=0):
    rep = {k: np.repeat(v, K, axis=0) for k, v in levels.items()}
    opt = oracle_lengths(rep)
    if solver == "bfs_oracle":
        return summarize(np.ones(len(opt), bool), opt, opt)
    if solver in ALGOS:
        o = run_batch(solver, rep, seed=seed)
    else:
        greedy = solver.endswith(":greedy")
        name = solver.split(":")[0]
        pol = load_policy(os.path.join(HERE, "checkpoints", name, "weights.safetensors"))
        o = run_policy(pol, rep, greedy=greedy, seed=seed)
    return summarize(o["success"], o["steps"], opt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-arch", type=int, default=0, help="add this many archive levels to each designer set")
    ap.add_argument("--n", type=int, default=500, help="levels per family test set")
    ap.add_argument("--solvers", nargs="*", default=SOLVERS)
    ap.add_argument("--designer-runs", nargs="*", default=None, help="override designer result names")
    ap.add_argument("--tables-only", action="store_true", help="rebuild the markdown from results/<out>.json")
    a = ap.parse_args(argv)
    out = a.out or ("eval_quick" if a.quick else "eval")
    if a.tables_only:
        res = json.load(open(os.path.join(HERE, "results", out + ".json")))
        sets = list(res["results"])
        solvers = list(res["results"][sets[0]])
        write_tables(res, sets, solvers, res["K"], out)
        return
    n = 100 if a.quick else a.n
    K = 1 if a.quick else a.K

    sets = {f: {k: v[:n] for k, v in test_set(f).items()} for f in ALL_FAMILIES}
    sets["probe: exit in view at spawn"] = {k: v[:n if a.quick else None] for k, v in exit_in_view_probe().items()}
    for name, label in (("v1_s1", "designer_vs_v1 (held-out run)"),
                        ("cur1_s2", "designer_vs_cur1"), ("ctl1_s2", "designer_vs_ctl1"),
                        ("wall_follow_s0", "designer_vs_wall_follow"), ("tremaux_s0", "designer_vs_tremaux"),
                        ("v1_drift", "neutral_drift_vs_v1")) if a.designer_runs is None else \
            [(r, "designer_" + r) for r in a.designer_runs]:
        L = designer_set(name, n_arch=a.n_arch)
        if L is not None:
            sets[label] = L

    solvers = [s for s in a.solvers if os.path.exists(os.path.join(HERE, "checkpoints", s, "weights.safetensors"))]
    solvers = solvers + ([solvers[0] + ":greedy"] if solvers else []) + list(ALGOS) + ["bfs_oracle"]

    res = dict(K=K, n_per_family=n, results={}, level_stats={})
    for sname, L in sets.items():
        res["level_stats"][sname] = levels_stats(L)
        res["results"][sname] = {}
        for sv in solvers:
            r = eval_on(L, sv, K, seed=4242)
            res["results"][sname][sv] = r
            print(f"{sname:32s} {sv:18s} succ={r['success']:.3f} eff={r['efficiency']:.3f} spl={r['spl']:.3f}", flush=True)

    # designer validity summaries
    res["designer"] = {}
    for f in sorted(os.listdir(os.path.join(HERE, "results"))):
        if f.startswith("designer_") and f.endswith(".json") and "smoke" not in f:
            with open(os.path.join(HERE, "results", f)) as fh:
                d = json.load(fh)
            res["designer"][f[9:-5]] = {k: d[k] for k in ("valid_proposal_rate", "final_solvable", "final_fit_selected",
                                                          "final_fit_fresh", "final_fail_fresh", "archive", "proposals")}
            res["designer"][f[9:-5]]["gen0_fit_mean"] = d["gen0"]["fit_mean"]
    with open(os.path.join(HERE, "results", out + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_tables(res, list(sets), solvers, K, out)


def write_tables(res, sets, solvers, K, out):
    lines = []
    def se(p, m):
        return np.sqrt(max(p * (1 - p), 1e-12) / m)
    lines.append(f"Success rate (± s.e.) / SPL. {K} rollout(s) per level; train families = {', '.join(TRAIN_FAMILIES)}; "
                 f"held-out = caves, spiral.\n")
    lines.append("| level set | " + " | ".join(solvers) + " |")
    lines.append("|---" * (len(solvers) + 1) + "|")
    for sname in sets:
        row = [sname + (" (held-out)" if sname in ("caves", "spiral") else "")]
        for sv in solvers:
            r = res["results"][sname][sv]
            row.append(f"{r['success']:.3f}±{se(r['success'], r['n']):.3f} / {r['spl']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\nPath efficiency (BFS length / steps, successful episodes only):\n")
    lines.append("| level set | " + " | ".join(solvers) + " |")
    lines.append("|---" * (len(solvers) + 1) + "|")
    for sname in sets:
        lines.append("| " + sname + " | " + " | ".join(f"{res['results'][sname][sv]['efficiency']:.3f}" for sv in solvers) + " |")
    lines.append("\nLevel structure (means):\n")
    keys = ["floor", "path_len", "dead_ends", "junctions", "loops", "open2x2"]
    lines.append("| level set | " + " | ".join(keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for sname in sets:
        lines.append("| " + sname + " | " + " | ".join(f"{res['level_stats'][sname][k]:.1f}" for k in keys) + " |")
    lines.append("\nDesigner runs:\n")
    dk = ["proposals", "valid_proposal_rate", "final_solvable", "gen0_fit_mean", "final_fit_selected", "final_fit_fresh", "final_fail_fresh", "archive"]
    lines.append("| run | " + " | ".join(dk) + " |")
    lines.append("|---" * (len(dk) + 1) + "|")
    for k, d in res["designer"].items():
        lines.append("| " + k + " | " + " | ".join(f"{d[x]:.3f}" if isinstance(d[x], float) else str(d[x]) for x in dk) + " |")
    with open(os.path.join(HERE, "results", out + "_tables.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
