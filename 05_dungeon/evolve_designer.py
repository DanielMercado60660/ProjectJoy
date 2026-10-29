"""Evolve nasty-but-solvable dungeons against a solver.

  ../.venv/bin/python evolve_designer.py --solver checkpoints/solver_v1 --out vs_v1 --gens 200
  ../.venv/bin/python evolve_designer.py --solver wall_follow --out vs_wall --gens 200
  ../.venv/bin/python evolve_designer.py --solver checkpoints/solver_v1 --out vs_v1_nocon --no-constraint --gens 30

Writes results/designer_<out>.npz (final population, archive of levels with fitness >= 0.5,
fresh re-evaluation of the final population with unseen seeds) and logs/designer_<out>.jsonl.
"""
import argparse, json, os
import numpy as np

from dungeon.data import train_pool
from dungeon.designer import evolve, policy_eval_fn, scripted_eval_fn
from dungeon.scripted import ALGOS

HERE = os.path.dirname(os.path.abspath(__file__))


def make_eval(solver, K):
    if solver in ALGOS:
        return scripted_eval_fn(solver, K)
    from dungeon.policy import load_policy
    return policy_eval_fn(load_policy(os.path.join(solver, "weights.safetensors")), K)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--solver", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--K", type=int, default=8, help="solver rollouts per level per generation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-constraint", action="store_true")
    p.add_argument("--no-select", action="store_true", help="neutral-drift control")
    p.add_argument("--fresh-K", type=int, default=32)
    p.add_argument("--min-dist", type=int, default=6, help="minimum BFS start-exit distance (constraint)")
    a = p.parse_args(argv)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = open(os.path.join(HERE, "logs", f"designer_{a.out}.jsonl"), "w")
    import dungeon.designer as D
    D.MIN_DIST = a.min_dist
    init = train_pool()
    if a.min_dist > 6:  # seed the population only with levels that already satisfy the constraint
        from dungeon.scripted import oracle_lengths
        keep = oracle_lengths(init) >= a.min_dist
        init = {k: v[keep] for k, v in init.items()}
    P, fit, A, hist, stats = evolve(make_eval(a.solver, a.K), init, gens=a.gens, pop=a.pop, seed=a.seed,
                                    constrained=not a.no_constraint, select=not a.no_select, log=log)
    fresh = make_eval(a.solver, a.fresh_K)(P, seed=10**7 + a.seed)  # unseen seeds: no winner's curse
    np.savez_compressed(os.path.join(HERE, "results", f"designer_{a.out}.npz"),
                        walls=P["walls"], start=P["start"], goal=P["goal"], fit=fit,
                        fresh_fitness=fresh["fitness"], fresh_fail=fresh["fail"], solvable=fresh["solvable"],
                        arch_walls=A["walls"], arch_start=A["start"], arch_goal=A["goal"],
                        arch_gen=A["gen"], arch_fit=A["fit"])
    summary = dict(args=vars(a), proposals=stats.proposed, valid_proposals=stats.valid,
                   valid_proposal_rate=stats.valid / max(1, stats.proposed),
                   final_solvable=float(fresh["solvable"].mean()),
                   final_fit_selected=float(fit.mean()), final_fit_fresh=float(fresh["fitness"].mean()),
                   final_fail_fresh=float(fresh["fail"].mean()), archive=int(len(A["walls"])),
                   gen0=hist[0], last=hist[-1])
    with open(os.path.join(HERE, "results", f"designer_{a.out}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("gen0", "last", "args")}))


if __name__ == "__main__":
    main()
