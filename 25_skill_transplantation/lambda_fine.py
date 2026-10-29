"""Follow-up to the merge stage: in the `pre` condition the validation pick for task arithmetic and
DARE sat on the lowest grid value (lambda = 0.3), so extend the grid downward to lambda in
{0 (= theta0), 0.1, 0.2} for task arithmetic / TIES / DARE, re-run the validation selection over the
union grid, and score the new picks on the held-out eval seeds. Results go to
results/lambda_fine_s<seed>.json (the original merge JSON is left untouched, because the adapt stage
started its task-arithmetic fine-tune from the original pick).

  ../bin/gpu_run ../.venv/bin/python lambda_fine.py --seeds 0 1 2 3
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FINE = [0.0, 0.1, 0.2]


def run(seed):
    from common import load_params, evaluate, EVAL_SEEDS, VAL_SEEDS, TASKS
    import merge as M
    d = os.path.join(HERE, "checkpoints", f"s{seed}")
    mres = json.load(open(os.path.join(HERE, "results", f"merge_s{seed}.json")))
    _, norm = load_params(os.path.join(d, "base.npz"))
    out = {"seed": seed}
    for cond in ("pre", "init"):
        t0, _ = load_params(os.path.join(d, "base.npz" if cond == "pre" else "init.npz"))
        A, _ = load_params(os.path.join(d, f"{cond}_fwd.npz"))
        B, _ = load_params(os.path.join(d, f"{cond}_bwd.npz"))
        R = mres[cond]["by_length"][str(max(int(k) for k in mres[cond]["by_length"]))]
        fns = {"task_arith": lambda l: M.task_arithmetic(t0, [A, B], l),
               "ties": lambda l: M.ties(t0, [A, B], l, k=0.2),
               "dare": lambda l: M.dare(t0, [A, B], l, p=0.9, seed=seed)}
        res = {}
        for name, fn in fns.items():
            fine_val = [float(sum(np.mean(evaluate(fn(l), norm, tk, VAL_SEEDS)) for tk in TASKS)) for l in FINE]
            grid = FINE + R[name]["grid"]
            scores = fine_val + R[name]["val_scores"]
            lam = grid[int(np.argmax(scores))]
            p = fn(lam)
            res[name] = dict(grid=grid, val_scores=scores, lam=lam,
                             **{tk: evaluate(p, norm, tk, EVAL_SEEDS).tolist() for tk in TASKS},
                             fine_eval={str(l): {tk: float(np.mean(evaluate(fn(l), norm, tk, EVAL_SEEDS)))
                                                 for tk in TASKS} for l in FINE[1:]})
        out[cond] = res
    json.dump(out, open(os.path.join(HERE, "results", f"lambda_fine_s{seed}.json"), "w"), indent=1)
    print(f"[lambda_fine] seed {seed}: " + ", ".join(f"{c}/{k} lam={out[c][k]['lam']}" for c in ("pre", "init")
                                                     for k in out[c]), flush=True)
    return seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(min(4, len(a.seeds))) as pool:
        list(pool.imap_unordered(run, a.seeds))
