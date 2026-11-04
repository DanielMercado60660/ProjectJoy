"""Held-out scoring of every WSP attempt and every baseline (runner: Claude). Written after the Astra
attempts finished; the HELDOUT bank below appears nowhere else in the project and was never given to Astra.

  ../bin/gpu_run ../.venv/bin/python wsp/heldout.py            # -> wsp/results/heldout.json

Networks per training seed (pre condition): oracle switch, theta0, average, task arithmetic / TIES / DARE
(original lambda grid and the extended grid, lambdas from results/), fwd-/bwd-specialist under both labels,
DAgger-200k (checkpoints/s*/distill_pre_theta0.npz), PPO fine-tune from the average 1M
(checkpoints/s*/ft_pre_ft_average.npz), the matched-budget arms in wsp/baselines/, and every
wsp/attempts/r6wsp_*/merged_s*.npz. Banks: HELDOUT (primary, 20 resets per task) and the canonical
EVAL_SEEDS 10000-10009 (secondary; readable in common.py, never used by the attempts' evaluator).
"""
import glob
import json
import multiprocessing as mp
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import numpy as np  # noqa: E402

HELDOUT = list(range(31000, 31020))


def networks(seed):
    from common import load_params
    import merge as M
    d = os.path.join(PROJ, "checkpoints", f"s{seed}")
    t0, norm = load_params(os.path.join(d, "base.npz"))
    A, _ = load_params(os.path.join(d, "pre_fwd.npz"))
    B, _ = load_params(os.path.join(d, "pre_bwd.npz"))
    mres = json.load(open(os.path.join(PROJ, "results", f"merge_s{seed}.json")))["pre"]["by_length"]["2000000"]
    lf = json.load(open(os.path.join(PROJ, "results", f"lambda_fine_s{seed}.json")))["pre"]
    nets = {"oracle": ("switch", A, B), "theta0": t0, "average": M.interpolate(A, B, 0.5),
            "task_arith": M.task_arithmetic(t0, [A, B], mres["task_arith"]["lam"]),
            "ties": M.ties(t0, [A, B], mres["ties"]["lam"], k=0.2),
            "dare": M.dare(t0, [A, B], mres["dare"]["lam"], p=0.9, seed=seed),
            "task_arith_ext": M.task_arithmetic(t0, [A, B], lf["task_arith"]["lam"]),
            "ties_ext": M.ties(t0, [A, B], lf["ties"]["lam"], k=0.2),
            "dare_ext": M.dare(t0, [A, B], lf["dare"]["lam"], p=0.9, seed=seed),
            "fwd_spec_both": A, "bwd_spec_both": B,
            "dagger_200k": load_params(os.path.join(d, "distill_pre_theta0.npz"))[0],
            "ppo_ft_avg_1M": load_params(os.path.join(d, "ft_pre_ft_average.npz"))[0]}
    for f in sorted(glob.glob(os.path.join(HERE, "baselines", f"*_s{seed}.npz"))):
        nets[os.path.basename(f)[:-len(f"_s{seed}.npz")]] = {k: np.load(f)[k].astype(np.float32) for k in
                                                               ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias"]}
    for f in sorted(glob.glob(os.path.join(HERE, "attempts", "r6wsp_*", f"merged_s{seed}.npz"))):
        z = np.load(f)
        nets[os.path.basename(os.path.dirname(f))] = {k: z[k].astype(np.float32) for k in
                                                       ["a1.weight", "a1.bias", "a2.weight", "a2.bias", "a3.weight", "a3.bias"]}
    return nets, norm


def job(args):
    seed, name = args
    from common import EVAL_SEEDS, evaluate
    nets, norm = networks(seed)
    n = nets[name]
    out = {}
    for bank_name, bank in (("heldout", HELDOUT), ("canonical", EVAL_SEEDS)):
        r = {}
        for ti, task in enumerate(("fwd", "bwd")):
            p = (n[1 + ti] if isinstance(n, tuple) else n)
            ret, xv = evaluate(p, norm, task, bank, return_xvel=True)
            r[task] = ret.tolist()
            r[task + "_xvel"] = float(xv.mean())
        out[bank_name] = r
    return seed, name, out


def main():
    seeds = [0, 1, 2, 3]
    jobs = [(s, k) for s in seeds for k in networks(s)[0]]
    res = {}
    with mp.get_context("spawn").Pool(4) as pool:
        for s, k, o in pool.imap_unordered(job, jobs):
            res.setdefault(str(s), {})[k] = o
            print(f"s{s} {k}: fwd {np.mean(o['heldout']['fwd']):.0f} bwd {np.mean(o['heldout']['bwd']):.0f}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(heldout_bank=HELDOUT, results=res), open(os.path.join(HERE, "results", "heldout.json"), "w"))


if __name__ == "__main__":
    main()
