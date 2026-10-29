"""The whole experiment, per training seed, in three stages.

  train : shared start -> two specialists, under two conditions
          "pre"  : common multitask PPO pretraining (P steps, 8 fwd + 8 bwd envs) is theta0
          "init" : common random init is theta0 (same seed, same frozen obs normalizer)
          each specialist = PPO on one task for S steps from theta0 (16 envs, task one-hot set)
          snapshots of each branch are kept for the branch-length study
  merge : training-free combinations (interpolation sweep, averaging, task arithmetic, TIES,
          DARE), lambda picked on VALIDATION seeds, scored on held-out EVAL seeds
  adapt : adaptation cost: multitask PPO fine-tuning from {theta0, average, best merge}, DAgger
          distillation of both specialists into one network, sequential transfer
          (specialist A -> task B), multitask PPO from scratch with the full matched budget

  ../bin/gpu_run ../.venv/bin/python pipeline.py --stage train --seeds 0 1 2 3
  ../bin/gpu_run ../.venv/bin/python pipeline.py --stage merge --seeds 0 1 2 3
  ../bin/gpu_run ../.venv/bin/python pipeline.py --stage adapt --seeds 0 1 2 3
Seeds run in a pool of <= 4 single-threaded worker processes. --tiny shrinks everything for smoke.sh.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from common import TASKS  # noqa: E402  (pair chosen by env SKILL_PAIR, default fwd_bwd)

T0, T1 = TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = dict(full=dict(P=500_000, S=2_000_000, snaps=[125_000, 250_000, 500_000, 1_000_000],
                       FT=1_000_000, ft_eval=100_000, D=200_000, D_evals=[0, 25_000, 50_000, 100_000, 200_000],
                       TR=1_000_000, MT=4_500_000, mt_eval=250_000, n_eval=10, n_val=5),
             tiny=dict(P=8192, S=16384, snaps=[8192], FT=8192, ft_eval=4096, D=8192, D_evals=[0, 4096, 8192],
                       TR=8192, MT=8192, mt_eval=4096, n_eval=2, n_val=2))
LAMS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5]
ALPHAS = [round(a, 2) for a in np.linspace(0, 1, 11)]


def cdir(root, seed):
    d = os.path.join(root, f"s{seed}")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- train
def stage_train(seed, Z, root, resdir):
    from common import Normalizer, init_params, save_params, evaluate, EVAL_SEEDS, TASKS, load_params
    from ppo import train
    d = cdir(root, seed)
    ev_seeds = EVAL_SEEDS[:Z["n_eval"]]
    t0 = time.time()
    init = init_params(seed)
    base, norm, h_base = train(init, Normalizer(), [T0] * 8 + [T1] * 8, Z["P"], seed,
                               os.path.join(d, "base"), update_norm=True, quiet=True)
    save_params(os.path.join(d, "init.npz"), init, norm)   # same frozen normalizer as every branch
    out = {"seed": seed, "base_log": h_base["log"][-1], "spec": {}}
    for cond, start in (("pre", base), ("init", init)):
        for task in TASKS:
            name = f"{cond}_{task}"

            def cb(step, p, nrm, name=name, task=task):
                r = {}
                for sn in Z["snaps"]:   # snapshot at the first update that reaches each listed length
                    if step - 4096 < sn <= step:
                        save_params(os.path.join(d, f"{name}_snap{sn}.npz"), p, nrm, step=step)
                for tk in TASKS:
                    r[tk] = float(np.mean(evaluate(p, nrm, tk, ev_seeds)))
                return r

            # eval every 125k (full) so curves exist and snapshots land on the listed steps
            every = Z["snaps"][0]
            p, _, h = train(start, Normalizer.from_state(norm.state()), [task] * 16, Z["S"], seed + 100,
                            os.path.join(d, name), eval_every=every, eval_cb=cb, quiet=True)
            out["spec"][name] = dict(evals=h["evals"], final_log=h["log"][-1])
            print(f"[s{seed}] {name} done: {h['evals'][-1]}  ({time.time() - t0:.0f}s)", flush=True)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(resdir, f"train_s{seed}.json"), "w"), indent=1)
    return seed


# ----------------------------------------------------------------------------- merge
def stage_merge(seed, Z, root, resdir):
    from common import load_params, evaluate, EVAL_SEEDS, VAL_SEEDS, TASKS
    import merge as M
    d = cdir(root, seed)
    ev_seeds, val_seeds = EVAL_SEEDS[:Z["n_eval"]], VAL_SEEDS[:Z["n_val"]]
    t0 = time.time()
    _, norm = load_params(os.path.join(d, "base.npz"))
    out = {"seed": seed}

    def ev(p, seeds):
        return {tk: evaluate(p, norm, tk, seeds).tolist() for tk in TASKS}

    for cond in ("pre", "init"):
        t_0, _ = load_params(os.path.join(d, "base.npz" if cond == "pre" else "init.npz"))
        lengths = Z["snaps"] + [Z["S"]]
        res = {"theta0": ev(t_0, ev_seeds), "by_length": {}}
        for L in lengths:
            suf = "" if L == Z["S"] else f"_snap{L}"
            A, _ = load_params(os.path.join(d, f"{cond}_{T0}{suf}.npz"))
            B, _ = load_params(os.path.join(d, f"{cond}_{T1}{suf}.npz"))
            spec = {T0: ev(A, ev_seeds), T1: ev(B, ev_seeds)}
            spec_val = {T0: float(np.mean(evaluate(A, norm, T0, val_seeds))),
                        T1: float(np.mean(evaluate(B, norm, T1, val_seeds)))}
            R = {"specialist_" + T0: spec[T0], "specialist_" + T1: spec[T1],
                 "stats": M.delta_stats(t_0, A, B), "spec_val": spec_val}

            def val_score(p):
                # summed raw validation return over both tasks (same reward scale); a ratio to the
                # specialist would blow up when a short branch's specialist return is near zero
                return float(sum(np.mean(evaluate(p, norm, tk, val_seeds)) for tk in TASKS))

            def pick(fn, grid):
                scores = [val_score(fn(g)) for g in grid]
                g = grid[int(np.argmax(scores))]
                return g, scores

            R["average"] = ev(M.interpolate(A, B, 0.5), ev_seeds)
            for mname, fn, grid in (
                    ("task_arith", lambda l: M.task_arithmetic(t_0, [A, B], l), LAMS),
                    ("ties", lambda l: M.ties(t_0, [A, B], l, k=0.2), LAMS + [2.0]),
                    ("dare", lambda l: M.dare(t_0, [A, B], l, p=0.9, seed=seed), LAMS)):
                g, sc = pick(fn, grid)
                R[mname] = dict(lam=g, val_scores=sc, grid=grid, **ev(fn(g), ev_seeds))
            # interference: same lambda, but only one task vector added
            lam = R["task_arith"]["lam"]
            R["ta_single"] = {T0 + "_only": ev(M.task_arithmetic(t_0, [A], lam), ev_seeds),
                              T1 + "_only": ev(M.task_arithmetic(t_0, [B], lam), ev_seeds),
                              "lam": lam}
            if L == Z["S"]:
                R["interp"] = {str(a): ev(M.interpolate(A, B, a), ev_seeds) for a in ALPHAS}
                R["ta_curve"] = {str(l): ev(M.task_arithmetic(t_0, [A, B], l), ev_seeds) for l in LAMS}
            res["by_length"][str(L)] = R
            print(f"[s{seed}] merge {cond} L={L} done ({time.time() - t0:.0f}s)", flush=True)
        out[cond] = res
    out["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(resdir, f"merge_s{seed}.json"), "w"), indent=1)
    return seed


# ----------------------------------------------------------------------------- adapt
def distill(student, teachers, norm, steps, evals, seed, ev_seeds, grad_per_iter=50, lr=1e-3):
    """DAgger: the student acts (mean + N(0, 0.1) noise), the oracle-labelled teacher's mean action
    is the target; MSE on the actor mean over an aggregated buffer. Rewards are never used."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from common import TASKS, make_env, np_mean, policy_input, evaluate
    from ppo import model_from_np, np_from_model
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    model = model_from_np(student)
    opt = optim.Adam(learning_rate=lr)
    tasks = [T0] * 8 + [T1] * 8
    tid = np.array([TASKS.index(t) for t in tasks])
    envs = [make_env() for _ in tasks]
    obs = np.stack([e.reset(seed=int(seed * 1000 + 50 + i))[0] for i, e in enumerate(envs)])
    BX, BY = [], []
    curve, done, T = [], 0, 256

    def loss_fn(m, x, y):
        return ((m.mean(x) - y) ** 2).mean()

    gfn = nn.value_and_grad(model, loss_fn)

    def record():
        p = np_from_model(model)
        curve.append(dict(step=done, **{tk: float(np.mean(evaluate(p, norm, tk, ev_seeds))) for tk in TASKS}))

    if 0 in evals:
        record()
    while done < steps:
        p = np_from_model(model)
        for _ in range(T):
            x = policy_input(norm, obs, tid)
            a = np.clip(np_mean(p, x) + 0.1 * rng.standard_normal((len(tasks), 6)), -1, 1)
            y = np.where((tid == 0)[:, None], np_mean(teachers[0], x), np_mean(teachers[1], x))
            BX.append(x)
            BY.append(y.astype(np.float32))
            for i, e in enumerate(envs):
                o, _, te, tr, _ = e.step(a[i])
                if te or tr:
                    o, _ = e.reset()
                obs[i] = o
        done += T * len(tasks)
        X, Y = np.concatenate(BX), np.concatenate(BY)
        for _ in range(grad_per_iter):
            idx = rng.integers(0, len(X), 1024)
            l, g = gfn(model, mx.array(X[idx]), mx.array(Y[idx]))
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
        if any(done - T * len(tasks) < e_ <= done for e_ in evals if e_ > 0):
            record()
    for e in envs:
        e.close()
    p = np_from_model(model)
    p["log_std"] = ((teachers[0]["log_std"] + teachers[1]["log_std"]) / 2).astype(np.float32)
    return p, curve


def stage_adapt(seed, Z, root, resdir, only=None):
    from common import load_params, evaluate, EVAL_SEEDS, TASKS, init_params, save_params
    from ppo import train
    import merge as M
    d = cdir(root, seed)
    ev_seeds = EVAL_SEEDS[:Z["n_eval"]]
    t0 = time.time()
    _, norm = load_params(os.path.join(d, "base.npz"))
    mres = json.load(open(os.path.join(resdir, f"merge_s{seed}.json")))
    path = os.path.join(resdir, f"adapt_s{seed}.json")
    out = json.load(open(path)) if os.path.exists(path) else {"seed": seed}

    def cb(step, p, nrm):
        return {tk: float(np.mean(evaluate(p, nrm, tk, ev_seeds))) for tk in TASKS}

    def run_ft(key, start, steps, every, tasks=(T0,) * 8 + (T1,) * 8):
        if key in out or (only and key not in only):
            return
        p, _, h = train(start, norm, list(tasks), steps, seed + 300, os.path.join(d, "ft_" + key),
                        eval_every=every, eval_cb=cb, quiet=True)
        out[key] = dict(evals=h["evals"], steps=steps, env_tasks=list(tasks))
        json.dump(out, open(path, "w"), indent=1)
        print(f"[s{seed}] adapt {key}: {h['evals'][-1]} ({time.time() - t0:.0f}s)", flush=True)

    S = str(Z["S"])
    for cond in ("pre", "init"):
        t_0, _ = load_params(os.path.join(d, "base.npz" if cond == "pre" else "init.npz"))
        A, _ = load_params(os.path.join(d, f"{cond}_{T0}.npz"))
        B, _ = load_params(os.path.join(d, f"{cond}_{T1}.npz"))
        lam = mres[cond]["by_length"][S]["task_arith"]["lam"]
        starts = {"theta0": t_0, "average": M.interpolate(A, B, 0.5),
                  "task_arith": M.task_arithmetic(t_0, [A, B], lam)}
        for sname, st in starts.items():
            run_ft(f"{cond}_ft_{sname}", st, Z["FT"], Z["ft_eval"])
        for sname in ("theta0", "average"):
            key = f"{cond}_distill_{sname}"
            if key not in out and (not only or key in only):
                p, curve = distill(starts[sname], [A, B], norm, Z["D"], Z["D_evals"], seed + 400, ev_seeds)
                save_params(os.path.join(d, f"distill_{cond}_{sname}.npz"), p, norm)
                out[key] = dict(curve=curve, env_steps=Z["D"], grad_steps_per_4096=50, batch=1024)
                json.dump(out, open(path, "w"), indent=1)
                print(f"[s{seed}] adapt {key}: {curve[-1]} ({time.time() - t0:.0f}s)", flush=True)
        if cond == "pre":
            # sequential transfer: specialist of one task -> PPO on the other task
            run_ft(f"pre_transfer_{T0}2{T1}", A, Z["TR"], Z["snaps"][0], tasks=(T1,) * 16)
            run_ft(f"pre_transfer_{T1}2{T0}", B, Z["TR"], Z["snaps"][0], tasks=(T0,) * 16)
    # multitask PPO from the random init with the same total env budget as base + 2 specialists
    run_ft("multitask_scratch", init_params(seed), Z["MT"], Z["mt_eval"])
    out["wall_s_last"] = time.time() - t0
    json.dump(out, open(path, "w"), indent=1)
    return seed


def _worker(args):
    stage, seed, size, root, resdir, only = args
    Z = SIZES[size]
    if stage == "train":
        return stage_train(seed, Z, root, resdir)
    if stage == "merge":
        return stage_merge(seed, Z, root, resdir)
    return stage_adapt(seed, Z, root, resdir, only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "merge", "adapt"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--root", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    size = "tiny" if args.tiny else "full"
    jobs = [(args.stage, s, size, args.root, args.results, args.only) for s in args.seeds]
    t0 = time.time()
    n = min(args.workers, 4, len(jobs))
    if n == 1:
        for j in jobs:
            _worker(j)
    else:
        with mp.get_context("spawn").Pool(n) as pool:
            for s in pool.imap_unordered(_worker, jobs):
                print(f"[pipeline] stage {args.stage} seed {s} finished at {time.time() - t0:.0f}s", flush=True)
    print(f"[pipeline] stage {args.stage} all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
