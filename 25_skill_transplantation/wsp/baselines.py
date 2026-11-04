"""Matched-budget gradient arms for the WSP comparison (runner: Claude).

For a given env-step budget S per training seed:
  ppo_ft_avg_<S>  multitask PPO fine-tune from the naive average (8 fwd + 8 bwd envs), the same
                  ppo.train call the adapt stage used for `pre_ft_average`, just with S steps
  dagger_<S>      DAgger distillation of both specialists from theta0 (pipeline.distill), S student steps
Outputs wsp/baselines/<name>_s<seed>.npz (+ PPO resume state under wsp/baselines/ckpt/).

  ../bin/gpu_run ../.venv/bin/python wsp/baselines.py --steps 73728 --seeds 0 1 2 3
"""
import argparse
import multiprocessing as mp
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)


def run(args):
    seed, steps = args
    import numpy as np
    from common import load_params, save_params
    import merge as M
    from ppo import train
    from pipeline import distill
    d = os.path.join(PROJ, "checkpoints", f"s{seed}")
    out = os.path.join(HERE, "baselines")
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    t0, norm = load_params(os.path.join(d, "base.npz"))
    A, _ = load_params(os.path.join(d, "pre_fwd.npz"))
    B, _ = load_params(os.path.join(d, "pre_bwd.npz"))
    p, _, h = train(M.interpolate(A, B, 0.5), norm, ["fwd"] * 8 + ["bwd"] * 8, steps, seed + 300,
                    os.path.join(out, "ckpt", f"ppo_ft_avg_{steps}_s{seed}"), quiet=True)
    save_params(os.path.join(out, f"ppo_ft_avg_{steps}_s{seed}.npz"), p, norm, env_steps=steps)
    q, _ = distill(t0, [A, B], norm, steps, [], seed + 400, [])
    save_params(os.path.join(out, f"dagger_{steps}_s{seed}.npz"), q, norm, env_steps=steps)
    return seed, int(np.ceil(steps / 4096) * 4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(min(4, len(a.seeds))) as pool:
        for s, used in pool.imap_unordered(run, [(s, a.steps) for s in a.seeds]):
            print(f"s{s}: steps {a.steps} (env steps actually run, rounded up to 4096: {used})", flush=True)


if __name__ == "__main__":
    main()
