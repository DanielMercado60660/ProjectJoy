"""Held-out evaluation of all methods under matched conditions.

Seeds >= 10000 (training used 0..~1e5-range env streams seeded from 0 and resume offsets; baseline
tuning used 999-2001), per-substep peak contact force (substep_force=True), deterministic (mean)
action for PPO. Writes results/eval.json and prints markdown tables.

  ../bin/gpu_run ../.venv/bin/python evaluate.py --ckpt checkpoints/run2/latest.npz
"""
import argparse
import json
import os
import time

import numpy as np

from baselines import Gait, Posture, Random, Zero
from egg_env import FORCE_BUDGET, EggVecEnv, run_episodes
from train import load_policy


def summarize(r):
    n = len(r["success"])
    p = r["success"].mean()
    return {"n": n, "success": float(p), "success_se": float(np.sqrt(p * (1 - p) / n)),
            "drop": float(r["dropped"].mean()), "ever_success": float(r["ever_success"].mean()),
            "frac_steps_success": float(r["frac_steps_success"].mean()),
            "final_err_med": float(np.median(r["final_err"])), "err0_med": float(np.median(r["err0"])),
            "improved": float(np.mean(r["final_err"] < r["err0"])),
            "peak_force_med": float(np.median(r["peak_force"])),
            "peak_force_p90": float(np.percentile(r["peak_force"], 90)),
            "peak_force_max": float(r["peak_force"].max()),
            "over_budget": float(r["over_budget"].mean()),
            "success_within_budget": float(np.mean(r["success"] & ~r["over_budget"])),
            "full_quat_err_med": float(np.median(r["full_err"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--baselines", default="checkpoints/baselines.json")
    ap.add_argument("--baselines_fa", default="checkpoints/baselines_forceaware.json")
    ap.add_argument("--ckpt_before", default=None, help="earlier PPO checkpoint (main cells only)")
    ap.add_argument("--n_main", type=int, default=256)
    ap.add_argument("--n_other", type=int, default=128)
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for smoke")
    a = ap.parse_args()
    if a.quick:
        a.n_main = a.n_other = 8
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    B = json.load(open(a.baselines))
    ppo = load_policy(a.ckpt)
    methods = {
        "zero (hold still)": lambda: Zero(),
        "posture (tuned hold)": lambda: Posture(B["posture"]["c"]),
        f"random (sigma={B['random']['sigma']})": lambda: Random(B["random"]["sigma"], 7),
        "scripted gait (tuned)": lambda: Gait(np.array(B["gait"]["c"]), np.array(B["gait"]["A"]),
                                              np.array(B["gait"]["phi"]), B["gait"]["f"]),
        "PPO (asym. critic, touch)": lambda: ppo,
    }
    rob_methods = ["zero (hold still)", "posture (tuned hold)", "scripted gait (tuned)", "PPO (asym. critic, touch)"]
    if os.path.exists(a.baselines_fa):
        G = json.load(open(a.baselines_fa))["gait"]
        methods["scripted gait (force-aware retune)"] = lambda: Gait(
            np.array(G["c"]), np.array(G["A"]), np.array(G["phi"]), G["f"])
        rob_methods.insert(3, "scripted gait (force-aware retune)")
    if a.ckpt_before:
        ppo0 = load_policy(a.ckpt_before)
        methods["PPO before force fine-tune"] = lambda: ppo0
    envs = {n: EggVecEnv(n, seed=10000, auto_reset=False, substep_force=True) for n in {a.n_main, a.n_other}}
    out = {"ckpt": a.ckpt, "ppo_meta_state": ppo.meta["state"], "ppo_final_std": ppo.std,
           "force_budget_N": FORCE_BUDGET, "cells": []}
    t0 = time.time()

    def cell(name, mk, n, seed, tmin, tmax, fric=None, mass=None, collect=False):
        env = envs[n]
        env.theta_min, env.theta_max = tmin, tmax
        env.fixed_fric, env.fixed_mass = fric, mass
        env.rngs = [np.random.default_rng([seed, i]) for i in range(n)]
        r = run_episodes(env, mk(), n, collect_traj=collect)
        s = summarize(r)
        s.update(method=name, theta=[tmin, tmax], fric=fric, mass=mass, seed=seed)
        if collect:
            s["err_curve_mean"] = r["traj"].mean(0).tolist()
            s["peak_forces"] = r["peak_force"].tolist()
        out["cells"].append(s)
        print(f"[{time.time() - t0:5.0f}s] {name:28s} th={tmin:.2f}-{tmax:.2f} fric={fric} mass={mass}: "
              f"succ {s['success']:.3f}±{s['success_se']:.3f} drop {s['drop']:.3f} "
              f"peakF med {s['peak_force_med']:.1f} over {s['over_budget']:.3f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    # 1) main condition: training distribution of friction/mass, theta in [0.15, 0.4]
    for name, mk in methods.items():
        cell(name, mk, a.n_main, 10001, 0.15, 0.4, collect=True)
    # 2) "already satisfied" rate: targets include near-zero rotations
    for name, mk in methods.items():
        cell(name, mk, a.n_other, 10002, 0.0, 0.4)
    # 3) wider targets than the first curriculum stage
    for name, mk in methods.items():
        cell(name, mk, a.n_other, 10003, 0.4, 0.8)
        cell(name, mk, a.n_other, 10004, 0.8, np.pi / 2)
    # 4) robustness: object friction / mass outside the training range [0.7, 1.3]
    for name in rob_methods:
        for fr in [0.3, 0.5, 1.0, 2.0]:
            cell(name, methods[name], a.n_other, 10005, 0.15, 0.4, fric=fr, mass=1.0)
        for ms in [0.4, 2.0, 3.0]:
            cell(name, methods[name], a.n_other, 10006, 0.15, 0.4, fric=1.0, mass=ms)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"saved {a.out} ({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
