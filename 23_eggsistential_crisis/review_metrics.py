"""Reviewer's re-evaluation: success-stability and gym full-quaternion metrics for every method.

Re-runs the four target-range cells of evaluate.py (same seeds 10001-10004, same n, same env
construction, so the plain "success" column must reproduce results/eval.json exactly), keeping the
per-episode error trajectories that evaluate.py throws away. Adds the wide-range gait retune
(checkpoints/baselines_wide.json) as an extra row, plus its robustness cells (seeds 10005/10006).

Per method and cell it reports:
  success        long-axis error < 0.1 rad at t = 4 s and not dropped (the project's criterion)
  held_0.5s      error < 0.1 rad at every one of the last 13 control steps (0.52 s) and not dropped:
                 rules out "passes the target at exactly 4 s" transients
  gym_fullquat   gymnasium-robotics' own criterion: full-quaternion distance < 0.1 rad (spin about
                 the egg's long axis counts), not dropped
  within budget  success and per-substep peak contact force <= 10 N for the whole episode

  ../bin/gpu_run ../.venv/bin/python review_metrics.py            # ~3-4 min under load
  ../.venv/bin/python review_metrics.py --quick                    # n=8 smoke
"""
import argparse
import json
import os
import time

import numpy as np

from baselines import Gait, Posture, Random, Zero
from egg_env import THR, EggVecEnv, run_episodes
from train import load_policy

HOLD = 13  # last 13 control steps = 0.52 s


def gait_from(path):
    G = json.load(open(path))["gait"]
    return lambda: Gait(np.array(G["c"]), np.array(G["A"]), np.array(G["phi"]), G["f"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/run3/latest.npz")
    ap.add_argument("--ckpt_before", default="checkpoints/run1/latest.npz")
    ap.add_argument("--wide", default="checkpoints/baselines_wide.json")
    ap.add_argument("--out", default="results/eval_review.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_main, n_other = (8, 8) if a.quick else (256, 128)
    B = json.load(open("checkpoints/baselines.json"))
    ppo, ppo0 = load_policy(a.ckpt), load_policy(a.ckpt_before)
    methods = {
        "zero (hold still)": lambda: Zero(),
        "posture (tuned hold)": lambda: Posture(B["posture"]["c"]),
        f"random (sigma={B['random']['sigma']})": lambda: Random(B["random"]["sigma"], 7),
        "scripted gait (tuned)": gait_from("checkpoints/baselines.json"),
        "PPO (asym. critic, touch)": lambda: ppo,
        "scripted gait (force-aware retune)": gait_from("checkpoints/baselines_forceaware.json"),
        "PPO before force fine-tune": lambda: ppo0,
    }
    if os.path.exists(a.wide):
        methods["scripted gait (wide retune, review)"] = gait_from(a.wide)
    envs = {n: EggVecEnv(n, seed=10000, auto_reset=False, substep_force=True) for n in {n_main, n_other}}
    out = {"ckpt": a.ckpt, "hold_steps": HOLD, "cells": []}
    t0 = time.time()

    def cell(name, mk, n, seed, tmin, tmax, fric=None, mass=None):
        env = envs[n]
        env.theta_min, env.theta_max = tmin, tmax
        env.fixed_fric, env.fixed_mass = fric, mass
        env.rngs = [np.random.default_rng([seed, i]) for i in range(n)]
        r = run_episodes(env, mk(), n, collect_traj=True)
        ok = ~r["dropped"]
        held = ok & (r["traj"][:, -HOLD:].max(1) < THR)
        gymq = ok & (r["full_err"] < THR)
        s = {"method": name, "theta": [tmin, tmax], "fric": fric, "mass": mass, "seed": seed, "n": n,
             "success": float(r["success"].mean()),
             "held_last_0.5s": float(held.mean()),
             "success_but_not_held": float((r["success"] & ~held).mean()),
             "gym_fullquat_success": float(gymq.mean()),
             "full_quat_err_med_deg": float(np.degrees(np.median(r["full_err"]))),
             "axis_err_med_deg": float(np.degrees(np.median(r["final_err"]))),
             "max_err_last_0.5s_med_deg": float(np.degrees(np.median(r["traj"][:, -HOLD:].max(1)))),
             "within_budget": float(np.mean(r["success"] & ~r["over_budget"])),
             "held_within_budget": float(np.mean(held & ~r["over_budget"])),
             "drop": float(r["dropped"].mean()),
             "peak_force_med": float(np.median(r["peak_force"])),
             "over_budget": float(r["over_budget"].mean())}
        if fric is None and mass is None:
            # first control step at which the per-episode error is below THR and stays there
            below = r["traj"] < THR
            settle = np.array([next((t for t in range(below.shape[1]) if below[i, t:].all()), -1)
                               for i in range(n)])
            st = settle[(settle >= 0) & ok]
            s["settle_step_med"] = float(np.median(st) + 1) if len(st) else None  # steps of 40 ms
        out["cells"].append(s)
        print(f"[{time.time() - t0:5.0f}s] {name:36s} th={tmin:.2f}-{tmax:.2f} f={fric} m={mass}: "
              f"succ {s['success']:.3f} held {s['held_last_0.5s']:.3f} gymq {s['gym_fullquat_success']:.3f} "
              f"fq_med {s['full_quat_err_med_deg']:.1f} wb {s['within_budget']:.3f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    for name, mk in methods.items():
        cell(name, mk, n_main, 10001, 0.15, 0.4)
        cell(name, mk, n_other, 10002, 0.0, 0.4)
        cell(name, mk, n_other, 10003, 0.4, 0.8)
        cell(name, mk, n_other, 10004, 0.8, np.pi / 2)
    w = "scripted gait (wide retune, review)"
    if w in methods:
        for fr in [0.3, 0.5, 1.0, 2.0]:
            cell(w, methods[w], n_other, 10005, 0.15, 0.4, fric=fr, mass=1.0)
        for ms in [0.4, 2.0, 3.0]:
            cell(w, methods[w], n_other, 10006, 0.15, 0.4, fric=1.0, mass=ms)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"saved {a.out} ({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
