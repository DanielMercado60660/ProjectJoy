"""Reviewer check: does the elapsed-time feature (obs[-1] = t/T, T = 6*cells) leak maze size?
Re-evaluates the trained bat on the held-out pools with that feature zeroed, or computed against a
fixed T = 216 (the 6x6 limit) for every size. The episode limit itself is unchanged.
  ../.venv/bin/python leak_check.py --ckpt checkpoints/bat --out results/leak_check.json
"""
import argparse, json
import numpy as np
import bat_env
from bat_env import BatEnv, MazePool, N_SECT, N_BINS
from evaluate import HELDOUT, LearnedAgent, run
from model import load

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/bat")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="results/leak_check.json")
a = ap.parse_args()
model, cfg = load(a.ckpt)
orig = BatEnv._obs
K = 2 * N_SECT * N_BINS + 6

def make(mode):
    def _obs(self, heard):
        o = orig(self, heard)
        if mode == "zero":
            o[:, K] = 0.0
        elif mode == "fixedT216":
            o[:, K] = self.t / 216.0
        return o
    return _obs

out = {"ckpt": a.ckpt, "n": a.n}
pools = {s: MazePool([s], a.n, seed=sd) for s, sd in HELDOUT.items()}
for mode in ["as_trained", "zero", "fixedT216"]:
    BatEnv._obs = make(mode)
    for s, pool in pools.items():
        for name, fn in {"bat_learned": lambda n: LearnedAgent(model, n, seed=1),
                         "bat_learned_greedy": lambda n: LearnedAgent(model, n, greedy=True)}.items():
            r = run(fn, pool)
            out.setdefault(mode, {}).setdefault(f"size{s}", {})[name] = r
            print(mode, s, name, f"succ {r['success']:.3f} pulses {r['pulses_per_ep']:.1f} steps {r['steps']:.1f} coll/step {r['collision_rate']:.4f}", flush=True)
BatEnv._obs = orig
json.dump(out, open(a.out, "w"), indent=2)
