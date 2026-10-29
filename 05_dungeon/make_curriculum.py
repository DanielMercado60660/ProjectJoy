"""Collect designer levels into a curriculum file.

  ../.venv/bin/python make_curriculum.py results/curriculum_r1.npz v1_s0 [more runs...]

Uses each run's archive (levels that scored regret >= 0.5 at some generation). If the archives
hold fewer than 64 levels, the top half of each final population (by fresh regret) is added.
"""
import sys
import numpy as np

out, runs = sys.argv[1], sys.argv[2:]
parts = {k: [] for k in ("walls", "start", "goal")}
for r in runs:
    z = np.load(f"results/designer_{r}.npz")
    for k in parts:
        parts[k].append(z["arch_" + k])
n = sum(len(x) for x in parts["walls"])
if n < 64:
    for r in runs:
        z = np.load(f"results/designer_{r}.npz")
        top = np.argsort(-z["fresh_fitness"])[: len(z["walls"]) // 2]
        for k in parts:
            parts[k].append(z[k][top])
L = {k: np.concatenate(v) for k, v in parts.items()}
assert len(L["walls"]) > 0, "no curriculum levels"
np.savez_compressed(out, **L)
print(f"curriculum levels: {len(L['walls'])} (archive {n})")
