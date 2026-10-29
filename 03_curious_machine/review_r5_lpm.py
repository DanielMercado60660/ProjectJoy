"""Claude reviewer's LPM diagnosis for Round 5 Turn 2 (NumPy only, no MLX, no training, <1 s).

Writes two receipts:
  results/r5/review_lpm_training.json  per-cell phase summaries of logs/r5/t2_ac/*/updates.jsonl
                                        (entropy, reward sign, log error, rooms, TV) + LPM collapse onset
  results/r5/review_discovery_vs_tasks.json  per-episode Spearman: discovery vs tasks solved (learned cells only)
  results/r5/review_lpm_probe.json     forward pass of each final LPM checkpoint's aux nets on saved replays:
                                        reward = g(o,a) - log(MSE(f(o,a), phi(o')) + 1e-6), unclipped, unscaled
Usage: ../.venv/bin/python review_r5_lpm.py
"""
import json
import math
import pickle
from pathlib import Path
import numpy as np

R = Path("results/r5/t2_ac"); L = Path("logs/r5/t2_ac")
LEARNED = ("lpm", "rnd_ep", "disagree", "ngu")
WORLDS = ("plain", "tv"); SEEDS = (610, 611, 612)
PHASES = ((1, 6), (7, 32), (33, 128), (129, 256), (257, 366))
KEYS = ("policy_entropy", "reward_mean", "progress_positive", "log_error", "expected_log_error",
        "reward_scale", "reward_tv_mean", "reward_non_tv_mean", "rooms", "interactions", "coverage", "tv_steps")


def training_summary():
    out = {}
    for m in LEARNED:
        for w in WORLDS:
            for s in SEEDS:
                rows = [json.loads(l) for l in (L / f"{m}_{w}_s{s}" / "updates.jsonl").open()]
                cell = dict(updates=len(rows), steps=rows[-1]["steps"], train_seconds=rows[-1]["elapsed_seconds"], phases={})
                for a, b in PHASES:
                    sel = [r for r in rows if a <= r["update"] <= b]
                    cell["phases"][f"u{a}-{b}"] = {k: float(np.mean(v)) for k in KEYS
                                                   if (v := [r[k] for r in sel if r.get(k) is not None])}
                tv = [(r.get("tv_steps") or 0.0, r["update"]) for r in rows]
                cell["max_training_tv_steps"], cell["max_tv_update"] = max(tv)
                if m == "lpm":
                    ready = next(r["update"] for r in rows if r["progress_ready"])
                    low = next((r["update"] for r in rows if r["policy_entropy"] < 0.05), None)
                    cell["progress_ready_update"] = ready
                    cell["first_update_entropy_below_0.05"] = low
                    cell["min_log_error"] = min(r["log_error"] for r in rows)
                out[f"{m}_{w}_s{s}"] = cell
    return dict(kind="Reviewer phase summary of training logs; means over updates in each phase",
                log_epsilon_floor=math.log(1e-6), sources=f"{L}/<cell>/updates.jsonl", cells=out)


def mlp(p, pre, x):
    n = 0
    while f"{pre}.layers.{n}.weight" in p:
        n += 1
    for j in range(n):
        x = x @ p[f"{pre}.layers.{j}.weight"].T + p[f"{pre}.layers.{j}.bias"]
        if j < n - 1:
            x = np.where(x > 0, x, 0.01 * x)  # mlx.nn.leaky_relu default slope
    return x


def lpm_reward(p, obs, act):
    x, xn = obs[:-1].astype(np.float32), obs[1:].astype(np.float32)
    inp = np.concatenate([x, np.eye(5, dtype=np.float32)[act]], 1)
    le = np.log(np.mean((mlp(p, "current", inp) - mlp(p, "phi", xn)) ** 2, 1) + 1e-6)
    g = mlp(p, "error_model", inp)[:, 0]
    return g - le, le, g


def probe():
    out = {}
    for w in WORLDS:
        for s in SEEDS:
            ck = Path(f"checkpoints/r5_t2_ac/lpm_{w}_s{s}.pkl")
            p = pickle.loads(ck.read_bytes())["aux"]
            own = np.load(R / f"lpm_{w}_s{s}.npz")
            rooms_own = {int(own["room"][tuple(q)]) for q in own["explore_pos"]}
            r, le, g = lpm_reward(p, own["observations"], own["actions"])
            row = dict(checkpoint=str(ck), own_rooms=sorted(rooms_own), own_reward=float(r.mean()),
                       own_log_error=float(le.mean()), own_expected=float(g.mean()))
            for other in ("frontier", "rnd_ep"):
                o = np.load(R / f"{other}_{w}_s610.npz")
                r, le, g = lpm_reward(p, o["observations"], o["actions"])
                rooms_t = np.array([int(o["room"][tuple(q)]) for q in o["explore_pos"][1:]])
                novel = ~np.isin(rooms_t, sorted(rooms_own))
                part = lambda v, m: float(v[m].mean()) if m.any() else None
                row[f"on_{other}_s610_replay"] = dict(
                    n_same_rooms=int((~novel).sum()), n_other_rooms=int(novel.sum()),
                    same_rooms_reward=part(r, ~novel), other_rooms_reward=part(r, novel),
                    other_rooms_frac_negative=part((r < 0).astype(float), novel),
                    same_rooms_log_error=part(le, ~novel), other_rooms_log_error=part(le, novel),
                    other_rooms_expected_log_error=part(g, novel))
            out[f"lpm_{w}_s{s}"] = row
    return dict(kind="NumPy forward pass of final LPM aux weights (no MLX, no training)",
                reward="g(o,a) - log(MSE(f(o,a),phi(o')) + 1e-6); raw, before clip and adaptive scale",
                rooms="'same' = rooms the checkpoint itself visited in its first fresh test episode (85400); "
                      "'other' = remaining rooms; transitions taken from saved first-episode replays of "
                      "frontier and rnd_ep (training seed 610 receipts, .npz)", results=out)


def discovery_vs_tasks():
    from scipy.stats import spearmanr
    mech, rooms, solved = [], [], []
    for m in LEARNED:
        for w in WORLDS:
            for s in SEEDS:
                d = json.loads((R / f"{m}_{w}_s{s}.json").read_text())
                for e in d["episodes"]:
                    mech.append(e["interactions"]); rooms.append(e["rooms"])
                    solved.append(sum(t["success"] for t in d["tasks"] if t["seed"] == e["seed"]))
    rm, rr = spearmanr(mech, solved), spearmanr(rooms, solved)
    return dict(kind="Per-episode Spearman correlation, learned methods only (4 methods x 2 worlds x 3 seeds x 16 episodes); "
                     "classical rows excluded because their three seeds are identical repeats",
                episodes=len(solved), mechanisms_vs_tasks_solved=dict(rho=float(rm.statistic), p=float(rm.pvalue)),
                rooms_vs_tasks_solved=dict(rho=float(rr.statistic), p=float(rr.pvalue)), sources=str(R) + "/<method>_<world>_s61*.json")


if __name__ == "__main__":
    for name, fn in (("review_lpm_training.json", training_summary), ("review_lpm_probe.json", probe),
                     ("review_discovery_vs_tasks.json", discovery_vs_tasks)):
        Path("results/r5", name).write_text(json.dumps(fn(), indent=1))
        print("WROTE", Path("results/r5", name))
