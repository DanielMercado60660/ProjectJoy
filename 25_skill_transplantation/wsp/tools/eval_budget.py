"""Budgeted evaluation call for the WSP (weight-space programming) attempts. Written by Claude (runner).

One call = one candidate network for one training seed, scored on BOTH tasks, 3 episodes each,
on the fixed TRAIN reset bank below (6000 env steps). Deterministic (mean action), 1000-step
episodes, same reward as the rest of project 25. Returns scalars only, no states.

  python wsp/tools/eval_budget.py --attempt <tag> --seed <0-3> --cand <path/to/merged_sK.npz>

Every call is appended to wsp/attempts/<tag>/ledger.jsonl. The call is refused once the attempt
has used its budget (wsp/tools/budgets.json). Attempts must not edit this file, budgets.json,
ref_train.json or the ledger.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)
import numpy as np  # noqa: E402
from common import Normalizer, evaluate  # noqa: E402

TRAIN_BANK = [60000, 60001, 60002]
SHAPES = {"a1.weight": (128, 19), "a1.bias": (128,), "a2.weight": (128, 128), "a2.bias": (128,),
          "a3.weight": (6, 128), "a3.bias": (6,)}


def load_cand(path):
    z = np.load(path)
    p = {}
    for k, s in SHAPES.items():
        if k not in z:
            raise SystemExit(f"candidate missing key {k}")
        a = np.asarray(z[k], np.float32)
        if a.shape != s:
            raise SystemExit(f"candidate {k} has shape {a.shape}, expected {s}")
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"candidate {k} has non-finite values")
        p[k] = a
    return p


def norm_for(seed):
    z = np.load(os.path.join(PROJ, "wsp", "data", f"s{seed}", "theta0.npz"))
    return Normalizer(z["norm_mean"], z["norm_var"], 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--cand", required=True)
    a = ap.parse_args()
    budgets = json.load(open(os.path.join(HERE, "budgets.json")))
    if a.attempt not in budgets:
        raise SystemExit(f"attempt {a.attempt} has no evaluation budget (one-shot attempts get none)")
    adir = os.path.join(PROJ, "wsp", "attempts", a.attempt)
    ledger = os.path.join(adir, "ledger.jsonl")
    used = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
    N = budgets[a.attempt]
    if used >= N:
        raise SystemExit(f"BUDGET EXHAUSTED: {used}/{N} evaluation calls used")
    p = load_cand(a.cand)
    norm = norm_for(a.seed)
    ref = json.load(open(os.path.join(HERE, "ref_train.json")))[str(a.seed)]
    t0 = time.time()
    out = {"call": used + 1, "budget": N, "seed": a.seed, "cand": os.path.abspath(a.cand),
           "cand_sha256": hashlib.sha256(open(a.cand, "rb").read()).hexdigest()[:16]}
    for task in ("fwd", "bwd"):
        ret, xv = evaluate(p, norm, task, TRAIN_BANK, return_xvel=True)
        out[task] = {"mean_return": round(float(ret.mean()), 1), "returns": [round(float(r), 1) for r in ret],
                     "mean_x_velocity": round(float(xv.mean()), 3),
                     "retained_vs_specialist": round(float(ret.mean() / ref[task]), 4)}
    out["mean_retained"] = round((out["fwd"]["retained_vs_specialist"] + out["bwd"]["retained_vs_specialist"]) / 2, 4)
    out["env_steps"] = 6000
    out["wall_s"] = round(time.time() - t0, 2)
    out["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(adir, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out, indent=1))
    print(f"[{used + 1}/{N} evaluation calls used]")


if __name__ == "__main__":
    main()
