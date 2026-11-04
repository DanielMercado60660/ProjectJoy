"""Exp 1 scoring: do the hidden units Astra named implement the wall-riding?

Each named unit's tanh output is clamped to its mean value over the policy's own soft-wall laps
on train tracks 0..31 ("mean ablation"). The ablated policy is then driven on the fresh held-out
tracks under soft-wall rules (the hack's home) and strict rules. Null distribution: the same
number of units per layer drawn at random (n_null draws).

    python wsp/ablate.py --policy es_soft --units h1:3 h2:17 ... --tag read_a1 --start 90000 --n 200
"""
import argparse
import json
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WSP, WeightsController, load_weights  # noqa: E402
from racing.evaluate import run_lap, summarize  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402


def unit_means(d):
    bank = pack(make_tracks("train", 32))
    c = WeightsController(d)
    c.record = []
    run_lap(c, bank, mode="soft")
    h1 = np.concatenate([r[0] for r in c.record]); h2 = np.concatenate([r[1] for r in c.record])
    return h1.mean(0), h2.mean(0)


def parse_units(us):
    out = {"h1": [], "h2": []}
    for u in us:
        l, j = u.split(":")
        out[l].append(int(j))
    return out


def run(d, units, means, bank):
    abl = {l: {j: float(means[0 if l == "h1" else 1][j]) for j in js} for l, js in units.items() if js}
    c = WeightsController(d, ablate=abl)
    soft = summarize(run_lap(c, bank, mode="soft"))
    strict = summarize(run_lap(c, bank, mode="strict"))
    return dict(soft_completion=soft["completion"], soft_lap=soft["mean_lap_time"],
                soft_wall_steps=soft["collisions_per_lap"], soft_near_wall=soft["near_wall_time"],
                strict_completion=strict["completion"], strict_progress=strict["mean_progress"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="es_soft")
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_null", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = load_weights(os.path.join(ROOT, "checkpoints", f"{a.policy}_best.npz"))
    means = unit_means(d)
    bank = pack(make_tracks(a.split, a.n, start=a.start))
    units = parse_units(a.units)
    res = dict(policy=a.policy, units=a.units, split=a.split, start=a.start, n=a.n)
    res["none"] = run(d, {"h1": [], "h2": []}, means, bank)
    res["named"] = run(d, units, means, bank)
    # each named unit alone
    res["single"] = {u: run(d, parse_units([u]), means, bank) for u in a.units}
    rng = np.random.default_rng(a.seed)
    null = []
    for _ in range(a.n_null):
        r = {l: sorted(rng.choice(32, len(js), replace=False).tolist()) for l, js in units.items()}
        null.append(dict(units=r, **run(d, r, means, bank)))
    res["null"] = null
    ws = np.array([x["soft_wall_steps"] for x in null])
    res["named_vs_null"] = dict(
        named_soft_wall_steps=res["named"]["soft_wall_steps"], none_soft_wall_steps=res["none"]["soft_wall_steps"],
        null_median=float(np.median(ws)), null_p10=float(np.percentile(ws, 10)), null_p90=float(np.percentile(ws, 90)),
        frac_null_lower_or_equal=float((ws <= res["named"]["soft_wall_steps"]).mean()))
    os.makedirs(os.path.join(WSP, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(WSP, "results", f"ablate_{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("none", "named", "single", "named_vs_null")}, indent=1))


if __name__ == "__main__":
    main()
