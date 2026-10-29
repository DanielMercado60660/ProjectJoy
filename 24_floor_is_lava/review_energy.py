"""Reviewer: energy per metre AND per second on layouts both methods completed (REVIEW.md).

Reads results/<name>_episodes.json (clean runs, same held-out seeds for every method) and writes
results/review_energy.json. Per second = mean mechanical power, energy_total / episode time.
Usage: ../.venv/bin/python review_energy.py eval_planner4_full review_eval_planner5_full
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main(names):
    os.chdir(HERE)
    out = {}
    for name in names:
        E = json.load(open(f"results/{name}_episodes.json"))
        pp = E["residual_ppo"]["clean"]
        for base in [k for k in E if k != "residual_ppo"]:
            bb = E[base]["clean"]
            both = [(a, b) for a, b in zip(pp, bb) if a["success"] and b["success"]]
            if not both:
                continue
            f = lambda key: {
                "J_per_m": boot([r["energy_total"] / r["progress"] for r in key]),
                "W_mean_power": boot([r["energy_total"] / r["t"] for r in key]),
                "J_total": boot([r["energy_total"] for r in key]),
                "time_s": boot([r["t"] for r in key]),
                "speed_mps": boot([r["progress"] / r["t"] for r in key]),
            }
            out[f"{name}:ppo_vs_{base}"] = dict(n_both_completed=len(both), residual_ppo=f([a for a, _ in both]),
                                                 baseline=f([b for _, b in both]))
            print(name, base, json.dumps(out[f"{name}:ppo_vs_{base}"], indent=None)[:400])
    json.dump(out, open("results/review_energy.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["eval_planner4_full"])
