"""Round 6 tables: goal rate [95% CI], conceded+own [CI], and the paired difference vs the scripted attacker
(same starts for every attacker, so the bootstrap resamples start indices jointly).

    ../.venv/bin/python r6_report.py results/r6_heldout.json [--sets heldout] [--ref scripted]
"""
import argparse, json
import numpy as np


def arr(m):
    return np.array([int(c) for c in m["outcomes"]])


def paired_ci(a, b, B=4000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), (B, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sets", default="heldout,corner,kickoff")
    ap.add_argument("--ref", default="scripted")
    args = ap.parse_args()
    T = {}
    for f in args.files:
        T.update(json.load(open(f))["table"])
    atts = list(dict.fromkeys(k.split("|")[0] for k in T))
    opps = list(dict.fromkeys(k.split("|")[1] for k in T))
    for s in args.sets.split(","):
        print(f"\n**Start set `{s}`**: goal % [95% CI] / conceded+own % [95% CI]; Δ = goal minus {args.ref} (paired, 95% CI)\n")
        print("| attacker | " + " | ".join(opps) + " |")
        print("|---" * (len(opps) + 1) + "|")
        for a in atts:
            cells = []
            for o in opps:
                k = f"{a}|{o}|{s}"
                if k not in T:
                    cells.append("–")
                    continue
                m = T[k]
                c = (f"{100 * m['goal']:.1f} [{100 * m['goal_ci'][0]:.1f}, {100 * m['goal_ci'][1]:.1f}] / "
                     f"{100 * m['conc_own']:.1f} [{100 * m['conc_own_ci'][0]:.1f}, {100 * m['conc_own_ci'][1]:.1f}]")
                rk = f"{args.ref}|{o}|{s}"
                if a != args.ref and rk in T and "outcomes" in m and "outcomes" in T[rk]:
                    d, ci = paired_ci((arr(m) == 1).astype(float), (arr(T[rk]) == 1).astype(float))
                    c += f"<br>Δ {100 * d:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
                cells.append(c)
            print(f"| {a} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
