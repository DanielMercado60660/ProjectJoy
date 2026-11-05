"""Round 6 done-bar check (criteria pre-registered in STATUS.md at the recipe freeze).
(a) every seed beats the untuned scripted attacker's goal rate vs every NEW held-out opponent (heldout start set),
    paired-bootstrap 95% CI of the difference above 0. Also reported vs the CEM-tuned attacker and on corner/kickoff.
(b) no collapse vs the original goalie: goal rate above the CEM-tuned scripted attacker, non-overlapping 95% CIs.
    ../.venv/bin/python r6_donebar.py results/r6_heldout_a.json results/r6_heldout_b.json --out results/r6_donebar.json"""
import argparse, json
import numpy as np
from r6_report import arr, paired_ci

NEW = ["interceptor", "zone", "presser"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    T = {}
    for f in args.files:
        T.update(json.load(open(f))["table"])
    seeds = sorted({k.split("|")[0] for k in T if k.startswith("r6_s")})
    res, lines = {}, []
    g = lambda k: (arr(T[k]) == 1).astype(float)
    for s in seeds:
        for ref in ["scripted", "tuned"]:
            for st in ["heldout", "corner", "kickoff"]:
                for o in NEW:
                    d, ci = paired_ci(g(f"{s}|{o}|{st}"), g(f"{ref}|{o}|{st}"))
                    res[f"{s}|{o}|{st}|vs_{ref}"] = dict(diff=d, ci=ci, beats=ci[0] > 0)
        a_ok = all(res[f"{s}|{o}|heldout|vs_scripted"]["beats"] for o in NEW)
        gs, gt = T[f"{s}|goalie|heldout"], T["tuned|goalie|heldout"]
        b_ok = gs["goal_ci"][0] > gt["goal_ci"][1]
        res[f"{s}|bar_a"], res[f"{s}|bar_b"] = a_ok, b_ok
        lines.append(f"{s}: (a) {'PASS' if a_ok else 'FAIL'} "
                     + ", ".join(f"{o} {100 * res[f'{s}|{o}|heldout|vs_scripted']['diff']:+.1f} "
                                 f"[{100 * res[f'{s}|{o}|heldout|vs_scripted']['ci'][0]:+.1f}, "
                                 f"{100 * res[f'{s}|{o}|heldout|vs_scripted']['ci'][1]:+.1f}]" for o in NEW)
                     + f" | (b) {'PASS' if b_ok else 'FAIL'} goalie {100 * gs['goal']:.1f} [{100 * gs['goal_ci'][0]:.1f}, "
                       f"{100 * gs['goal_ci'][1]:.1f}] vs tuned {100 * gt['goal']:.1f} [{100 * gt['goal_ci'][0]:.1f}, {100 * gt['goal_ci'][1]:.1f}]")
        lines.append("    vs tuned (heldout set): " + ", ".join(
            f"{o} {100 * res[f'{s}|{o}|heldout|vs_tuned']['diff']:+.1f} [{100 * res[f'{s}|{o}|heldout|vs_tuned']['ci'][0]:+.1f}, "
            f"{100 * res[f'{s}|{o}|heldout|vs_tuned']['ci'][1]:+.1f}]" for o in NEW))
        lines.append("    all sets vs scripted beaten: " + ", ".join(
            f"{o}/{st}:{'Y' if res[f'{s}|{o}|{st}|vs_scripted']['beats'] else 'N'}" for o in NEW for st in ["heldout", "corner", "kickoff"]))
        lines.append("    all sets vs tuned beaten: " + ", ".join(
            f"{o}/{st}:{'Y' if res[f'{s}|{o}|{st}|vs_tuned']['beats'] else 'N'}" for o in NEW for st in ["heldout", "corner", "kickoff"]))
    json.dump(res, open(args.out, "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
