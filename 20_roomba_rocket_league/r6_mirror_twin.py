"""Round 6 reverse-dribbling test: the 'mirror twin' of a policy.
A differential-drive disc with equal forward/reverse speed is symmetric under (heading -> heading + pi,
(left, right) -> (-right, -left)): both describe the same motion. The twin evaluates the policy as if the robot
were turned round and maps the action back, i.e. it is the same strategy using the other end of the robot.
If backwards driving is just an arbitrary symmetry-breaking convention, the twin drives FORWARD into the ball and
scores the same. Dev opponents (goalie, keeper), seed 1501, 500 matches.
    ../.venv/bin/python r6_mirror_twin.py --att run4=checkpoints/run4_tanh_relp.npz --out results/r6_mirror_twin.json"""
import argparse, json
import numpy as np
from r6_eval import make_attacker, run_matches, summarize, twin
from r6_probe_reverse import probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+", required=True)
    ap.add_argument("--opps", default="goalie,keeper")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for s in args.att:
        name, pol = make_attacker(s)
        for nm, p in [(name, pol), (name + "_twin", twin(pol))]:
            for o in args.opps.split(","):
                m = summarize(run_matches(p, o, "heldout", 500, 1501))
                pr = probe(p, o)
                res[f"{nm}|{o}"] = dict({k: m[k] for k in ["goal", "goal_ci", "conc_own", "rev_frac"]}, behind=pr["behind"])
                print(f"{nm:14s} {o:8s} goal {m['goal']:.3f} [{m['goal_ci'][0]:.3f},{m['goal_ci'][1]:.3f}] rev {m['rev_frac']:.2f} "
                      f"ball-behind (near-ball steps) {pr['behind']:.2f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
