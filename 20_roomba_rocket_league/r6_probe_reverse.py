"""Round 6 probe of the reverse-dribbling behaviour.

    ../.venv/bin/python r6_probe_reverse.py --att run4=checkpoints/run4_tanh_relp.npz r6_s0=checkpoints/r6_s0.pkl \
        --opp goalie --out results/r6_probe_reverse.json

For each policy, 500 matches (seed 1501, dev seed), first 150 steps:
  rev        fraction of steps with mean wheel speed < -0.1 (driving backwards)
  behind     fraction of near-ball steps (centre distance < 0.13 m) with the ball behind the robot's heading
  behind_by_start  the same, split by the start heading: 'facing' (ball within +-90 deg of the nose at t=0)
             vs 'away'. A fixed rear-dribbling convention shows high 'behind' for BOTH groups (the robot turns
             its back to the ball even when it starts facing it); an end-agnostic robot shows facing << away.
  rev_first1s  reversing in the first second, by start group (does it turn round or just back up?)
  goal_by_start  goal rate over the full match by start group
"""
import argparse, json
import numpy as np
from env import T_MAX, OUT_GOAL
from r6_common import R6Arena
from r6_eval import make_attacker, make_opponent


def probe(pol, opp_name, n=500, seed=1501):
    sk, opp = make_opponent(opp_name, n, seed)
    a = R6Arena(n, seed=seed, opponent=sk)
    a.curriculum = 1.0
    a.reset(np.ones(n, bool))
    d0 = a.ball - a.pos[:, 0]
    rel = np.arctan2(d0[:, 1], d0[:, 0]) - a.th[:, 0]
    facing = np.cos(rel) > 0
    rev, near_n, behind_n, rev1 = [], np.zeros(n), np.zeros(n), np.zeros(n)
    live = np.ones(n, bool)
    out = np.zeros(n, int)
    for t in range(T_MAX):
        act = np.clip(pol(a), -1, 1)
        r = (act[:, 0] + act[:, 1]) / 2 < -0.1
        if t < 150:
            rev.append(r[live].mean() if live.any() else 0)
            d = a.ball - a.pos[:, 0]
            fwd = d[:, 0] * np.cos(a.th[:, 0]) + d[:, 1] * np.sin(a.th[:, 0])
            near = (np.linalg.norm(d, axis=1) < 0.13) & live
            near_n += near
            behind_n += near & (fwd < 0)
        if t < 20:
            rev1 += r
        _, dn, tr, oc, _ = a.step(act, opp(a))
        fin = (dn | tr) & live
        out[fin] = oc[fin]
        live &= ~fin
        a.bvel[fin] = 0
        if not live.any():
            break
    beh = lambda m: float(behind_n[m].sum() / max(near_n[m].sum(), 1))
    return dict(rev=float(np.mean(rev)), behind=beh(np.ones(n, bool)),
                behind_by_start=dict(facing=beh(facing), away=beh(~facing)),
                rev_first1s=dict(facing=float(rev1[facing].mean() / 20), away=float(rev1[~facing].mean() / 20)),
                goal_by_start=dict(facing=float((out[facing] == OUT_GOAL).mean()), away=float((out[~facing] == OUT_GOAL).mean())),
                n_facing=int(facing.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+", required=True)
    ap.add_argument("--opp", default="goalie")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for s in args.att:
        name, pol = make_attacker(s)
        for opp in args.opp.split(","):
            res[f"{name}|{opp}"] = r = probe(pol, opp)
            print(name, opp, json.dumps(r), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
