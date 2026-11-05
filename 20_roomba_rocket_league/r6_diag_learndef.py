"""Round 6 diagnostic: what did the stopped learn-def pilot (pilot 2, seed 11) learn?
Plays pilot 2's learned defender in slot 1 against the scripted attacker and pilot 1's attacker, and pilot 2's
attacker against dev opponents. Dev seeds (1501), 300 matches, heldout start set. -> results/r6_diag_learndef.json
Also measures where the learned defender sits (mean x, distance to own goal centre) and how far the pilot-2
attacker stays from the ball (parking)."""
import json
import numpy as np
from env import T_MAX, W, OUT_GOAL, OUT_TIMEOUT
import scripted
from r6_common import R6Arena, load_snap, np_policy
from r6_eval import summarize, make_opponent

def play(att, opp, n=300, seed=1501, start="goalie"):
    a = R6Arena(n, seed=seed, opponent=start); a.curriculum = 1.0; a.reset(np.ones(n, bool))
    live = np.ones(n, bool); out = np.zeros(n, int); first = np.zeros(n, int); tt = np.zeros(n, int)
    dx, dgoal, dball0 = [], [], []
    for _ in range(T_MAX):
        if live.any():
            dx.append(a.pos[live, 1, 0].mean()); dgoal.append(np.linalg.norm(a.pos[live, 1] - [W, 0], axis=1).mean())
            dball0.append(np.linalg.norm(a.pos[live, 0] - a.ball[live], axis=1).mean())
        _, d, tr, oc, _ = a.step(np.clip(att(a), -1, 1), np.clip(opp(a), -1, 1))
        fin = (d | tr) & live; out[fin] = oc[fin]; first[fin] = a.first_touch[fin]; tt[fin] = a.t[fin]
        live &= ~fin; a.bvel[fin] = 0
        if not live.any(): break
    m = summarize(dict(outcome=out, first_touch=first, t=tt, rev_frac=np.zeros(n)))
    m.update(def_mean_x=float(np.mean(dx)), def_dist_own_goal=float(np.mean(dgoal)), att_dist_ball=float(np.mean(dball0)))
    return m

dsnap = load_snap("checkpoints/r6_pilot_def_s11.pkl", "dsnap")
asnap2 = load_snap("checkpoints/r6_pilot_def_s11.pkl")
asnap1 = load_snap("checkpoints/r6_pilot_s10.pkl")
D = lambda a: np_policy(dsnap, a.obs_r6(1))
A1 = lambda a: np_policy(asnap1, a.obs_r6(0))
A2 = lambda a: np_policy(asnap2, a.obs_r6(0))
res = {}
for an, att in [("scripted", scripted.attacker), ("pilot1_att", A1), ("pilot2_att", A2)]:
    for on, opp in [("pilot2_def", D), ("goalie", scripted.goalie), ("keeper", scripted.keeper)]:
        res[f"{an}|{on}"] = m = play(att, opp)
        print(an, on, {k: round(m[k], 3) for k in ["goal", "conc_own", "timeout", "first_touch", "def_mean_x", "def_dist_own_goal", "att_dist_ball"]}, flush=True)
json.dump(res, open("results/r6_diag_learndef.json", "w"), indent=1)
