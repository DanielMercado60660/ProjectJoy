"""Round 6 round-robin + Elo (Bradley-Terry) between attackers (slot 0) and opponents (slot 1).

    ../bin/gpu_run ../.venv/bin/python r6_elo.py --seed-base <post-freeze seed> --out results/r6_elo.json \
        --att r6_s0=checkpoints/r6_s0.pkl ... --arch checkpoints/r6_s0_arch,checkpoints/r6_s1_arch,...

Match score for the attacker: goal 1, timeout 0.5, conceded/own goal 0. One rating per player; the model is
E[score] = 1 / (1 + 10^((R_def - R_att)/400)), fitted by maximum likelihood (fractional outcomes) with the
scripted attacker anchored at 1000. 95% CIs: bootstrap over matches within each pair (B resamples, refit).
Learned archive snapshots appear on BOTH sides: as attackers (slot 0) and, mirrored, as opponents (slot 1).
"""
import argparse, glob, json, os, time
import numpy as np
from scipy.optimize import minimize
from env import OUT_GOAL, OUT_TIMEOUT
import r6_opponents as ro
from r6_eval import make_attacker, run_matches
from r6_common import load_snap, r6_policy

SCRIPTED_OPPS = list(ro.HELDOUT) + ["goalie", "keeper", "chaser"]


def pick_snaps(d, k):
    fs = sorted(glob.glob(os.path.join(d, "snap_*.pkl")))
    if len(fs) <= k:
        return fs
    return [fs[int(round(i))] for i in np.linspace(len(fs) / k - 1, len(fs) - 1, k)]


def fit(S, N, na, nd, anchor):
    """S[a,d] mean score, N[a,d] match counts. Returns ratings (na + nd)."""
    k = np.log(10) / 400
    mask = N > 0

    def nll(r):
        ra, rd = r[:na], r[na:]
        p = 1 / (1 + np.exp(-k * (ra[:, None] - rd[None, :])))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        l = -(N * (S * np.log(p) + (1 - S) * np.log(1 - p)))[mask].sum()
        return l + 1e-6 * np.sum((r - 1000) ** 2)       # tiny ridge fixes the gauge; shifted to the anchor below

    r0 = np.full(na + nd, 1000.0)
    res = minimize(nll, r0, method="L-BFGS-B")
    r = res.x
    return r - (r[anchor] - 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="*", default=[])
    ap.add_argument("--arch", default="", help="comma list of *_arch dirs; k snapshots each join both sides")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--opps", default=",".join(SCRIPTED_OPPS), help="scripted slot-1 opponents")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    atts = [make_attacker(s) for s in ["scripted"] + args.att]
    opps = args.opps.split(",")
    for s in args.att:                       # final learned checkpoints also play slot 1
        name, _, path = s.partition("=")
        if path.endswith(".pkl"):
            opps.append(f"snap:{path}")
    for d in [x for x in args.arch.split(",") if x]:
        tag = os.path.basename(d.rstrip("/")).replace("_arch", "")
        for f in pick_snaps(d, args.k):
            nm = f"{tag}@{os.path.basename(f)[5:-5]}"
            atts.append((nm, r6_policy(load_snap(f), 0)))
            opps.append(f"snap:{f}")
    na, nd = len(atts), len(opps)
    scores = {}
    S = np.zeros((na, nd))
    N = np.zeros((na, nd))
    for i, (an, pol) in enumerate(atts):
        for j, on in enumerate(opps):
            res = run_matches(pol, on, "heldout", args.n, args.seed_base)
            o = res["outcome"]
            sc = np.where(o == OUT_GOAL, 1.0, np.where(o == OUT_TIMEOUT, 0.5, 0.0))
            scores[(i, j)] = sc
            S[i, j], N[i, j] = sc.mean(), len(sc)
        print(f"{an:24s} " + " ".join(f"{S[i, j]:.2f}" for j in range(nd)), flush=True)
    R = fit(S, N, na, nd, 0)
    rng = np.random.default_rng(0)
    boots = []
    for b in range(args.boot):
        Sb = np.array([[scores[(i, j)][rng.integers(0, args.n, args.n)].mean() for j in range(nd)] for i in range(na)])
        boots.append(fit(Sb, N, na, nd, 0))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    onames = [o if not o.startswith("snap:") else
              (os.path.basename(os.path.dirname(o[5:])).replace("_arch", "") + "@" + os.path.basename(o)[5:-5]
               if "_arch" in o else os.path.basename(o[5:]).replace(".pkl", "") + "@final")
              for o in opps]
    out = dict(seed_base=args.seed_base, n=args.n, boot=args.boot,
               attackers=[dict(name=a[0], elo=float(R[i]), ci=[float(lo[i]), float(hi[i])]) for i, a in enumerate(atts)],
               opponents=[dict(name=onames[j], elo=float(R[na + j]), ci=[float(lo[na + j]), float(hi[na + j])])
                          for j in range(nd)],
               score_matrix=S.tolist(), att_names=[a[0] for a in atts], opp_names=onames, wall=time.time() - t0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for e in sorted(out["attackers"], key=lambda e: -e["elo"]):
        print(f"ATT {e['name']:24s} {e['elo']:7.0f} [{e['ci'][0]:.0f}, {e['ci'][1]:.0f}]")
    for e in sorted(out["opponents"], key=lambda e: -e["elo"]):
        print(f"OPP {e['name']:24s} {e['elo']:7.0f} [{e['ci'][0]:.0f}, {e['ci'][1]:.0f}]")
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
