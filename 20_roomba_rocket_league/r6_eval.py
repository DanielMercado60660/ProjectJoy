"""Round 6 evaluation with bootstrap CIs.

    ../.venv/bin/python r6_eval.py --mode check  --out results/r6_check_heldout_opponents.json
    ../.venv/bin/python r6_eval.py --mode dev    --att r6_s0=checkpoints/r6_s0.pkl ... --out results/r6_dev_*.json
    ../.venv/bin/python r6_eval.py --mode heldout --seed-base <drawn after freeze> --att ... --out results/r6_heldout.json

Modes (the gate that keeps held-out opponents out of development):
  check    scripted + random attackers only, vs the 3 held-out opponents (validity check before training)
  dev      any attacker vs the TRAINING-visible opponents (goalie, keeper, chaser, pool families), dev seeds
  heldout  any attacker vs held-out + legacy opponents, seeds given on the command line (fresh, post-freeze)
Attackers: scripted, random, tuned=<json>, run4=<npz (ppo.py checkpoint)>, <name>=<r6 .pkl>.
Each cell: n matches on one start set; deterministic (mean-action) learned policies.
"""
import argparse, json, os, time
import numpy as np
from env import T_MAX, OUT_GOAL, OUT_CONCEDED, OUT_OWN_GOAL, OUT_TIMEOUT
import scripted
import r6_opponents as ro
from r6_common import R6Arena, r6_policy, load_snap, np_policy, KINDS, K

START_SETS = ["heldout", "corner", "kickoff"]
LEGACY = ["goalie", "keeper", "chaser"]
HELDOUT = list(ro.HELDOUT)


def twin(pol):
    """Mirror twin: the same policy using the other end of the robot (heading + pi, (l, r) -> (-r, -l))."""
    def f(a):
        th = a.th.copy()
        a.th[:, 0] = th[:, 0] + np.pi
        pa = a.prev_act.copy() if hasattr(a, "prev_act") else None
        if pa is not None:                  # the r6 obs includes the own previous action: map it into the twin frame
            a.prev_act[:, 0] = np.stack([-pa[:, 0, 1], -pa[:, 0, 0]], 1)
        try:
            act = np.clip(pol(a), -1, 1)
        finally:
            a.th = th
            if pa is not None:
                a.prev_act = pa
        return np.stack([-act[:, 1], -act[:, 0]], 1)
    return f


def make_attacker(spec):
    name, _, path = spec.partition("=")
    if name.startswith("twin_"):
        base, pol = make_attacker(name[5:] + "=" + path)
        return name, twin(pol)
    if name == "scripted":
        return "scripted", scripted.attacker
    if name == "random":
        rng = np.random.default_rng(7)
        return "random", lambda a: rng.uniform(-1, 1, (a.n, 2))
    if name == "tuned":
        P = json.load(open(path))["best"]
        return "tuned", lambda a: ro.attacker_param(a, P)
    if name == "run4":
        from ppo import load_ckpt, make_policy
        m, nrm, _ = load_ckpt(path)
        return "run4", make_policy(m, nrm)
    return name, r6_policy(load_snap(path), 0)


def make_opponent(name, n, seed):
    """Returns (start_kind, fn(arena)->(n,2))."""
    if name in LEGACY:
        return name, scripted.OPPONENTS[name]
    if name in ro.HELDOUT:
        return "goalie", ro.HELDOUT[name]
    if name in ro.POOL_FAMILIES:          # dev: fixed random params per match, seeded
        P = ro.sample_params(np.random.default_rng(seed + 77), n)
        sub = ro.family_params(name, P)
        fn = ro.POOL_FAMILIES[name]

        def f(a):
            act = fn(a.delayed_view(np.arange(a.n), P["delay"]), sub) * P["speed"][:, None]
            return np.clip(act, -1, 1)
        return ("chaser" if name == "chaser_r" else "goalie"), f
    if name.startswith("dsnap:"):         # final learned defender of an r6 checkpoint
        snap = load_snap(name[6:], "dsnap")
        return "goalie", lambda a: np_policy(snap, a.obs_r6(1))
    if name.startswith("snap:"):          # learned snapshot (attacker or archived defender) driving slot 1
        snap = load_snap(name[5:])
        return "goalie", lambda a: np_policy(snap, a.obs_r6(1))
    raise KeyError(name)


def run_matches(att, opp_name, start_set, n, seed, record=()):
    start_kind, opp = make_opponent(opp_name, n, seed)
    a = R6Arena(n, seed=seed, opponent=start_kind)
    a.start_mode = "train" if start_set == "heldout" else start_set
    a.curriculum = 1.0
    a.reset(np.ones(n, dtype=bool))
    live = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=np.int64)
    first = np.zeros(n, dtype=np.int64)
    tt = np.zeros(n, dtype=np.int64)
    rev = np.zeros(n)
    steps = np.zeros(n)
    frames = {i: [] for i in record}
    for _ in range(T_MAX):
        for i in frames:
            if live[i]:
                frames[i].append(a.snapshot(i))
        act0 = np.clip(att(a), -1, 1)
        rev += live * ((act0[:, 0] + act0[:, 1]) / 2 < -0.1)
        steps += live
        _, d, tr, outc, _ = a.step(act0, opp(a))
        fin = (d | tr) & live
        out[fin] = outc[fin]
        for i in frames:
            if fin[i]:
                frames[i].append(a.snapshot(i))
        first[fin] = a.first_touch[fin]
        tt[fin] = a.t[fin]
        live &= ~fin
        a.bvel[fin] = 0
        if not live.any():
            break
    res = dict(outcome=out, first_touch=first, t=tt, rev_frac=rev / np.maximum(steps, 1))
    if record:
        res["frames"] = frames
    return res


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    bs = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarize(res):
    o = res["outcome"]
    g = (o == OUT_GOAL).astype(float)
    c = ((o == OUT_CONCEDED) | (o == OUT_OWN_GOAL)).astype(float)
    return dict(n=len(o), goal=float(g.mean()), goal_ci=boot_ci(g), conc_own=float(c.mean()), conc_own_ci=boot_ci(c),
                conceded=float((o == OUT_CONCEDED).mean()), own_goal=float((o == OUT_OWN_GOAL).mean()),
                timeout=float((o == OUT_TIMEOUT).mean()), first_touch=float((res["first_touch"] == 0).mean()),
                rev_frac=float(res["rev_frac"].mean()),
                score=float(np.mean(np.where(o == OUT_GOAL, 1.0, np.where(o == OUT_TIMEOUT, 0.5, 0.0)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "dev", "heldout"])
    ap.add_argument("--att", nargs="*", default=["scripted"])
    ap.add_argument("--opps", default=None, help="comma list; defaults per mode")
    ap.add_argument("--sets", default="heldout,corner,kickoff")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--save-outcomes", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "check":
        atts = ["scripted", "random"]
        opps = HELDOUT
        base = 2001
    elif args.mode == "dev":
        atts = args.att
        opps = args.opps.split(",") if args.opps else LEGACY + list(ro.POOL_FAMILIES)
        bad = [o for o in opps if o in ro.HELDOUT]
        assert not bad, f"held-out opponents are not allowed in dev mode: {bad}"
        base = 1501
    else:
        assert args.seed_base is not None, "heldout mode needs --seed-base (drawn after the recipe freeze)"
        atts = args.att
        opps = args.opps.split(",") if args.opps else HELDOUT + LEGACY
        base = args.seed_base
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table = {}
    t0 = time.time()
    for spec in atts:
        name, pol = make_attacker(spec)
        for opp in opps:
            for si, s in enumerate(args.sets.split(",")):
                res = run_matches(pol, opp, s, args.n, base + 10 * START_SETS.index(s))
                m = summarize(res)
                if args.save_outcomes:
                    m["outcomes"] = "".join(str(int(x)) for x in res["outcome"])   # per-match, for paired bootstrap
                table[f"{name}|{opp}|{s}"] = m
                print(f"{name:12s} {opp:12s} {s:8s} goal {m['goal']:.3f} [{m['goal_ci'][0]:.3f},{m['goal_ci'][1]:.3f}] "
                      f"conc+own {m['conc_own']:.3f} to {m['timeout']:.3f} 1st {m['first_touch']:.2f} "
                      f"rev {m['rev_frac']:.2f}", flush=True)
    json.dump(dict(mode=args.mode, seed_base=base, n=args.n, table=table, wall=time.time() - t0), open(args.out, "w"),
              indent=1)
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
