"""Round-robin tournament, cross-play, robustness and diversity for a trained checkpoint.

Everything uses fixed EVAL seeds (5000+), disjoint from training env seeds (0..15) and the script-tuning
seeds (100, 101). Every bout alternates which side (robot a/b) each entrant drives. PPO entrants act with
the mean action (deterministic). Results -> results/tournament.json (or --out).

  ../bin/gpu_run ../.venv/bin/python tournament.py --ckpt checkpoints/main.pkl --n 200
"""
import argparse, itertools, json, os, pickle, time
from multiprocessing import Pool
import numpy as np
from sumo_env import SumoEnv, make_script, Idle, RandomPolicy, SCRIPT_CLASSES, MAX_STEPS
from ppo import NumpyPolicy, load_tuned

CK = None
TUNED = None


def init(ckpt):
    global CK, TUNED
    CK = pickle.load(open(ckpt, "rb"))
    TUNED = load_tuned()


def make(spec, seed):
    if spec == "ppo":
        return NumpyPolicy(CK["final_params"], name="ppo")
    if spec.startswith("snap"):          # snap<k>: k-th archive entry
        return NumpyPolicy(CK["archive"][int(spec[4:])]["params"], name=spec)
    if spec == "idle":
        return Idle()
    if spec == "random":
        return RandomPolicy(seed)
    return make_script(spec, TUNED.get(spec))


def run_cell(task):
    a, b, n, seed, R, fric = task
    env = SumoEnv(seed=seed)
    W = L = D = tip_a = tip_b = out_a = out_b = 0
    lens, contact = [], []
    for i in range(n):
        pa, pb = make(a, seed * 7 + i), make(b, seed * 11 + i)
        if i % 2 == 0:
            r = play_bout_any(env, pa, pb, R, fric)
            o, ta, tb, oa, ob = r["outcome"], r["tip_a"], r["tip_b"], r["out_a"], r["out_b"]
        else:
            r = play_bout_any(env, pb, pa, R, fric)
            o, ta, tb, oa, ob = -r["outcome"], r["tip_b"], r["tip_a"], r["out_b"], r["out_a"]
        W += o == 1; L += o == -1; D += o == 0
        tip_a += ta; tip_b += tb; out_a += oa; out_b += ob
        lens.append(r["steps"]); contact.append(r["contact_frac"])
    return dict(a=a, b=b, n=n, seed=seed, R=R, fric=fric, W=int(W), L=int(L), D=int(D),
                tip_a=int(tip_a), tip_b=int(tip_b), out_a=int(out_a), out_b=int(out_b),
                mean_len=float(np.mean(lens)), contact_frac=float(np.mean(contact)),
                win_ci=wilson(W, n), loss_ci=wilson(L, n), score=float((W - L) / n))


def play_bout_any(env, pa, pb, R, fric):
    from sumo_env import play_bout
    return play_bout(env, pa, pb, R=R, fric=fric)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - h), 4), round(float(c + h), 4)]


def behaviour(task):
    """Per-snapshot behaviour descriptors from bouts vs the tuned charger, plus probe-set actions."""
    spec, n, seed, probe = task
    env = SumoEnv(seed=seed)
    pol = make(spec, seed)
    fwd, turn, margin, dist = [], [], [], []
    for i in range(n):
        oa, ob = env.reset()
        opp = make("charger", 0)
        while True:
            act = pol(oa)
            fwd.append((act[0] + act[1]) / 2); turn.append(abs(act[1] - act[0]) / 2)
            margin.append(oa[5]); dist.append(oa[8])
            (oa, ob), out, info = env.step(act, opp(ob))
            if out is not None:
                break
    probe_act = pol.mean(probe) if hasattr(pol, "mean") else np.array([pol(o) for o in probe])
    return spec, dict(mean_fwd=float(np.mean(fwd)), mean_abs_turn=float(np.mean(turn)),
                      frac_reverse=float(np.mean(np.array(fwd) < -0.1)), mean_edge_margin=float(np.mean(margin)),
                      mean_opp_dist=float(np.mean(dist))), probe_act.tolist()


def probe_set(n=400, seed=4242):
    """Observations visited in bouts between scripts (policy-independent probe set)."""
    env = SumoEnv(seed=seed)
    names = list(SCRIPT_CLASSES)
    rng = np.random.default_rng(seed)
    obs = []
    while len(obs) < n:
        a, b = make_script(names[rng.integers(4)]), make_script(names[rng.integers(4)])
        oa, ob = env.reset()
        while True:
            if rng.random() < 0.05:
                obs.append(oa)
            (oa, ob), out, _ = env.step(a(oa), b(ob))
            if out is not None:
                break
    return np.array(obs[:n], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/main.pkl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_small", type=int, default=100)
    ap.add_argument("--snaps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/tournament.json")
    args = ap.parse_args()
    t0 = time.time()
    ck = pickle.load(open(args.ckpt, "rb"))
    na = len(ck["archive"])
    snap_idx = sorted(set(np.linspace(0, na - 1, args.snaps).round().astype(int).tolist())) if na else []
    snaps = [f"snap{k}" for k in snap_idx]
    scripts = ["charger", "flanker", "circler", "counter"]
    entrants = ["ppo"] + scripts + ["random"]

    tasks, seed = [], 5000
    # 1. round robin (training conditions: ring/friction sampled from training ranges, eval seeds)
    for a, b in itertools.combinations(entrants, 2):
        tasks.append(("rr", (a, b, args.n, seed, None, None))); seed += 1
    # 2. cross-play: final vs snapshots, snapshots vs each other, snapshots vs scripts
    for s in snaps:
        tasks.append(("xp", ("ppo", s, args.n_small, seed, None, None))); seed += 1
    for s1, s2 in itertools.combinations(snaps, 2):
        tasks.append(("xp", (s1, s2, args.n_small, seed, None, None))); seed += 1
    for s in snaps:
        for sc in scripts:
            tasks.append(("snapvs", (s, sc, args.n_small, seed, None, None))); seed += 1
    # 3. robustness: ring radius / friction outside training ranges (train: R 0.9-1.1, friction 0.8-1.2)
    conds = [(1.0, 1.0), (0.7, 1.0), (1.4, 1.0), (1.0, 0.5), (1.0, 1.6), (0.7, 0.5), (1.4, 1.6)]
    for R, f in conds:
        for sc in scripts:
            tasks.append(("rob", ("ppo", sc, args.n_small, seed, R, f))); seed += 1
        for a, b in itertools.combinations(scripts, 2):       # script-vs-script reference under the same shift
            tasks.append(("rob_s", (a, b, args.n_small, seed, R, f))); seed += 1

    with Pool(args.workers, initializer=init, initargs=(args.ckpt,)) as pool:
        res = pool.map(run_cell, [t for _, t in tasks], chunksize=1)
        probe = probe_set()
        beh = pool.map(behaviour, [(s, 20, 9000, probe) for s in snaps + ["ppo"]])
    out = dict(ckpt=args.ckpt, n=args.n, n_small=args.n_small, snap_idx=snap_idx,
               snap_updates=[ck["archive"][k]["update"] for k in snap_idx],
               snap_steps=[ck["archive"][k]["steps"] for k in snap_idx],
               train_updates=ck["update"], train_steps=ck["steps"], max_steps=MAX_STEPS)
    for kind in ("rr", "xp", "snapvs", "rob", "rob_s"):
        out[kind] = [r for (k, _), r in zip(tasks, res) if k == kind]
    # diversity: pairwise mean |delta action| on the shared probe set
    names = [b[0] for b in beh]
    acts = {b[0]: np.array(b[2]) for b in beh}
    out["behaviour"] = {b[0]: b[1] for b in beh}
    out["probe_n"] = len(probe)
    out["action_disagreement"] = {f"{x}|{y}": float(np.abs(acts[x] - acts[y]).mean())
                                  for x, y in itertools.combinations(names, 2)}
    out["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("saved", args.out, f"{out['wall_s']:.0f}s")
    for r in out["rr"]:
        print(f"{r['a']:>8} vs {r['b']:<8} W {r['W']:3d} L {r['L']:3d} D {r['D']:3d}  score {r['score']:+.3f} "
              f"winCI {r['win_ci']} tips {r['tip_a']}/{r['tip_b']}")


if __name__ == "__main__":
    main()
