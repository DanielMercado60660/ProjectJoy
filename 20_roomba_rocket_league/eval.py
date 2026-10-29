"""Held-out evaluation: learned policy vs scripted attacker vs random, against two scripted defenders.

    ../.venv/bin/python eval.py --ckpt checkpoints/ppo.npz --episodes 500

Start sets (fixed seeds, never used in training: training seeds are 0/1, eval uses 1000+):
  heldout  - the full training distribution (curriculum=1) from an unseen seed, corners excluded
  corner   - ball starts in the four corners, a region excluded from every training reset
  kickoff  - ball at centre, learner in its own half
Opponents: goalie (training opponent); keeper (line keeper) and chaser (a striker that attacks the
learner's goal), two styles never seen in training.
"""
import argparse, json, os
import numpy as np
from env import Arena, T_MAX, OUT_GOAL, OUT_CONCEDED, OUT_OWN_GOAL, OUT_TIMEOUT
import scripted

SETS = {"heldout": 1001, "corner": 1002, "kickoff": 1003}


def run_episodes(policy, opponent, start_set, n, seed=None, record=None):
    """Each of n envs plays exactly one match. Returns per-episode metric arrays.
    record: list of env indices to snapshot every step (for clips)."""
    a = Arena(n, seed=SETS[start_set] if seed is None else seed, opponent=opponent)
    a.start_mode = "train" if start_set == "heldout" else start_set
    a.curriculum = 1.0
    a.reset(np.ones(n, dtype=bool))
    opp = scripted.OPPONENTS[opponent]
    live = np.ones(n, dtype=bool)
    res = {k: np.zeros(n, dtype=np.int64) for k in
           ["outcome", "t", "first_touch", "touches0", "touches1", "pin_steps", "near_steps", "intercepts",
            "t_first_touch"]}
    res["t_first_touch"][:] = -1
    frames = {i: [] for i in (record or [])}
    for step in range(T_MAX):
        for i in frames:
            if live[i]:
                frames[i].append(a.snapshot(i))
        prev_last = a.last_touch.copy()
        _, d, tr, outc, info = a.step(policy(a), opp(a))
        regain = (prev_last == 1) & (a.last_touch == 0) & live
        res["intercepts"] += regain
        newtouch = (a.touches[:, 0] > 0) & (res["t_first_touch"] < 0) & live
        res["t_first_touch"][newtouch] = a.t[newtouch]
        fin = (d | tr) & live
        if fin.any():
            res["outcome"][fin] = outc[fin]
            res["t"][fin] = a.t[fin]
            res["first_touch"][fin] = a.first_touch[fin]
            res["touches0"][fin] = a.touches[fin, 0]
            res["touches1"][fin] = a.touches[fin, 1]
            res["pin_steps"][fin] = a.pin_steps[fin]
            res["near_steps"][fin] = a.near_steps[fin]
            for i in frames:
                if fin[i]:
                    frames[i].append(a.snapshot(i))
            live &= ~fin
            # freeze finished envs: park their ball so they never "score" again
            a.bvel[fin] = 0
        if not live.any():
            break
    return res, frames


def summarize(res):
    o = res["outcome"]
    n = len(o)
    g = o == OUT_GOAL
    return dict(
        n=n,
        goal=float(g.mean()), conceded=float((o == OUT_CONCEDED).mean()),
        own_goal=float((o == OUT_OWN_GOAL).mean()), timeout=float((o == OUT_TIMEOUT).mean()),
        first_touch_learner=float((res["first_touch"] == 0).mean()),
        first_touch_opp=float((res["first_touch"] == 1).mean()),
        possession=float((res["near_steps"] / np.maximum(res["t"], 1)).mean()),
        touch_share=float((res["touches0"] / np.maximum(res["touches0"] + res["touches1"], 1)).mean()),
        intercepts_per_match=float(res["intercepts"].mean()),
        time_to_first_touch_s=float(np.mean(res["t_first_touch"][res["t_first_touch"] > 0]) * 0.05)
        if (res["t_first_touch"] > 0).any() else None,
        time_to_goal_s=float(res["t"][g].mean() * 0.05) if g.any() else None,
        pin_frac=float((res["pin_steps"] / np.maximum(res["t"], 1)).mean()),
        pin_matches=float((res["pin_steps"] >= 40).mean()),   # >= 2 s of learner-on-ball-on-wall
    )


def get_policies(ckpt):
    pols = {"scripted_attacker": lambda a: scripted.attacker(a)}
    rng = np.random.default_rng(7)
    pols["random"] = lambda a: scripted.random_policy(a, rng)
    if ckpt and os.path.exists(ckpt):
        from ppo import load_ckpt, make_policy
        model, norm, _ = load_ckpt(ckpt)
        pols = {"ppo": make_policy(model, norm), **pols}
    return pols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo.npz")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pols = get_policies(args.ckpt)
    table = {}
    for pname, pol in pols.items():
        for opp in ["goalie", "keeper", "chaser"]:
            for s in SETS:
                res, _ = run_episodes(pol, opp, s, args.episodes)
                table[f"{pname}|{opp}|{s}"] = summarize(res)
                m = table[f"{pname}|{opp}|{s}"]
                print(f"{pname:18s} {opp:7s} {s:8s} goal {m['goal']:.3f} conc {m['conceded']:.3f} "
                      f"own {m['own_goal']:.3f} to {m['timeout']:.3f} 1st {m['first_touch_learner']:.2f} "
                      f"poss {m['possession']:.2f} int {m['intercepts_per_match']:.2f} pin {m['pin_frac']:.3f}",
                      flush=True)
    json.dump(table, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
