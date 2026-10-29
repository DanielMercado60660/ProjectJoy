"""Render the demo: residual PPO policy vs tuned CPG baselines on the same HELD-OUT layout.

Layout choice is mechanical (no hand-picking): --index i uses held-out layout seed 10_000_000 + i;
by default the script picks the lowest held-out index where the policy completes the course
("--pick first_success"), and prints how the baselines did on it.

Usage: ../.venv/bin/python render_demo.py --policy checkpoints/policy.npz
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/policy.npz")
    ap.add_argument("--cpg", default="checkpoints/cpg_params.json")
    ap.add_argument("--planner", default="checkpoints/planner_params.json")
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--max_search", type=int, default=40)
    ap.add_argument("--out", default="media")
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=320)
    a = ap.parse_args()
    os.chdir(HERE)
    import task as T, lava_env as E, ppo, render_util as R

    pol, cfg = ppo.make_policy_fn(a.policy)
    methods = [("residual PPO", T.LavaTask(cfg["cpg_params"]), pol)]
    cpg = json.load(open(a.cpg))["params"]
    methods.append(("open-loop CPG", T.LavaTask(cpg), None))
    if os.path.exists(a.planner):
        methods.append(("CPG + foothold planner", T.LavaTask(json.load(open(a.planner))["params"], planner=True), None))

    def run(m, i, rec=None):
        name, tk, p = m
        s = E.HELDOUT_SEED_BASE + i
        return T.run_episode(tk, E.make_layout(s), policy=p, rng=np.random.default_rng(s + 50_000),
                             init_noise=0.05, record=rec)

    idx = a.index
    if idx < 0:
        for i in range(a.max_search):
            if run(methods[0], i)["success"]:
                idx = i
                break
        idx = max(idx, 0)
    L = E.make_layout(E.HELDOUT_SEED_BASE + idx)
    os.makedirs(a.out, exist_ok=True)
    clips, summary = [], {}
    for m in methods:
        rec = []
        r = run(m, idx, rec)
        outcome = "CROSSED" if r["success"] else ("lava" if r["lava"] else "fell" if (r["flip"] or r["collapsed"]) else "stopped")
        label = f"{m[0]}: {outcome} t={r['t']:.1f}s"
        print(f"held-out layout {idx}: {label}  progress {r['frac']:.2f}", flush=True)
        summary[m[0]] = dict(success=r["success"], lava=r["lava"], flip=r["flip"], frac=r["frac"], t=r["t"])
        fr = R.render_frames(m[1].env, L, rec, a.width, a.height, every=2, label=label)
        slug = {"residual PPO": "ppo", "open-loop CPG": "cpg_openloop", "CPG + foothold planner": "cpg_planner"}[m[0]]
        fn = os.path.join(a.out, f"demo_{a.tag}_{slug}_{idx}.mp4")
        R.save_video(fr, fn)
        clips.append(fr)
    # side-by-side (pad shorter clips by holding their last frame)
    n = max(len(c) for c in clips)
    side = [np.concatenate([c[min(k, len(c) - 1)] for c in clips], 1) for k in range(n)]
    R.save_video(side, os.path.join(a.out, f"demo_{a.tag}_side_by_side_{idx}.mp4"))
    small = [f[::2, ::2] for f in side[::2]]
    R.save_video(small, os.path.join(a.out, f"demo_{a.tag}_side_by_side_{idx}.gif"), fps=12)
    json.dump(dict(layout_index=idx, layout_seed=E.HELDOUT_SEED_BASE + idx, layout=L.tolist(), results=summary),
              open(os.path.join(a.out, f"demo_{a.tag}_{idx}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
