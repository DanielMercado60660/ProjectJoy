"""Render demo clips from the saved checkpoint into media/.

    ../.venv/bin/python demo.py --ckpt checkpoints/ppo.npz

Clips (blue = learner / scripted attacker, attacks the right goal; red = scripted defender):
  ppo_vs_goalie.{gif,mp4}      first 4 held-out matches, unselected
  ppo_vs_keeper.{gif,mp4}      first 4 cross-play matches vs the unseen line keeper, unselected
  ppo_vs_chaser.{gif,mp4}      first 4 cross-play matches vs the unseen chaser, unselected
  scripted_vs_goalie.{gif,mp4} baseline on the same 4 held-out starts
  ppo_goals.{gif,mp4}          4 matches the learner won (selected: first 4 goals in 200 held-out starts)
  ppo_pin_exploit.{gif,mp4}    4 matches with the most learner-on-ball-against-wall time (selected)
"""
import argparse, os
import numpy as np
from PIL import Image
import render
from env import OUT_NAMES
from eval import run_episodes, get_policies


def grid_clip(policy, opp, sset, idx, caption, path, n=200):
    res, frames = run_episodes(policy, opp, sset, n, record=list(idx))
    seqs = [[render.draw_frame(s, f"{caption} #{i}: {OUT_NAMES.get(int(res['outcome'][i]), '?')}")
             for s in frames[i]] for i in idx]
    L = max(len(s) for s in seqs)
    w, h = seqs[0][0].size
    out = []
    for t in range(L):
        g = Image.new("RGB", (2 * w, 2 * h))
        for k, s in enumerate(seqs[:4]):
            g.paste(s[min(t, len(s) - 1)], ((k % 2) * w, (k // 2) * h))
        out.append(g)
    out += [out[-1]] * 20
    render.save_clip(out, path + ".mp4", fps=20)
    small = [f.resize((w, h)) for f in out[::2]]
    render.save_clip(small, path + ".gif", fps=10)
    print("wrote", path, [OUT_NAMES.get(int(res["outcome"][i])) for i in idx])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo.npz")
    ap.add_argument("--outdir", default="media")
    ap.add_argument("--quick", action="store_true", help="only one clip (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pols = get_policies(args.ckpt)
    ppo = pols["ppo"]
    o = lambda name: os.path.join(args.outdir, name)
    res = grid_clip(ppo, "goalie", "heldout", range(4), "PPO vs goalie", o("ppo_vs_goalie"))
    if args.quick:
        return
    grid_clip(ppo, "keeper", "heldout", range(4), "PPO vs keeper (unseen)", o("ppo_vs_keeper"))
    grid_clip(ppo, "chaser", "heldout", range(4), "PPO vs chaser (unseen)", o("ppo_vs_chaser"))
    grid_clip(pols["scripted_attacker"], "goalie", "heldout", range(4), "scripted vs goalie",
              o("scripted_vs_goalie"))
    full, _ = run_episodes(ppo, "goalie", "heldout", 200)
    goals = np.flatnonzero(full["outcome"] == 1)[:4]
    if len(goals):
        grid_clip(ppo, "goalie", "heldout", list(goals) + [goals[0]] * (4 - len(goals)), "PPO goal",
                  o("ppo_goals"))
    pin = np.argsort(-full["pin_steps"])[:4]
    grid_clip(ppo, "goalie", "heldout", pin, "PPO pin", o("ppo_pin_exploit"))


if __name__ == "__main__":
    main()
