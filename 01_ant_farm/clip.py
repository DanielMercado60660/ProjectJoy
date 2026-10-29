"""Render a 2x2 grid of worlds for one colony (used for the failure exhibits).

  ../.venv/bin/python clip.py --policy checkpoints/ppo_nocompass_collapsed --out media/nocompass_collapse
  ../.venv/bin/python clip.py --policy compass --out media/handwritten_grid
"""
import argparse, os
import numpy as np
import sim, render
import handwritten as hw
from evaluate import load_policy

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="hand-written name or checkpoint dir")
ap.add_argument("--seeds", type=int, nargs=4, default=[0, 1, 2, 3])
ap.add_argument("--family", default="train")
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--title", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
pol = hw.POLICIES[a.policy] if a.policy in hw.POLICIES else load_policy(a.policy)
w = sim.make_worlds(np.array(a.seeds), 64, a.family)
frames = []
for t in range(a.steps):
    pm, pd = pol(sim.observe(w))
    m, d = sim.sample_actions(w, pm, pd)
    sim.step(w, m, d)
    if t % a.every == 0:
        tiles = [render.frame(w, b, 3) for b in range(4)]
        grid = np.concatenate([render.hstack(tiles[:2], 3), np.full((3, 2 * 192 + 3, 3), 12, np.uint8),
                               render.hstack(tiles[2:], 3)], 0)
        frames.append(render.label(grid, [a.title or os.path.basename(a.policy.rstrip("/")),
                                          f"step {t}  food home {int(w.delivered.sum())}  ants alive {int(w.alive.sum())}/256"]))
render.write_video(frames, a.out + ".gif", fps=20)
print("wrote", a.out + ".gif", "delivered", w.delivered, "alive", w.alive.mean(1))
