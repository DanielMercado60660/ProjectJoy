"""Render the demo videos.

  ../.venv/bin/python demo.py --ppo checkpoints/ppo1 [--es checkpoints/es1]

media/side_by_side.mp4 / .gif : hand-written vs learned colonies on the same unseen (held-out)
                               world; a wall is dropped across the busiest trail at step 500.
"""
import argparse, os
import numpy as np
import sim, render
import handwritten as hw
from evaluate import load_policy

ap = argparse.ArgumentParser()
ap.add_argument("--ppo", default="checkpoints/ppo1")
ap.add_argument("--es", default=None)
ap.add_argument("--ppo_label", default="PPO shared policy (learned)")
ap.add_argument("--seed", type=int, default=50_001)
ap.add_argument("--family", default="heldout")
ap.add_argument("--steps", type=int, default=1400)
ap.add_argument("--block_at", type=int, default=500)
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--scale", type=int, default=5)
ap.add_argument("--out", default="media/side_by_side")
a = ap.parse_args()

cols = [("hand-written (2-pheromone + compass)", hw.compass), (a.ppo_label, load_policy(a.ppo))]
if a.es:
    cols.append(("ES colony (learned)", load_policy(a.es)))
worlds = [sim.make_worlds(np.array([a.seed]), 64, a.family) for _ in cols]
for w in worlds:
    w.food[w.food > 0] = 120  # plenty of food so the movie is about routes, not depletion
frames, wall_src = [], None
for t in range(a.steps):
    if t == a.block_at:
        # block the route to the source the *hand-written* colony uses most, in every panel,
        # so all colonies face the identical wall
        wall_src = sim.busiest_source(worlds[0], 0)
        for w in worlds:
            sim.block_route(w, 0, food_xy=wall_src)
    for (name, pol), w in zip(cols, worlds):
        pm, pd = pol(sim.observe(w))
        m, d = sim.sample_actions(w, pm, pd)
        sim.step(w, m, d)
    if t % a.every == 0 or t == a.steps - 1:
        panels = []
        for (name, _), w in zip(cols, worlds):
            note = "  WALL DROPPED" if a.block_at <= t < a.block_at + 60 else ""
            panels.append(render.label(render.frame(w, 0, a.scale),
                                       [name, f"step {t:4d}  food home {int(w.delivered[0]):4d}  ants {int(w.alive[0].sum())}/64{note}"]))
        frames.append(render.hstack(panels))
os.makedirs("media", exist_ok=True)
render.write_video(frames, a.out + ".mp4", fps=30)
small = [f[::2, ::2] for f in frames[::2]]
render.write_video(small, a.out + ".gif", fps=15)
from PIL import Image
Image.fromarray(np.concatenate([frames[len(frames) * k // 4] for k in (1, 2, 3)], 0)).save(a.out + "_strip.png")
print("wrote", a.out, [int(w.delivered[0]) for w in worlds])
