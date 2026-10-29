"""Find the first eval seed (from 8000) where the PPO robot tips itself vs the held-out counter at friction x1.6,
and render it to media/exploit_tip_highfriction.mp4/.gif.

  ../.venv/bin/python render_tip_clip.py
"""
import numpy as np, imageio.v2 as imageio
import tournament
from sumo_env import SumoEnv, play_bout
from render_demo import render_bout, label

tournament.init("checkpoints/main.pkl")
env = SumoEnv(seed=0)
for s in range(8000, 8100):
    env.rng = np.random.default_rng(s)
    r = play_bout(env, tournament.make("ppo", 0), tournament.make("counter", 0), R=1.0, fric=1.6)
    if r["tip_a"]:
        print("tip seed", s, r["steps"])
        break
frames, r = render_bout(env, tournament.make("ppo", 0), tournament.make("counter", 0), s, R=1.0, fric=1.6, every=1)
frames = label(frames, f"friction x1.6 (outside training): PPO (red) tips itself vs counter (blue), seed {s}")
imageio.mimsave("media/exploit_tip_highfriction.mp4", frames, fps=25)
imageio.mimsave("media/exploit_tip_highfriction.gif", frames[::2], duration=0.08, loop=0)
print("wrote media/exploit_tip_highfriction.mp4/.gif")
