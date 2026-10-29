"""Fixed-horizon training episodes shared by ES, baseline tuning and PPO eval.

fitness episode: car starts at rest at a random point of a training track and
drives for `steps` steps (default 60 s). Score = signed progress in meters, the
episode stops at the first wall contact, and a crash costs CRASH_PENALTY meters.
"""
import numpy as np

from .sim import VecRace

CRASH_PENALTY = 100.0


def episodes(act_fn, bank, track_ids, starts, steps=600, grip=1.0):
    """act_fn(obs (E,D)) -> (E,2). Returns progress (E,), crashed (E,)."""
    env = VecRace(bank, len(track_ids))
    obs = env.reset(track_ids, start_frac=starts, grip=grip)
    for _ in range(steps):
        obs, _, done = env.step(act_fn(obs))
        if done.all():
            break
    return env.prog.copy(), env.crashed.copy()


def score(prog, crashed):
    return prog - CRASH_PENALTY * crashed
