"""Reward-hacking exhibit: what happens if walls are 'soft' (push back, keep sliding)
instead of ending the episode? Compares, on the 100 held-out tracks:
  * a brainless car: steering 0, full throttle, always
  * es_soft: ES trained with soft walls (train_es.py --soft_walls)
  * es: ES trained with the real rules (wall = crash)
under both soft-wall and strict rules. Writes results/hack.json.

    ../.venv/bin/python hack_check.py
"""
import json
import os

import numpy as np

from racing.controllers import get_controller
from racing.evaluate import run_lap, summarize
from racing.track import make_tracks, pack


def brainless(obs, env=None):
    return np.tile([0.0, 1.0], (len(obs), 1))


def main():
    bank = pack(make_tracks("heldout", 100))
    ctrls = {"no steering, full throttle": brainless, "es_soft": get_controller("es_soft", bank),
             "es": get_controller("es", bank), "ray": get_controller("ray", bank)}
    out = []
    for name, c in ctrls.items():
        for mode in ("soft", "strict"):
            s = summarize(run_lap(c, bank, mode=mode))
            s.update(controller=name, mode=mode)
            out.append(s)
            print(f"{name:28s} {mode:6s} done={s['completion']:.2f} lap={s['mean_lap_time']:.1f}s "
                  f"wall-contact steps/lap={s['collisions_per_lap']:.1f} near-wall time={s['near_wall_time']:.1%} "
                  f"progress={s['mean_progress']:.2f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/hack.json", "w"), indent=1)


if __name__ == "__main__":
    main()
