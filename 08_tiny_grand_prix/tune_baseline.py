"""Tune the matched-information RayFollower baseline on TRAINING tracks.

Cross-entropy method over its 7 knobs, using exactly the ES fitness
(60 s episodes from random start points on training tracks, crash = -100 m).
Writes checkpoints/ray_follower.json. With --grip_lo/--grip_hi the episodes use
random grip (the same randomization as the *_dr learners) -> ray_follower_dr.json.

    ../.venv/bin/python tune_baseline.py [--iters 15 --pop 40]
"""
import argparse
import json
import time

import numpy as np

from racing.baselines import DEFAULT_RAY_PARAMS, RayFollower
from racing.rollout import episodes, score
from racing.track import make_tracks, pack

KEYS = ["k_steer", "p", "k_center", "brake", "margin", "v_max", "v_min"]
LO = np.array([0.3, 0.5, 0.0, 2.0, 0.0, 15.0, 2.0])
HI = np.array([4.0, 10.0, 3.0, 12.0, 20.0, 40.0, 10.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--n_train_tracks", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/ray_follower.json")
    ap.add_argument("--grip_lo", type=float, default=1.0)
    ap.add_argument("--grip_hi", type=float, default=1.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bank = pack(make_tracks("train", a.n_train_tracks))

    def fit(params, n_ep, seed):
        r = np.random.default_rng(seed)
        tids = r.integers(0, a.n_train_tracks, n_ep)
        starts = r.random(n_ep)
        grips = r.uniform(a.grip_lo, a.grip_hi, n_ep)
        ctrl = RayFollower(**dict(zip(KEYS, params)))
        prog, crashed = episodes(ctrl, bank, tids, starts, grip=grips)
        return score(prog, crashed).mean(), crashed.mean()

    mu = np.array([DEFAULT_RAY_PARAMS[k] for k in KEYS])
    sd = (HI - LO) / 4
    t0 = time.time()
    for it in range(a.iters):
        cand = np.clip(mu + sd * rng.standard_normal((a.pop, len(KEYS))), LO, HI)
        cand[0] = mu
        fits = np.array([fit(c, a.episodes, 1000 + it)[0] for c in cand])
        elite = cand[np.argsort(-fits)[: a.pop // 5]]
        mu, sd = elite.mean(0), elite.std(0) + 0.02 * (HI - LO)
        print(f"iter {it} best {fits.max():.1f} mean {fits.mean():.1f} mu "
              + " ".join(f"{k}={v:.2f}" for k, v in zip(KEYS, mu)) + f" ({time.time() - t0:.0f}s)", flush=True)
    final, crash = fit(mu, 512, 99)
    print(f"final fitness on 512 fresh training episodes: {final:.1f} m, crash rate {crash:.3f}")
    params = {k: float(v) for k, v in zip(KEYS, mu)}
    json.dump(dict(params=params, train_fitness=float(final), train_crash=float(crash),
                   env_steps_approx=int(a.iters * a.pop * a.episodes * 600)), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
