"""Why do the learned policies finish MORE hard (narrow, twisty) tracks at grip 0.8
than at their training grip 1.0?

Hypothesis: the policies steer twitchily; at full grip every steering spike turns
into real yaw, while at lower grip the yaw-rate cap (mu*g/v) clips those spikes.
Test 1: keep grip 1.0 but low-pass the policy's steering command (EMA), and see if
that recovers the grip-0.8 completion rate.
Test 2 (alternative: the policy is simply too fast for these unfamiliar corners and
lower grip makes it slow down): keep grip 1.0 but cap the speed it may throttle to. Writes results/diagnose_grip.json.

    ../.venv/bin/python diagnose_grip.py --controllers es ppo
"""
import argparse
import json
import os

import numpy as np

from racing.controllers import get_controller
from racing.evaluate import run_lap, summarize
from racing.sim import N_RAYS, V_SCALE
from racing.track import make_tracks, pack


class SmoothedSteer:
    """Wraps a controller; steer_t = a*cmd + (1-a)*previous steer (read from obs)."""

    def __init__(self, ctrl, a):
        self.ctrl, self.a = ctrl, a

    def __call__(self, obs, env=None):
        act = self.ctrl(obs, env).copy()
        act[:, 0] = self.a * act[:, 0] + (1 - self.a) * obs[:, N_RAYS + 1]
        return act


class SpeedCap:
    """Wraps a controller; no throttle above v_cap m/s (brakes are untouched)."""

    def __init__(self, ctrl, v_cap):
        self.ctrl, self.v_cap = ctrl, v_cap

    def __call__(self, obs, env=None):
        act = self.ctrl(obs, env).copy()
        v = obs[:, N_RAYS] * V_SCALE
        act[:, 1] = np.where(v > self.v_cap, np.minimum(act[:, 1], 0.0), act[:, 1])
        return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["es", "ppo", "ray"])
    a = ap.parse_args()
    bank = pack(make_tracks("hard", 50))
    out = []
    for name in a.controllers:
        base = get_controller(name, bank)
        for label, ctrl, grip in [("grip 1.0", base, 1.0), ("grip 0.8", base, 0.8),
                                  ("grip 1.0 + steer EMA 0.5", SmoothedSteer(base, 0.5), 1.0),
                                  ("grip 1.0 + steer EMA 0.3", SmoothedSteer(base, 0.3), 1.0),
                                  ("grip 1.0 + speed cap 22 m/s", SpeedCap(base, 22.0), 1.0),
                                  ("grip 1.0 + speed cap 18 m/s", SpeedCap(base, 18.0), 1.0)]:
            s = summarize(run_lap(ctrl, bank, grip=grip))
            s.update(controller=name, variant=label)
            out.append(s)
            print(f"{name:6s} {label:26s} done={s['completion']:.2f} lap={s['mean_lap_time']:.1f}s "
                  f"jitter={s['steer_jitter']:.3f} p95lat={s['lat_p95']:.2f} inside-wall share of crashes={s['inside_wall_share']:.2f}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/diagnose_grip.json", "w"), indent=1)


if __name__ == "__main__":
    main()
