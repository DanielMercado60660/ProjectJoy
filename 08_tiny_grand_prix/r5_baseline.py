"""Matched-sensor adaptive ray follower with a scalar RLS grip estimator.

Unclipped motion supplies only a LOWER BOUND on grip. Saturated steering or
braking supplies an equality measurement. No simulator/track object is accepted.
The simulator has noiseless odometry: this deliberately gives classical physics
its full strength, rather than hiding an easy grip estimate from the baseline.
"""
import numpy as np
from racing.sim import A_MAX, B_MAX, DRAG, DT, G, MAX_STEER, RAY_ANGLES, RAY_LEN, V_SCALE, WHEELBASE

# Inclusive search bounds. The search also includes the old tuned DR ray knobs.
RANGES = {
    'k_steer': (.5, 3.0), 'power': (1., 8.), 'k_center': (0., 2.),
    'brake_fraction': (.25, 1.2), 'margin': (2., 18.), 'v_max': (18., 40.),
    'v_min': (2., 8.), 'prior': (.4, 1.1), 'forget': (.8, 1.),
    'rise': (0., .15), 'safety_mu': (.55, 1.), 'slip_tol': (.005, .15),
    'steer_mu': (0., .8),
}
KEYS = tuple(RANGES)
LO = np.array([RANGES[k][0] for k in KEYS])
HI = np.array([RANGES[k][1] for k in KEYS])
DEFAULT = dict(k_steer=1.5, power=4., k_center=.6, brake_fraction=.8, margin=8.,
               v_max=35., v_min=4., prior=.7, forget=.95, rise=.03,
               safety_mu=.9, slip_tol=.02, steer_mu=0.)


def vector(p=DEFAULT):
    return np.array([p[k] for k in KEYS])


def as_dict(x):
    return dict(zip(KEYS, map(float, x)))


class AdaptiveRay:
    def __init__(self, params=None, frozen=False):
        self.params = vector() if params is None else (vector(params) if isinstance(params, dict) else np.asarray(params))
        self.frozen = frozen
        self.mu = None

    def reset(self, n):
        q = np.broadcast_to(self.params, (n, len(KEYS)))
        self.p = {k: q[:, i] for i, k in enumerate(KEYS)}
        self.mu = self.p['prior'].copy()
        self.cov = np.ones(n)
        self.prev_v = np.zeros(n)
        self.seen = np.zeros(n, bool)
        self.measurements = np.zeros(n, int)

    def estimate(self, obs):
        if self.mu is None or len(self.mu) != len(obs):
            self.reset(len(obs))
        p = self.p
        v = obs[:, 10] * V_SCALE
        vm = .5 * (v + self.prev_v)
        demand_yaw = vm * np.tan(obs[:, 11] * MAX_STEER) / WHEELBASE
        measured_yaw = obs[:, 13]
        yaw_mu = np.abs(measured_yaw) * np.maximum(vm, 1.) / G
        yaw_sat = np.abs(demand_yaw) > np.abs(measured_yaw) + p['slip_tol']
        demand_a = obs[:, 12] * np.where(obs[:, 12] > 0, A_MAX, B_MAX)
        measured_a = (v - self.prev_v) / DT + DRAG * self.prev_v ** 2
        accel_mu = np.abs(measured_a) / G
        accel_sat = np.abs(demand_a) > np.abs(measured_a) + 10 * p['slip_tol']
        valid = self.seen & (self.prev_v > 3) & (v > 1)
        sat = valid & (yaw_sat | accel_sat)
        measurement = np.where(yaw_sat, yaw_mu, accel_mu)
        if not self.frozen:
            # A sudden change raises RLS uncertainty. Otherwise use forgetting.
            self.cov = np.where(sat & (np.abs(measurement - self.mu) > .12), 1., self.cov)
            prior_cov = self.cov / p['forget']
            gain = prior_cov / (prior_cov + .02)
            self.mu = np.where(sat, self.mu + gain * (measurement - self.mu), self.mu)
            self.cov = np.where(sat, (1 - gain) * prior_cov, np.minimum(prior_cov, 2.))
            lower_bound = np.maximum(yaw_mu, accel_mu)
            self.mu = np.where(valid & ~sat, np.maximum(self.mu, lower_bound), self.mu)
            # Slow optimism permits probing for improved grip; tuned including zero.
            self.mu = np.where(valid & ~sat, self.mu + p['rise'] * DT, self.mu)
            self.mu = np.clip(self.mu, .25, 1.2)
        self.measurements += sat
        self.prev_v = v.copy()
        self.seen[:] = True
        return self.mu.copy()

    def __call__(self, obs):
        mu = self.estimate(obs)
        p = self.p
        r = np.clip(obs[:, :10], 1e-3, 1.)
        w = r ** p['power'][:, None]
        target = (w * RAY_ANGLES).sum(1) / np.maximum(w.sum(1), 1e-12)
        steer = (p['k_steer'] * target / MAX_STEER + p['k_center'] * (r[:, -1] - r[:, 0]))
        steer *= (p['prior'] / np.maximum(mu, .25)) ** p['steer_mu']
        front = r[:, np.abs(RAY_ANGLES) < np.deg2rad(20)].max(1) * RAY_LEN
        brake = np.minimum(B_MAX, G * mu * p['safety_mu']) * p['brake_fraction']
        vt = np.sqrt(np.maximum(2 * brake * np.maximum(front - p['margin'], 0.), 1e-8))
        vt = np.clip(vt, p['v_min'], p['v_max'])
        return np.stack([np.clip(steer, -1, 1), np.clip(.5 * (vt - obs[:, 10] * V_SCALE), -1, 1)], 1)
