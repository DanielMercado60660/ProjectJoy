"""Hand-written controllers.

RayFollower  -- matched information: sees exactly the learned policy's observation
               vector (10 wall rays, speed, previous action). Steers toward the
               open space (ray-length weighted mean direction) plus a centering
               term, and picks a target speed it could still brake from given the
               free distance straight ahead. Its 7 knobs are tuned by the cross-entropy
               method on the *training* tracks (see tune_baseline.py).

PurePursuit  -- PRIVILEGED: reads the true centerline and curvature. Chases a
               lookahead point on the centerline and follows a precomputed
               curvature speed profile assuming grip MU_ASSUMED. It is a reference
               for "what good driving looks like", not a fair competitor.
"""
import numpy as np

from .sim import B_MAX, G, MAX_STEER, N_RAYS, RAY_ANGLES, RAY_LEN, V_SCALE, WHEELBASE

DEFAULT_RAY_PARAMS = dict(k_steer=1.5, p=4.0, k_center=0.6, brake=7.0, margin=8.0, v_max=35.0, v_min=4.0)


class RayFollower:
    name = "ray_follower"

    def __init__(self, **params):
        self.p = dict(DEFAULT_RAY_PARAMS)
        self.p.update(params)

    def __call__(self, obs, env=None):
        p = self.p
        r = np.clip(obs[:, :N_RAYS], 1e-3, 1.0)
        v = obs[:, N_RAYS] * V_SCALE
        w = r ** p["p"]
        target = (w * RAY_ANGLES).sum(1) / w.sum(1)
        steer = p["k_steer"] * target / MAX_STEER + p["k_center"] * (r[:, -1] - r[:, 0])
        # free distance in the direction we are heading (best of the 4 central rays)
        front = r[:, np.abs(RAY_ANGLES) < np.deg2rad(20)].max(1) * RAY_LEN
        v_t = np.sqrt(2 * p["brake"] * np.maximum(front - p["margin"], 0.0))
        v_t = np.clip(v_t, p["v_min"], p["v_max"])
        pedal = np.clip(0.5 * (v_t - v), -1, 1)
        return np.stack([np.clip(steer, -1, 1), pedal], 1)


class PurePursuit:
    name = "pure_pursuit_privileged"
    MU_ASSUMED = 0.9

    def __init__(self, bank, mu_assumed=None):
        self.bank = bank
        mu = self.MU_ASSUMED if mu_assumed is None else mu_assumed
        self.profiles = []
        for t in bank.tracks:
            ds = t.length / len(t.center)
            k = np.abs(t.curvature)
            # smooth curvature a little so the profile sees corners slightly early
            k = np.maximum(k, np.roll(k, -2))
            vp = np.minimum(np.sqrt(mu * G / np.maximum(k, 1e-4)), 38.0)
            b = min(B_MAX, mu * G) * 0.8
            for _ in range(3):  # backward braking pass, wrapped around the loop
                for i in range(len(vp) - 1, -1, -1):
                    j = (i + 1) % len(vp)
                    vp[i] = min(vp[i], np.sqrt(vp[j] ** 2 + 2 * b * ds))
            self.profiles.append(vp)

    def __call__(self, obs, env):
        b = self.bank
        out = np.zeros((env.E, 2))
        for e in range(env.E):
            tid = env.track_id[e]
            n = b.n_pts[tid]
            v = env.v[e]
            ld = max(6.0, 0.6 * v)
            j = (env.idx[e] + int(ld / b.seglen[tid])) % n
            tx, ty = b.center[tid, j] - np.array([env.x[e], env.y[e]])
            c, s = np.cos(env.yaw[e]), np.sin(env.yaw[e])
            lx, ly = c * tx + s * ty, -s * tx + c * ty
            alpha = np.arctan2(ly, lx)
            delta = np.arctan(2 * WHEELBASE * np.sin(alpha) / np.hypot(lx, ly))
            vt = self.profiles[tid][(env.idx[e] + 2) % n]
            out[e] = [np.clip(delta / MAX_STEER, -1, 1), np.clip(0.5 * (vt - v), -1, 1)]
        return out
