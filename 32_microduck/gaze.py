"""'Chicken head': adaptive gaze stabilisation for the walking duck (bot side; uses only the gyro
samples, the encoders and the bot's own camera model; results/m3/bot_contract.md).

While walking, the heavy head swings on its neck at the stepping rhythm (~1.86 Hz): the camera wobbles
~6.7 deg in yaw and ~5 deg in pitch (std) while the trunk wobbles only 2.1 / 0.7 deg. A chicken holds
its head still while its body bobs. Here the bot adds a counter-swing to its head-yaw and head-pitch
commands and adapts it as it walks (sinusoidal-reference LMS, one per axis):
    offset_axis = w_axis . [sin(w t), cos(w t)]          w = 2 pi * 1.86 Hz
    error_axis  = camera angle now - its mean over the last second   (from the bot's own camera model)
    w_axis     -= mu * error_axis * [sin(w t - phi_axis), cos(w t - phi_axis)]
phi_axis compensates the delay from head command to head motion (calibrated per axis, m3_gaze.py).
It only runs while walking: a lock-in on the trunk's yaw rate (the gyro samples) measures the gait's
strength; below a threshold (standing, looking around) the offsets fade to zero and adaptation stops.
The interface and the walking policy are unchanged: this only shapes the head commands.
"""
from collections import deque

import numpy as np

GAZE_PARAMS = {
    "freq_hz": 1.86,
    "mu": 0.5,
    "phi_yaw": 2.0, "phi_pitch": 4.4,     # rad; m3_gaze.py / results/m3/gaze.json
    "max_offset": 0.35,                   # rad per axis
    "window": 50,                         # gyro samples (1 s) for the gait-strength lock-in
    "a_on": 0.08, "a_full": 0.25,         # gait lock-in amplitude (rad/s): standing ~0.002, walking 0.4-0.5
    "err_window": 10,                     # ticks for the camera-angle mean
}


class Gaze:
    def __init__(self, params=None):
        self.P = dict(GAZE_PARAMS, **(params or {}))
        self.t = deque(maxlen=self.P["window"])
        self.g = deque(maxlen=self.P["window"])
        self.hist = deque(maxlen=self.P["err_window"])
        self.W = np.zeros((2, 2))         # rows: yaw, pitch; columns: sin, cos
        self.enabled = True

    def gait_strength(self):
        if len(self.g) < self.P["window"]:
            return 0.0
        w = 2 * np.pi * self.P["freq_hz"]
        t, g = np.array(self.t), np.array(self.g)
        g = g - g.mean()
        amp = 2.0 / len(g) * np.hypot((g * np.cos(w * t)).sum(), (g * np.sin(w * t)).sum())
        return float(np.clip((amp - self.P["a_on"]) / (self.P["a_full"] - self.P["a_on"]), 0.0, 1.0))

    def update(self, obs, cam_mat):
        """Call once per tick after sensing: gyro samples and the camera orientation estimate."""
        gz = np.asarray(obs.body["gyro_substeps"])[:, 2]
        n = len(gz)
        for i, v in enumerate(gz):
            self.t.append(obs.sim_time - (n - 1 - i) * 0.02)
            self.g.append(float(v))
        look = -np.asarray(cam_mat)[:, 2]
        self.hist.append([np.arctan2(look[1], look[0]), np.arcsin(np.clip(look[2], -1, 1))])
        gate = self.gait_strength()
        if not self.enabled or gate <= 0.0 or len(self.hist) < 6:
            return
        H = np.unwrap(np.array(self.hist), axis=0)
        e = H[-1] - H.mean(0)
        w = 2 * np.pi * self.P["freq_hz"]
        for ax, phi in enumerate((self.P["phi_yaw"], self.P["phi_pitch"])):
            xr = np.array([np.sin(w * obs.sim_time - phi), np.cos(w * obs.sim_time - phi)])
            self.W[ax] -= self.P["mu"] * gate * e[ax] * xr

    def offsets(self, t_now):
        """(head_yaw, head_pitch) offsets for the tick starting at t_now."""
        if not self.enabled:
            return 0.0, 0.0
        gate = self.gait_strength()
        if gate <= 0.0:
            return 0.0, 0.0
        w = 2 * np.pi * self.P["freq_hz"]
        x = np.array([np.sin(w * (t_now + 0.05)), np.cos(w * (t_now + 0.05))])
        u = np.clip(gate * (self.W @ x), -self.P["max_offset"], self.P["max_offset"])
        return float(u[0]), float(u[1])

    def state(self):
        return {"W": self.W.tolist()}
