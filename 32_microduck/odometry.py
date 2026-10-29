"""The reference bot's odometry (bot side): where am I relative to where I started?

Frame: the "odometry frame" = the duck's start pose (origin at the start, x forward). Heading comes
from integrating the tick's five 50 Hz gyro samples, rotated into a yaw-free frame with the gravity
samples. Position, two candidates (m3_odometry.py measures both):
  "leg": the stance foot stays put, so the trunk moved by minus the stance foot's change in the
         world-aligned frame (joint encoders through the body model + IMU orientation).
  "cmd": dead reckoning from the commanded twist with M0's measured gains (the brief's "from
         commands"), counted only when the command is above the gait's start threshold.
Integration freezes while the trunk is tilted more than 45 deg (a fall) and the estimate is flagged.
"""
import numpy as np

from bodymodel import BodyModel, rot_z, tilt_rotation

SUB_DT = 0.02
CMD_GAIN = {"fwd": 0.44, "back": 0.52}      # achieved/commanded, results/m0/gate.json
FALL_TILT = np.radians(45)


class Odometry:
    def __init__(self, body: BodyModel | None = None, mode="leg", stance_rule="switch_mean", leg_scale=1.3):
        self.body = body or BodyModel()
        self.mode, self.stance_rule, self.leg_scale = mode, stance_rule, leg_scale
        self.x = self.y = self.psi = 0.0
        self.prev_q = None          # world-aligned feet vectors at the previous tick (2x3)
        self.prev_stance = None
        self.degraded = False       # set after a fall
        self.R = np.eye(3)          # current body -> odometry-frame rotation

    def pose(self):
        return self.x, self.y, self.psi

    def state(self):
        return {"x": self.x, "y": self.y, "psi": self.psi, "degraded": self.degraded,
                "prev_q": None if self.prev_q is None else self.prev_q.tolist(), "prev_stance": self.prev_stance}

    def update(self, obs):
        b = obs.body
        gs, gr = np.asarray(b["gyro_substeps"]), np.asarray(b["gravity_substeps"])
        fallen = False
        for w_b, g_b in zip(gs, gr):
            Rt = tilt_rotation(g_b)
            if np.arccos(np.clip(-np.asarray(g_b)[2] / max(np.linalg.norm(g_b), 1e-9), -1, 1)) > FALL_TILT:
                fallen = True
            self.psi += float((rot_z(self.psi) @ Rt @ w_b)[2]) * SUB_DT
        self.psi = float((self.psi + np.pi) % (2 * np.pi) - np.pi)
        self.R = rot_z(self.psi) @ tilt_rotation(gr[-1])
        if fallen:
            self.degraded = True
            self.prev_q = None
            return self.pose()
        if self.mode == "cmd":
            c = obs.last_command
            v = 0.0
            if c.vx >= 0.15:
                v = CMD_GAIN["fwd"] * c.vx
            elif c.vx <= -0.25:
                v = CMD_GAIN["back"] * c.vx
            self.x += v * 0.1 * np.cos(self.psi)
            self.y += v * 0.1 * np.sin(self.psi)
            return self.pose()
        feet, _, _ = self.body.fk(b["joint_pos"])
        q = (self.R @ feet.T).T                     # trunk-origin, odometry-frame-aligned feet vectors
        stance = int(np.argmin(q[:, 2]))
        if self.prev_q is not None:
            if self.stance_rule == "lower" or stance == self.prev_stance:
                s = stance if self.stance_rule == "lower" else stance
                d = self.prev_q[s] - q[s]
            else:                                    # "switch_mean": stance changed within the tick
                d = 0.5 * ((self.prev_q[stance] - q[stance]) + (self.prev_q[self.prev_stance] - q[self.prev_stance]))
            self.x += self.leg_scale * float(d[0])
            self.y += self.leg_scale * float(d[1])
        self.prev_q, self.prev_stance = q, stance
        return self.pose()
