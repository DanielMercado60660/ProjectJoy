"""The reference bot's model of its own body (bot side; see results/m3/bot_contract.md).

A separate kinematic copy of the robot (robot/robot_allcollisions.xml, no room) that turns joint
encoder readings into foot and head-camera poses in the trunk frame. Real robots know their own
kinematics; nothing here reads the simulated world.
"""
from pathlib import Path

import mujoco
import numpy as np

from policy.velstand_np import DEFAULT_POSE

HERE = Path(__file__).resolve().parent
ROBOT_XML = HERE / "robot" / "robot_allcollisions.xml"
CAM_FOVY_DEG = 45.0
FRAME = 128


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def tilt_rotation(gravity_body):
    """Rotation R (body -> yaw-free world) with R @ g_body = (0, 0, -1): the minimal rotation, so the
    heading of the body x-axis is preserved as far as the tilt allows."""
    g = np.asarray(gravity_body, float)
    g = g / max(np.linalg.norm(g), 1e-9)
    t = np.array([0.0, 0.0, -1.0])
    v = np.cross(g, t)
    c = float(g @ t)
    if c < -0.999999:                       # upside down: any 180 deg turn about x
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class BodyModel:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jq = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        # the same 180 deg turn the simulator's Duck applies (the MJCF camera faces backwards)
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, m.cam_quat[self.cam], np.array([0.0, 1.0, 0.0, 0.0]))
        m.cam_quat[self.cam] = fixed
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qadr = int(m.jnt_qposadr[jid])
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    def fk(self, joint_pos_rel):
        """Trunk at the origin, unrotated. Returns feet (2x3) and camera (pos 3, mat 3x3) in the trunk frame."""
        d = self.data
        d.qpos[:] = 0
        d.qpos[self.qadr + 3] = 1.0
        d.qpos[self.jq] = DEFAULT_POSE + np.asarray(joint_pos_rel)
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_camlight(self.model, d)          # camera poses are not part of mj_kinematics
        feet = np.array([d.site_xpos[s] for s in self.feet])
        return feet, d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy()
