"""Workaround for gymnasium-robotics 1.4.2 + mujoco 3.16 (applied at import, in-process only).

The bug: `mujoco_utils.get_joint_qpos` and friends test
    joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
where `joint_type` is a numpy.int32 read from `model.jnt_type`. Under mujoco 3.16 the enum's
`__eq__` with numpy.int32 makes tuple membership return False, even though
`np.int32(3) == mjJNT_HINGE` is True. So every *hinge* joint of the hand trips the assert
(the shelf note guessed a free joint; it is the hinges). We cast the type to a plain int.
Nothing in site-packages is edited; the patch replaces the four module functions in memory.
"""
import mujoco
import numpy as np
from gymnasium_robotics.utils import mujoco_utils as _mu

_FREE, _BALL = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)
_HINGE, _SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)


def _jid(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert j != -1, f"Joint with name '{name}' is not part of the model!"
    return j, int(model.jnt_type[j])


def _ndim(t, vel):
    if t == _FREE:
        return 6 if vel else 7
    if t == _BALL:
        # upstream returns 4 for BALL qvel in get_joint_qvel (a bug); keep upstream behaviour
        return 3 if vel else 4
    assert t in (_HINGE, _SLIDE), t
    return 1


def get_joint_qpos(model, data, name):
    j, t = _jid(model, name)
    a = model.jnt_qposadr[j]
    return data.qpos[a:a + _ndim(t, False)].copy()


def get_joint_qvel(model, data, name):
    j, t = _jid(model, name)
    a = model.jnt_dofadr[j]
    n = 4 if t == _BALL else _ndim(t, True)  # upstream quirk preserved
    return data.qvel[a:a + n].copy()


def set_joint_qpos(model, data, name, value):
    j, t = _jid(model, name)
    a, n = model.jnt_qposadr[j], _ndim(t, False)
    value = np.array(value)
    if n > 1:
        assert value.shape == (n,), f"Value has incorrect shape {name}: {value}"
    data.qpos[a:a + n] = value


def set_joint_qvel(model, data, name, value):
    j, t = _jid(model, name)
    a, n = model.jnt_dofadr[j], _ndim(t, True)
    value = np.array(value)
    if n > 1:
        assert value.shape == (n,), f"Value has incorrect shape {name}: {value}"
    data.qvel[a:a + n] = value


def apply():
    _mu.get_joint_qpos = get_joint_qpos
    _mu.get_joint_qvel = get_joint_qvel
    _mu.set_joint_qpos = set_joint_qpos
    _mu.set_joint_qvel = set_joint_qvel


apply()
