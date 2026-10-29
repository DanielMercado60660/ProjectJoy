"""Checks that (1) the gymfix monkeypatch makes the official env run and returns the same values as
direct qpos/qvel slicing, and (2) egg_env's base observation and dynamics match the patched gym
HandManipulateEggRotate_ContinuousTouchSensors-v1 on identical states."""
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import gymfix  # noqa: F401  (applies the patch)
import gymnasium as gym
import gymnasium_robotics
import mujoco

from egg_env import EggVecEnv

gym.register_envs(gymnasium_robotics)


def _gym_env():
    return gym.make("HandManipulateEggRotate_ContinuousTouchSensors-v1").unwrapped


def test_patch_matches_direct_slicing():
    env = _gym_env()
    env.reset(seed=1)
    m, d = env.model, env.data
    for j in range(m.njnt):
        name = m.joint(j).name
        n = 7 if m.jnt_type[j] == 0 else 1
        nv = 6 if m.jnt_type[j] == 0 else 1
        a, b = m.jnt_qposadr[j], m.jnt_dofadr[j]
        assert np.array_equal(gymfix.get_joint_qpos(m, d, name), d.qpos[a:a + n])
        assert np.array_equal(gymfix.get_joint_qvel(m, d, name), d.qvel[b:b + nv])


def test_obs_and_step_parity():
    genv = _gym_env()
    mine = EggVecEnv(1, seed=0, threads=1, fric=(1, 1), mass=(1, 1))
    rng = np.random.default_rng(0)
    for ep in range(3):
        go, _ = genv.reset(seed=ep)
        gd, md, mm = genv.data, mine.datas[0], mine.models[0]
        for k in range(20):
            # copy the full gym state into our env, then compare observation and one control step
            md.qpos[:], md.qvel[:], md.ctrl[:] = gd.qpos, gd.qvel, gd.ctrl
            md.qacc_warmstart[:] = gd.qacc_warmstart
            md.time = gd.time
            mujoco.mj_forward(mm, md)
            # first 61 dims (joints, object) must match exactly; touch dims (61:) are sensor values
            # computed inside mj_step, which mj_forward recomputes at the new state, so copy them
            assert np.allclose(mine.base_obs(0)[:61], genv._get_obs()["observation"][:61], atol=1e-9)
            md.sensordata[:] = gd.sensordata
            assert np.array_equal(mine.base_obs(0), genv._get_obs()["observation"])
            a = rng.uniform(-1, 1, 20) * 0.3
            go, *_ = genv.step(a)
            mine.done[:] = False
            mine._step_one(0, a)
            assert np.allclose(mine.base_obs(0), go["observation"], atol=1e-6), k
