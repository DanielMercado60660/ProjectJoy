"""M0 checks: policy port parity, observation layout, delay buffers, determinism, camera, a short walk."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duck import CTRL_DT, BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY, goto_command  # noqa: E402
from policy.velstand_np import DEFAULT_ONNX, DEFAULT_POSE, VelstandPolicy  # noqa: E402


@pytest.fixture(scope="module")
def policy():
    return VelstandPolicy()


def test_parity_with_onnxruntime(policy):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = (policy.mean + policy.std * rng.standard_normal((64, 61))).astype(np.float32)
    ref = np.concatenate([sess.run(None, {"obs": o[None]})[0] for o in obs])
    assert np.abs(ref - policy(obs)).max() <= 1e-5


def test_observation_layout(policy):
    duck = Duck(BodyConfig(), policy=policy)
    duck.set_command(0.3, 0.0, -0.5, head=(0.1, 0.2, 0.3, 0.05))
    obs = duck.observation()
    assert obs.shape == (61,) and obs.dtype == np.float32
    assert np.allclose(obs[3:6], [0, 0, -1], atol=1e-6)          # projected gravity, upright at reset
    assert np.allclose(obs[6:20], 0, atol=1e-6)                  # q - DEFAULT_POSE at reset
    assert np.allclose(obs[34:48], 0)                            # last action
    assert np.allclose(obs[48:51], [0.3, 0.0, -0.5])             # twist
    assert np.allclose(obs[51:55], [0.1, 0.2, 0.3, 0.05])        # head pose deltas
    assert np.allclose(obs[55:61], 0)                            # body pose
    assert np.allclose(duck.data.ctrl, DEFAULT_POSE)


def test_head_command_is_capped(policy):
    duck = Duck(BodyConfig(), policy=policy)
    duck.set_command(head=(5, -5, 5, 5))
    assert np.allclose(duck.command[3:7], [1.1, -1.1, 1.4, 0.31])


def test_policy_rate_delay_buffer(policy):
    duck = Duck(BodyConfig(delay_min=2, delay_max=2), policy=policy)
    duck.set_command(0.3)
    actions = [duck.step()[1] for _ in range(5)]
    assert duck.lag == 2
    assert np.array_equal(duck.action_buf[0], actions[-3])
    assert np.allclose(duck.data.ctrl, DEFAULT_POSE + actions[-3])


def test_physics_rate_delay(policy):
    duck = Duck(BodyConfig(phys_delay=4), policy=policy)
    duck.set_command(0.3)
    _, a1 = duck.step()
    # 4 physics steps per control step and a 4-step lag: the first action reaches ctrl
    # exactly at the start of the next control step.
    assert np.allclose(duck.data.ctrl, DEFAULT_POSE)
    _, a2 = duck.step()
    assert np.allclose(duck.data.ctrl, DEFAULT_POSE + a1)


def test_bit_exact_replay(policy):
    def run():
        duck = Duck(BodyConfig(**FROZEN_BODY), policy=policy, seed=3)
        for cmd in ((0.3, 0, 0), (0, 0, 0.6), (0.2, 0, -0.5)):
            duck.set_command(*cmd)
            for _ in range(40):
                duck.step()
        return duck.data.qpos.copy(), duck.data.qvel.copy()
    (q1, v1), (q2, v2) = run(), run()
    assert np.array_equal(q1, q2) and np.array_equal(v1, v2)


def test_head_camera_looks_forward(policy):
    duck = Duck(BodyConfig(), policy=policy)
    mujoco.mj_forward(duck.model, duck.data)
    look = -duck.data.cam_xmat[duck.cam_id].reshape(3, 3)[:, 2]
    assert look[0] > 0.9  # duck faces +x at yaw 0
    r = mujoco.Renderer(duck.model, 64, 64)
    img = duck.head_image(r)
    assert img.shape == (64, 64, 3) and img.std() > 1.0  # not a blank frame


def test_walks_to_near_pad(policy):
    duck = Duck(BodyConfig(**FROZEN_BODY), policy=policy)
    target = np.array([0.5, 0.0])
    for _ in range(int(15 / CTRL_DT)):
        if np.linalg.norm(duck.pos()[:2] - target) < 0.10:
            break
        duck.set_command(*goto_command(duck, target))
        duck.step()
    assert np.linalg.norm(duck.pos()[:2] - target) < 0.10
    assert duck.tilt() < np.radians(20)
