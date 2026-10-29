"""Parity: our copy of the Dungeon PPO-GRU step (numpy + MLX) and env against 05_dungeon's own code
(verification only)."""
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB = os.path.join(HERE, "..", "05_dungeon")
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import dungeon as dg  # noqa: E402


def _sibling():
    if SIB not in sys.path:
        sys.path.insert(0, SIB)
    from dungeon import env, policy, rollout  # noqa: E402
    return env, policy, rollout


def test_step_parity():
    _, policy, _ = _sibling()
    ref = policy.load_policy(os.path.join(SIB, "checkpoints", "solver_cur3", "weights.safetensors"))
    rng = np.random.default_rng(0)
    obs = (rng.random((128, dg.OBS_DIM)) < 0.4).astype(np.float32)
    h = rng.uniform(-1, 1, (128, dg.HID)).astype(np.float32)
    with mx.stream(mx.cpu):
        r = [np.array(a) for a in ref.step(mx.array(obs), mx.array(h))]
        ours = [np.array(a) for a in dg.load_mlx()(mx.array(obs), mx.array(h))]
    for a, b in zip(ours, r):
        assert np.abs(a - b).max() < 1e-5
    for a, b in zip(dg.NumpyPolicy()(obs, h), r):
        assert np.abs(a - b).max() < 1e-4


def test_rollout_parity():
    """Greedy rollouts: our env + numpy policy escape exactly the levels 05's run_policy does,
    in the same number of steps."""
    _, policy, rollout = _sibling()
    ref = policy.load_policy(os.path.join(SIB, "checkpoints", "solver_cur3", "weights.safetensors"))
    levels = dg.load_levels(6)
    with mx.stream(mx.cpu):
        r = rollout.run_policy(ref, levels, greedy=True)
    pol = dg.NumpyPolicy()
    env = dg.VecMaze(levels)
    h = np.zeros((env.n, dg.HID), np.float32)
    succ = np.zeros(env.n, bool)
    for _ in range(dg.MAX_STEPS):
        live = ~env.done
        logits, _, h2 = pol(env.obs(), h)
        h = np.where(live[:, None], h2, h).astype(np.float32)
        succ |= env.step(logits.argmax(-1))[1]
        if env.done.all():
            break
    np.testing.assert_array_equal(succ, r["success"])
    np.testing.assert_array_equal(env.t[succ], r["steps"][succ])
