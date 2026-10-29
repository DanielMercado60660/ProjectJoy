import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lava_env as E
import task as T


def test_ik_home_pose():
    q1, q2 = T.leg_ik(0.0, -0.2650)
    assert abs(q1 - 0.9) < 0.01 and abs(q2 + 1.8) < 0.01


def test_layout_shapes_and_gaps():
    L = E.make_layout(3)
    assert L.shape == (E.N_SLOTS, 5)
    gaps = L[1:, 0] - L[:-1, 1]
    lo, hi = E.LAYOUT_CFG["gap"]
    assert np.all(gaps >= lo - 1e-9) and np.all(gaps <= hi + 1e-9)
    assert np.allclose(E.make_layout(3), L)          # deterministic per seed
    flat = E.make_layout(3, flat=True)
    assert np.allclose(flat[1:, 0], flat[:-1, 1])     # no gaps


def test_resized_stone_collides_and_lava_detects():
    env = E.LavaEnv()
    L = E.make_layout(1)
    L[0] = [-0.3, 0.9, 0.25, 0.5, 0.1]   # moved + resized: the compiled 0.2 m box would miss every foot
    env.reset(L)
    for _ in range(50):
        env.step_targets(E.HOME_Q)
    assert not env.lava and env.foot_contacts().sum() == 4
    L[0] = [0.3, 1.5, 0.0, 0.5, 0.1]     # rear feet over lava
    env.reset(L)
    for _ in range(50):
        env.step_targets(E.HOME_Q)
    assert env.lava


def test_zero_residual_is_cpg_and_deterministic():
    tk = T.LavaTask()
    L = E.make_layout(5)
    r1 = T.run_episode(tk, L, rng=np.random.default_rng(0), init_noise=0.05)
    r2 = T.run_episode(tk, L, policy=lambda o: np.zeros(T.ACT_DIM), rng=np.random.default_rng(0), init_noise=0.05)
    assert r1["x"] == r2["x"] and r1["t"] == r2["t"]
    assert len(tk.reset(L)) == T.LavaTask.obs_dim
