import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sim


def test_speed_caps_and_stamina():
    env = sim.Arena(64, seed=0)
    a = np.full(64, 9)                      # sprint east forever
    for _ in range(200):
        env.step(a, a)
        sp = np.linalg.norm(env.vel, axis=2)
        assert (sp[:, 0] <= sim.P_SPRINT + 1e-9).all() and (sp[:, 1] <= sim.E_SPRINT + 1e-9).all()
    assert env.stam.min() >= 0.0
    env2 = sim.Arena(8, seed=1)
    for _ in range(int(3.0 / sim.DT)):      # 3 s of sprinting drains a 2.5 s tank
        env2.step(np.full(8, 9), np.full(8, 9))
    assert (env2.stam[env2.t > 50] < 0.05).all()


def test_walls_and_obstacles_hold():
    env = sim.Arena(256, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(300):
        env.step(rng.integers(0, sim.N_ACT, 256), rng.integers(0, sim.N_ACT, 256))
        assert (np.abs(env.pos) <= sim.L).all()
        for ox, oy, r in sim.OBSTACLES:
            d = np.linalg.norm(env.pos - np.array([ox, oy]), axis=2)
            assert (d >= r + sim.BODY_R - 1e-6).all()


def test_capture_ends_episode_and_rewards_are_zero_sum_up_to_effort():
    env = sim.Arena(4, seed=3)
    env.pos[:, 0] = [0.0, 1.5]; env.pos[:, 1] = [0.1, 1.5]
    rp, re, done, cap, _ = env.step(np.zeros(4, int), np.zeros(4, int))
    assert cap.all() and done.all() and (env.t == 0).all()   # auto-reset
    assert np.allclose(rp + re, 0, atol=1e-6)                  # no thrust -> exactly zero-sum


def test_obs_shape_and_scripted_actions():
    env = sim.Arena(16, seed=4)
    op, oe = env.obs()
    assert op.shape == (16, sim.OBS_DIM) and oe.shape == (16, sim.OBS_DIM)
    for k in sim.PURSUER_SUITE:
        a = sim.Scripted(k, "p").act(env); assert a.shape == (16,) and a.max() < sim.N_ACT
    for k in sim.EVADER_SUITE:
        a = sim.Scripted(k, "e").act(env); assert a.shape == (16,) and a.max() < sim.N_ACT


def test_policy_roundtrip(tmp_path):
    from policy import new_model, get_np, save_pair, load_pair, set_np, act
    m = new_model(0)
    save_pair(tmp_path / "x.npz", get_np(m), get_np(m), {"iter": 3})
    p, e, meta = load_pair(tmp_path / "x.npz")
    m2 = new_model(1); set_np(m2, p)
    obs = np.random.default_rng(0).normal(size=(5, sim.OBS_DIM)).astype(np.float32)
    rng = np.random.default_rng(0)
    assert (act(m, obs, rng, True)[0] == act(m2, obs, rng, True)[0]).all() and int(meta["iter"]) == 3
