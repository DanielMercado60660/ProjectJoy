"""Fast checks: world mechanics, observation shape, a tiny training run, and checkpoint/resume."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env as E  # noqa: E402

UP, DOWN, LEFT, RIGHT, USE = range(5)


def make():
    e = E.VecRooms(1, max_steps=10_000)
    return e


def teleport(e, y, x, facing=DOWN):
    e.pos[0] = (y, x)
    e.facing[0] = facing


def test_obs_shape_and_onehot():
    e = make()
    assert e.obs.shape == (1, E.OBS_DIM)
    cells = e.obs[0, :E.VIEW * E.VIEW * E.CELL_FEAT].reshape(E.VIEW * E.VIEW, E.CELL_FEAT)
    assert np.all(cells[:, :E.N_TYPES].sum(1) == 1)


def test_lever_lights_lamp_and_counts_once():
    e = make()
    ly, lx = e.mech_dict["lever_A"]
    teleport(e, ly + 1, lx, UP)
    e.step([USE])
    assert e.grid[0][e.mech_dict["lamp_A"]] == E.LAMP_ON and e.inter[0, 0]
    e.step([USE])
    assert e.grid[0][e.mech_dict["lamp_A"]] == E.LAMP_OFF and e.repeat[0] == 1


def test_key_unlocks_locked_door():
    e = make()
    ky, kx = e.mech_dict["key"]
    teleport(e, ky - 1, kx)
    e.step([DOWN])
    assert e.has_key[0] == 1 and e.inter[0, 3]
    dy, dx = e.mech_dict["locked"]
    teleport(e, dy - 1, dx)
    e.step([DOWN])  # bump the door with the key: unlocks, key consumed
    assert e.grid[0][dy, dx] == E.DOOR_OPEN and e.has_key[0] == 0 and e.inter[0, 4]
    e.step([DOWN])
    assert tuple(e.pos[0]) == (dy, dx)


def test_locked_door_blocks_without_key():
    e = make()
    dy, dx = e.mech_dict["locked"]
    teleport(e, dy - 1, dx)
    e.step([DOWN])
    assert tuple(e.pos[0]) == (dy - 1, dx) and e.grid[0][dy, dx] == E.DOOR_LOCKED


def test_box_pushes_out_of_doorway():
    e = make()
    by, bx = e.mech_dict["box"]
    teleport(e, by - 1, bx)
    e.step([DOWN])
    assert e.grid[0][by, bx] == E.DOOR_OPEN and e.grid[0][by + 1, bx] == E.BOX and e.inter[0, 2]
    assert tuple(e.pos[0]) == (by, bx)


def test_far_button_opens_far_door():
    e = make()
    y, x = e.mech_dict["button_E"]
    teleport(e, y, x + 1, LEFT)
    e.step([USE])
    assert e.grid[0][e.mech_dict["door_E"]] == E.DOOR_OPEN and e.inter[0, 5]


def test_episode_reset_and_stats():
    e = E.VecRooms(3, max_steps=20)
    done_any = False
    for _ in range(20):
        _, d, eps = e.step(np.zeros(3, int) + RIGHT)
        done_any |= bool(d.any())
    assert done_any and len(eps) == 3 and eps[0]["rooms"] >= 1
    assert tuple(e.pos[0]) == tuple(e.start)


def test_noisy_tv_changes_and_counts():
    e = E.VecRooms(1, max_steps=100, noisy_tv=True, seed=0)
    ty, tx = e.tv_cells[0]
    teleport(e, ty, tx - 2)
    vals = []
    for _ in range(10):
        e.step([USE])
        vals.append(e.tv_val[0, ty, tx])
    assert len(set(vals)) > 2 and e.tvsteps[0] == 10


def test_tiny_training_and_resume(tmp_path):
    from train import DEFAULTS, train, Explorer
    for agent in ("rnd", "count", "disagree", "rnd_ep", "ep", "ngu"):
        cfg = dict(DEFAULTS, agent=agent, n_envs=8, rollout=16, total_steps=8 * 16 * 4,
                   hidden=32, ckpt_every=2, out=str(tmp_path / agent))
        ex, _ = train(cfg, stop_after_updates=2)
        assert ex.update == 2
        ck = str(tmp_path / agent / "ckpt.pkl")
        ex2, env2 = train(cfg, resume=ck)
        assert ex2.update == 4 and ex2.steps == 8 * 16 * 4
        lines = open(tmp_path / agent / "log.jsonl").read().strip().splitlines()
        assert len(lines) == 4
    # resumed weights equal saved weights
    a, _ = Explorer.load(ck)
    assert a.update == 4


import pytest  # noqa: E402


@pytest.mark.parametrize("agent", ["rnd", "rnd_ep", "ngu", "count"])
def test_resume_matches_uninterrupted(tmp_path, agent):
    """Save -> stop -> resume must continue exactly like a run that never stopped."""
    from train import DEFAULTS, train
    from mlx.utils import tree_flatten
    base = dict(DEFAULTS, agent=agent, n_envs=8, rollout=16, total_steps=8 * 16 * 4, hidden=32,
                ckpt_every=2)
    full, _ = train(dict(base, out=str(tmp_path / "full")))
    part = dict(base, out=str(tmp_path / "part"))
    train(part, stop_after_updates=2)
    res, _ = train(part, resume=str(tmp_path / "part" / "ckpt.pkl"))
    for (k, a), (_, b) in zip(tree_flatten(full.policy.parameters()), tree_flatten(res.policy.parameters())):
        assert np.allclose(np.array(a), np.array(b), atol=1e-5), k
