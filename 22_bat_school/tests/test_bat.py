import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from bat_env import BatEnv, MazePool, WallFollower, decode_open, OBS_DIM, bfs
from evaluate import HELDOUT
from model import BatPolicy
from train import TRAIN_SEED


def test_heldout_disjoint_from_training():
    train = set(MazePool([4, 5, 6], 3000, seed=TRAIN_SEED, rays=False).hashes)
    for s, seed in HELDOUT.items():
        held = MazePool([s], 200, seed=seed, rays=False).hashes
        assert not (set(held) & train), f"size {s} held-out overlaps training"
        assert len(set(held)) >= 0.98 * len(held)   # a rare duplicate inside the held-out pool is fine


def test_maze_is_perfect_and_echo_decodes():
    pool = MazePool([5], 5, seed=3)
    for m in range(5):
        assert (bfs(pool.open[m, :5, :5], (0, 0)) >= 0).all()      # connected
        assert pool.open[m].sum() == 2 * (25 - 1)                    # tree: 24 edges
    env = BatEnv(pool, 64, seed=0)
    o = env.reset()
    assert o.shape == (64, OBS_DIM)
    true = np.stack([pool.open[env.m, env.r, env.c, (env.hd + d) % 4] for d in range(4)], 1)
    assert (decode_open(o) == true).mean() > 0.97


def test_wall_follower_solves_with_full_sensing():
    pool = MazePool([5], 20, seed=11)
    env = BatEnv(pool, 20, seed=0, fixed_order=True)
    ag = WallFollower(20, 1)
    o = env.reset(); first = env.ep_id.copy(); res = {}
    while len(res) < 20:
        mv, pu = ag.act(o); o, r, d, info = env.step(mv, pu); ag.on_done(d)
        for i in np.nonzero(d)[0]:
            if info["ep_id"][i] == first[i]:
                res.setdefault(int(first[i]), bool(info["success"][i]))
    assert np.mean(list(res.values())) >= 0.95


def test_policy_sequence_matches_steps():
    m = BatPolicy(OBS_DIM, 32)
    x = mx.random.normal((5, 3, OBS_DIM))
    st = mx.array(np.array([[1, 1, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]], np.float32))
    lm, lp, v = m.sequence(x, mx.zeros((3, 32)), st)
    h = mx.zeros((3, 32))
    for t in range(5):
        h = h * (1 - st[t])[:, None]
        a, b, c, h = m.step(x[t], h)
        assert np.allclose(np.array(a), np.array(lm[t]), atol=1e-4)
