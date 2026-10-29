import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dungeon.mazes import ALL_FAMILIES, make_levels, is_valid, bfs_path, SIZE  # noqa: E402
from dungeon.env import VecMaze, OBS_DIM, VIEW  # noqa: E402
from dungeon.designer import make_child, Stats, solvable  # noqa: E402
from dungeon.scripted import run_batch, oracle_lengths  # noqa: E402


def test_families_valid():
    for f in ALL_FAMILIES:
        L = make_levels(f, 40, seed=5)
        for w, s, g in zip(L["walls"], L["start"], L["goal"]):
            assert is_valid(w, s, g, min_dist=6)
            p = bfs_path(w, s[0], s[1], g[0], g[1])
            assert tuple(p[0]) == tuple(s) and tuple(p[-1]) == tuple(g)
            assert all(w[r, c] == 0 for r, c in p)


def test_env_follows_bfs_path():
    L = make_levels("dfs", 8, seed=1)
    env = VecMaze(8, None, auto_reset=False)
    obs = env.reset(L)
    assert obs.shape == (8, OBS_DIM)
    paths = [bfs_path(w, s[0], s[1], g[0], g[1]) for w, s, g in zip(L["walls"], L["start"], L["goal"])]
    T = max(len(p) for p in paths) - 1
    succ = np.zeros(8, bool)
    for t in range(T):
        a = np.zeros(8, np.int64)
        for i, p in enumerate(paths):
            if t + 1 < len(p):
                d = p[t + 1] - p[t]
                a[i] = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[tuple(d)]
        obs, r, done, info = env.step(a)
        succ |= info["success"]
    assert succ.all()
    assert (env.t == np.array([len(p) - 1 for p in paths])).all()


def test_walls_block_and_goal_visible():
    w = np.ones((SIZE, SIZE), np.uint8)
    w[1, 1:6] = 0
    L = dict(walls=w[None], start=np.array([[1, 1]]), goal=np.array([[1, 3]]))
    env = VecMaze(1, None, auto_reset=False)
    obs = env.reset(L)
    gwin = obs[0, VIEW * VIEW:2 * VIEW * VIEW].reshape(VIEW, VIEW)
    assert gwin[2, 4] == 1  # goal two cells to the east
    obs, r, done, info = env.step(np.array([0]))  # north is wall
    assert tuple(env.pos[0]) == (1, 1) and obs[0, -1] == 1  # bumped flag
    env.step(np.array([1]))
    obs, r, done, info = env.step(np.array([1]))
    assert done[0] and info["success"][0] and r[0] > 0.9


def test_constrained_designer_never_invalid():
    rng = np.random.default_rng(0)
    L = make_levels("dfs", 20, seed=3)
    st = Stats()
    for i in range(20):
        w, s, g = L["walls"][i], L["start"][i], L["goal"][i]
        for _ in range(30):
            w, s, g = make_child(w, s, g, rng, True, st)
            assert solvable(w, s, g)
            assert w[0].all() and w[-1].all() and w[:, 0].all() and w[:, -1].all()
    assert st.valid < st.proposed  # some raw proposals really were invalid


def test_tremaux_solves_perfect_mazes():
    L = make_levels("dfs", 50, seed=2)
    o = run_batch("tremaux", L)
    assert o["success"].all()
    assert (o["steps"] >= oracle_lengths(L)).all()
