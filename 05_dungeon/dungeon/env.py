"""Vectorised numpy grid-maze environment with a partial (egocentric-position) view.

Observation per env (flat float32 vector, OBS_DIM):
  - V x V window of walls centred on the agent (outside the map counts as wall)
  - V x V window with a 1 where the goal is, if it is inside the window
  - one-hot of the previous action (4) and a "bumped into a wall last step" flag (1)
The window is not rotated (actions are absolute N/E/S/W) and there is no occlusion:
the agent sees every cell within Chebyshev radius R = V // 2. The goal is invisible
until it is within that radius, so the agent has to explore and remember.

Reward: 1 - 0.9 * t / T on reaching the goal (MiniGrid-style), 0 otherwise, plus an optional
episodic novelty bonus (`novelty` per first visit of a cell) used only as training shaping.
"""
import numpy as np

from .mazes import SIZE, DIRS

VIEW = 5
R = VIEW // 2
OBS_DIM = 2 * VIEW * VIEW + 4 + 1
MAX_STEPS = 200


def make_obs(walls_pad, pos, goal, prev_a, bumped):
    """walls_pad: (n, S+2R, S+2R) padded with walls."""
    n = pos.shape[0]
    ar = np.arange(VIEW)
    rr = pos[:, 0, None, None] + ar[None, :, None]  # padded coords: pos + R + (k - R)
    cc = pos[:, 1, None, None] + ar[None, None, :]
    win = walls_pad[np.arange(n)[:, None, None], rr, cc].astype(np.float32)
    gwin = np.zeros((n, VIEW, VIEW), np.float32)
    rel = goal - pos + R
    vis = (rel >= 0).all(1) & (rel < VIEW).all(1)
    gwin[np.nonzero(vis)[0], rel[vis, 0], rel[vis, 1]] = 1.0
    pa = np.zeros((n, 4), np.float32)
    has = prev_a >= 0
    pa[np.nonzero(has)[0], prev_a[has]] = 1.0
    return np.concatenate([win.reshape(n, -1), gwin.reshape(n, -1), pa, bumped[:, None].astype(np.float32)], 1)


class VecMaze:
    """n parallel mazes. `sampler(k, rng)` returns k levels (dict walls/start/goal) on reset."""

    def __init__(self, n, sampler, seed=0, max_steps=MAX_STEPS, auto_reset=True, novelty=0.0):
        self.n, self.sampler, self.T, self.auto_reset = n, sampler, max_steps, auto_reset
        self.novelty = novelty
        self.rng = np.random.default_rng(seed)
        S = SIZE
        self.walls = np.ones((n, S, S), np.uint8)
        self.walls_pad = np.ones((n, S + 2 * R, S + 2 * R), np.uint8)
        self.pos = np.zeros((n, 2), np.int64)
        self.goal = np.zeros((n, 2), np.int64)
        self.t = np.zeros(n, np.int64)
        self.prev_a = -np.ones(n, np.int64)
        self.bumped = np.zeros(n, bool)
        self.level_id = np.zeros(n, np.int64)
        self.done = np.zeros(n, bool)
        self.visited = np.zeros((n, S, S), bool)

    def _load(self, idx, lv):
        self.walls[idx] = lv["walls"]
        self.walls_pad[idx, R:R + SIZE, R:R + SIZE] = lv["walls"]
        self.pos[idx] = lv["start"]
        self.goal[idx] = lv["goal"]
        self.t[idx] = 0
        self.prev_a[idx] = -1
        self.bumped[idx] = False
        self.done[idx] = False
        self.visited[idx] = False
        self.visited[idx, self.pos[idx, 0], self.pos[idx, 1]] = True
        if "id" in lv:
            self.level_id[idx] = lv["id"]

    def reset(self, levels=None):
        lv = levels if levels is not None else self.sampler(self.n, self.rng)
        self._load(np.arange(self.n), lv)
        return self.obs()

    def obs(self):
        return make_obs(self.walls_pad, self.pos, self.goal, self.prev_a, self.bumped)

    def step(self, a):
        a = np.asarray(a, np.int64)
        live = ~self.done
        new = self.pos + DIRS[a]
        blocked = self.walls[np.arange(self.n), new[:, 0], new[:, 1]] == 1
        move = live & ~blocked
        self.pos[move] = new[move]
        self.bumped = live & blocked
        self.prev_a = np.where(live, a, self.prev_a)
        self.t += live
        success = live & (self.pos == self.goal).all(1)
        timeout = live & ~success & (self.t >= self.T)
        done = success | timeout
        reward = np.where(success, 1.0 - 0.9 * self.t / self.T, 0.0).astype(np.float32)
        if self.novelty:
            ar = np.arange(self.n)
            new_cell = live & ~self.visited[ar, self.pos[:, 0], self.pos[:, 1]]
            self.visited[ar, self.pos[:, 0], self.pos[:, 1]] = True
            reward = reward + self.novelty * new_cell.astype(np.float32)
        info = dict(success=success, timeout=timeout, steps=self.t.copy(), level_id=self.level_id.copy())
        self.done |= done
        if self.auto_reset and done.any():
            idx = np.nonzero(done)[0]
            self._load(idx, self.sampler(len(idx), self.rng))
        return self.obs(), reward, done, info
