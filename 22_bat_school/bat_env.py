"""Bat School environment: grid mazes seen only through simulated echoes.

World
  A perfect maze of h x w cells, stored as an occupancy grid of shape (2h+1, 2w+1) (1 = rock).
  The bat sits at a cell centre facing N/E/S/W. Moves: 0 = forward one cell, 1 = turn left,
  2 = turn right. Walking into rock is a collision: the bat stays put and feels a bump.
  The exit of the cave holds a moth (the goal). Reaching the moth's cell ends the episode.

Echo model (the only sense besides touch)
  When the bat pulses, K = 32 rays leave its head. Each ray travels to the first rock surface
  (distance d, in corridor widths) and returns. Time of flight is 2d / c, so the arrival time
  bin is proportional to d. Amplitude falls with spreading and absorption: a = exp(-0.1 d) / (1 + 0.5 d).
  Rays are grouped into 8 head-relative sectors (front, front-right, right, ...), and arrivals are
  binned into 12 time bins, giving an "echo profile" of 8 x 12 for rock plus 8 x 12 for the moth
  (a moth echo is a distinct spectral signature, heard when a ray reaches the moth before rock).
  Noise: Gaussian noise on every bin and per-ray dropout. No pulse = silence (zeros).

The policy never sees the map, its position, or its heading. Reward shaping (new-cell bonus) uses
true state, which is allowed because reward is not an observation.
"""
from __future__ import annotations

import hashlib
import numpy as np

K_RAYS = 32
N_SECT = 8
N_BINS = 12
BIN_W = 1.0          # distance units (corridor widths) per time bin
MAX_RANGE = N_BINS * BIN_W
MOTH_R = 0.35
HMAX = 8
DR = np.array([-1, 0, 1, 0])  # N E S W
DC = np.array([0, 1, 0, -1])

# ray angles, offset by half a step so sectors are symmetric about the head axis
_ANG = (np.arange(K_RAYS) + 0.5) * 2 * np.pi / K_RAYS        # 0 = north, clockwise
RAY_DIR = np.stack([-np.cos(_ANG), np.sin(_ANG)], 1)          # (row, col) unit vectors
# relative ray j belongs to sector ((j + 2) // 4) % 8  -> sector 0 = rays -2..1 = front
RAY_SECT = ((np.arange(K_RAYS) + 2) // 4) % N_SECT
OBS_DIM = 2 * N_SECT * N_BINS + 7


def amp(d):
    return np.exp(-0.1 * d) / (1.0 + 0.5 * d)


# ----------------------------------------------------------------------------- mazes
def make_maze(h, w, rng):
    """Perfect maze by randomized DFS. Returns open[h, w, 4] (N, E, S, W) booleans."""
    opn = np.zeros((h, w, 4), bool)
    seen = np.zeros((h, w), bool)
    stack = [(int(rng.integers(h)), int(rng.integers(w)))]
    seen[stack[0]] = True
    while stack:
        r, c = stack[-1]
        nb = [d for d in range(4) if 0 <= r + DR[d] < h and 0 <= c + DC[d] < w and not seen[r + DR[d], c + DC[d]]]
        if not nb:
            stack.pop()
            continue
        d = nb[int(rng.integers(len(nb)))]
        r2, c2 = r + DR[d], c + DC[d]
        opn[r, c, d] = True
        opn[r2, c2, (d + 2) % 4] = True
        seen[r2, c2] = True
        stack.append((r2, c2))
    return opn


def occupancy(opn):
    h, w, _ = opn.shape
    g = np.ones((2 * h + 1, 2 * w + 1), np.uint8)
    for r in range(h):
        for c in range(w):
            g[2 * r + 1, 2 * c + 1] = 0
            if opn[r, c, 1]:
                g[2 * r + 1, 2 * c + 2] = 0
            if opn[r, c, 2]:
                g[2 * r + 2, 2 * c + 1] = 0
    return g


def cell_xy(r, c):
    return np.stack([2 * np.asarray(r) + 1.5, 2 * np.asarray(c) + 1.5], -1).astype(np.float64)


def ray_distances(grid, h, w, step=0.02):
    """Distance to first rock along each absolute ray from every cell centre: (h, w, K)."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    p = cell_xy(rr, cc).reshape(-1, 1, 1, 2)                       # (C,1,1,2)
    ts = np.arange(step, MAX_RANGE + 1.0, step)                     # (S,)
    pts = p + ts[None, None, :, None] * RAY_DIR[None, :, None, :]   # (C,K,S,2)
    iy = np.clip(np.floor(pts[..., 0]).astype(int), 0, grid.shape[0] - 1)
    ix = np.clip(np.floor(pts[..., 1]).astype(int), 0, grid.shape[1] - 1)
    hit = grid[iy, ix] == 1
    first = np.where(hit.any(-1), hit.argmax(-1), len(ts) - 1)
    d = ts[first]
    return d.reshape(h, w, K_RAYS)


def maze_hash(opn):
    return hashlib.sha1(np.ascontiguousarray(opn).tobytes() + bytes(opn.shape[:2])).hexdigest()


def bfs(opn, start):
    h, w, _ = opn.shape
    dist = -np.ones((h, w), int)
    dist[start] = 0
    q = [start]
    for r, c in q:
        for d in range(4):
            if opn[r, c, d]:
                n = (r + DR[d], c + DC[d])
                if dist[n] < 0:
                    dist[n] = dist[r, c] + 1
                    q.append(n)
    return dist


class MazePool:
    """A fixed set of mazes with precomputed ray distances, padded to HMAX x HMAX."""

    def __init__(self, sizes, n, seed, rays=True):
        rng = np.random.default_rng(seed)
        self.n = n
        self.hw = np.zeros((n, 2), int)
        self.open = np.zeros((n, HMAX, HMAX, 4), bool)
        self.dist = np.full((n, HMAX, HMAX, K_RAYS), 0.5)
        self.grids = []
        self.hashes = []
        for i in range(n):
            s = int(sizes[i % len(sizes)])
            o = make_maze(s, s, rng)
            g = occupancy(o)
            self.hw[i] = (s, s)
            self.open[i, :s, :s] = o
            if rays:
                self.dist[i, :s, :s] = ray_distances(g, s, s)
            self.grids.append(g)
            self.hashes.append(maze_hash(o))
        self.seed = seed


# ----------------------------------------------------------------------------- env
class BatEnv:
    """Vectorized bat-in-a-maze. step(move, pulse) -> obs, reward, done, info."""

    def __init__(self, pool: MazePool, n_envs, seed=0, steps_per_cell=6, pulse_cost=0.01,
                 collision_cost=0.1, step_cost=0.01, new_cell_bonus=0.03, goal_reward=1.0,
                 noise=0.03, dropout=0.05, pulse_budget=None, fixed_order=False):
        self.pool, self.N = pool, n_envs
        self.rng = np.random.default_rng(seed)
        self.spc = steps_per_cell
        self.pulse_cost, self.collision_cost, self.step_cost = pulse_cost, collision_cost, step_cost
        self.new_cell_bonus, self.goal_reward = new_cell_bonus, goal_reward
        self.noise, self.dropout = noise, dropout
        self.pulse_budget = pulse_budget
        self.fixed_order = fixed_order       # eval: maze i goes to the i-th episode, in order
        self.next_maze = 0
        N = n_envs
        self.m = np.zeros(N, int); self.r = np.zeros(N, int); self.c = np.zeros(N, int)
        self.hd = np.zeros(N, int); self.gr = np.zeros(N, int); self.gc = np.zeros(N, int)
        self.t = np.zeros(N, int); self.T = np.zeros(N, int)
        self.pulses = np.zeros(N, int); self.colls = np.zeros(N, int)
        self.visited = np.zeros((N, HMAX, HMAX), bool)
        self.prev_move = np.zeros(N, int); self.prev_pulse = np.zeros(N, int)
        self.bump = np.zeros(N, bool)
        self.ep_id = -np.ones(N, int)

    # --- episode setup
    def _reset_one(self, i):
        if self.fixed_order:
            m = self.next_maze % self.pool.n
            self.ep_id[i] = self.next_maze
            self.next_maze += 1
            erng = np.random.default_rng(10_000 + m)       # start/goal fixed per eval maze
        else:
            m = int(self.rng.integers(self.pool.n))
            erng = self.rng
        h, w = self.pool.hw[m]
        opn = self.pool.open[m, :h, :w]
        while True:
            g = (int(erng.integers(h)), int(erng.integers(w)))
            dist = bfs(opn, g)
            cand = np.argwhere(dist >= max(3, (h + w) // 2))
            if len(cand):
                break
        s = cand[int(erng.integers(len(cand)))]
        self.m[i], self.r[i], self.c[i] = m, s[0], s[1]
        self.gr[i], self.gc[i] = g
        self.hd[i] = int(erng.integers(4))
        self.t[i] = 0
        self.T[i] = self.spc * h * w
        self.pulses[i] = 0; self.colls[i] = 0
        self.visited[i] = False
        self.visited[i, s[0], s[1]] = True
        self.prev_move[i] = 0; self.prev_pulse[i] = 0; self.bump[i] = False

    def reset(self):
        for i in range(self.N):
            self._reset_one(i)
        heard = np.zeros(self.N, bool)
        # first observation: the bat gets one free pulse on arrival
        return self._obs(np.ones(self.N, bool) if True else heard)

    # --- echoes
    def echo(self, idx=None):
        """Echo profiles (wall, moth) for envs idx: (n, 8, 12) each, noisy."""
        if idx is None:
            idx = np.arange(self.N)
        n = len(idx)
        roll = (np.arange(K_RAYS)[None, :] + 8 * self.hd[idx, None]) % K_RAYS        # rel -> abs
        dw = self.pool.dist[self.m[idx, None], self.r[idx, None], self.c[idx, None], roll]  # (n,K)
        # moth: line of sight from bat to moth before rock
        p = cell_xy(self.r[idx], self.c[idx])
        g = cell_xy(self.gr[idx], self.gc[idx])
        u = RAY_DIR[roll]                                    # (n,K,2)
        rel = (g - p)[:, None, :]
        tproj = (rel * u).sum(-1)
        perp = np.abs(rel[..., 0] * u[..., 1] - rel[..., 1] * u[..., 0])
        moth = (tproj > 0) & (tproj < dw) & (perp < MOTH_R) & (tproj < MAX_RANGE)
        keep = self.rng.random((n, K_RAYS)) >= self.dropout
        prof = np.zeros((n, 2, N_SECT * N_BINS))
        for ch, d, valid in ((0, dw, keep & (dw < MAX_RANGE)), (1, tproj, keep & moth)):
            b = d / BIN_W - 0.5
            i0 = np.clip(np.floor(b).astype(int), 0, N_BINS - 1)
            fr = np.clip(b - np.floor(b), 0, 1)
            i1 = np.clip(i0 + 1, 0, N_BINS - 1)
            a = amp(d) * valid / 4.0
            base = RAY_SECT[None, :] * N_BINS
            rows = np.repeat(np.arange(n), K_RAYS)
            np.add.at(prof[:, ch], (rows, (base + i0).ravel()), (a * (1 - fr)).ravel())
            np.add.at(prof[:, ch], (rows, (base + i1).ravel()), (a * fr).ravel())
        prof += self.rng.normal(0, self.noise, prof.shape)
        return prof.reshape(n, 2, N_SECT, N_BINS)

    def _obs(self, heard):
        o = np.zeros((self.N, OBS_DIM), np.float32)
        idx = np.nonzero(heard)[0]
        if len(idx):
            o[idx, :2 * N_SECT * N_BINS] = self.echo(idx).reshape(len(idx), -1) * 3.0
        k = 2 * N_SECT * N_BINS
        o[:, k] = heard
        o[:, k + 1] = self.bump
        o[np.arange(self.N), k + 2 + self.prev_move] = 1
        o[:, k + 5] = self.prev_pulse
        o[:, k + 6] = self.t / np.maximum(self.T, 1)
        self.last_heard = heard
        return o

    # --- dynamics
    def step(self, move, pulse):
        move = np.asarray(move); pulse = np.asarray(pulse).astype(bool)
        N = self.N
        rew = np.full(N, -self.step_cost)
        fwd = move == 0
        ok = self.pool.open[self.m, self.r, self.c, self.hd]
        blocked = fwd & ~ok
        go = fwd & ok
        self.r = self.r + go * DR[self.hd]
        self.c = self.c + go * DC[self.hd]
        self.hd = np.where(move == 1, (self.hd + 3) % 4, np.where(move == 2, (self.hd + 1) % 4, self.hd))
        self.bump = blocked
        rew -= self.collision_cost * blocked
        self.colls += blocked
        if self.pulse_budget is not None:
            pulse = pulse & (self.pulses < self.pulse_budget)
        self.pulses += pulse
        rew -= self.pulse_cost * pulse
        new = ~self.visited[np.arange(N), self.r, self.c]
        rew += self.new_cell_bonus * new
        self.visited[np.arange(N), self.r, self.c] = True
        self.t += 1
        success = (self.r == self.gr) & (self.c == self.gc)
        rew += self.goal_reward * success
        timeout = (self.t >= self.T) & ~success
        done = success | timeout
        info = {"success": success.copy(), "timeout": timeout.copy(), "done": done.copy(),
                "t": self.t.copy(), "pulses": self.pulses.copy(), "collisions": self.colls.copy(),
                "ep_id": self.ep_id.copy(), "maze": self.m.copy()}
        self.prev_move = move.copy(); self.prev_pulse = pulse.astype(int)
        heard = pulse.copy()
        for i in np.nonzero(done)[0]:
            self._reset_one(i)
            heard[i] = True                     # free arrival pulse at a new episode
        return self._obs(heard), rew.astype(np.float32), done, info


# ----------------------------------------------------------------------------- baselines
def decode_open(obs, thresh=0.25):
    """From an echo profile in obs, decide whether front/right/back/left cells are open.
    Blocked = strong rock echo in the first time bin (rock 0.5 widths away) of that sector."""
    w = obs[:, :N_SECT * N_BINS].reshape(-1, N_SECT, N_BINS)
    near = w[:, [0, 2, 4, 6], 0]          # front, right, back, left
    return near < thresh


def decode_moth(obs, thresh=0.3):
    m = obs[:, N_SECT * N_BINS:2 * N_SECT * N_BINS].reshape(-1, N_SECT, N_BINS)
    e = m.max(-1)[:, [0, 2, 4, 6]]
    has = e.max(-1) > thresh
    return has, e.argmax(-1)              # 0 front 1 right 2 back 3 left


class WallFollower:
    """Right-hand wall follower using the same echo observations, pulsing every k steps.
    Between pulses it executes its queued plan; with an empty queue and no fresh echo it walks
    forward blind, turning right after a bump."""

    def __init__(self, n, k=1):
        self.n, self.k = n, k
        self.queue = [[] for _ in range(n)]
        self.t = np.zeros(n, int)

    def reset_idx(self, i):
        self.queue[i] = []
        self.t[i] = 0

    def act(self, obs):
        k0 = 2 * N_SECT * N_BINS
        heard = obs[:, k0] > 0.5
        bump = obs[:, k0 + 1] > 0.5
        openf = decode_open(obs)
        has_moth, moth_dir = decode_moth(obs)
        move = np.zeros(self.n, int)
        for i in range(self.n):
            if heard[i]:
                if has_moth[i] and not self.queue[i]:
                    self.queue[i] = {0: [0], 1: [2, 0], 2: [2, 2, 0], 3: [1, 0]}[int(moth_dir[i])]
                elif not self.queue[i]:
                    F, R, B, L = openf[i]
                    if R:
                        self.queue[i] = [2, 0]
                    elif F:
                        self.queue[i] = [0]
                    elif L:
                        self.queue[i] = [1, 0]
                    else:
                        self.queue[i] = [2, 2, 0]
            if self.queue[i]:
                move[i] = self.queue[i].pop(0)
            else:
                move[i] = 2 if bump[i] else 0
        # pulse so the echo arrives on steps that are multiples of k
        self.t += 1
        pulse = (self.t % self.k) == 0
        return move, pulse

    def on_done(self, done):
        for i in np.nonzero(done)[0]:
            self.reset_idx(i)


class WallFollowerEvent(WallFollower):
    """Reviewer baseline: the same wall follower, but it pulses only when its plan queue is empty
    after this move, i.e. when it will need a fresh echo to make the next decision. This is the
    scripted version of "pulse after you move, not after you turn"."""

    def act(self, obs):
        move, _ = super().act(obs)
        return move, np.array([len(q) == 0 for q in self.queue])


class RandomWalk:
    def __init__(self, n, seed=0):
        self.n, self.rng = n, np.random.default_rng(seed)

    def act(self, obs):
        return self.rng.integers(0, 3, self.n), np.zeros(self.n, bool)

    def on_done(self, done):
        pass
