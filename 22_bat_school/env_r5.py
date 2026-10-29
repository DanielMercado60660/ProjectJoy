"""Round 5 toy acoustics. NumPy only; the original exhibit is untouched.

Four cardinal beams, twelve range bins, rock/shutter and moth channels. Shutters
move across edges, never through occupied cell centres. A pulse observes AFTER
the move; there is no free initial echo. Public policy interface is obs only.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import numpy as np

from bat_env import DR, DC, HMAX, make_maze, maze_hash, bfs, MazePool

GROUPS = ("static", "moving", "ambiguous", "hard")
# size, shutter density, closed fraction, multipath probability, noise, dropout
GROUP_SPEC = ((5, 0., 0., 0., .025, .03), (5, .35, .35, 0., .04, .08),
              (5, 0., 0., .35, .07, .15), (6, .55, .5, .55, .10, .22))
BUDGETS = (8, 16, 32, 64)
MAX_STEPS = 320
ECHO = 96
HEARD, BUMP, MOVE, PULSE, REMAIN, INITIAL, CLOCK = 96, 97, 98, 103, 104, 105, 106
OBS_DIM = 107
FORWARD, LEFT, RIGHT, WAIT, BACK = range(5)
SPLIT_SEEDS = {"train": 225100, "val": 225200, "test": 225300,
               "test_turn2": 225400, "legacy": 5005}


@lru_cache(maxsize=1)
def old_layout_hashes():
    """Exclude the original training AND test layouts from all fresh R5 splits."""
    old = set(MazePool([4, 5, 6], 3000, 1000, rays=False).hashes)
    for size, seed in ((5, 5005), (6, 6006), (7, 7007)):
        old.update(MazePool([size], 200, seed, rays=False).hashes)
    return old


@dataclass
class CavePool:
    open: np.ndarray
    group: np.ndarray
    sizes: np.ndarray
    hashes: list
    start: np.ndarray
    goal: np.ndarray
    heading: np.ndarray
    period: np.ndarray
    phase: np.ndarray
    closed: np.ndarray
    alias: np.ndarray
    ray_r: np.ndarray
    ray_c: np.ndarray
    ray_valid: np.ndarray
    wall_distance: np.ndarray
    split: str
    seed: int

    @property
    def n(self):
        return len(self.group)

    def manifest(self):
        return {"split": self.split, "seed": self.seed, "n": self.n,
                "hashes": self.hashes, "groups": self.group.tolist(),
                "group_spec": GROUP_SPEC, "budgets": BUDGETS, "max_steps": MAX_STEPS}


def make_pool(split="train", per_group=256):
    """Fixed geometry/start/goal and reciprocal edge obstacles. No MLX imports."""
    if split not in SPLIT_SEEDS or per_group < 1:
        raise ValueError("invalid split or per_group")
    n = 4 * per_group
    opn = np.zeros((n, HMAX, HMAX, 4), bool)
    sizes = np.repeat([s[0] for s in GROUP_SPEC], per_group)
    groups = np.repeat(np.arange(4), per_group)
    hashes = []
    seen = set()
    partition = {"train": 0, "val": 1, "test": 2, "test_turn2": 3}.get(split)
    # Keep geometry RNG separate from acoustic parameters. Legacy uses original
    # size-5/6 test layouts (first per_group members), with NEW R5 physics.
    for g in range(4):
        seed = (5005 if g < 3 else 6006) if split == "legacy" else SPLIT_SEEDS[split] + g
        rng = np.random.default_rng(seed)
        size = GROUP_SPEC[g][0]
        for j in range(per_group):
            i = g * per_group + j
            while True:
                maze = make_maze(size, size, rng)
                digest = maze_hash(maze)
                if split == "legacy" or (int(digest[:8], 16) % 4 == partition and
                                          digest not in seen and digest not in old_layout_hashes()):
                    break
            opn[i, :size, :size] = maze
            hashes.append(digest)
            seen.add(digest)
    rng = np.random.default_rng(SPLIT_SEEDS[split] + 1000000)
    start = np.zeros((n, 2), int)
    goal = np.zeros((n, 2), int)
    heading = rng.integers(4, size=n)
    period = np.full_like(opn, 8, dtype=int)
    phase = np.zeros_like(period)
    closed = np.zeros_like(period)
    for i, size in enumerate(sizes):
        start[i] = rng.integers(size, size=2)
        distance = bfs(opn[i, :size, :size], tuple(start[i]))
        candidates = np.argwhere(distance >= size)
        goal[i] = candidates[rng.integers(len(candidates))]
        density, duty = GROUP_SPEC[groups[i]][1:3]
        for r in range(size):
            for c in range(size):
                for d in (1, 2):
                    if opn[i, r, c, d]:
                        p = int(rng.integers(6, 15))
                        ph = int(rng.integers(p))
                        shut = max(1, round(p * duty)) if rng.random() < density else 0
                        rr, cc = r + DR[d], c + DC[d]
                        for a, b, e in ((r, c, d), (rr, cc, (d+2) % 4)):
                            period[i, a, b, e] = p
                            phase[i, a, b, e] = ph
                            closed[i, a, b, e] = shut
    alias = rng.random((n, HMAX, HMAX, 4, 4))
    # Every possible origin and direction: six consecutive edges in a straight
    # corridor. Stop at first wall; gates are checked at the actual echo time.
    mi, rr, cc, dd = np.indices(opn.shape)
    valid = np.ones(opn.shape, bool)
    wall = np.full(opn.shape, 12., np.float32)
    ray_r, ray_c, ray_valid = [], [], []
    for k in range(6):
        ray_r.append(rr.copy()); ray_c.append(cc.copy())
        go = valid & opn[mi, rr, cc, dd]
        wall[valid & ~go] = .5 + 2 * k
        ray_valid.append(go.copy())
        rr = np.clip(rr + go * DR[dd], 0, HMAX-1)
        cc = np.clip(cc + go * DC[dd], 0, HMAX-1)
        valid = go
    return CavePool(opn, groups, sizes, hashes, start, goal, heading, period, phase,
                    closed, alias, np.stack(ray_r, -1), np.stack(ray_c, -1),
                    np.stack(ray_valid, -1), wall, split, SPLIT_SEEDS[split])


def assert_disjoint(*pools):
    for i, a in enumerate(pools):
        for b in pools[i+1:]:
            overlap = set(a.hashes) & set(b.hashes)
            if overlap:
                raise AssertionError(f"layout overlap: {a.split}/{b.split}: {len(overlap)}")


def splat(distance, amplitude):
    """Linear time-of-flight interpolation; distance .5 maps to bin zero."""
    x = np.clip(distance - .5, 0., 11.)
    lo = x.astype(int)
    frac = x - lo
    bins = np.arange(12)
    return amplitude[..., None] * ((lo[..., None] == bins) * (1-frac[..., None])
                                   + (np.minimum(lo+1, 11)[..., None] == bins) * frac[..., None])


class CaveEnv:
    def __init__(self, pool, n=128, seed=0, budget=None, fixed=False, auto_reset=True):
        self.pool, self.n = pool, n
        self.rng = np.random.default_rng(seed)
        self.fixed, self.auto_reset, self.budget_override = fixed, auto_reset, budget
        if fixed and n != pool.n:
            raise ValueError("fixed evaluation uses one slot per cave")
        for name in ("m", "r", "c", "hd", "t", "budget", "pulses", "collisions", "prev_move"):
            setattr(self, name, np.zeros(n, int))
        for name in ("bump", "last_pulse", "finished", "success"):
            setattr(self, name, np.zeros(n, bool))
        self.visited = np.zeros((n, HMAX, HMAX), bool)
        self.episode = np.zeros(n, int)
        self.last_obs = np.zeros((n, OBS_DIM), np.float32)

    def reset_indices(self, ids):
        self.m[ids] = ids if self.fixed else self.rng.integers(self.pool.n, size=len(ids))
        m = self.m[ids]
        self.r[ids], self.c[ids] = self.pool.start[m].T
        self.hd[ids] = self.pool.heading[m]
        self.budget[ids] = (self.budget_override if self.budget_override is not None else
                            self.rng.choice(BUDGETS, len(ids)))
        for name in ("t", "pulses", "collisions", "bump", "last_pulse", "finished", "success"):
            getattr(self, name)[ids] = 0
        self.prev_move[ids] = WAIT
        self.visited[ids] = False
        self.visited[ids, self.r[ids], self.c[ids]] = True
        self.episode[ids] += 1

    def reset(self):
        self.reset_indices(np.arange(self.n))
        return self.observe(np.zeros(self.n, bool))

    def shut(self, m, r, c, d, t):
        p = self.pool
        return ((t + p.phase[m, r, c, d]) % p.period[m, r, c, d]) < p.closed[m, r, c, d]

    def echo(self):
        p = self.pool
        m, r, c = self.m[:, None], self.r[:, None], self.c[:, None]
        d = (self.hd[:, None] + np.arange(4)) % 4
        rr, cc = p.ray_r[m, r, c, d], p.ray_c[m, r, c, d]
        gates = self.shut(m[..., None], rr, cc, d[..., None], self.t[:, None, None])
        gates &= p.ray_valid[m, r, c, d]
        gd = np.min(np.where(gates, 1. + 2*np.arange(6), 12.), -1)
        dist = np.minimum(p.wall_distance[m, r, c, d], gd)
        group = p.group[self.m]
        spec = np.asarray(GROUP_SPEC)[group]
        # Draw for ALL slots at EVERY step, including silent/dead slots. Under
        # fixed evaluation, schedules share noise at each (cave,time,beam,bin).
        keep = self.rng.random((self.n, 4)) >= spec[:, 5, None]
        amplitude = np.exp(-.08 * dist) / (1+.4*dist)
        wall = splat(dist, amplitude * keep * (dist < 12))
        # A weak delayed return plus a viewpoint-dependent near ghost. This is
        # deliberately a toy multipath sensor, not a physical acoustic solver.
        alias = p.alias[m, r, c, d, self.hd[:, None]] < spec[:, 3, None]
        ghost = alias & (dist > 1.5)
        wall += splat(np.full_like(dist, .65), ghost * (.25 + .25*self.rng.random(dist.shape)))
        wall += splat(np.minimum(dist+2, 12), alias * amplitude * .3 * (dist < 10))
        delta = p.goal[self.m] - np.stack([self.r, self.c], 1)
        proj = delta[:, 0, None] * DR[d] + delta[:, 1, None] * DC[d]
        perp = delta[:, 0, None] * DC[d] - delta[:, 1, None] * DR[d]
        visible = (proj > 0) & (perp == 0) & (2*proj < dist)
        moth = splat(np.maximum(2*proj, .5), visible * keep / (1+.4*np.maximum(2*proj, 0)))
        sound = np.stack([wall, moth], 1)
        sound += self.rng.normal(size=sound.shape) * spec[:, 4, None, None, None]
        return sound.astype(np.float32)

    def observe(self, heard):
        obs = np.zeros((self.n, OBS_DIM), np.float32)
        obs[:, :ECHO] = self.echo().reshape(self.n, -1) * heard[:, None]
        obs[:, HEARD], obs[:, BUMP] = heard, self.bump
        obs[np.arange(self.n), MOVE+self.prev_move] = 1
        obs[:, PULSE] = self.last_pulse
        obs[:, REMAIN] = (self.budget-self.pulses) / 64.
        obs[:, INITIAL] = self.budget / 64.
        obs[:, CLOCK] = self.t / MAX_STEPS
        self.last_obs = obs
        return obs

    def step(self, move, pulse):
        move = np.asarray(move, int)
        pulse = np.asarray(pulse, bool)
        if move.shape != (self.n,) or pulse.shape != (self.n,) or not np.isin(move, range(5)).all():
            raise ValueError("action shape/range")
        active = ~self.finished
        direction = (self.hd + 2*(move == BACK)) % 4
        trying = active & ((move == FORWARD) | (move == BACK))
        arrival = self.t + 1
        open_edge = self.pool.open[self.m, self.r, self.c, direction]
        shutter = self.shut(self.m, self.r, self.c, direction, arrival)
        self.bump = trying & (~open_edge | shutter)
        go = trying & ~self.bump
        self.r += go * DR[direction]; self.c += go * DC[direction]
        self.hd = (self.hd + active * ((move == RIGHT).astype(int) - (move == LEFT))) % 4
        self.t += active
        used = pulse & active & (self.pulses < self.budget)
        self.pulses += used
        self.collisions += self.bump
        new = ~self.visited[np.arange(self.n), self.r, self.c] & active
        self.visited[np.arange(self.n), self.r, self.c] = True
        self.success |= active & (self.r == self.pool.goal[self.m, 0]) & (self.c == self.pool.goal[self.m, 1])
        done = active & (self.success | (self.t >= MAX_STEPS))
        reward = (-.005*active - .08*self.bump - .004*used + .08*new
                  + 3*(done & self.success) - .25*(done & ~self.success)).astype(np.float32)
        self.prev_move, self.last_pulse = move.copy(), used.copy()
        info = {"done": done.copy(), "success": self.success.copy(), "steps": self.t.copy(),
                "pulses": self.pulses.copy(), "collisions": self.collisions.copy(),
                "used": used.copy(), "moved": go.copy(), "bump": self.bump.copy()}
        self.finished |= done
        heard = used.copy()
        if self.auto_reset:
            self.reset_indices(np.flatnonzero(done))
            heard[done] = False
        return self.observe(heard), reward, done, info

    def state_dict(self):
        return {"arrays": {k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)},
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        for k, v in state["arrays"].items():
            setattr(self, k, v.copy())
        self.rng.bit_generator.state = state["rng"]


def manifest_digest(pool):
    return hashlib.sha256("".join(pool.hashes).encode()).hexdigest()
