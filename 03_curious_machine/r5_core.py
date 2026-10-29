"""Round-5 observation-only mapping, classical exploration, replay planning and metrics.
No MLX imports. Environment internals are used by the evaluator, never these agents.
"""
from collections import deque
import hashlib
import numpy as np
from env import (VIEW, HALF, CELL_FEAT, N_TYPES, N_COLORS, OBS_DIM, DIRS,
                 UNSEEN, FLOOR, WALL, DOOR_OPEN, DOOR_LOCKED, DOOR_SWITCH, KEY,
                 BUTTON_UP, BUTTON_DOWN, LEVER_OFF, LEVER_ON, LAMP_ON, BOX)

TASKS = (("lamp", 0), ("lamp", 4), ("lamp", 8), ("key_room", 0),
         ("key_room", 3), ("reach", 6), ("reach", 8))
TRAIN_SEEDS = (510, 511, 512)
VAL_SEEDS = (61500, 61501, 61502, 61503)
TEST_SEEDS = tuple(range(72500, 72532))
OLD_SEEDS = tuple(range(10000, 10004))  # secondary only; earlier results used these


def decode(obs):
    obs = np.asarray(obs)
    assert obs.shape == (OBS_DIM,)
    cells = obs[:-5].reshape(VIEW, VIEW, CELL_FEAT)
    return (cells[..., :N_TYPES].argmax(-1),
            cells[..., N_TYPES:N_TYPES + N_COLORS].argmax(-1) - 1,
            bool(obs[-5]), int(obs[-4:].argmax()))


def movement(prev, action, obs):
    """Infer odometry from overlapping wall/room-colour patches (no true position).

    Wall/colour are static, even for box pushes, key pickup and remote door changes.
    A 7x7 view in 5x5 rooms disambiguates motion from a blocked action. Fail on ties.
    """
    if action == 4:
        return (0, 0)
    pt, pc, _, _ = decode(prev)
    nt, nc, _, _ = decode(obs)
    scores = []
    moves = [(0, 0), tuple(DIRS[action])]
    for dy, dx in moves:
        yy, xx = np.indices((VIEW, VIEW))
        py, px = yy + dy, xx + dx
        mask = (py >= 0) & (py < VIEW) & (px >= 0) & (px < VIEW)
        scores.append(float(np.mean((pc[py[mask], px[mask]] != nc[mask]) |
                                    ((pt[py[mask], px[mask]] == WALL) != (nt[mask] == WALL)))))
    if abs(scores[0] - scores[1]) < 1e-12:
        raise ValueError("Ambiguous observation-only odometry; do not use simulator position")
    return moves[int(np.argmin(scores))]


class SeenMap:
    def __init__(self):
        self.pos = (0, 0)  # coordinates relative to start, not known layout coordinates
        self.cells = {}
        self.visits = {}
        self.prev = None
        self.key = False
        self.facing = 1
        self.room = -1

    def observe(self, obs, action=None):
        if self.prev is not None:
            if action is None:
                raise ValueError("Missing action between observations")
            dy, dx = movement(self.prev, action, obs)
            self.pos = (self.pos[0] + dy, self.pos[1] + dx)
        types, colors, self.key, self.facing = decode(obs)
        self.room = int(colors[HALF, HALF])
        for y in range(VIEW):
            for x in range(VIEW):
                if types[y, x] != UNSEEN:
                    self.cells[(self.pos[0] + y - HALF, self.pos[1] + x - HALF)] = (
                        int(types[y, x]), int(colors[y, x]))
        self.visits[self.pos] = self.visits.get(self.pos, 0) + 1
        self.prev = np.asarray(obs).copy()

    def signature(self):
        # Full remembered semantic map prevents merging states with different remote
        # button/lever histories. TV intensities were never included; TV tile remains.
        state = (self.pos, self.key, self.facing, tuple(sorted(self.cells.items())))
        return hashlib.sha256(repr(state).encode()).digest()

    def goals(self):
        g = {("reach", self.room)}
        if self.key:
            g.add(("key_room", self.room))
        # Retained observations, not a privileged lamp lookup.
        g.update(("lamp", room) for typ, room in self.cells.values() if typ == LAMP_ON)
        return g


class Frontier:
    """BFS on observed traversable cells, with unvisited/frontier and mechanism targets.

    Generic object affordances are a disclosed classical prior. No layout, coordinates,
    task labels or mechanism links are supplied. Remote switches trigger a door recheck.
    """
    def __init__(self, seed=0, mechanism_bonus=6.0, frontier_bonus=2.0):
        self.map = SeenMap()
        self.rng = np.random.default_rng(seed)
        self.mechanism_bonus = mechanism_bonus
        self.frontier_bonus = frontier_bonus
        self.used = set()
        self.tried_doors = set()
        self.last_switches = 0

    def observe(self, obs, action=None):
        self.map.observe(obs, action)
        switches = sum(t == BUTTON_DOWN for t, _ in self.map.cells.values())
        if switches > self.last_switches:
            self.tried_doors.clear()
        self.last_switches = switches

    def act(self):
        m = self.map
        q = deque([m.pos]); routes = {m.pos: ()}
        candidates = []
        def offer(cost, route, kind, target):
            if route:
                candidates.append((cost + float(self.rng.uniform(0, 0.01)), route, kind, target))
        while q:
            p = q.popleft(); path = routes[p]
            unseen = sum((p[0] + dy, p[1] + dx) not in m.cells for dy, dx in DIRS)
            if p != m.pos and (p not in m.visits or unseen):
                offer(len(path) - self.frontier_bonus * unseen, path, "walk", p)
            for a, (dy, dx) in enumerate(DIRS):
                n = (p[0] + int(dy), p[1] + int(dx))
                typ = m.cells.get(n, (UNSEEN, -1))[0]
                if typ in (FLOOR, DOOR_OPEN, KEY):
                    if n not in routes:
                        routes[n] = path + (a,); q.append(n)
                    if typ == KEY:
                        offer(len(path) + 1 - self.mechanism_bonus, path + (a,), "walk", n)
                elif typ in (LEVER_OFF, LEVER_ON, BUTTON_UP) and n not in self.used:
                    # A move into the solid object changes facing, then USE activates it.
                    route = path + (a, 4)
                    if not path and m.facing == a:
                        route = (4,)
                    offer(len(route) - self.mechanism_bonus, route, "use", n)
                elif typ == BOX:
                    behind = (n[0] + int(dy), n[1] + int(dx))
                    if m.cells.get(behind, (UNSEEN, -1))[0] in (FLOOR, DOOR_OPEN):
                        if n not in self.used:
                            offer(len(path) + 1 - self.mechanism_bonus, path + (a,), "push", n)
                elif typ == DOOR_LOCKED and m.key:
                    offer(len(path) + 1 - self.mechanism_bonus, path + (a,), "unlock", n)
                elif typ == DOOR_SWITCH and n not in self.tried_doors:
                    offer(len(path) + 1, path + (a,), "door", n)
        if not candidates:
            return int(self.rng.integers(5))
        _, route, kind, target = min(candidates, key=lambda c: c[0])
        if len(route) == 1:
            if kind in ("use", "push"):
                self.used.add(target)
            if kind == "door":
                self.tried_doors.add(target)
        return int(route[0])


class ReplayGraph:
    """Same transfer for every explorer: one bounded action/observation trajectory.

    Build a graph of remembered observable states. BFS only traverses recorded edges;
    no simulator rollouts, hidden grid, learned weights or object-link oracle.
    """
    def __init__(self, observations, actions):
        assert len(observations) == len(actions) + 1
        self.edges = {}; self.labels = {}; self.start = None
        m = SeenMap(); previous = None
        for i, obs in enumerate(observations):
            m.observe(obs, None if i == 0 else int(actions[i - 1]))
            key = m.signature()
            if i == 0:
                self.start = key
            if previous is not None:
                self.edges.setdefault(previous, []).append((int(actions[i - 1]), key))
            self.labels[key] = m.goals()
            previous = key

    def plan(self, task, budget=128, max_expansions=4096):
        queue = deque([(self.start, ())]); seen = {self.start}; expanded = 0
        while queue and expanded < max_expansions:
            node, route = queue.popleft(); expanded += 1
            if tuple(task) in self.labels[node]:
                return list(route), expanded
            if len(route) >= budget:
                continue
            for action, nxt in self.edges.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt); queue.append((nxt, route + (action,)))
        return None, expanded


def signed_progress(old_error, new_error, valid, clip=1.0):
    """No positive-part rectification: stationary noise must not receive a noise floor."""
    old_error, new_error, valid = map(np.asarray, (old_error, new_error, valid))
    assert old_error.shape == new_error.shape == valid.shape
    return np.clip(old_error - new_error, -clip, clip) * valid


def transfer_metrics(rows, budget):
    if not rows:
        raise ValueError("No task trials")
    successes = [r["success"] for r in rows]
    times = [r["steps"] for r in rows if r["success"]]
    return dict(success_rate=float(np.mean(successes)),
                capped_steps=float(np.mean([r["steps"] if r["success"] else budget for r in rows])),
                success_steps=float(np.mean(times)) if times else None,
                plan_coverage=float(np.mean([r["planned"] for r in rows])), trials=len(rows))
