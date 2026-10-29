"""Compositional task planner. Normal knowledge comes ONLY from the same replay.
Oracle construction is an explicit, separate privileged diagnostic, not an explorer.
"""
from collections import deque
from dataclasses import dataclass, field
import heapq
import itertools
from r5_core import SeenMap
from env import (DIRS, FLOOR, WALL, DOOR_OPEN, DOOR_LOCKED, DOOR_SWITCH, KEY,
                 BUTTON_UP, BUTTON_DOWN, LEVER_OFF, LEVER_ON, LAMP_OFF, LAMP_ON, BOX)


@dataclass
class Knowledge:
    cells: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    source: str = 'replay'

    @classmethod
    def from_replay(cls, observations, actions):
        assert len(observations) == len(actions) + 1
        out = cls(); memory = SeenMap(); seen_at = {}; activations = []
        for t, obs in enumerate(observations):
            before = dict(memory.cells)
            memory.observe(obs, None if t == 0 else int(actions[t - 1]))
            # Only cells visible NOW constitute fresh evidence (not the remembered map).
            from r5_core import decode, HALF, VIEW
            types, colors, _, _ = decode(obs)
            visible = {}
            for y in range(VIEW):
                for x in range(VIEW):
                    p = (memory.pos[0] + y - HALF, memory.pos[1] + x - HALF)
                    if int(types[y, x]) != 0:
                        visible[p] = (int(types[y, x]), int(colors[y, x]))
            if t and actions[t - 1] == 4:
                for p, (typ, _) in visible.items():
                    old = before.get(p, (None,))[0]
                    if (old, typ) in ((BUTTON_UP, BUTTON_DOWN), (LEVER_OFF, LEVER_ON)):
                        activations.append((t, p, 'button' if typ == BUTTON_DOWN else 'lever'))
            for p, value in visible.items():
                old = before.get(p, (None,))[0]; typ = value[0]
                kind = 'button' if (old, typ) == (DOOR_SWITCH, DOOR_OPEN) else (
                       'lever' if (old, typ) == (LAMP_OFF, LAMP_ON) else None)
                if kind:
                    candidates = {q for when, q, k in activations if k == kind and seen_at.get(p, -1) < when <= t}
                    if len(candidates) == 1:
                        cause = next(iter(candidates))
                        out.links.setdefault(cause, set()).add(p)
                # Earliest observed object state defines reset knowledge. Observation
                # before each action exposes a key/box before pickup/push occurs.
                out.cells.setdefault(p, value)
                seen_at[p] = t
        return out

    @classmethod
    def oracle(cls, env):
        """Full INITIAL map and exact mechanism relations; scorer/test code calls this only."""
        start = env.start
        rel = lambda p: (int(p[0] - start[0]), int(p[1] - start[1]))
        out = cls(source='oracle_full_initial_state')
        for y in range(env.template.shape[0]):
            for x in range(env.template.shape[1]):
                out.cells[rel((y, x))] = (int(env.template[y, x]), int(env.room[y, x]))
        for a, b in (('button_B', 'door_B'), ('button_E', 'door_E'), ('lever_A', 'lamp_A'),
                     ('lever_C', 'lamp_C'), ('lever_D', 'lamp_D')):
            out.links[rel(env.mech_dict[a])] = {rel(env.mech_dict[b])}
        return out

    def stats(self):
        return dict(cells=len(self.cells), links=sum(len(x) for x in self.links.values()), source=self.source)


# position, has_key, key_taken, opened_doors, pressed_buttons, lit_lamps, boxes
INITIAL = ((0, 0), False, False, frozenset(), frozenset(), frozenset(), ())


class CompositionalPlanner:
    def __init__(self, knowledge):
        self.k = knowledge
        self.initial = (*INITIAL[:-1], tuple(sorted(p for p, (t, _) in knowledge.cells.items() if t == BOX)))
        self.stats = {}

    def tile(self, state, p):
        pos, key, taken, opened, pressed, lit, boxes = state
        if p in boxes:
            return BOX
        typ, room = self.k.cells.get(p, (WALL, -1))
        if p in opened:
            return DOOR_OPEN
        if typ == BOX:
            return DOOR_OPEN if room == -1 else FLOOR
        if typ == KEY and taken:
            return FLOOR
        return typ

    def goal(self, state, task):
        p, key, _, _, _, lit, _ = state
        kind, room = task
        if kind == 'lamp':
            return any(self.k.cells.get(q, (None, -1))[1] == room for q in lit)
        return self.k.cells.get(p, (None, -1))[1] == room and (kind != 'key_room' or key)

    def reachable(self, state, remaining):
        queue = deque([state[0]]); seen = {state[0]}; used = 0
        while queue and used < remaining:
            p = queue.popleft(); used += 1
            for dy, dx in DIRS:
                q = (p[0] + int(dy), p[1] + int(dx))
                if q not in seen and self.tile(state, q) in (FLOOR, DOOR_OPEN):
                    seen.add(q); queue.append(q)
        return seen, used

    def plan(self, task, budget=128, max_expansions=100000):
        """Dijkstra over navigation + manipulation macros; all primitive actions cost 1.
        Expansion cap counts BOTH outer states and inner navigation BFS cells.
        Unknown cells/causal links are unavailable, never filled with the true map.
        """
        counter = itertools.count(); heap = [(0, next(counter), self.initial, ())]
        best = {self.initial: 0}; expanded = 0; outer = 0
        while heap and expanded < max_expansions:
            cost, _, state, route = heapq.heappop(heap)
            if cost != best.get(state):
                continue
            outer += 1; expanded += 1
            if self.goal(state, task):
                self.stats = dict(expansions=expanded, macro_states=outer, exhausted=False)
                return list(route), expanded
            if cost >= budget:
                continue
            pos, key, taken, opened, pressed, lit, boxes = state
            queue = deque([pos]); paths = {pos: ()}
            events = []
            while queue and expanded < max_expansions:
                p = queue.popleft(); path = paths[p]; expanded += 1
                if self.goal((p, *state[1:]), task):
                    events.append(((p, *state[1:]), path))
                for a, (dy, dx) in enumerate(DIRS):
                    q = (p[0] + int(dy), p[1] + int(dx)); typ = self.tile(state, q)
                    if typ in (FLOOR, DOOR_OPEN):
                        if q not in paths:
                            paths[q] = path + (a,); queue.append(q)
                    elif typ == KEY and not taken:
                        events.append(((q, True, True, opened, pressed, lit, boxes), path + (a,)))
                    elif typ == DOOR_LOCKED and key:
                        # Unlock consumes the key and does NOT move until the next action.
                        events.append(((p, False, taken, opened | {q}, pressed, lit, boxes), path + (a,)))
                    elif typ == BUTTON_UP and q not in pressed and q in self.k.links:
                        effects = {v for v in self.k.links[q] if self.k.cells[v][0] == DOOR_SWITCH}
                        events.append(((p, key, taken, opened | effects, pressed | {q}, lit, boxes), path + (a, 4)))
                    elif typ in (LEVER_OFF, LEVER_ON) and task[0] == 'lamp' and q in self.k.links:
                        effects = {v for v in self.k.links[q] if self.k.cells[v][1] == task[1]}
                        if effects and not effects <= lit:
                            events.append(((p, key, taken, opened, pressed, lit | effects, boxes), path + (a, 4)))
                    elif typ == BOX:
                        behind = (q[0] + int(dy), q[1] + int(dx))
                        if self.tile(state, behind) in (FLOOR, DOOR_OPEN):
                            moved = tuple(sorted((set(boxes) - {q}) | {behind}))
                            events.append(((q, key, taken, opened, pressed, lit, moved), path + (a,)))
            for nxt, suffix in events:
                distance = cost + len(suffix)
                if nxt[-1] != boxes:
                    # Box is an access obstacle, not a placement goal. Only retain
                    # pushes that increase the navigable region. This prunes box
                    # wandering; it is not a general Sokoban-completeness claim.
                    region, used = self.reachable(nxt, max(0, max_expansions - expanded))
                    expanded += used
                    if len(region) <= len(paths):
                        continue
                if distance <= budget and distance < best.get(nxt, float('inf')):
                    best[nxt] = distance
                    heapq.heappush(heap, (distance, next(counter), nxt, route + suffix))
        self.stats = dict(expansions=expanded, macro_states=outer, exhausted=expanded >= max_expansions)
        return None, expanded
