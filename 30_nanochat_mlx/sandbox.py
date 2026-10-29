"""
Grid Courier: a tiny text sandbox for the model's "tiny job".

A 5x5 grid holds 3 items (from key, apple, gem, coin, book, cup) and 2 containers
(from box, basket, chest). The task is one or two deliveries, e.g.
    "put the gem in the basket, then put the key in the box".
Actions (the whole action language, 28 strings):
    move north | move south | move east | move west
    pick <item>                       (item on your cell, hands empty)
    put <item> in <container>         (holding item, standing on the container)
x grows to the east, y grows to the north. An action that is malformed or illegal in the
current state is "invalid": the state does not change and the episode continues.
The episode succeeds when every delivery is done, and fails after MAX_STEPS actions.

Held-out split: some (item, container) delivery pairs never appear in training tasks.
Test tasks contain at least one held-out pair (all objects were seen, only the combination is new).
"""
import random

import numpy as np

ITEMS = ["key", "apple", "gem", "coin", "book", "cup"]
CONTAINERS = ["box", "basket", "chest"]
DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
N = 5
MAX_STEPS = 40
# 5 of the 18 delivery pairs are held out from training (each item / container still appears)
HELDOUT_PAIRS = {("gem", "basket"), ("key", "chest"), ("cup", "box"), ("book", "basket"), ("coin", "chest")}
ALL_PAIRS = [(i, c) for i in ITEMS for c in CONTAINERS]
TRAIN_PAIRS = [p for p in ALL_PAIRS if p not in HELDOUT_PAIRS]

ACTIONS = [f"move {d}" for d in DIRS] + [f"pick {i}" for i in ITEMS] + \
          [f"put {i} in {c}" for i in ITEMS for c in CONTAINERS]
ACTION_ID = {a: k for k, a in enumerate(ACTIONS)}


class Env:
    def __init__(self, goals, rng):
        """goals: list of (item, container) deliveries, done in order."""
        self.goals = list(goals)
        items_needed = list(dict.fromkeys(g[0] for g in goals))
        conts_needed = list(dict.fromkeys(g[1] for g in goals))
        others_i = [i for i in ITEMS if i not in items_needed]
        others_c = [c for c in CONTAINERS if c not in conts_needed]
        rng.shuffle(others_i)
        rng.shuffle(others_c)
        self.items = items_needed + others_i[: 3 - len(items_needed)]
        self.conts = conts_needed + others_c[: max(0, 2 - len(conts_needed))]
        cells = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(cells)
        # items and containers on distinct cells; agent anywhere
        self.item_loc = {it: cells[k] for k, it in enumerate(self.items)}  # cell, "held", or container name
        self.cont_loc = {c: cells[3 + k] for k, c in enumerate(self.conts)}
        n_occ = 3 + len(self.conts)
        self.agent = cells[n_occ + rng.randrange(len(cells) - n_occ)] if rng.random() < 0.7 else cells[rng.randrange(n_occ)]
        self.holding = None
        self.steps = 0

    def done_goals(self):
        return [g for g in self.goals if self.item_loc[g[0]] == g[1]]

    def success(self):
        return len(self.done_goals()) == len(self.goals)

    def task_text(self):
        parts = [f"put the {i} in the {c}" for i, c in self.goals]
        return ", then ".join(parts)

    def observation(self):
        def where(it):
            loc = self.item_loc[it]
            if loc == "held":
                return f"{it} held"
            if isinstance(loc, str):
                return f"{it} in {loc}"
            return f"{it} ({loc[0]},{loc[1]})"
        lines = [
            f"task: {self.task_text()}",
            f"you: ({self.agent[0]},{self.agent[1]}) holding: {self.holding or 'nothing'}",
            "items: " + " | ".join(where(i) for i in self.items),
            "containers: " + " | ".join(f"{c} ({self.cont_loc[c][0]},{self.cont_loc[c][1]})" for c in self.conts),
        ]
        return "\n".join(lines)

    def legal(self, action):
        a = action.strip().split()
        if action.strip() not in ACTION_ID:
            return False
        if a[0] == "move":
            dx, dy = DIRS[a[1]]
            x, y = self.agent[0] + dx, self.agent[1] + dy
            return 0 <= x < N and 0 <= y < N
        if a[0] == "pick":
            return self.holding is None and a[1] in self.item_loc and self.item_loc[a[1]] == self.agent
        if a[0] == "put":
            return self.holding == a[1] and a[3] in self.cont_loc and self.cont_loc[a[3]] == self.agent
        return False

    def legal_actions(self):
        return [a for a in ACTIONS if self.legal(a)]

    def step(self, action):
        """Returns True if the action was valid."""
        self.steps += 1
        if not self.legal(action):
            return False
        a = action.strip().split()
        if a[0] == "move":
            dx, dy = DIRS[a[1]]
            self.agent = (self.agent[0] + dx, self.agent[1] + dy)
        elif a[0] == "pick":
            self.item_loc[a[1]] = "held"
            self.holding = a[1]
        elif a[0] == "put":
            self.item_loc[a[1]] = a[3]
            self.holding = None
        return True

    def over(self):
        return self.success() or self.steps >= MAX_STEPS

    # ---- structured features for the non-language baseline (same information as the text) ----
    def features(self):
        f = []
        oh = lambda k, n: [1.0 if j == k else 0.0 for j in range(n)]
        cell = lambda p: p[0] * N + p[1]
        f += oh(cell(self.agent), N * N)
        f += oh(ITEMS.index(self.holding) if self.holding else len(ITEMS), len(ITEMS) + 1)
        for it in ITEMS:
            loc = self.item_loc.get(it)
            present = loc is not None
            on_floor = present and isinstance(loc, tuple)
            f += oh(cell(loc), N * N) if on_floor else [0.0] * (N * N)
            f += [float(present), float(loc == "held")] + [float(loc == c) for c in CONTAINERS]
        for c in CONTAINERS:
            loc = self.cont_loc.get(c)
            f += oh(cell(loc), N * N) if loc is not None else [0.0] * (N * N)
        for k in range(2):
            if k < len(self.goals):
                i, c = self.goals[k]
                f += oh(ITEMS.index(i), len(ITEMS)) + oh(CONTAINERS.index(c), len(CONTAINERS)) + [1.0]
            else:
                f += [0.0] * (len(ITEMS) + len(CONTAINERS) + 1)
        return np.array(f, dtype=np.float32)


def expert_action(env):
    """Scripted optimal policy: next unfinished delivery; go to item (x first, then y), pick,
    go to container, put."""
    for item, cont in env.goals:
        if env.item_loc[item] == cont:
            continue
        if env.holding is not None and env.holding != item:
            # never happens for the expert (it only picks goal items in order)
            raise RuntimeError("expert holding wrong item")
        target = env.cont_loc[cont] if env.holding == item else env.item_loc[item]
        if env.agent == target:
            return f"put {item} in {cont}" if env.holding == item else f"pick {item}"
        dx, dy = target[0] - env.agent[0], target[1] - env.agent[1]
        if dx:
            return "move east" if dx > 0 else "move west"
        return "move north" if dy > 0 else "move south"
    return None


def sample_task(rng, split):
    """split: 'train' (only train pairs), 'test' (at least one held-out pair), 'test_seen'
    (seen pairs, new layouts) or 'test3' (THREE deliveries, any pairs: longer than any
    training task)."""
    if split == "test3":
        goals = []
        for _ in range(3):
            goals.append(rng.choice([p for p in ALL_PAIRS if p[0] not in [g[0] for g in goals]]))
        return goals
    while True:
        n = 1 if rng.random() < 0.3 else 2
        pairs = TRAIN_PAIRS if split == "train" else ALL_PAIRS
        goals = []
        for _ in range(n):
            choices = [p for p in pairs if p[0] not in [g[0] for g in goals]]
            goals.append(rng.choice(choices))
        has_heldout = any(g in HELDOUT_PAIRS for g in goals)
        if split == "train" and not has_heldout:
            return goals
        if split == "test" and has_heldout:
            return goals
        if split == "test_seen" and not has_heldout:
            return goals


def make_env(seed, split):
    rng = random.Random(seed)
    return Env(sample_task(rng, "train" if split == "train" else split), rng)


def expert_trace(env):
    """List of (observation, action) pairs following the expert until success."""
    out = []
    while not env.success():
        a = expert_action(env)
        out.append((env.observation(), a))
        assert env.step(a)
    return out


# seeds: train episodes use [0, 1e6); eval episodes use disjoint ranges
TRAIN_SEED0, TEST_SEED0, TESTSEEN_SEED0, TEST3_SEED0 = 0, 10_000_000, 20_000_000, 30_000_000


def parse_action(text):
    t = text.strip().lower()
    return t if t in ACTION_ID else None


if __name__ == "__main__":
    env = make_env(TEST_SEED0, "test")
    print(env.observation())
    for o, a in expert_trace(env):
        print(">", a)
    print("success", env.success(), "steps", env.steps)
    lens = [len(expert_trace(make_env(s, "train"))) for s in range(2000)]
    print("expert steps: mean", np.mean(lens), "max", max(lens), "features", make_env(0, "train").features().shape)
