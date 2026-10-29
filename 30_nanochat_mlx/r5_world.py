"""Round 5 courier. NumPy/stdlib only; policies receive a public notebook, never Env.

Three rooms, four parcels, three labelled receptacles. Orders name labels; commands
name receptacles. Looking discovers locations/bindings. Failed actions cost a step.
The notebook is shared external memory, so this is replanning, not a memory benchmark.
"""
import copy
import itertools
import numpy as np

ITEMS = ("gem", "book", "cup", "coin")
BINS = ("red", "blue", "green")
LABELS = ("sun", "moon", "star")
MODES = ("none", "locked", "missing", "redirect")
SPLITS = ("seen", "pairs", "order", "long3", "long4")
HELDOUT = {(0, 0), (1, 1), (2, 2), (3, 0)}
MAX_STEPS = 64
FEEDBACK = ("start", "seen", "moved", "taken", "delivered", "dropped",
            "unlocked", "locked", "missing", "redirected", "illegal")
# (verb, item, argument). -1 means the factor is unused, not an action class.
FACTORS = [(0, -1, -1)] + [(1, -1, r) for r in range(3)] + \
          [(2, i, -1) for i in range(4)] + [(3, -1, c) for c in range(3)] + \
          [(4, i, c) for i in range(4) for c in range(3)] + [(5, i, -1) for i in range(4)]


def action_text(f):
    v, i, a = f
    return ("INSPECT" if v == 0 else f"GO {a}" if v == 1 else
            f"TAKE {ITEMS[i]}" if v == 2 else f"OPEN {BINS[a]}" if v == 3 else
            f"SEND {ITEMS[i]} {BINS[a]}" if v == 4 else f"DROP {ITEMS[i]}")


ACTIONS = tuple(map(action_text, FACTORS))
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}


def factor_scores(log_probs):
    """Log p(command) from normalized verb/item/argument heads, grammar only."""
    verb, item, arg = log_probs
    return np.array([verb[v] + (item[i] if i >= 0 else 0) + (arg[a] if a >= 0 else 0)
                     for v, i, a in FACTORS])


class Env:
    def __init__(self, seed, split="train", mode=None):
        if split not in ("train",) + SPLITS:
            raise ValueError(split)
        self.rng = np.random.default_rng(seed)
        self.seed, self.split = int(seed), split
        self.mode = mode or MODES[int(self.rng.integers(4))]
        if self.mode not in MODES:
            raise ValueError(self.mode)
        n = 3 if split == "long3" else 4 if split == "long4" else 2 if split == "order" else int(self.rng.integers(1, 3))
        its = sorted(map(int, self.rng.choice(4, n, replace=False)))
        if split == "order":
            its.reverse()
        elif split.startswith("long"):
            while True:
                perm = list(map(int, self.rng.permutation(its)))
                if perm != its:
                    its = perm
                    break
        self.orders = [[i, int(self.rng.choice(self.allowed_labels(i)))] for i in its]
        if split == "pairs":
            j = int(self.rng.integers(n))
            self.orders[j][1] = next(l for i, l in sorted(HELDOUT) if i == self.orders[j][0])
        self.item_at = list(map(int, self.rng.integers(0, 3, 4)))
        self.bin_at = list(map(int, self.rng.permutation(3)))
        self.bin_label = list(map(int, self.rng.permutation(3)))
        self.locked = [False] * 3
        if self.mode == "locked":
            self.locked[self.bin_label.index(self.orders[0][1])] = True
        self.room = int(self.rng.integers(3))
        self.holding, self.cursor, self.steps = -1, 0, 0
        self.known_items = [-1] * 4
        self.known_bins = [[-1, -1, -1] for _ in range(3)]
        self.seen = [False] * 3
        self.feedback = "start"
        self.disruption = None
        self.event_used = False

    def allowed_labels(self, item):
        # Neither initial nor updated train/seen/order orders introduce held-out pairs.
        return [l for l in range(3) if self.split not in ("train", "seen", "order") or (item, l) not in HELDOUT]

    def observe(self):
        return copy.deepcopy(dict(room=self.room, holding=self.holding, cursor=self.cursor,
                                  orders=self.orders, items=self.known_items, bins=self.known_bins,
                                  seen=self.seen, feedback=self.feedback, steps=self.steps))

    def success(self):
        return self.cursor == len(self.orders)

    def over(self):
        return self.success() or self.steps >= MAX_STEPS

    def _event(self, kind):
        self.event_used = True
        self.disruption = dict(kind=kind, step=self.steps, affected_cursor=self.cursor,
                               recovered_step=None)

    def step(self, action):
        if self.over():
            raise RuntimeError("episode is over")
        self.steps += 1
        self.feedback = "illegal"
        if action not in ACTION_ID:
            return False
        v, i, a = FACTORS[ACTION_ID[action]]
        if v == 0:
            self.seen[self.room] = True
            for j, loc in enumerate(self.item_at):
                if loc == self.room:
                    self.known_items[j] = loc
                elif self.known_items[j] == self.room:
                    self.known_items[j] = -1
            for c in range(3):
                if self.bin_at[c] == self.room:
                    self.known_bins[c][:2] = [self.room, self.bin_label[c]]
            self.feedback = "seen"
            return True
        if v == 1 and a != self.room:
            self.room, self.feedback = a, "moved"
            return True
        if v == 2 and self.holding == -1 and self.item_at[i] == self.room:
            if self.mode == "missing" and not self.event_used and i == self.orders[0][0]:
                self.item_at[i] = (self.room + 1 + int(self.rng.integers(2))) % 3
                self.known_items[i] = -1
                self.seen = [False] * 3  # the failure explicitly invalidates the location survey
                self._event("missing")
                self.feedback = "missing"
                return False
            self.holding = i
            self.item_at[i] = self.known_items[i] = 3
            self.feedback = "taken"
            if self.mode == "redirect" and not self.event_used and i == self.orders[0][0]:
                old = self.orders[0][1]
                self.orders[0][1] = int(self.rng.choice([l for l in self.allowed_labels(i) if l != old]))
                self._event("redirect")
                self.feedback = "redirected"
            return True
        if v == 3 and self.bin_at[a] == self.room and self.locked[a]:
            self.locked[a] = False
            self.known_bins[a][0] = self.room
            self.known_bins[a][2] = 0
            self.feedback = "unlocked"
            return True
        if v == 4 and self.holding == i and self.bin_at[a] == self.room:
            if self.locked[a]:
                self.known_bins[a][0] = self.room
                self.known_bins[a][2] = 1
                if not self.event_used:
                    self._event("locked")
                self.feedback = "locked"
                return False
            if [i, self.bin_label[a]] != self.orders[self.cursor]:
                return False
            self.item_at[i] = self.known_items[i] = 4
            self.known_bins[a] = [self.room, self.bin_label[a], 0]
            self.holding = -1
            self.cursor += 1
            self.feedback = "delivered"
            if self.disruption and self.disruption["recovered_step"] is None:
                self.disruption["recovered_step"] = self.steps
            return True
        if v == 5 and self.holding == i:
            self.item_at[i] = self.known_items[i] = self.room
            self.holding = -1
            self.feedback = "dropped"
            return True
        return False


def prompt(o):
    # No command history, command list, hidden state, or complete target command.
    orders = ";".join(f"{ITEMS[i]}:{LABELS[l]}" for i, l in o["orders"])
    items = ";".join(f"{name}:{loc}" for name, loc in zip(ITEMS, o["items"]))
    bins = ";".join(f"{name}:{r},{LABELS[l] if l >= 0 else '?'},{k}"
                    for name, (r, l, k) in zip(BINS, o["bins"]))
    active_i, active_l = o["orders"][o["cursor"]] if o["cursor"] < len(o["orders"]) else (-1, -1)
    active = f"{ITEMS[active_i]}:{LABELS[active_l]}" if active_i >= 0 else "finished"
    return (f"ordered requests {orders}\ncurrent request {active}\ncompleted {o['cursor']}; room {o['room']}; "
            f"hand {ITEMS[o['holding']] if o['holding'] >= 0 else 'empty'}; clock {o['steps']}\n"
            f"parcel locations (-1 unknown,3 carried,4 finished) {items}\n"
            f"receptacles (room,label,lock: -1 unknown/0 open/1 shut) {bins}\n"
            f"surveyed {','.join(str(int(x)) for x in o['seen'])}; feedback {o['feedback']}")


def features(o):
    f = []
    def oh(k, n):
        f.extend(float(j == k) for j in range(n))
    oh(o["room"], 3)
    oh(o["holding"] + 1, 5)
    oh(o["cursor"], 5)
    # Shared with the text prompt: no untrained slot-specific head is needed to
    # read the current request when evaluation reaches queue positions 3 or 4.
    i, l = o["orders"][o["cursor"]] if o["cursor"] < len(o["orders"]) else (-1, -1)
    oh(i + 1, 5)
    oh(l + 1, 4)
    for j in range(4):
        i, l = o["orders"][j] if j < len(o["orders"]) else (-1, -1)
        oh(i + 1, 5)
        oh(l + 1, 4)
    for loc in o["items"]:
        oh(loc + 1, 6)
    for r, l, k in o["bins"]:
        oh(r + 1, 4)
        oh(l + 1, 4)
        oh(k + 1, 3)
    f.extend(map(float, o["seen"]))
    oh(FEEDBACK.index(o["feedback"]), len(FEEDBACK))
    f.append(o["steps"] / MAX_STEPS)
    return np.array(f, dtype=np.float32)


class Planner:
    """Uses only the same notebook as each learned policy; tunes survey order/eagerness."""
    def __init__(self, room_order=(0, 1, 2), survey_first=False):
        self.room_order, self.survey_first = room_order, survey_first

    def __call__(self, o):
        r, hold = o["room"], o["holding"]
        i, label = o["orders"][o["cursor"]]
        if hold != -1 and hold != i:
            return f"DROP {ITEMS[hold]}"
        if not o["seen"][r]:
            return "INSPECT"
        unknown = [j for j in self.room_order if not o["seen"][j]]
        if self.survey_first and unknown:
            return f"GO {unknown[0]}"
        if hold == -1 and o["items"][i] >= 0:
            loc = o["items"][i]
            return f"TAKE {ITEMS[i]}" if loc == r else f"GO {loc}"
        if hold == i:
            for c, (loc, lab, lock) in enumerate(o["bins"]):
                if lab == label:
                    return f"GO {loc}" if loc != r else f"OPEN {BINS[c]}" if lock == 1 else f"SEND {ITEMS[i]} {BINS[c]}"
        return f"GO {unknown[0]}" if unknown else "INSPECT"


def rollout(policy, env, trace=False, deadline=None):
    import time
    records, valid, lat = [], 0, []
    while not env.over():
        if deadline is not None and time.monotonic() >= deadline:
            return None  # incomplete episodes never become successes/failures in metrics
        o = env.observe()
        start = time.monotonic()
        a = policy(o)
        lat.append(time.monotonic() - start)
        ok = env.step(a)
        valid += int(ok)
        if trace:
            records.append(dict(observation=o, action=a, valid=ok, feedback=env.feedback))
    return dict(seed=env.seed, split=env.split, mode=env.mode, success=env.success(),
                steps=env.steps, invalid=env.steps - valid, disruption=env.disruption,
                latency_ms=1000 * float(np.mean(lat)), trace=records)


def summarize(rows):
    if not rows:
        return dict(episodes=0)
    events = [r for r in rows if r["disruption"]]
    recovered = [r["disruption"] for r in events if r["disruption"]["recovered_step"] is not None]
    assigned = [r for r in rows if r["mode"] != "none"]
    return dict(episodes=len(rows), success_rate=float(np.mean([r["success"] for r in rows])),
                invalid_action_rate=sum(r["invalid"] for r in rows) / sum(r["steps"] for r in rows),
                mean_steps=float(np.mean([r["steps"] for r in rows])),
                disruption_assigned=len(assigned), disruption_exposed=len(events),
                exposure_coverage=len(events) / len(assigned) if assigned else None,
                success_given_disruption=float(np.mean([r["success"] for r in events])) if events else None,
                recovery_rate=len(recovered) / len(events) if events else None,
                recovery_steps_mean=float(np.mean([d["recovered_step"] - d["step"] for d in recovered])) if recovered else None,
                recovery_unresolved=len(events) - len(recovered),
                # Censor unresolved recovery at the remaining episode budget, not zero.
                recovery_steps_capped_mean=float(np.mean([
                    (r["disruption"]["recovered_step"] or MAX_STEPS) - r["disruption"]["step"]
                    for r in events])) if events else None,
                decision_latency_ms_mean=float(np.mean([r["latency_ms"] for r in rows])))


def tune_planner(seeds=range(4_000_000, 4_000_016)):
    trials = []
    for order, survey in itertools.product(itertools.permutations(range(3)), (False, True)):
        p = Planner(order, survey)
        rows = [rollout(p, Env(s, "train", mode)) for s in seeds for mode in MODES]
        m = summarize(rows)
        trials.append(dict(room_order=order, survey_first=survey, **m))
    best = min(trials, key=lambda t: (-t["success_rate"], t["mean_steps"], t["invalid_action_rate"]))
    return Planner(best["room_order"], best["survey_first"]), dict(selected=best, trials=trials)
