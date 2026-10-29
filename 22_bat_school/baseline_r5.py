"""Memory/odometry + soft echo filtering, common flight controller, two schedulers.

Only act(obs) is exposed: no environment, true position, map, or goal access.
All tunable parameters are sampled for BOTH event and uncertainty variants.
"""
import numpy as np
from env_r5 import (ECHO, HEARD, BUMP, MOVE, REMAIN, INITIAL, FORWARD, LEFT,
                    RIGHT, WAIT, BACK, DR, DC)

DEFAULT = dict(threshold=.36, wall_threshold=.43, temperature=.12, decay=.10, blend=.7, interval=2,
               uncertainty=.57, reserve=.4, caution=.42, repeat=2,
               explore=1.8, moth=.24, route="least_visited", handed=1)


def candidate(rng, index):
    if index == 0:
        return dict(DEFAULT)
    return dict(threshold=float(rng.uniform(.16, .60)), wall_threshold=float(rng.uniform(.30, .65)), temperature=float(rng.uniform(.05, .24)),
                decay=float(rng.uniform(.02, .30)), blend=float(rng.uniform(.3, 1.)),
                interval=int(rng.integers(1, 7)), uncertainty=float(rng.uniform(.25, .95)),
                reserve=float(rng.uniform(.05, 1.)), caution=float(rng.uniform(.2, .7)),
                repeat=int(rng.integers(0, 4)), explore=float(rng.uniform(.2, 4.)),
                moth=float(rng.uniform(.12, .5)), route=str(rng.choice(["least_visited", "wall_follow"])),
                handed=int(rng.choice([-1, 1])))


class AcousticPlanner:
    def __init__(self, n, mode="event", config=None):
        if mode not in ("event", "uncertainty", "every"):
            raise ValueError(mode)
        self.n, self.mode = n, mode
        self.cfg = dict(DEFAULT, **(config or {}))
        # Relative coordinates from odometry, padded beyond max maze diameter.
        self.width = 19
        self.reset()

    def reset(self):
        n, w = self.n, self.width
        self.pos = np.full((n, 2), w//2, int)
        self.heading = np.zeros(n, int)
        self.visits = np.zeros((n, w, w), int)
        self.popen = np.full((n, w, w, 4), .5)
        self.wall_logit = np.full((n, w, w, 4), -1.4)
        self.traversed = np.zeros((n, w, w, 4), bool)
        self.seen = np.zeros((n, w, w, 4), bool)
        self.goal_signal = np.zeros((n, w, w, 4))
        self.age = np.full(n, 100, int)
        self.stationary_listens = np.zeros(n, int)
        self.started = False
        self.target = -np.ones(n, int)

    def act(self, obs):
        p = self.cfg; n = self.n; ids = np.arange(n)
        prev = obs[:, MOVE:MOVE+5].argmax(-1)
        bump = obs[:, BUMP] > .5
        heard = obs[:, HEARD] > .5
        if self.started:
            d = (self.heading + 2*(prev == BACK)) % 4
            moved = np.isin(prev, [FORWARD, BACK]) & ~bump
            old_r, old_c = self.pos.T.copy()
            self.traversed[ids[moved], old_r[moved], old_c[moved], d[moved]] = True
            self.pos[:, 0] += moved * DR[d]; self.pos[:, 1] += moved * DC[d]
            self.pos = np.clip(self.pos, 0, self.width-1)
            self.traversed[ids[moved], self.pos[moved, 0], self.pos[moved, 1], (d[moved]+2) % 4] = True
            self.heading = (self.heading + (prev == RIGHT) - (prev == LEFT).astype(int)) % 4
            self.target[moved | bump] = -1
            self.stationary_listens[moved] = 0
            self.stationary_listens += heard & ~moved
        else:
            moved = np.ones(n, bool)
            self.started = True
        r, c = self.pos.T
        self.visits[ids, r, c] += moved
        self.age += 1
        self.age[heard] = 0
        # Forget dynamic occupancy toward 0.5, while retaining visit memory.
        self.popen += p["decay"] * (.5-self.popen)
        rel = (self.heading[:, None] + np.arange(4)) % 4
        echo = obs[:, :ECHO].reshape(n, 2, 4, 12)
        near = echo[:, 0, :, :2].sum(-1)
        likelihood = 1/(1+np.exp(np.clip((near-p["threshold"])/p["temperature"], -30, 30)))
        old = self.popen[ids[:, None], r[:, None], c[:, None], rel]
        new = (1-p["blend"])*old + p["blend"]*likelihood
        self.popen[ids[:, None], r[:, None], c[:, None], rel] = np.where(heard[:, None], new, old)
        self.seen[ids[:, None], r[:, None], c[:, None], rel] |= heard[:, None]
        # Near-bin asymmetry distinguishes a wall at d=.5 from a shutter at
        # d=1 (split across bins). Accumulate static topology separately from
        # dynamic blockage; completed moves establish a permanently open edge.
        cue = echo[:, 0, :, 0]-echo[:, 0, :, 1]
        evidence = np.clip((cue-p["wall_threshold"])/p["temperature"], -2., 2.)
        wall = self.wall_logit[ids[:, None], r[:, None], c[:, None], rel]
        self.wall_logit[ids[:, None], r[:, None], c[:, None], rel] = np.clip(wall+heard[:, None]*evidence, -8., 8.)
        self.wall_logit[self.traversed] = -8.
        goal = np.maximum(echo[:, 1], 0).max(-1)
        old_goal = self.goal_signal[ids[:, None], r[:, None], c[:, None], rel]
        self.goal_signal[ids[:, None], r[:, None], c[:, None], rel] = np.where(heard[:, None], goal, old_goal)
        # A bump is an informative contact, available to every agent.
        bump_direction = (self.heading+2*(prev == BACK)) % 4
        self.popen[ids[bump], r[bump], c[bump], bump_direction[bump]] = .05
        po = self.popen[ids, r, c]
        wall_probability = 1/(1+np.exp(-self.wall_logit[ids, r, c]))
        known = self.seen[ids, r, c]
        nr = np.clip(r[:, None]+DR, 0, self.width-1)
        nc = np.clip(c[:, None]+DC, 0, self.width-1)
        visits = self.visits[ids[:, None], nr, nc]
        goal = self.goal_signal[ids, r, c]
        relative = (np.arange(4)[None, :]-self.heading[:, None]) % 4
        preference = np.array([.15, .3, -.1, 0.]) if p["handed"] == 1 else np.array([.15, 0., -.1, .3])
        score = 2*po - 12*wall_probability + preference[relative] + 3*(goal > p["moth"])*goal
        if p["route"] == "least_visited":
            score -= p["explore"]*np.log1p(visits)
        else:
            score += .9*preference[relative]
        score -= 3*(po < p["caution"])
        # Finish a turn before selecting a new direction; fresh echoes may veto
        # it if the corridor has closed in the meantime.
        target = score.argmax(-1)
        hold = (self.target >= 0) & ~bump
        target[hold] = self.target[hold]
        bad = po[ids, target] < p["caution"]
        target[bad] = score.argmax(-1)[bad]
        self.target = target.copy()
        delta = (target-self.heading) % 4
        move = np.array([FORWARD, RIGHT, BACK, LEFT])[delta]
        remaining = np.rint(obs[:, REMAIN]*64).astype(int)
        initial = np.maximum(np.rint(obs[:, INITIAL]*64), 1)
        # Budget-aware spacing grows as the remaining fraction shrinks.
        spacing = np.maximum(1, np.ceil(p["interval"] * (1+p["reserve"]*(1-remaining/initial)*4))).astype(int)
        uncertainty = (4*po*(1-po))[ids, target]
        novel = ~known.any(-1)
        event = novel | bump | (self.age >= spacing)
        if self.mode == "event":
            ask = event
        elif self.mode == "uncertainty":
            ask = novel | bump | ((uncertainty > p["uncertainty"]) & (self.age >= spacing))
        else:
            ask = np.ones(n, bool)
        unsafe = po[ids, target] < p["caution"]
        # Wait/listen while resolving an unknown or blocked crossing. Repeat
        # allowance is bounded; endlessly abstaining must incur timeout cost.
        resolve = (novel | unsafe) & (self.stationary_listens < p["repeat"]+1) & (remaining > 0)
        move[resolve] = WAIT
        ask |= resolve
        self.target[resolve] = -1
        ask &= remaining > 0
        return move, ask
