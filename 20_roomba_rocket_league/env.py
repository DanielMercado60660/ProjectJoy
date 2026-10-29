"""Roomba Rocket League: a batched top-down 2D soccer arena in numpy.

Field: x in [-1, 1], y in [-0.7, 0.7]. Goals are gaps of half-width GOAL_HALF in the end walls.
The learner (robot 0) attacks the +x goal and defends the -x goal. The opponent (robot 1) is a
scripted defender of the +x goal. Both robots are differential-drive discs: action = (left, right)
wheel commands in [-1, 1]. Kinematic drive (no inertia), ball has rolling friction and bounces.

Everything is batched over a leading env axis so N matches step at once.
"""
import numpy as np

W, H = 1.0, 0.7            # half-length, half-width
GOAL_HALF = 0.3
R_ROBOT = 0.07
R_BALL = 0.04
DT = 0.05
T_MAX = 300                # 15 s matches
V_LEARNER = 0.8            # m/s top speed
V_OPP = 0.56               # scripted opponent is 0.7x slower
W_MAX = 6.0                # rad/s top turn rate
BALL_FRICTION = 0.8        # 1/s exponential decay
BALL_VMAX = 2.5
E_ROBOT = 0.5              # robot-ball restitution
E_WALL = 0.7
PIN_MARGIN = 0.02          # ball within R_BALL + this of a side wall counts as "on the wall"

OUT_NONE, OUT_GOAL, OUT_CONCEDED, OUT_OWN_GOAL, OUT_TIMEOUT = 0, 1, 2, 3, 4
OUT_NAMES = {OUT_GOAL: "goal", OUT_CONCEDED: "conceded", OUT_OWN_GOAL: "own_goal", OUT_TIMEOUT: "timeout"}
OBS_DIM = 23


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def in_corner(bx, by):
    """Held-out start region: ball in the four corners (never used for training starts)."""
    return (np.abs(bx) > 0.5) & (np.abs(by) > 0.42)


class Arena:
    def __init__(self, n, seed=0, opponent="goalie"):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.opponent = opponent
        self.touch_bonus = "first"  # "first" (run 2) or "every" (run 1, farmable)
        self.curriculum = 1.0     # 0 = ball near the opponent goal, 1 = anywhere
        self.p_anywhere = 0.2     # fraction of resets that ignore the curriculum
        self.start_mode = "train"  # "train" | "corner" | "kickoff"
        self.pos = np.zeros((n, 2, 2))   # [env, robot, xy]
        self.th = np.zeros((n, 2))
        self.vel = np.zeros((n, 2, 2))
        self.ball = np.zeros((n, 2))
        self.bvel = np.zeros((n, 2))
        self.t = np.zeros(n, dtype=np.int64)
        self.last_touch = -np.ones(n, dtype=np.int64)
        self.first_touch = -np.ones(n, dtype=np.int64)
        self.touches = np.zeros((n, 2), dtype=np.int64)
        self.pin_steps = np.zeros(n, dtype=np.int64)
        self.near_steps = np.zeros(n, dtype=np.int64)   # possession proxy: learner closer to ball
        self.reset(np.ones(n, dtype=bool))

    # ------------------------------------------------------------------ resets
    def _sample_starts(self, k):
        r = self.rng
        mode = self.start_mode
        if mode == "kickoff":
            ball = np.stack([np.zeros(k), r.uniform(-0.1, 0.1, k)], 1)
            p0 = np.stack([r.uniform(-0.7, -0.5, k), r.uniform(-0.2, 0.2, k)], 1)
            th0 = r.uniform(-0.5, 0.5, k)
        else:
            ball = np.zeros((k, 2))
            todo = np.ones(k, dtype=bool)
            while todo.any():
                m = todo.sum()
                if mode == "corner":
                    bx = r.choice([-1, 1], m) * r.uniform(0.55, 0.85, m)
                    by = r.choice([-1, 1], m) * r.uniform(0.45, 0.62, m)
                    ok = np.ones(m, dtype=bool)
                else:
                    c = np.where(r.random(m) < self.p_anywhere, 1.0, self.curriculum)
                    xlo = 0.35 + (-0.85 - 0.35) * c
                    bx = r.uniform(xlo, 0.8)
                    by = r.uniform(-1, 1, m) * (0.3 + 0.32 * c)
                    ok = ~in_corner(bx, by)
                idx = np.flatnonzero(todo)[ok]
                ball[idx] = np.stack([bx[ok], by[ok]], 1)
                todo[idx] = False
            # learner: somewhere behind (in -x direction of) the ball, within reach that grows with curriculum
            c = self.curriculum if mode == "train" else 1.0
            back = r.uniform(0.15, 0.25 + 0.9 * c, k)
            p0 = np.stack([ball[:, 0] - back, ball[:, 1] + r.uniform(-1, 1, k) * (0.15 + 0.5 * c)], 1)
            th0 = r.uniform(-np.pi, np.pi, k)
        p0[:, 0] = np.clip(p0[:, 0], -W + R_ROBOT, W - R_ROBOT)
        p0[:, 1] = np.clip(p0[:, 1], -H + R_ROBOT, H - R_ROBOT)
        p1 = np.stack([np.full(k, 0.85), r.uniform(-0.2, 0.2, k)], 1)
        if self.opponent == "chaser":
            p1 = np.stack([r.uniform(0.4, 0.85, k), r.uniform(-0.4, 0.4, k)], 1)
        th1 = np.full(k, np.pi) + r.uniform(-0.3, 0.3, k)
        # keep robots off the ball and off each other
        for p in (p0, p1):
            d = p - ball
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            too = dist[:, 0] < R_ROBOT + R_BALL + 0.03
            p[too] = ball[too] + d[too] / dist[too] * (R_ROBOT + R_BALL + 0.03)
        return ball, p0, th0, p1, th1

    def reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        ball, p0, th0, p1, th1 = self._sample_starts(k)
        self.ball[mask] = ball
        self.bvel[mask] = 0
        self.pos[mask, 0] = p0
        self.pos[mask, 1] = p1
        self.th[mask, 0] = th0
        self.th[mask, 1] = th1
        self.vel[mask] = 0
        self.t[mask] = 0
        self.last_touch[mask] = -1
        self.first_touch[mask] = -1
        self.touches[mask] = 0
        self.pin_steps[mask] = 0
        self.near_steps[mask] = 0

    # ------------------------------------------------------------------ observation
    def obs(self):
        p, th, b, bv = self.pos, self.th, self.ball, self.bvel
        c0, s0 = np.cos(th[:, 0]), np.sin(th[:, 0])

        def ego(v):  # world vector -> learner frame
            return np.stack([c0 * v[:, 0] + s0 * v[:, 1], -s0 * v[:, 0] + c0 * v[:, 1]], 1)

        goal = np.array([W, 0.0])
        own = np.array([-W, 0.0])
        o = np.concatenate([
            p[:, 0], np.stack([c0, s0], 1),
            b, bv,
            ego(b - p[:, 0]), ego(bv),
            p[:, 1], np.stack([np.cos(th[:, 1]), np.sin(th[:, 1])], 1),
            ego(p[:, 1] - p[:, 0]),
            ego(goal - p[:, 0]),
            ego(own - p[:, 0]),
            (self.t / T_MAX)[:, None],
        ], 1)
        return o.astype(np.float32)

    def potential(self):
        d_rb = np.linalg.norm(self.ball - self.pos[:, 0], axis=1)
        d_bg = np.linalg.norm(self.ball - np.array([W, 0.0]), axis=1)
        return -0.5 * d_rb - 2.0 * d_bg

    # ------------------------------------------------------------------ physics
    def step(self, a0, a1):
        """a0, a1: (n, 2) wheel commands in [-1, 1] for learner and opponent.
        Returns obs, reward, done (terminal), trunc (timeout), outcome (n,), info dict of finished episodes."""
        n = self.n
        phi0 = self.potential()
        acts = np.stack([np.clip(a0, -1, 1), np.clip(a1, -1, 1)], 1)    # n,2,2
        vmax = np.array([V_LEARNER, V_OPP])
        v = vmax * (acts[..., 0] + acts[..., 1]) / 2
        w = W_MAX * (acts[..., 1] - acts[..., 0]) / 2
        self.th = wrap(self.th + w * DT)
        heading = np.stack([np.cos(self.th), np.sin(self.th)], -1)
        self.vel = v[..., None] * heading
        self.pos = self.pos + self.vel * DT
        self.pos[..., 0] = np.clip(self.pos[..., 0], -W + R_ROBOT, W - R_ROBOT)
        self.pos[..., 1] = np.clip(self.pos[..., 1], -H + R_ROBOT, H - R_ROBOT)
        # robot-robot separation
        d = self.pos[:, 0] - self.pos[:, 1]
        dist = np.linalg.norm(d, axis=1) + 1e-9
        ov = np.maximum(0, 2 * R_ROBOT - dist)
        push = (d / dist[:, None]) * (ov / 2)[:, None]
        self.pos[:, 0] += push
        self.pos[:, 1] -= push

        # ball
        self.bvel *= np.exp(-BALL_FRICTION * DT)
        self.ball = self.ball + self.bvel * DT
        touched = np.zeros((n, 2), dtype=bool)
        for i in range(2):
            d = self.ball - self.pos[:, i]
            dist = np.linalg.norm(d, axis=1) + 1e-9
            hit = dist < R_ROBOT + R_BALL
            if hit.any():
                nrm = d[hit] / dist[hit, None]
                self.ball[hit] = self.pos[hit, i] + nrm * (R_ROBOT + R_BALL)
                u = self.bvel[hit] - self.vel[hit, i]
                un = np.sum(u * nrm, 1)
                imp = np.where(un < 0, -(1 + E_ROBOT) * un, 0.0)
                self.bvel[hit] += imp[:, None] * nrm
                touched[hit, i] = True
        sp = np.linalg.norm(self.bvel, axis=1)
        self.bvel *= (np.minimum(sp, BALL_VMAX) / (sp + 1e-9))[:, None]

        # side walls
        for sgn in (1, -1):
            m = sgn * self.ball[:, 1] > H - R_BALL
            self.ball[m, 1] = sgn * (H - R_BALL)
            self.bvel[m, 1] = -E_WALL * np.abs(self.bvel[m, 1]) * sgn
        # end walls / goals
        mouth = np.abs(self.ball[:, 1]) < GOAL_HALF
        scored = (self.ball[:, 0] > W) & mouth
        against = (self.ball[:, 0] < -W) & mouth
        for sgn in (1, -1):
            m = (sgn * self.ball[:, 0] > W - R_BALL) & ~mouth
            self.ball[m, 0] = sgn * (W - R_BALL)
            self.bvel[m, 0] = -E_WALL * np.abs(self.bvel[m, 0]) * sgn
        # a ball sitting in a goal mouth but not over the line can still be past the robots' clamp: fine.

        # bookkeeping
        self._first_touch_now = touched[:, 0] & (self.touches[:, 0] == 0)
        for i in range(2):
            ti = touched[:, i]
            self.last_touch[ti] = i
            self.first_touch[ti & (self.first_touch < 0)] = i
            self.touches[ti, i] += 1
        bx, by = self.ball[:, 0], self.ball[:, 1]
        on_wall = (np.abs(by) > H - R_BALL - PIN_MARGIN) | ((np.abs(bx) > W - R_BALL - PIN_MARGIN) & ~mouth)
        self.pin_steps += (on_wall & touched[:, 0])
        dl = np.linalg.norm(self.ball - self.pos[:, 0], axis=1)
        do = np.linalg.norm(self.ball - self.pos[:, 1], axis=1)
        self.near_steps += dl < do
        self.t += 1

        outcome = np.zeros(n, dtype=np.int64)
        outcome[scored] = OUT_GOAL
        own = against & (self.last_touch == 0)
        outcome[against & ~own] = OUT_CONCEDED
        outcome[own] = OUT_OWN_GOAL
        done = outcome > 0
        trunc = (~done) & (self.t >= T_MAX)
        outcome[trunc] = OUT_TIMEOUT

        reward = (self.potential() - phi0)
        reward = reward + 10.0 * scored - 10.0 * against
        if self.touch_bonus == "every":   # run 1 (bug): paid every contact step -> wall-pinning farm
            reward = reward + 0.05 * touched[:, 0]
        else:                              # run 2: one-shot bonus for the learner's first touch only
            reward = reward + 0.5 * self._first_touch_now

        fin = done | trunc
        info = None
        if fin.any():
            info = dict(outcome=outcome[fin].copy(), t=self.t[fin].copy(),
                        first_touch=self.first_touch[fin].copy(), touches=self.touches[fin].copy(),
                        pin_steps=self.pin_steps[fin].copy(), near_steps=self.near_steps[fin].copy(),
                        idx=np.flatnonzero(fin))
        return reward.astype(np.float32), done, trunc, outcome, info

    def snapshot(self, i=0):
        return dict(pos=self.pos[i].copy(), th=self.th[i].copy(), ball=self.ball[i].copy(), t=int(self.t[i]))
