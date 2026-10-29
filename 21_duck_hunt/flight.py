"""2D duck flight sim (vectorized numpy) + tunable PD/pursuit baseline controller.

Body: point mass in the x (downrange) / z (up) plane.
  - gravity 9.81 m/s^2, quadratic air drag on air-relative velocity
  - thrust command a in [-1,1]^2, scaled by AMAX and clipped to |thrust| <= AMAX,
    passed through a first-order actuator lag (tau = 0.1 s)
  - wind gusts: Ornstein-Uhlenbeck process (unobserved by both controllers)
Course: N_WP waypoints of radius WP_R, visited in order, within T_LIM seconds.
Fail: touching the ground (z < 0), leaving the arena, or running out of time.
"""
import numpy as np

DT = 0.05
G = 9.81
AMAX = 2.0 * G          # max thrust acceleration (2 g)
DRAG = 0.04             # quadratic drag coefficient (1/m)
TAU_ACT = 0.10          # actuator lag
WP_R = 1.0              # waypoint radius (m)
N_WP = 6
T_LIM = 25.0
MAX_STEPS = int(T_LIM / DT)
X_LO, X_HI, Z_LO, Z_HI = 0.0, 30.0, 2.0, 15.0     # waypoint box
ARENA = (-8.0, 38.0, 0.0, 24.0)                    # leave this -> fail
OBS_DIM = 11


def make_courses(rng, n, n_wp=N_WP, dmin=5.0, dmax=12.0):
    """Waypoint sequences, consecutive spacing in [dmin, dmax], inside the box.
    Start point is placed 3-6 m from the first waypoint."""
    W = np.zeros((n, n_wp + 1, 2))
    for i in range(n):
        p = np.array([rng.uniform(X_LO, X_HI), rng.uniform(Z_LO, Z_HI)])
        W[i, 0] = p
        for k in range(1, n_wp + 1):
            lo = dmin if k > 1 else 3.0
            hi = dmax if k > 1 else 6.0
            for _ in range(200):
                ang = rng.uniform(-np.pi, np.pi)
                d = rng.uniform(lo, hi)
                q = p + d * np.array([np.cos(ang), np.sin(ang)])
                if X_LO <= q[0] <= X_HI and Z_LO <= q[1] <= Z_HI:
                    break
            W[i, k] = q
            p = q
    return W  # W[:,0] = start, W[:,1:] = waypoints


class FlightEnv:
    """Vectorized env over n courses. Each env has its own course and wind seed."""

    def __init__(self, n, seed, wind_sigma=2.0, wind_tau=2.0, fixed_courses=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.wind_sigma = wind_sigma
        self.wind_tau = wind_tau
        self.fixed = fixed_courses  # if given, reset() cycles through these once
        self.reset_all()

    # ---------- state ----------
    def _new(self, idx):
        if self.fixed is not None:
            W = self.fixed[idx]
        else:
            W = make_courses(self.rng, len(idx))
        self.W[idx] = W
        self.p[idx] = W[:, 0]
        self.v[idx] = 0.0
        self.thr[idx] = np.array([0.0, G])  # start hovering
        self.wind[idx] = self.rng.normal(0, self.wind_sigma, (len(idx), 2)) * np.array([1.0, 0.4])
        self.k[idx] = 1
        self.t[idx] = 0

    def reset_all(self):
        n = self.n
        self.W = np.zeros((n, N_WP + 1, 2))
        self.p = np.zeros((n, 2)); self.v = np.zeros((n, 2)); self.thr = np.zeros((n, 2))
        self.wind = np.zeros((n, 2)); self.k = np.ones(n, int); self.t = np.zeros(n, int)
        self._new(np.arange(n))
        return self.obs()

    def targets(self):
        ar = np.arange(self.n)
        k = np.minimum(self.k, N_WP)
        cur = self.W[ar, k]
        nxt = self.W[ar, np.minimum(k + 1, N_WP)]
        last = (self.k >= N_WP)
        return cur, nxt, last

    def obs(self):
        cur, nxt, last = self.targets()
        e = cur - self.p
        leg = np.where(last[:, None], 0.0, nxt - cur)
        o = np.concatenate([
            e / 10.0, leg / 10.0, self.v / 10.0, self.thr / AMAX,
            (self.p[:, 1:2] - 8.0) / 8.0,
            1.0 - self.t[:, None] / MAX_STEPS,
            ((N_WP - self.k)[:, None]) / N_WP,
        ], axis=1)
        return o.astype(np.float32)

    # ---------- dynamics ----------
    def step(self, a):
        """a: (n,2) in [-1,1]. Returns obs, reward, done, info dict."""
        a = np.clip(a, -1.0, 1.0) * AMAX
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        a = a * np.minimum(1.0, AMAX / np.maximum(nrm, 1e-9))
        cur, _, _ = self.targets()
        d0 = np.linalg.norm(cur - self.p, axis=1)
        # actuator lag
        self.thr += (a - self.thr) * (DT / TAU_ACT)
        # gusts (OU)
        s = self.wind_sigma * np.array([1.0, 0.4])
        self.wind += -self.wind * (DT / self.wind_tau) + s * np.sqrt(2 * DT / self.wind_tau) * self.rng.normal(size=(self.n, 2))
        vrel = self.v - self.wind
        drag = -DRAG * np.linalg.norm(vrel, axis=1, keepdims=True) * vrel
        acc = self.thr + drag + np.array([0.0, -G])
        # segment check for waypoint pass (catch fast fly-through)
        p_old = self.p.copy()
        self.v += acc * DT
        self.p += self.v * DT
        self.t += 1
        seg = self.p - p_old
        w = cur - p_old
        u = np.clip((w * seg).sum(1) / np.maximum((seg * seg).sum(1), 1e-9), 0, 1)
        closest = np.linalg.norm(p_old + u[:, None] * seg - cur, axis=1)
        reached = closest < WP_R
        d1 = np.linalg.norm(cur - self.p, axis=1)
        rew = (d0 - d1) * 1.0 - 0.01
        rew += reached * 2.0
        self.k += reached
        complete = self.k > N_WP
        crash = (self.p[:, 1] < ARENA[2]) | (self.p[:, 1] > ARENA[3]) | (self.p[:, 0] < ARENA[0]) | (self.p[:, 0] > ARENA[1])
        timeout = self.t >= MAX_STEPS
        rew += complete * 5.0 - crash * 5.0
        done = complete | crash | timeout
        info = dict(reached=reached, complete=complete, crash=crash, timeout=timeout,
                    t=self.t.copy(), k=self.k.copy())
        idx = np.nonzero(done)[0]
        if len(idx) and self.fixed is None:
            self._new(idx)
        return self.obs(), rew.astype(np.float32), done, info


# ---------------- PD / pursuit baseline ----------------
PD_DEFAULT = dict(vmax=8.0, kv=4.0, r_slow=3.0, w_next=0.5, r_blend=2.0, drag_comp=1.0, kz=1.0)


def pd_action(env, prm):
    """Pursuit guidance: desired velocity points at the current waypoint (blended toward
    the next leg when close), speed capped at vmax and ramped down inside r_slow;
    acceleration = kv*(v_des - v) + gravity (+ drag) compensation. Same thrust limits."""
    cur, nxt, last = env.targets()
    e = cur - env.p
    d = np.linalg.norm(e, axis=1, keepdims=True)
    u = e / np.maximum(d, 1e-6)
    leg = nxt - cur
    ul = leg / np.maximum(np.linalg.norm(leg, axis=1, keepdims=True), 1e-6)
    ul = np.where(last[:, None], 0.0, ul)
    blend = prm["w_next"] * np.exp(-d / max(prm["r_blend"], 1e-3))
    dirv = u + blend * ul
    dirv = dirv / np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-6)
    # slow down only for the final waypoint; mid-course waypoints keep (blended) speed
    slow = np.where(last[:, None], np.minimum(1.0, d / max(prm["r_slow"], 1e-3)),
                    np.minimum(1.0, (d + prm["r_slow"] * prm["w_next"]) / max(prm["r_slow"], 1e-3)))
    vdes = prm["vmax"] * dirv * slow
    ev = vdes - env.v
    along = (ev * dirv).sum(1, keepdims=True) * dirv
    acc = prm["kv"] * along + prm.get("kperp", prm["kv"]) * (ev - along)   # separate cross-track gain
    acc[:, 1] *= prm["kz"]
    acc[:, 1] += G
    acc += prm["drag_comp"] * DRAG * np.linalg.norm(env.v, axis=1, keepdims=True) * env.v
    return acc / AMAX


def run_episodes(courses, policy_fn, seed, wind_sigma=2.0, record=False):
    """Run each fixed course once. policy_fn(env, obs) -> action. Returns per-course metrics."""
    n = len(courses)
    env = FlightEnv(n, seed, wind_sigma=wind_sigma, fixed_courses=courses)
    obs = env.obs()
    alive = np.ones(n, bool)
    res = dict(complete=np.zeros(n, bool), crash=np.zeros(n, bool), t=np.full(n, np.nan),
               wps=np.zeros(n, int))
    traj = [] if record else None
    for _ in range(MAX_STEPS):
        a = policy_fn(env, obs)
        a = np.where(alive[:, None], a, 0.0)
        obs, r, done, info = env.step(a)
        if record:
            traj.append(np.where(alive[:, None], env.p, np.nan).copy())
        newly = done & alive
        res["complete"][newly] = info["complete"][newly]
        res["crash"][newly] = info["crash"][newly]
        res["t"][newly & info["complete"]] = info["t"][newly & info["complete"]] * DT
        res["wps"][alive] = np.minimum(info["k"][alive] - 1, N_WP)
        alive &= ~done
        # freeze finished envs
        env.v[~alive] = 0.0
        env.thr[~alive] = np.array([0.0, G])
        if not alive.any():
            break
    if record:
        res["traj"] = np.stack(traj, 1)
    return res


def summarize(res):
    c = res["complete"]
    return dict(completion=float(c.mean()), crash=float(res["crash"].mean()),
                mean_time_completed=float(np.nanmean(res["t"])) if c.any() else float("nan"),
                median_time_completed=float(np.nanmedian(res["t"])) if c.any() else float("nan"),
                mean_waypoints=float(res["wps"].mean()), n=int(len(c)))


def score(res):
    """Tuning objective used for BOTH methods' model selection: completion first, then speed."""
    c = res["complete"]
    t = np.where(c, res["t"], T_LIM)
    return float(np.mean(c * (2.0 - t / T_LIM) + (~c) * res["wps"] / N_WP))


# held-out / training course sets (fixed seeds)
def train_courses(n=256):
    return make_courses(np.random.default_rng(1), n)


def heldout_courses(n=500):
    return make_courses(np.random.default_rng(777_000), n)
