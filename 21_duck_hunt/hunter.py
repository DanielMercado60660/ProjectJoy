"""Hunter game (R6, "Go deeper"): an evasive duck on a timed course vs a crosshair hunter.
State-based (no pixels). Design: HUNTER_DESIGN.md.

Duck physics are the flight.py physics (copied step code, same constants). The hunter moves a
crosshair at <= V_CROSS, has AMMO shells, a DELAY-step shot delay (the shot lands where the
crosshair was at the trigger pull) and a COOLDOWN between shots. The hunter observes the duck's
position with Gaussian noise only.
"""
import numpy as np
import flight as F
import pd_plus as P

DT = F.DT
T_LIM = 11.0
MAXS = int(round(T_LIM / DT))          # 220 steps
V_CROSS = 14.0
AMMO = 3
DELAY = 5                              # 0.25 s from trigger to impact
COOLDOWN = 8                           # 0.4 s between shots
HIT_R = 0.6
OBS_NOISE = 0.3
HIST = 6
X0, X1, Z0, Z1 = F.ARENA
DUCK_OBS = 21
HUNT_OBS = 2 * HIST + 7


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def _clipn(x, m=1.0):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x * np.minimum(1.0, m / np.maximum(n, 1e-9))


class DuelEnv:
    """Vectorized duel. Attributes W, p, v, k, n, targets() match flight.FlightEnv so the PD+
    controller runs on it unchanged. If fixed_courses is given, envs are not auto-reset."""

    def __init__(self, n, seed, wind_sigma=2.0, fixed_courses=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.wind_sigma = wind_sigma
        self.fixed = fixed_courses
        self.W = np.zeros((n, F.N_WP + 1, 2)); self.p = np.zeros((n, 2)); self.v = np.zeros((n, 2))
        self.thr = np.zeros((n, 2)); self.wind = np.zeros((n, 2)); self.k = np.ones(n, int); self.t = np.zeros(n, int)
        self.c = np.zeros((n, 2)); self.cv = np.zeros((n, 2)); self.ammo = np.zeros(n, int); self.cool = np.zeros(n, int)
        self.pend = np.zeros(n, int); self.aim = np.zeros((n, 2)); self.H = np.zeros((n, HIST, 2))
        self.reset(np.arange(n))

    def reset(self, idx):
        if len(idx) == 0:
            return
        W = self.fixed[idx] if self.fixed is not None else F.make_courses(self.rng, len(idx))
        self.W[idx] = W; self.p[idx] = W[:, 0]; self.v[idx] = 0.0; self.thr[idx] = np.array([0.0, F.G])
        self.wind[idx] = self.rng.normal(0, self.wind_sigma, (len(idx), 2)) * np.array([1.0, 0.4])
        self.k[idx] = 1; self.t[idx] = 0
        # crosshair starts at a random point 6-12 m from the duck, inside the waypoint box
        ang = self.rng.uniform(-np.pi, np.pi, len(idx)); r = self.rng.uniform(6, 12, len(idx))
        c = W[:, 0] + np.stack([np.cos(ang), np.sin(ang)], 1) * r[:, None]
        self.c[idx] = np.clip(c, [F.X_LO, F.Z_LO], [F.X_HI, F.Z_HI]); self.cv[idx] = 0.0
        self.ammo[idx] = AMMO; self.cool[idx] = 0; self.pend[idx] = 0; self.aim[idx] = 0.0
        z = self.p[idx] + self.rng.normal(0, OBS_NOISE, (len(idx), 2))
        self.H[idx] = z[:, None, :]

    def targets(self):
        ar = np.arange(self.n)
        k = np.minimum(self.k, F.N_WP)
        return self.W[ar, k], self.W[ar, np.minimum(k + 1, F.N_WP)], self.k >= F.N_WP

    # ---------------- observations ----------------
    def duck_obs(self):
        cur, nxt, last = self.targets()
        leg = np.where(last[:, None], 0.0, nxt - cur)
        pending = (self.pend > 0)[:, None]
        o = np.concatenate([
            (cur - self.p) / 10.0, leg / 10.0, self.v / 10.0, self.thr / F.AMAX,
            (self.p[:, 1:2] - 8.0) / 8.0, 1.0 - self.t[:, None] / MAXS, ((F.N_WP - self.k)[:, None]) / F.N_WP,
            (self.c - self.p) / 10.0, self.cv / V_CROSS, self.ammo[:, None] / AMMO, self.cool[:, None] / COOLDOWN,
            pending.astype(float), np.where(pending, (self.aim - self.p) / 5.0, 0.0), self.pend[:, None] / DELAY,
        ], 1)
        return o.astype(np.float32)

    def hunter_obs(self):
        rel = (self.H - self.c[:, None, :]) / 10.0
        o = np.concatenate([
            rel.reshape(self.n, -1), (self.c - [15.0, 8.0]) / 10.0, self.cv / V_CROSS,
            self.ammo[:, None] / AMMO, self.cool[:, None] / COOLDOWN, 1.0 - self.t[:, None] / MAXS,
        ], 1)
        return o.astype(np.float32)

    # ---------------- dynamics ----------------
    def step(self, a_duck, a_hunt):
        """a_duck: (n,2) thrust in [-1,1] (norm-clipped); a_hunt: (n,3) crosshair velocity in
        [-1,1]^2 (x V_CROSS, norm-clipped) + trigger (fires if > 0)."""
        n = self.n
        # hunter
        self.cv = _clipn(np.clip(a_hunt[:, :2], -1, 1)) * V_CROSS
        self.c = np.clip(self.c + self.cv * DT, [X0, Z0], [X1, Z1])
        fire = (a_hunt[:, 2] > 0) & (self.ammo > 0) & (self.cool == 0) & (self.pend == 0)
        self.ammo -= fire; self.cool = np.where(fire, COOLDOWN, np.maximum(self.cool - 1, 0))
        self.aim = np.where(fire[:, None], self.c, self.aim)
        self.pend = np.where(fire, DELAY, self.pend)
        # duck (flight.py physics)
        a = _clipn(np.clip(a_duck, -1, 1)) * F.AMAX
        cur, _, _ = self.targets()
        self.thr += (a - self.thr) * (DT / F.TAU_ACT)
        s = self.wind_sigma * np.array([1.0, 0.4])
        self.wind += -self.wind * (DT / 2.0) + s * np.sqrt(2 * DT / 2.0) * self.rng.normal(size=(n, 2))
        vrel = self.v - self.wind
        acc = self.thr - F.DRAG * np.linalg.norm(vrel, axis=1, keepdims=True) * vrel + np.array([0.0, -F.G])
        p_old = self.p.copy()
        self.v += acc * DT
        self.p += self.v * DT
        self.t += 1
        seg = self.p - p_old; w = cur - p_old
        u = np.clip((w * seg).sum(1) / np.maximum((seg * seg).sum(1), 1e-9), 0, 1)
        reached = np.linalg.norm(p_old + u[:, None] * seg - cur, axis=1) < F.WP_R
        # shot resolution (after the duck moved)
        impact = self.pend == 1
        hit = impact & (np.linalg.norm(self.p - self.aim, axis=1) < HIT_R)
        self.pend = np.maximum(self.pend - 1, 0)
        self.k += reached & ~hit
        complete = (self.k > F.N_WP) & ~hit
        crash = ((self.p[:, 1] < Z0) | (self.p[:, 1] > Z1) | (self.p[:, 0] < X0) | (self.p[:, 0] > X1)) & ~hit & ~complete
        timeout = (self.t >= MAXS) & ~hit & ~complete & ~crash
        done = hit | complete | crash | timeout
        # hunter's noisy view
        z = self.p + self.rng.normal(0, OBS_NOISE, (n, 2))
        self.H = np.concatenate([self.H[:, 1:], z[:, None]], 1)
        info = dict(hit=hit, complete=complete, crash=crash, timeout=timeout, fired=fire, reached=reached,
                    impact=impact, t=self.t.copy(), k=self.k.copy())
        return done, info


# ---------------- scripted evaders ----------------
PDP = None


def _pdp():
    global PDP
    if PDP is None:
        import json, os
        PDP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/pd_plus_params.json")))
    return PDP


class Straight:
    name = "straight"

    def __init__(self, n, rng):
        self.n, self.rng = n, rng

    def reset(self, idx):
        pass

    def act(self, env):
        return P.pd_plus_action(env, _pdp())


class Jink(Straight):
    """PD+ plus a random-telegraph lateral thrust (side flips every 0.2-0.6 s)."""
    name = "jink"
    MAG = 0.25

    def __init__(self, n, rng):
        super().__init__(n, rng)
        self.side = rng.choice([-1.0, 1.0], n); self.left = rng.integers(4, 13, n)

    def reset(self, idx):
        self.side[idx] = self.rng.choice([-1.0, 1.0], len(idx)); self.left[idx] = self.rng.integers(4, 13, len(idx))

    def lateral(self, env):
        self.left -= 1
        flip = self.left <= 0
        self.side = np.where(flip, -self.side, self.side)
        self.left = np.where(flip, self.rng.integers(4, 13, self.n), self.left)
        vd = _unit(env.v + 1e-3)
        return self.side[:, None] * np.stack([-vd[:, 1], vd[:, 0]], 1) * self.MAG

    def act(self, env):
        return P.norm_clip((P.pd_plus_action(env, _pdp()) + self.lateral(env)) * F.AMAX)


class Dodge(Straight):
    """PD+, but while a shot is pending: full thrust away from the aim point."""
    name = "dodge"

    def act(self, env, base=None):
        base = P.pd_plus_action(env, _pdp()) if base is None else base
        away = env.p - env.aim
        d = np.linalg.norm(away, axis=1, keepdims=True)
        vd = _unit(env.v + 1e-3)
        away = np.where(d < 1e-3, np.stack([-vd[:, 1], vd[:, 0]], 1), away)
        dodge = P.norm_clip(_unit(away) * F.AMAX + np.array([0.0, F.G]))
        threat = ((env.pend > 0) & (d[:, 0] < 2.5 * HIT_R))[:, None]
        return np.where(threat, dodge, base)


class DodgeJink(Jink):
    """Held-out evader: jink + dodge."""
    name = "dodge_jink"

    def act(self, env):
        base = Jink.act(self, env)
        return Dodge.act(self, env, base)


# ---------------- scripted hunters ----------------
class Naive:
    """Chase the latest noisy position; fire when the crosshair is within 0.5 m of it."""
    name = "naive"

    def __init__(self, n, rng):
        self.n = n

    def reset(self, idx):
        pass

    def act(self, env):
        tgt = env.H[:, -1]
        vel = _clipn((tgt - env.c) / DT / V_CROSS)
        near = np.linalg.norm(tgt - (env.c + vel * V_CROSS * DT), axis=1) < 0.5
        return np.concatenate([vel, np.where(near, 1.0, -1.0)[:, None]], 1)


class KF:
    """Vectorized Kalman filter on noisy 2-D positions; order 1 = constant velocity,
    order 2 = constant acceleration (white-noise jerk/accel process noise q)."""

    def __init__(self, n, order, q):
        self.n, self.o, self.q = n, order, q
        m = order + 1
        A = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                A[i, j] = DT ** (j - i) / np.prod(range(1, j - i + 1))
        self.A1 = A
        Gm = np.array([DT ** (m - i) / np.prod(range(1, m - i + 1)) for i in range(m)])[:, None]
        self.Q1 = q * (Gm @ Gm.T)
        self.x = np.zeros((n, 2, m)); self.P = np.zeros((n, 2, m, m)); self.cnt = np.zeros(n, int)

    def reset(self, idx, z):
        m = self.o + 1
        self.x[idx] = 0.0; self.x[idx, :, 0] = z
        P0 = np.diag([OBS_NOISE ** 2] + [100.0] * (m - 1))
        self.P[idx] = P0; self.cnt[idx] = 0

    def update(self, z):
        A, Q = self.A1, self.Q1
        self.x = np.einsum("ij,ncj->nci", A, self.x)
        self.P = np.einsum("ij,ncjk,lk->ncil", A, self.P, A) + Q
        S = self.P[:, :, 0, 0] + OBS_NOISE ** 2
        K = self.P[:, :, :, 0] / S[..., None]
        y = z - self.x[:, :, 0]
        self.x = self.x + K * y[..., None]
        self.P = self.P - K[..., :, None] * self.P[:, :, 0, None, :]
        self.cnt += 1

    def predict(self, h):
        m = self.o + 1
        coef = np.array([h ** i / np.prod(range(1, i + 1)) for i in range(m)])
        return (self.x * coef).sum(-1)

    def pos_std(self):
        return np.sqrt(self.P[:, :, 0, 0].sum(1))


class Lead:
    """Lead pursuit with a constant-velocity Kalman predictor: aim at the predicted position at
    impact time; fire when the crosshair is on that point and the filter has settled."""
    name = "lead"
    ORDER, Q, TOL, MIN_OBS, STD_MAX = 1, 400.0, 0.25, 4, 0.35
    H_MULT = 1.0                       # aim at the prediction H_MULT * (shot delay) ahead

    def __init__(self, n, rng):
        self.n = n
        self.kf = KF(n, self.ORDER, self.Q)
        self.started = np.zeros(n, bool)

    def reset(self, idx):
        self.started[idx] = False

    def act(self, env):
        z = env.H[:, -1]
        new = ~self.started
        if new.any():
            self.kf.reset(np.nonzero(new)[0], z[new]); self.started[:] = True
        self.kf.update(z)
        aim = self.kf.predict(DELAY * DT * self.H_MULT)
        vel = _clipn((aim - env.c) / DT / V_CROSS)
        c_next = env.c + vel * V_CROSS * DT
        ok = (np.linalg.norm(aim - c_next, axis=1) < self.TOL) & (self.kf.cnt >= self.MIN_OBS) & \
             (self.kf.pos_std() < self.STD_MAX)
        return np.concatenate([vel, np.where(ok, 1.0, -1.0)[:, None]], 1)


class LeadCA(Lead):
    """Held-out hunter: constant-acceleration Kalman predictor, tighter fire rule."""
    name = "lead_ca"
    ORDER, Q, TOL, MIN_OBS, STD_MAX = 2, 4000.0, 0.2, 6, 0.4


SCRIPTED = {c.name: c for c in [Straight, Jink, Dodge, DodgeJink, Naive, Lead, LeadCA]}
LEAD_PARAMS = ["ORDER", "Q", "TOL", "MIN_OBS", "STD_MAX", "H_MULT"]


def lead_class(prm, name="lead_tuned"):
    return type("LeadTuned", (Lead,), dict(name=name, **{k: prm[k] for k in LEAD_PARAMS}))


def _register_tuned():
    import json, os
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "r6h_lead_tuned.json")
    if os.path.exists(f):
        SCRIPTED["lead_tuned"] = lead_class(json.load(open(f)))


_register_tuned()
EVADERS = ["straight", "jink", "dodge", "dodge_jink"]
HUNTERS = ["naive", "lead", "lead_ca"]


# ---------------- numpy MLP for frozen learned opponents ----------------
class NetAgent:
    """Frozen learned agent (numpy forward of the MLX policy mean). role 'duck' or 'hunter'."""

    def __init__(self, params, role, name, residual_scale=None):
        self.p, self.role, self.name, self.res = params, role, name, residual_scale

    def reset(self, idx):
        pass

    def mean(self, x):
        p = self.p
        h = np.tanh(x @ p["pi.0.weight"].T + p["pi.0.bias"])
        h = np.tanh(h @ p["pi.1.weight"].T + p["pi.1.bias"])
        return np.tanh(h @ p["pi.2.weight"].T + p["pi.2.bias"])

    def act(self, env, rows=None):
        if self.role == "duck":
            m = self.mean(env.duck_obs())
            return duck_action(env, m, self.res)
        return self.mean(env.hunter_obs())


def duck_action(env, m, res_scale):
    """Learned duck: residual on PD+ when res_scale is set, else raw thrust."""
    if res_scale is None:
        return m
    return P.norm_clip((P.pd_plus_action(env, _pdp()) + res_scale * m) * F.AMAX)


def make_agent(spec, n, rng):
    """spec: scripted name, or dict(kind='net', path=..., role=..., res_scale=...)."""
    if isinstance(spec, str):
        return SCRIPTED[spec](n, rng)
    d = np.load(spec["path"])
    params = {k: d[k].astype(np.float64) for k in d.files}
    return NetAgent(params, spec["role"], spec.get("name", spec["path"]), spec.get("res_scale"))


def play(courses, duck, hunter, seed=7, wind_sigma=2.0, record=False):
    """Run each fixed course once with the given agent objects (built for len(courses))."""
    n = len(courses)
    env = DuelEnv(n, seed, wind_sigma, fixed_courses=courses)
    duck.reset(np.arange(n)); hunter.reset(np.arange(n))
    alive = np.ones(n, bool)
    res = dict(hit=np.zeros(n, bool), complete=np.zeros(n, bool), crash=np.zeros(n, bool),
               timeout=np.zeros(n, bool), t=np.full(n, np.nan), shots=np.zeros(n, int), hit_t=np.full(n, np.nan))
    rec = [] if record else None
    for _ in range(MAXS):
        ad = duck.act(env); ah = hunter.act(env)
        ad = np.where(alive[:, None], ad, 0.0); ah = np.where(alive[:, None], ah, -1.0)
        if record:
            rec.append(dict(p=env.p.copy(), c=env.c.copy(), k=env.k.copy(), pend=env.pend.copy(),
                            aim=env.aim.copy(), z=env.H[:, -1].copy(), alive=alive.copy()))
        done, info = env.step(ad, ah)
        res["shots"] += info["fired"] & alive
        newly = done & alive
        for key in ["hit", "complete", "crash", "timeout"]:
            res[key][newly] = info[key][newly]
        res["t"][newly & info["complete"]] = info["t"][newly & info["complete"]] * DT
        res["hit_t"][newly & info["hit"]] = info["t"][newly & info["hit"]] * DT
        alive &= ~done
        env.v[~alive] = 0.0; env.thr[~alive] = np.array([0.0, F.G]); env.pend[~alive] = 0
        if not alive.any():
            break
    if record:
        res["rec"] = rec
    return res


def summarize(r):
    n = len(r["hit"])
    return dict(hit_rate=float(r["hit"].mean()), survival=float((r["complete"] & ~r["hit"]).mean()),
                completion=float(r["complete"].mean()), crash=float(r["crash"].mean()),
                timeout=float(r["timeout"].mean()), shots=float(r["shots"].mean()),
                mean_time_survivors=float(np.nanmean(r["t"])) if r["complete"].any() else float("nan"), n=n)


def train_courses(n=256):
    return F.make_courses(np.random.default_rng(5001), n)


def heldout_courses(n=500):
    return F.make_courses(np.random.default_rng(5_777_000), n)
