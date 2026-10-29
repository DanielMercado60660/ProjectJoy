"""Vectorized 1-vs-1 pursuit-evasion arena (numpy).

Square arena [-L, L]^2 with three circular obstacles. Both agents are point masses with
acceleration control, a cruise speed cap, and a sprint (higher cap) that drains stamina.
The pursuer is faster in a straight line; the evader turns harder (higher acceleration).
Capture when the distance falls below CAPTURE_R. Episode ends at capture or after T_MAX steps.

Action space (both roles, 17 discrete): 0 = no thrust (drift), 1..8 = thrust in one of eight
compass directions at cruise cap, 9..16 = the same eight directions while sprinting.
"""
import numpy as np

L = 2.0                    # arena half-size
DT = 0.05
T_MAX = 300                # 15 s episodes
CAPTURE_R = 0.20
BODY_R = 0.08
OBSTACLES = np.array([[-0.8, 0.7, 0.45],
                      [0.9, -0.4, 0.50],
                      [-0.2, -1.2, 0.30]], dtype=np.float64)   # x, y, radius

# role parameters: accel, cruise cap, sprint cap
P_ACC, P_CRUISE, P_SPRINT = 2.5, 1.00, 1.50
E_ACC, E_CRUISE, E_SPRINT = 6.0, 0.90, 1.35
DRAG = 0.5                 # 1/s linear drag
STAM_DRAIN = 1.0 / 2.5     # full stamina = 2.5 s of sprint
STAM_REGEN = 1.0 / 8.0     # 8 s to refill from empty

# reward constants (pursuer view; the evader gets the negative of the zero-sum part)
R_CAPTURE = 1.0
R_SHAPE = 0.2              # per unit of distance closed
C_THRUST = 0.0005          # own effort costs (not zero-sum)
C_SPRINT = 0.0015

N_ACT = 17
_ang = np.arange(8) * (np.pi / 4)
DIRS = np.zeros((N_ACT, 2))
DIRS[1:9] = np.stack([np.cos(_ang), np.sin(_ang)], 1)
DIRS[9:17] = DIRS[1:9]
SPRINT = np.zeros(N_ACT, bool)
SPRINT[9:] = True
OBS_DIM = 18


def dir_to_action(d, sprint):
    """Map desired direction vectors (n,2) and sprint flags (n,) to the nearest discrete action."""
    ang = np.arctan2(d[:, 1], d[:, 0])
    k = np.round(ang / (np.pi / 4)).astype(int) % 8
    a = 1 + k + 8 * sprint.astype(int)
    a[np.linalg.norm(d, axis=1) < 1e-9] = 0
    return a


class Arena:
    """N parallel matches. State arrays have a leading env axis."""

    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((n, 2, 2))   # env, agent(0=pursuer,1=evader), xy
        self.vel = np.zeros((n, 2, 2))
        self.stam = np.ones((n, 2))
        self.t = np.zeros(n, int)
        self.reset(np.arange(n))

    def _sample_free(self, k):
        out = np.zeros((k, 2))
        todo = np.arange(k)
        while len(todo):
            p = self.rng.uniform(-L + 0.2, L - 0.2, (len(todo), 2))
            d = np.linalg.norm(p[:, None, :] - OBSTACLES[None, :, :2], axis=2)
            ok = (d > OBSTACLES[None, :, 2] + 0.15).all(1)
            out[todo[ok]] = p[ok]
            todo = todo[~ok]
        return out

    def reset(self, idx):
        k = len(idx)
        if k == 0:
            return
        pp = self._sample_free(k)
        ee = self._sample_free(k)
        bad = np.linalg.norm(pp - ee, axis=1) < 1.5
        while bad.any():                       # start at least 1.5 apart
            ee[bad] = self._sample_free(int(bad.sum()))
            bad = np.linalg.norm(pp - ee, axis=1) < 1.5
        self.pos[idx, 0] = pp
        self.pos[idx, 1] = ee
        self.vel[idx] = 0.0
        self.stam[idx] = 1.0
        self.t[idx] = 0

    def dist(self):
        return np.linalg.norm(self.pos[:, 0] - self.pos[:, 1], axis=1)

    def obs(self):
        """Observation for each role, same layout: (n,18) pursuer view, (n,18) evader view."""
        out = []
        for me, op in ((0, 1), (1, 0)):
            vs = P_SPRINT if me == 0 else E_SPRINT
            rel = self.pos[:, op] - self.pos[:, me]
            ob = self.pos[:, me][:, None, :2] - OBSTACLES[None, :, :2]
            o = np.concatenate([
                self.pos[:, me] / L,
                self.vel[:, me] / vs,
                self.stam[:, me:me + 1] * 2 - 1,
                rel / (2 * L),
                self.vel[:, op] / 1.5,
                self.stam[:, op:op + 1] * 2 - 1,
                np.linalg.norm(rel, axis=1, keepdims=True) / (2 * L),
                (self.t / T_MAX)[:, None] * 2 - 1,
                -ob.reshape(self.n, 6) / (2 * L),
            ], 1)
            out.append(o.astype(np.float32))
        return out[0], out[1]

    def step(self, a_p, a_e):
        """Advance one tick. Returns (r_p, r_e, done, captured, info) and auto-resets finished envs.
        info holds per-agent effort this tick: thrust (n,2) bool and sprint (n,2) bool."""
        d0 = self.dist()
        acts = np.stack([a_p, a_e], 1)
        thrust = acts > 0
        want_sprint = SPRINT[acts]
        sprint = want_sprint & (self.stam > 0.0)
        acc = np.array([P_ACC, E_ACC])[None, :, None]
        self.vel += DIRS[acts] * acc * DT
        self.vel *= (1.0 - DRAG * DT)
        cap = np.where(sprint, np.array([P_SPRINT, E_SPRINT])[None], np.array([P_CRUISE, E_CRUISE])[None])
        sp = np.linalg.norm(self.vel, axis=2)
        scale = np.minimum(1.0, cap / np.maximum(sp, 1e-9))
        self.vel *= scale[..., None]
        self.stam = np.clip(self.stam + np.where(sprint & thrust, -STAM_DRAIN, STAM_REGEN) * DT, 0.0, 1.0)
        self.pos += self.vel * DT
        # walls
        lim = L - BODY_R
        for ax in range(2):
            hi = self.pos[..., ax] > lim
            lo = self.pos[..., ax] < -lim
            self.pos[..., ax] = np.clip(self.pos[..., ax], -lim, lim)
            self.vel[..., ax] = np.where(hi, np.minimum(self.vel[..., ax], 0), self.vel[..., ax])
            self.vel[..., ax] = np.where(lo, np.maximum(self.vel[..., ax], 0), self.vel[..., ax])
        # obstacles: push out to surface, remove inward normal velocity
        for ox, oy, r in OBSTACLES:
            rel = self.pos - np.array([ox, oy])
            dd = np.linalg.norm(rel, axis=2)
            inside = dd < r + BODY_R
            if inside.any():
                nrm = rel / np.maximum(dd, 1e-9)[..., None]
                self.pos = np.where(inside[..., None], np.array([ox, oy]) + nrm * (r + BODY_R), self.pos)
                vn = (self.vel * nrm).sum(2)
                self.vel = np.where((inside & (vn < 0))[..., None], self.vel - vn[..., None] * nrm, self.vel)
        self.t += 1
        d1 = self.dist()
        captured = d1 < CAPTURE_R
        timeout = self.t >= T_MAX
        done = captured | timeout
        zs = R_SHAPE * (d0 - d1) + R_CAPTURE * captured
        eff = C_THRUST * thrust + C_SPRINT * (sprint & thrust)
        r_p = zs - eff[:, 0]
        r_e = -zs - eff[:, 1]
        info = {"thrust": thrust, "sprint": sprint & thrust, "t": self.t.copy(),
                "pos": self.pos.copy()}
        self.reset(np.nonzero(done)[0])
        return r_p.astype(np.float32), r_e.astype(np.float32), done, captured, info


# ---------------------------------------------------------------- scripted opponents
class Scripted:
    """Stateless-ish scripted controllers that read the Arena state directly."""

    def __init__(self, kind, role, seed=0):
        self.kind, self.role = kind, role
        self.rng = np.random.default_rng(seed)
        self.wander_dir = None

    def act(self, env):
        n = env.n
        me, op = (0, 1) if self.role == "p" else (1, 0)
        pm, po = env.pos[:, me], env.pos[:, op]
        rel = po - pm
        dist = np.linalg.norm(rel, axis=1)
        if self.kind == "random":
            if self.wander_dir is None or len(self.wander_dir) != n:
                self.wander_dir = self.rng.integers(0, N_ACT, n)
            flip = self.rng.random(n) < 0.05          # keep a heading ~20 ticks
            self.wander_dir[flip] = self.rng.integers(0, N_ACT, int(flip.sum()))
            return self.wander_dir.copy()
        if self.kind == "greedy":                      # pursuer: run straight at the evader
            return dir_to_action(rel, dist < 1.2)
        if self.kind == "lead":                        # pursuer: aim at predicted evader position
            tl = np.clip(dist / P_SPRINT, 0, 1.0)[:, None]
            tgt = po + env.vel[:, op] * tl
            return dir_to_action(tgt - pm, dist < 1.2)
        if self.kind == "flee":                        # evader: run directly away
            return dir_to_action(-rel, dist < 0.9)
        if self.kind == "smartflee":                   # evader: away + wall/obstacle repulsion
            away = -rel / np.maximum(dist, 1e-9)[:, None] / np.maximum(dist, 0.3)[:, None]
            rep = np.zeros_like(pm)
            for ax in range(2):
                rep[:, ax] += 0.3 / np.maximum(L - pm[:, ax], 0.05) ** 2 * -1
                rep[:, ax] += 0.3 / np.maximum(L + pm[:, ax], 0.05) ** 2
            for ox, oy, r in OBSTACLES:
                ro = pm - np.array([ox, oy])
                do = np.maximum(np.linalg.norm(ro, axis=1) - r, 0.05)
                rep += 0.15 * ro / np.linalg.norm(ro, axis=1, keepdims=True) / do[:, None] ** 2
            d = away + rep
            # if repulsion fights the escape direction (cornered), slide tangentially
            corner = (d * away).sum(1) < 0
            tang = np.stack([-away[:, 1], away[:, 0]], 1)
            tang *= np.sign((tang * -pm).sum(1) + 1e-9)[:, None]   # tangent that heads to centre
            d[corner] = tang[corner]
            return dir_to_action(d, dist < 0.9)
        raise ValueError(self.kind)


PURSUER_SUITE = ["random", "greedy", "lead"]
EVADER_SUITE = ["random", "flee", "smartflee"]
