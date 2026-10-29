"""Minimal vendored copy of 08_tiny_grand_prix racing sim (racing/track.py + racing/sim.py, verbatim
except: numba parallel=True/prange dropped (single-threaded, polite on a shared Mac) and a stripped-down
strict one-lap time trial at the end, mirroring racing/evaluate.py run_lap(mode="strict")).
Used only for the event-08 task check and to record realistic observations."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
import math
import numba as nb

DS = 2.0  # centerline sample spacing in meters

SPLITS = {"train": 0, "heldout": 1_000_000, "hard": 2_000_000}


@dataclass
class Track:
    seed: int
    center: np.ndarray   # (N, 2) centerline points, closed (point N wraps to 0)
    tangent: np.ndarray  # (N, 2) unit tangent
    normal: np.ndarray   # (N, 2) unit left normal
    curvature: np.ndarray  # (N,) signed curvature 1/m (positive = left turn)
    half_width: float
    length: float

    @property
    def left(self):
        return self.center + self.normal * self.half_width

    @property
    def right(self):
        return self.center - self.normal * self.half_width


def _params(split):
    if split == "hard":
        return dict(n_ctrl=(12, 18), radius=(55, 85), jitter=(0.55, 0.30), half_width=(4.0, 5.0))
    return dict(n_ctrl=(8, 16), radius=(60, 95), jitter=(0.55, 0.35), half_width=(5.0, 7.0))


def _candidate(rng, p):
    n = int(rng.integers(p["n_ctrl"][0], p["n_ctrl"][1] + 1))
    R = rng.uniform(*p["radius"])
    ang = np.sort((np.arange(n) + rng.uniform(-0.3, 0.3, n)) * 2 * np.pi / n)
    r = R * (1.0 + rng.uniform(-p["jitter"][0], p["jitter"][1], n))
    # random anisotropic stretch so tracks are not all roundish
    sx, sy = np.exp(rng.uniform(-0.35, 0.35, 2))
    pts = np.stack([sx * r * np.cos(ang), sy * r * np.sin(ang)], 1)
    rot = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(rot), np.sin(rot)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    if rng.random() < 0.5:  # half the tracks run clockwise
        pts = pts[::-1]
    hw = rng.uniform(*p["half_width"])
    return pts, hw


def _resample(pts):
    closed = np.vstack([pts, pts[:1]])
    chord = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    spl = CubicSpline(chord, closed, bc_type="periodic")
    # dense sample to get arclength
    t = np.linspace(0, chord[-1], 4000, endpoint=False)
    dense = spl(t)
    seg = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    L = arc[-1]
    N = int(round(L / DS))
    s_target = np.arange(N) * (L / N)
    t_of_s = np.interp(s_target, arc, np.r_[t, chord[-1]])
    center = spl(t_of_s)
    d1 = spl(t_of_s, 1)
    d2 = spl(t_of_s, 2)
    speed = np.linalg.norm(d1, axis=1)
    tangent = d1 / speed[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    curv = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed**3
    return center, tangent, normal, curv, N * (L / N)


def _valid(center, curv, hw):
    # tightest corner must leave room for the inside wall
    if np.max(np.abs(curv)) * hw > 0.6:
        return False
    # far-apart parts of the track must not come within a track width + margin
    N = len(center)
    d = np.linalg.norm(center[:, None, :] - center[None, :, :], axis=-1)
    idx = np.arange(N)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, N - gap) * DS
    far = gap > 4 * hw + 10.0
    return d[far].min() > 2 * hw + 8.0 if far.any() else True


def make_track(seed, split="train"):
    p = _params(split)
    rng = np.random.default_rng(SPLITS[split] + seed)
    for _ in range(200):
        pts, hw = _candidate(rng, p)
        center, tangent, normal, curv, L = _resample(pts)
        if _valid(center, curv, hw):
            return Track(SPLITS[split] + seed, center, tangent, normal, curv, hw, L)
    raise RuntimeError(f"could not generate a valid track for seed {seed}")


def make_tracks(split, n, start=0):
    return [make_track(i, split) for i in range(start, start + n)]


@dataclass
class TrackBank:
    """All tracks packed into padded arrays for the numba simulator."""
    center: np.ndarray    # (T, Nmax, 2)
    tangent: np.ndarray   # (T, Nmax, 2)
    seg_a: np.ndarray     # (T, Smax, 2) wall segment start points
    seg_b: np.ndarray     # (T, Smax, 2) wall segment end points
    n_pts: np.ndarray     # (T,) int
    n_seg: np.ndarray     # (T,) int
    seglen: np.ndarray    # (T,) float centerline spacing
    length: np.ndarray    # (T,)
    half_width: np.ndarray  # (T,)
    tracks: list


def pack(tracks):
    T = len(tracks)
    Nmax = max(len(t.center) for t in tracks)
    center = np.zeros((T, Nmax, 2))
    tangent = np.zeros((T, Nmax, 2))
    seg_a = np.zeros((T, 2 * Nmax, 2))
    seg_b = np.zeros((T, 2 * Nmax, 2))
    n_pts = np.zeros(T, np.int64)
    n_seg = np.zeros(T, np.int64)
    for i, t in enumerate(tracks):
        N = len(t.center)
        center[i, :N] = t.center
        tangent[i, :N] = t.tangent
        for k, wall in enumerate((t.left, t.right)):
            seg_a[i, k * N:(k + 1) * N] = wall
            seg_b[i, k * N:(k + 1) * N] = np.roll(wall, -1, axis=0)
        n_pts[i] = N
        n_seg[i] = 2 * N
    return TrackBank(center, tangent, seg_a, seg_b, n_pts, n_seg,
                     np.array([t.length / len(t.center) for t in tracks]),
                     np.array([t.length for t in tracks]),
                     np.array([t.half_width for t in tracks]), tracks)
DT = 0.1
WHEELBASE = 2.6
MAX_STEER = 0.45
A_MAX = 6.0
B_MAX = 12.0
DRAG = A_MAX / 40.0**2
G = 9.81
CAR_R = 1.0
RAY_LEN = 60.0
WALL_KEEP = 0.97  # soft-wall mode only: speed kept when sliding along a wall
V_SCALE = 30.0
RAY_ANGLES = np.deg2rad(np.array([-90, -60, -35, -18, -6, 6, 18, 35, 60, 90], dtype=np.float64))
N_RAYS = len(RAY_ANGLES)
OBS_DIM = N_RAYS + 3
ACT_DIM = 2


@nb.njit(cache=True, inline="always")
def _wrap_idx(i, n):
    i = i % n
    return i + n if i < 0 else i


@nb.njit(cache=True)
def _locate(px, py, idx0, center, n, seglen, window_back, window_fwd):
    """Nearest point on the centerline polyline near index idx0.
    Returns (segment index, fraction t, signed lateral offset)."""
    best_d = 1e18
    best_i = idx0
    best_t = 0.0
    best_lat = 0.0
    for k in range(-window_back, window_fwd + 1):
        i = _wrap_idx(idx0 + k, n)
        j = _wrap_idx(i + 1, n)
        ax, ay = center[i, 0], center[i, 1]
        dx, dy = center[j, 0] - ax, center[j, 1] - ay
        l2 = dx * dx + dy * dy
        t = ((px - ax) * dx + (py - ay) * dy) / l2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx, qy = ax + t * dx - px, ay + t * dy - py
        d = qx * qx + qy * qy
        if d < best_d:
            best_d = d
            best_i = i
            best_t = t
            l = math.sqrt(l2)
            best_lat = (dx * (py - ay) - dy * (px - ax)) / l
    return best_i, best_t, best_lat


@nb.njit(cache=True)
def step_kernel(act, grip, x, y, yaw, v, steer_prev, pedal_prev, idx, s_frac, prog, lat,
                done, crashed, track_id, center, n_pts, seglen, length, half_width, soft_walls, wall_hits):
    E = x.shape[0]
    dprog = np.zeros(E)
    for e in range(E):
        if done[e]:
            continue
        st = min(1.0, max(-1.0, act[e, 0]))
        pd = min(1.0, max(-1.0, act[e, 1]))
        steer_prev[e] = st
        pedal_prev[e] = pd
        mu = grip[e]
        a = pd * A_MAX if pd > 0 else pd * B_MAX
        lim = mu * G
        if a > lim:
            a = lim
        elif a < -lim:
            a = -lim
        a -= DRAG * v[e] * v[e]
        vn = v[e] + a * DT
        if vn < 0.0:
            vn = 0.0
        vm = 0.5 * (v[e] + vn)
        omega = vm * math.tan(st * MAX_STEER) / WHEELBASE
        wlim = lim / max(vm, 1.0)
        if omega > wlim:
            omega = wlim
        elif omega < -wlim:
            omega = -wlim
        yaw[e] += omega * DT
        x[e] += vm * math.cos(yaw[e]) * DT
        y[e] += vm * math.sin(yaw[e]) * DT
        v[e] = vn
        tid = track_id[e]
        n = n_pts[tid]
        i, t, la = _locate(x[e], y[e], idx[e], center[tid], n, seglen[tid], 4, 6)
        s_old = (idx[e] + s_frac[e]) * seglen[tid]
        s_new = (i + t) * seglen[tid]
        ds = s_new - s_old
        L = length[tid]
        if ds > 0.5 * L:
            ds -= L
        elif ds < -0.5 * L:
            ds += L
        idx[e] = i
        s_frac[e] = t
        lat[e] = la
        prog[e] += ds
        dprog[e] = ds
        room = half_width[tid] - CAR_R
        if abs(la) > room:
            if not soft_walls:
                crashed[e] = True
                done[e] = True
            else:
                # SOFT WALLS (only for the reward-hacking demo): push the car back
                # inside, keep the velocity component along the wall, count a hit.
                j = _wrap_idx(i + 1, n)
                tx, ty = center[tid][j, 0] - center[tid][i, 0], center[tid][j, 1] - center[tid][i, 1]
                tl = math.sqrt(tx * tx + ty * ty)
                tx, ty = tx / tl, ty / tl
                excess = la - room if la > 0 else la + room
                x[e] -= -ty * excess
                y[e] -= tx * excess
                lat[e] = room if la > 0 else -room
                hc, hs = math.cos(yaw[e]), math.sin(yaw[e])
                along = hc * tx + hs * ty
                v[e] = v[e] * abs(along) * WALL_KEEP
                yaw[e] = math.atan2(ty, tx) if along >= 0 else math.atan2(-ty, -tx)
                wall_hits[e] += 1
    return dprog


@nb.njit(cache=True)
def _cast(px, py, ang, seg_a, seg_b, ns, maxlen):
    dx, dy = math.cos(ang), math.sin(ang)
    best = maxlen
    for k in range(ns):
        ax, ay = seg_a[k, 0], seg_a[k, 1]
        # cheap reject: segment start too far away (segments are ~2 m long)
        ox, oy = ax - px, ay - py
        if ox * ox + oy * oy > (maxlen + 4.0) ** 2:
            continue
        ex, ey = seg_b[k, 0] - ax, seg_b[k, 1] - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = (ox * ey - oy * ex) / den   # distance along ray
        u = (ox * dy - oy * dx) / den   # fraction along segment
        if t >= 0.0 and 0.0 <= u <= 1.0 and t < best:
            best = t
    return best


@nb.njit(cache=True)
def obs_kernel(x, y, yaw, v, steer_prev, pedal_prev, track_id, seg_a, seg_b, n_seg, ray_angles, out):
    E = x.shape[0]
    R = ray_angles.shape[0]
    for e in range(E):
        tid = track_id[e]
        for r in range(R):
            out[e, r] = _cast(x[e], y[e], yaw[e] + ray_angles[r], seg_a[tid], seg_b[tid],
                              n_seg[tid], RAY_LEN) / RAY_LEN
        out[e, R] = v[e] / V_SCALE
        out[e, R + 1] = steer_prev[e]
        out[e, R + 2] = pedal_prev[e]


class VecRace:
    """A batch of independent cars, each on its own track from a TrackBank.

    reset(track_ids, start_frac=None, grip=1.0) puts car e on track track_ids[e]
    at fraction start_frac[e] of the lap (default: the start line), at rest,
    facing along the track. step(actions) returns (obs, progress_delta_m, done).
    """

    def __init__(self, bank, n_envs, soft_walls=False):
        self.bank = bank
        self.soft_walls = soft_walls
        self.wall_hits = np.zeros(n_envs, np.int64)
        self.E = n_envs
        self.x = np.zeros(n_envs)
        self.y = np.zeros(n_envs)
        self.yaw = np.zeros(n_envs)
        self.v = np.zeros(n_envs)
        self.steer_prev = np.zeros(n_envs)
        self.pedal_prev = np.zeros(n_envs)
        self.idx = np.zeros(n_envs, np.int64)
        self.s_frac = np.zeros(n_envs)
        self.prog = np.zeros(n_envs)
        self.lat = np.zeros(n_envs)
        self.done = np.zeros(n_envs, np.bool_)
        self.crashed = np.zeros(n_envs, np.bool_)
        self.track_id = np.zeros(n_envs, np.int64)
        self.grip = np.ones(n_envs)
        self.t = np.zeros(n_envs, np.int64)
        self._obs = np.zeros((n_envs, OBS_DIM))

    def reset(self, track_ids, start_frac=None, grip=1.0, v0=0.0, mask=None):
        m = np.ones(self.E, bool) if mask is None else mask
        b = self.bank
        tid = np.broadcast_to(np.asarray(track_ids, np.int64), (self.E,))
        self.track_id[m] = tid[m]
        n = b.n_pts[self.track_id]
        if start_frac is None:
            i0 = np.zeros(self.E, np.int64)
        else:
            i0 = (np.broadcast_to(start_frac, (self.E,)) * n).astype(np.int64) % n
        tt = self.track_id
        c = b.center[tt, i0]
        tg = b.tangent[tt, i0]
        self.x[m] = c[m, 0]
        self.y[m] = c[m, 1]
        self.yaw[m] = np.arctan2(tg[m, 1], tg[m, 0])
        self.v[m] = np.broadcast_to(v0, (self.E,))[m]
        self.steer_prev[m] = 0
        self.pedal_prev[m] = 0
        self.idx[m] = i0[m]
        self.s_frac[m] = 0
        self.prog[m] = 0
        self.lat[m] = 0
        self.done[m] = False
        self.crashed[m] = False
        self.wall_hits[m] = 0
        self.grip[m] = np.broadcast_to(grip, (self.E,))[m]
        self.t[m] = 0
        return self.observe()

    def observe(self):
        b = self.bank
        obs_kernel(self.x, self.y, self.yaw, self.v, self.steer_prev, self.pedal_prev,
                   self.track_id, b.seg_a, b.seg_b, b.n_seg, RAY_ANGLES, self._obs)
        return self._obs.copy()

    def step(self, actions):
        b = self.bank
        alive = ~self.done
        dprog = step_kernel(np.ascontiguousarray(actions, np.float64), self.grip, self.x, self.y,
                            self.yaw, self.v, self.steer_prev, self.pedal_prev, self.idx,
                            self.s_frac, self.prog, self.lat, self.done, self.crashed,
                            self.track_id, b.center, b.n_pts, b.seglen, b.length, b.half_width,
                            self.soft_walls, self.wall_hits)
        self.t[alive] += 1
        return self.observe(), dprog, self.done.copy()

    def snapshot(self):
        return np.stack([self.x, self.y, self.yaw, self.v, self.steer_prev, self.pedal_prev], 1).copy()


# ---------- stripped-down strict one-lap time trial (racing/evaluate.py, mode="strict") ----------
MIN_AVG_SPEED = 4.0  # m/s; time limit = length / this


def run_lap_strict(controller, bank, grip=1.0, record_obs=False):
    """controller(obs (T, 13) float64) -> actions (T, 2). Each car starts at rest on the start
    line of its own track; touching a wall is a DNF. Returns finished, lap_time (+ obs log)."""
    T = len(bank.tracks)
    env = VecRace(bank, T)
    obs = env.reset(np.arange(T), grip=grip)
    length = bank.length
    limit = np.ceil(length / MIN_AVG_SPEED / DT).astype(int)
    finished = np.zeros(T, bool)
    lap_time = np.full(T, np.nan)
    active = np.ones(T, bool)
    log = []
    for t in range(int(limit.max())):
        if record_obs:
            log.append(obs[active].copy())
        act = np.array(controller(obs), np.float64)
        act[~active] = 0
        env.done[~active] = True
        obs, dprog, done = env.step(act)
        newly = active & (env.prog >= length) & ~env.crashed
        finished |= newly
        lap_time[newly] = (t + 1) * DT
        active &= ~newly
        active &= ~env.crashed
        active &= (t + 1) < limit
        if not active.any():
            break
    res = dict(finished=finished, lap_time=lap_time)
    if record_obs:
        res["obs"] = np.concatenate(log, 0)
    return res
