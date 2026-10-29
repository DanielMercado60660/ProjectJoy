"""Procedural closed race tracks.

A track is a closed centerline (periodic cubic spline through randomly perturbed
points around a circle) resampled at uniform arclength, plus a constant
half-width. Every track comes from one integer seed, so splits are reproducible:

    train     seeds 0 .. 9_999          (we use the first N)
    heldout   seeds 1_000_000 + i        same distribution, never trained on
    hard      seeds 2_000_000 + i        narrower + twistier (out of distribution)

Generation is rejection sampled: a candidate is rejected if a corner is tighter
than the track allows (boundary would fold over) or if two far-apart parts of
the track come closer than one track width plus a margin (walls would overlap).
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

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
