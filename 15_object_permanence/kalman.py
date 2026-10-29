"""Classical baseline: color-threshold detector + constant-velocity Kalman filter per ball.

Sees exactly the same rendered frames as the network and outputs the same thing: for every
frame and slot, the filtered position mean and 2x2 covariance (pixels).

* Detector: pixels within `tol` (RGB distance) of a ball color.  For distinct colors, each color
  has one track and the measurement is the centroid of that color's pixels.  For identical colors
  (hard variant), connected components are detections and are assigned to tracks with the
  Hungarian algorithm on Mahalanobis distance. The chi2_2(0.999) gate applies only when there are
  fewer detections than live tracks (otherwise every track takes its assigned detection).
* Measurement noise scales with how much of the ball is visible: R = r^2 * full_area / area.
* No detection -> pure prediction (the filter coasts behind the wall).
* bounce=True: the prediction step reflects the mean and covariance off the frame edges
  (the walls of the box, not the occluders - occluders don't affect motion).
"""
import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

import sim

FULL_AREA = np.pi * sim.R ** 2
LO, HI = sim.R, sim.W - sim.R


EDGE_OFF = 0.1   # calibrated on val visible balls: true center - (right edge + 1 - R) = 0.10 px


def _is_wall(frame, ys, x):
    if x < 0 or x >= frame.shape[1]:
        return False
    return bool((np.abs(frame[ys, x] - 0.45) < 0.15).all(-1).mean() > 0.5)


def _measure(frame, ys, xs, wall_aware):
    """Centroid of a blob; if the blob is cut by an occluder (narrow and touching wall-gray
    pixels on one side), estimate x from the uncut edge instead (center = edge -/+ R)."""
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
    if wall_aware:
        x0, x1 = xs.min(), xs.max()
        if x1 - x0 + 1 < 2 * sim.R - 1:
            rows = np.unique(ys)
            left, right = _is_wall(frame, rows, x0 - 1), _is_wall(frame, rows, x1 + 1)
            if left and not right:
                cx = x1 + 1 - sim.R + EDGE_OFF
            elif right and not left:
                cx = x0 + sim.R - EDGE_OFF
    return np.array([cx, cy])


def detect(frame, color, tol, components, wall_aware=True):
    m = np.sqrt(((frame - color) ** 2).sum(-1)) < tol
    if not components:
        n = m.sum()
        if n < 3:
            return []
        ys, xs = np.nonzero(m)
        return [(_measure(frame, ys, xs, wall_aware), n)]
    lab, nl = label(m)
    out = []
    for i in range(1, nl + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= 3:
            out.append((_measure(frame, ys, xs, wall_aware), len(xs)))
    return out


class KF:
    def __init__(self, q, r, v0):
        self.q, self.r, self.v0 = q, r, v0
        self.x = None
        self.P = None

    def init(self, z, area):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        rr = self.r ** 2 * FULL_AREA / max(area, 1)
        self.P = np.diag([rr, rr, self.v0 ** 2, self.v0 ** 2])

    def predict(self, bounce):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = 1
        G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q ** 2 * G @ G.T
        if bounce:
            for d in range(2):
                if self.x[d] < LO or self.x[d] > HI:
                    edge = LO if self.x[d] < LO else HI
                    self.x[d] = 2 * edge - self.x[d]
                    self.x[d + 2] *= -1
                    J = np.eye(4)
                    J[d, d] = J[d + 2, d + 2] = -1
                    self.P = J @ self.P @ J.T

    def update(self, z, area):
        Hm = np.zeros((2, 4))
        Hm[0, 0] = Hm[1, 1] = 1
        Rm = np.eye(2) * self.r ** 2 * FULL_AREA / max(area, 1)
        S = Hm @ self.P @ Hm.T + Rm
        Kg = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ (z - Hm @ self.x)
        self.P = (np.eye(4) - Kg @ Hm) @ self.P

    def maha(self, z, area):
        S = self.P[:2, :2] + np.eye(2) * self.r ** 2 * FULL_AREA / max(area, 1)
        d = z - self.x[:2]
        return d @ np.linalg.solve(S, d)


def track_sequence(frames, colors, present, q=0.3, r=0.5, v0=2.0, tol=0.35, bounce=True):
    """frames (T,H,W,3); returns mean (T,K,2), cov (T,K,2,2)."""
    Tn = frames.shape[0]
    K = len(present)
    mean = np.full((Tn, K, 2), 32.0)
    cov = np.tile(np.eye(2) * 1e4, (Tn, K, 1, 1))
    slots = [k for k in range(K) if present[k]]
    # group slots by color
    groups = {}
    for k in slots:
        groups.setdefault(tuple(np.round(colors[k], 3)), []).append(k)
    kfs = {k: KF(q, r, v0) for k in slots}
    for t in range(Tn):
        for col, ks in groups.items():
            comps = len(ks) > 1
            dets = detect(frames[t], np.array(col), tol, comps)
            for k in ks:
                if kfs[k].x is not None:
                    kfs[k].predict(bounce)
            if t == 0 and comps:
                # identity is defined as left-to-right order at frame 0
                dets = sorted(dets, key=lambda d: d[0][0])
                for k, dt in zip(ks, dets):
                    kfs[k].init(*dt)
            elif not comps:
                k = ks[0]
                if dets:
                    if kfs[k].x is None:
                        kfs[k].init(*dets[0])
                    else:
                        kfs[k].update(*dets[0])
            else:
                live = [k for k in ks if kfs[k].x is not None]
                if dets and live:
                    C = np.array([[kfs[k].maha(*dt) for dt in dets] for k in live])
                    rows, cols = linear_sum_assignment(np.minimum(C, 1e6))
                    gate = 13.8 if len(dets) < len(live) else np.inf  # chi2_2 0.999; no gate if
                    for i, j in zip(rows, cols):                     # every track has a detection
                        if C[i, j] < gate:
                            kfs[live[i]].update(*dets[j])
            for k in ks:
                if kfs[k].x is not None:
                    mean[t, k] = kfs[k].x[:2]
                    cov[t, k] = kfs[k].P[:2, :2]
    return mean, cov


def track_split(d, idx, frames=None, **kw):
    frames = sim.render_split(d, idx) if frames is None else frames
    M, C = [], []
    for j, i in enumerate(idx):
        m, c = track_sequence(frames[j], d["colors"][i], d["present"][i], **kw)
        M.append(m)
        C.append(c)
    return np.stack(M), np.stack(C)
