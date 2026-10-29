"""The reference bot's map (bot side): a per-room floor colour model, an occupancy grid built from the
head camera, and A* planning on it. All in the bot's odometry frame; nothing reads the world.

Floor model: colours sampled where the floor is guaranteed (a ring 0.28-0.40 m around the start: the
room validator keeps every obstacle >= 0.4 m from the home mat's centre), clustered by chromaticity;
a pixel is "floor-like" if its chromaticity is near a cluster and its brightness is 0.35-1.6x the
cluster's (shadows darken the floor without changing its tint).
Occupancy: below the horizon, floor-like or walkable-marker pixels (pads, button and its frame, mat)
mark their floor cells free; in each image column the first run of other pixels (walls, partition,
occluder) marks the floor cell just below it as an obstacle. Log-odds per 5 cm cell; unknown = free.
"""
import heapq

import numpy as np
from scipy import ndimage

from perception import CameraPose, masks

CELL = 0.05
HALF = 2.2            # grid covers +-2.2 m around the start (the 3 m room fits from any start)
L_FREE, L_OCC, L_MAX = -0.4, 0.9, 3.0
MAX_RANGE = 2.2


FLOOR_PARAMS = {"tol_min": 0.015, "tol_k": 0.0, "bright_lo": 0.35, "bright_hi": 1.4}   # m3_floor.py, tuning.jsonl


class FloorModel:
    def __init__(self, params=None):
        self.P = dict(FLOOR_PARAMS, **(params or {}))
        self.samples = []
        self.centres = None    # (k, 3) chromaticity
        self.bright = None     # (k,) mean brightness
        self.tol = None

    @staticmethod
    def features(rgb):
        rgb = rgb.astype(np.float32) + 1.0
        s = rgb.sum(-1, keepdims=True)
        return rgb / s, s[..., 0] / 3.0

    def add_samples(self, frame, cam: CameraPose, walkable):
        """Pixels whose floor point is 0.28-0.40 m from the trunk (guaranteed floor at the start)."""
        below = cam.rays[..., 2] < -1e-3
        t = -cam.pos[2] / np.where(below, cam.rays[..., 2], -1)
        p = cam.pos[:2] + t[..., None] * cam.rays[..., :2]
        trunk_xy = cam.trunk_xy
        dist = np.linalg.norm(p - trunk_xy, axis=-1)
        sel = below & (dist > 0.28) & (dist < 0.40) & ~walkable
        if sel.sum():
            self.samples.append(frame[sel])

    def fit(self, k=3):
        x = np.concatenate(self.samples) if self.samples else np.zeros((0, 3))
        if len(x) < 20:
            return False
        chroma, bright = self.features(x)
        # k-means on chromaticity (few iterations, deterministic init from quantiles of brightness)
        order = np.argsort(bright)
        cent = chroma[order[np.linspace(0, len(order) - 1, k).astype(int)]]
        for _ in range(15):
            lab = np.argmin(((chroma[:, None] - cent[None]) ** 2).sum(-1), 1)
            cent = np.array([chroma[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        keep = [j for j in range(k) if (lab == j).mean() > 0.10]
        self.centres = cent[keep]
        self.bright = np.array([bright[lab == j].mean() for j in keep])
        spread = np.array([np.sqrt(((chroma[lab == j] - cent[j]) ** 2).sum(-1)).mean() for j in keep])
        self.tol = np.maximum(self.P["tol_min"], self.P["tol_k"] * spread)
        return True

    def floor_like(self, frame):
        chroma, bright = self.features(frame)
        ok = np.zeros(frame.shape[:2], bool)
        for c, b, tl in zip(self.centres, self.bright, self.tol):
            d = np.sqrt(((chroma - c) ** 2).sum(-1))
            ok |= (d < tl) & (bright > self.P["bright_lo"] * b) & (bright < self.P["bright_hi"] * b)
        return ok


def walkable_mask(frame):
    m = masks(frame)
    w = m["pad_red"] | m["pad_yellow"] | m["pad_green"] | m["pad_blue"] | m["mat"]
    w |= ndimage.binary_dilation(m["button"], iterations=3)      # the plate and its dark frame
    w |= m["ball"]
    return w


class OccupancyGrid:
    def __init__(self):
        self.n = int(2 * HALF / CELL)
        self.L = np.zeros((self.n, self.n), np.float32)

    def idx(self, xy):
        xy = np.atleast_2d(xy)
        return np.clip(((xy + HALF) / CELL).astype(int), 0, self.n - 1)

    def update(self, frame, cam: CameraPose, floor: FloorModel):
        if floor.centres is None:
            return
        below = cam.rays[..., 2] < -0.02
        t = -cam.pos[2] / np.where(below, cam.rays[..., 2], -1)
        p = cam.pos[:2] + t[..., None] * cam.rays[..., :2]
        dist = np.linalg.norm(p - cam.pos[:2], axis=-1)
        inrange = below & (dist < MAX_RANGE)
        free = (floor.floor_like(frame) | walkable_mask(frame)) & inrange
        # free cells
        fi = self.idx(p[free])
        np.add.at(self.L, (fi[:, 1], fi[:, 0]), L_FREE)
        # obstacle boundary per column: first non-free run (>= 2 px) going up from the bottom row
        H = frame.shape[0]
        occ_pts = []
        for j in range(0, frame.shape[1], 2):
            col_free, col_below = free[:, j], inrange[:, j]
            for i in range(H - 1, 0, -1):
                if not col_below[i]:
                    break
                if not col_free[i] and not col_free[i - 1]:
                    if i + 1 < H and col_below[i + 1]:
                        occ_pts.append(p[i + 1, j] + 0.03 * (p[i + 1, j] - cam.pos[:2]) / max(dist[i + 1, j], 1e-6))
                    break
        if occ_pts:
            oi = self.idx(np.array(occ_pts))
            np.add.at(self.L, (oi[:, 1], oi[:, 0]), L_OCC + 0.4)    # cancels this frame's free vote
        np.clip(self.L, -L_MAX, L_MAX, out=self.L)

    def occupied(self, thresh=0.5):
        return self.L > thresh

    def plan(self, start, goal, clearance=0.15, blocked=()):
        """A* on the inflated occupancy grid; unknown cells count as free. Waypoints or None."""
        raw = self.occupied()
        occ = ndimage.binary_dilation(raw, iterations=max(1, int(round(clearance / CELL))))
        s, g = tuple(self.idx(start)[0][::-1]), tuple(self.idx(goal)[0][::-1])
        # a duck standing (or a goal) near an obstacle sits inside its inflation: free the inflated
        # (not the raw) cells within the clearance of start and goal so A* can leave and arrive
        rr, cc = np.mgrid[0:self.n, 0:self.n]
        k = int(round(clearance / CELL)) + 1
        for (r0, c0) in (s, g):
            near = (rr - r0) ** 2 + (cc - c0) ** 2 <= k * k
            occ[near & ~raw] = False
        occ[s] = occ[g] = False
        for (bxy, r) in blocked:            # no-go discs (e.g. the button plate when not pressing)
            br, bc = self.idx(bxy)[0][::-1]
            k2 = int(round(r / CELL))
            disc = (rr - br) ** 2 + (cc - bc) ** 2 <= k2 * k2
            disc[s] = disc[g] = False
            occ |= disc
        if s == g:
            return [list(map(float, goal))]
        n = self.n
        h = lambda a: np.hypot(a[0] - g[0], a[1] - g[1])  # noqa: E731
        openq, came, cost = [(h(s), 0.0, s)], {s: None}, {s: 0.0}
        steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while openq:
            _, c, cur = heapq.heappop(openq)
            if cur == g:
                break
            if c > cost[cur]:
                continue
            for dr, dc, wgt in steps:
                nb = (cur[0] + dr, cur[1] + dc)
                if 0 <= nb[0] < n and 0 <= nb[1] < n and not occ[nb]:
                    nc = c + wgt
                    if nc < cost.get(nb, 1e18):
                        cost[nb], came[nb] = nc, cur
                        heapq.heappush(openq, (nc + h(nb), nc, nb))
        if g not in came:
            return None
        cells, cur = [], g
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells.reverse()
        pts = [np.array([c * CELL - HALF + CELL / 2, r * CELL - HALF + CELL / 2]) for r, c in cells]

        def clear(a, b):
            for tt in np.linspace(0, 1, int(np.hypot(*(b - a)) / (CELL / 2)) + 2):
                r, c = self.idx(a + tt * (b - a))[0][::-1]
                if occ[r, c]:
                    return False
            return True

        out, i = [], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        out = out or [pts[-1]]
        out[-1] = np.asarray(goal, float)
        return [q.round(3).tolist() for q in out]
