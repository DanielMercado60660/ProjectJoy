"""Procedural terrain: fBm elevation, rivers that follow downhill flow, lakes, bridges,
forest and rock. Everything is numpy and deterministic given a seed.

A terrain is a dict of HxW arrays:
  elev    float32 in [0, 1]   (relative elevation; cost.py scales it to "meters")
  kind    int8                one of GRASS, FOREST, ROCK, WATER, BRIDGE
  bridges list of lists of (r, c) cells, one entry per bridge
"""
import heapq
import numpy as np
from scipy import ndimage

GRASS, FOREST, ROCK, WATER, BRIDGE = 0, 1, 2, 3, 4
KIND_NAMES = ["grass", "forest", "rock", "water", "bridge"]

# D8 neighbour offsets (r, c), shared by flow routing
D8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def fbm(rng, h, w, octaves=5, base=4, persistence=0.5):
    """Fractal Brownian motion from upsampled white noise (value noise), range [0, 1]."""
    out = np.zeros((h, w), np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        coarse = rng.random((n + 3, n + 3))
        up = ndimage.zoom(coarse, ((h + 3 * h / n) / (n + 3), (w + 3 * w / n) / (n + 3)), order=3)
        out += amp * up[:h, :w]
        total += amp
        amp *= persistence
    out /= total
    out -= out.min()
    out /= max(out.max(), 1e-9)
    return out


def priority_flood(elev):
    """Fill depressions so every cell drains to the border (Barnes et al. 2014, plain version).
    Returns filled elevation (>= elev) with a tiny epsilon gradient so flow is defined."""
    h, w = elev.shape
    filled = elev.copy()
    closed = np.zeros((h, w), bool)
    pq = []
    for r in range(h):
        for c in (0, w - 1):
            heapq.heappush(pq, (filled[r, c], r, c)); closed[r, c] = True
    for c in range(1, w - 1):
        for r in (0, h - 1):
            heapq.heappush(pq, (filled[r, c], r, c)); closed[r, c] = True
    eps = 1e-6
    while pq:
        z, r, c = heapq.heappop(pq)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not closed[rr, cc]:
                closed[rr, cc] = True
                if filled[rr, cc] <= z:
                    filled[rr, cc] = z + eps
                heapq.heappush(pq, (filled[rr, cc], rr, cc))
    return filled


def flow_accumulation(filled):
    """D8 steepest-descent receiver for each cell, and upstream cell count."""
    h, w = filled.shape
    recv = -np.ones((h, w, 2), np.int32)
    for r in range(h):
        for c in range(w):
            best, br, bc = 0.0, -1, -1
            for dr, dc in D8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    drop = (filled[r, c] - filled[rr, cc]) / (1.4142 if dr and dc else 1.0)
                    if drop > best:
                        best, br, bc = drop, rr, cc
            recv[r, c] = (br, bc)
    acc = np.ones((h, w), np.float64)
    order = np.argsort(-filled, axis=None)  # high to low
    for idx in order:
        r, c = divmod(int(idx), w)
        br, bc = recv[r, c]
        if br >= 0:
            acc[br, bc] += acc[r, c]
    return recv, acc


def _river_mask(recv, acc, threshold, h, w):
    """Cells whose accumulation exceeds threshold, made 8-connected-safe: we also mark the
    orthogonal 'elbow' of every diagonal flow step so the river is 4-connected. Together with
    the no-corner-cutting rule in cost.py, that makes rivers watertight barriers."""
    river = acc >= threshold
    rs, cs = np.nonzero(river)
    extra = []
    for r, c in zip(rs, cs):
        br, bc = recv[r, c]
        if br >= 0 and br != r and bc != c:
            extra.append((r, bc))
    for r, c in extra:
        river[r, c] = True
    return river


def hydrology(elev, river_threshold=None):
    """Rivers and lakes implied by an elevation map (the generator's own water model; also
    used as the hand-built 'hydrology prior' in the partial-information scenario)."""
    h, w = elev.shape
    elev = elev.astype(np.float64)
    filled = priority_flood(elev)
    recv, acc = flow_accumulation(filled)
    if river_threshold is None:
        river_threshold = 0.02 * h * w
    river = _river_mask(recv, acc, river_threshold, h, w)
    # widen big rivers downstream
    wide = acc >= river_threshold * 6
    river |= ndimage.binary_dilation(wide, structure=np.ones((3, 3), bool)) & ndimage.binary_dilation(river)
    # lakes: depressions filled more than a little
    lake = (filled - elev) > 0.045
    lake = ndimage.binary_opening(lake, iterations=1)
    return river, lake, recv, acc


def generate(seed, size=96, river_threshold=None, n_bridges=3, height_scale=1.0):
    """Generate a terrain. Same seed -> same map."""
    rng = np.random.default_rng(seed)
    h = w = size
    elev = fbm(rng, h, w, octaves=5, base=3)
    # a gentle tilt so rivers have a preferred direction and reach the border
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w] / size
    elev = 0.75 * elev + 0.25 * (np.cos(ang) * (xx - 0.5) + np.sin(ang) * (yy - 0.5) + 0.5)
    # sharpen peaks a bit
    elev = (elev - elev.min()) / (elev.max() - elev.min())
    elev = elev ** 1.3

    river, lake, recv, acc = hydrology(elev, river_threshold)
    water = river | lake

    kind = np.full((h, w), GRASS, np.int8)
    veg = fbm(rng, h, w, octaves=4, base=5)
    kind[(veg > 0.55) & (elev < 0.6)] = FOREST
    rocky = fbm(rng, h, w, octaves=3, base=6)
    kind[(elev > 0.7) | ((elev > 0.55) & (rocky > 0.72))] = ROCK
    kind[water] = WATER

    bridges = place_bridges(rng, kind, recv, acc, river, n_bridges)
    for br in bridges:
        for r, c in br:
            kind[r, c] = BRIDGE
    return {"elev": elev.astype(np.float32), "kind": kind, "bridges": bridges,
            "seed": int(seed), "size": size}


def place_bridges(rng, kind, recv, acc, river, n_bridges, max_len=7):
    """Bridges: short straight spans (horizontal or vertical) across water, each ending on land
    at both sides. Chosen at random among valid spans, kept apart from each other."""
    h, w = kind.shape
    water = kind == WATER
    cands = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if not water[r, c]:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                # span must start right after land
                pr, pc = r - dr, c - dc
                if water[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and water[rr, cc] and len(cells) <= max_len:
                    cells.append((rr, cc)); rr += dr; cc += dc
                if not (0 <= rr < h and 0 <= cc < w) or water[rr, cc] or len(cells) > max_len:
                    continue
                # land on both ends must not be a lake shore pocket: require land cells
                # beyond the ends to be land too (a real crossing)
                if 0 <= rr + dr < h and 0 <= cc + dc < w and water[rr + dr, cc + dc]:
                    continue
                cands.append(cells)
    rng.shuffle(cands)
    chosen = []
    for cells in cands:
        r0, c0 = cells[len(cells) // 2]
        if all(abs(r0 - b[len(b) // 2][0]) + abs(c0 - b[len(b) // 2][1]) > h // 4 for b in chosen):
            chosen.append(cells)
        if len(chosen) >= n_bridges:
            break
    return chosen
