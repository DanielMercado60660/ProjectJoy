"""Maze families, BFS verification and level statistics.

A level is (walls, start, goal):
  walls : uint8 array (S, S), 1 = wall, 0 = floor. The border is always wall.
  start, goal : (row, col) of two distinct floor cells, goal reachable from start.

Families
  dfs     perfect maze (recursive backtracker on the odd lattice)      [train]
  braid   dfs maze with ~half of the dead ends knocked through (loops) [train]
  rooms   rectangular rooms joined by L-shaped corridors               [train]
  caves   cellular-automaton caves, largest connected region kept      [HELD OUT]
  spiral  concentric rings with gaps + radial blockers, or serpentines [HELD OUT]
"""
import numpy as np
from numba import njit

SIZE = 13
TRAIN_FAMILIES = ("dfs", "braid", "rooms")
HELDOUT_FAMILIES = ("caves", "spiral")
ALL_FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)  # N E S W


# ----------------------------------------------------------------------------- BFS
@njit(cache=True)
def bfs_dist(walls, sr, sc):
    """Shortest path distance from (sr, sc) to every cell (-1 = unreachable)."""
    H, W = walls.shape
    dist = -np.ones((H, W), dtype=np.int64)
    if walls[sr, sc]:
        return dist
    qr = np.empty(H * W, dtype=np.int64)
    qc = np.empty(H * W, dtype=np.int64)
    head, tail = 0, 1
    qr[0], qc[0] = sr, sc
    dist[sr, sc] = 0
    while head < tail:
        r, c = qr[head], qc[head]
        head += 1
        for k in range(4):
            nr = r + (-1, 0, 1, 0)[k]
            nc = c + (0, 1, 0, -1)[k]
            if 0 <= nr < H and 0 <= nc < W and walls[nr, nc] == 0 and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                qr[tail], qc[tail] = nr, nc
                tail += 1
    return dist


def shortest_path_len(walls, start, goal):
    return int(bfs_dist(walls, int(start[0]), int(start[1]))[goal[0], goal[1]])


def is_valid(walls, start, goal, min_dist=1):
    """Hard validity: border intact, start/goal on floor and distinct, goal reachable."""
    S = walls.shape[0]
    if not (walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all()):
        return False
    if walls[start[0], start[1]] or walls[goal[0], goal[1]] or tuple(start) == tuple(goal):
        return False
    d = shortest_path_len(walls, start, goal)
    return d >= min_dist


@njit(cache=True)
def bfs_path(walls, sr, sc, gr, gc):
    """Shortest path as an (n, 2) array of cells from start to goal (inclusive)."""
    dist = bfs_dist(walls, gr, gc)  # distance-to-goal; walk downhill from start
    n = dist[sr, sc]
    path = np.empty((max(n, 0) + 1, 2), dtype=np.int64)
    if n < 0:
        return path[:0]
    r, c = sr, sc
    path[0, 0], path[0, 1] = r, c
    for i in range(1, n + 1):
        for k in range(4):
            nr = r + (-1, 0, 1, 0)[k]
            nc = c + (0, 1, 0, -1)[k]
            if dist[nr, nc] == dist[r, c] - 1 and walls[nr, nc] == 0:
                r, c = nr, nc
                break
        path[i, 0], path[i, 1] = r, c
    return path


# ----------------------------------------------------------------------------- families
def _blank(S):
    return np.ones((S, S), dtype=np.uint8)


def gen_dfs(rng, S=SIZE):
    w = _blank(S)
    cells = [(r, c) for r in range(1, S - 1, 2) for c in range(1, S - 1, 2)]
    r, c = cells[rng.integers(len(cells))]
    w[r, c] = 0
    stack = [(r, c)]
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((-2, 0), (0, 2), (2, 0), (0, -2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < S - 1 and 1 <= nc < S - 1 and w[nr, nc]:
                nbrs.append((nr, nc, dr, dc))
        if not nbrs:
            stack.pop()
            continue
        nr, nc, dr, dc = nbrs[rng.integers(len(nbrs))]
        w[r + dr // 2, c + dc // 2] = 0
        w[nr, nc] = 0
        stack.append((nr, nc))
    return w


def gen_braid(rng, S=SIZE, p=0.5):
    w = gen_dfs(rng, S)
    for r in range(1, S - 1, 2):
        for c in range(1, S - 1, 2):
            open_n = [(dr, dc) for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)) if w[r + dr, c + dc] == 0]
            if len(open_n) == 1 and rng.random() < p:
                cand = [(dr, dc) for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1))
                        if w[r + dr, c + dc] == 1 and 1 <= r + 2 * dr < S - 1 and 1 <= c + 2 * dc < S - 1]
                if cand:
                    dr, dc = cand[rng.integers(len(cand))]
                    w[r + dr, c + dc] = 0
    return w


def gen_rooms(rng, S=SIZE):
    w = _blank(S)
    rooms = []
    for _ in range(40):
        if len(rooms) >= rng.integers(4, 7):
            break
        h, wd = rng.integers(2, 5), rng.integers(2, 5)
        r0, c0 = rng.integers(1, S - 1 - h + 1), rng.integers(1, S - 1 - wd + 1)
        # keep a 1-cell wall margin between rooms
        if any(r0 - 1 < rr + hh and rr - 1 < r0 + h and c0 - 1 < cc + ww and cc - 1 < c0 + wd
               for rr, cc, hh, ww in rooms):
            continue
        rooms.append((r0, c0, h, wd))
        w[r0:r0 + h, c0:c0 + wd] = 0
    centers = [(r0 + rng.integers(h), c0 + rng.integers(wd)) for r0, c0, h, wd in rooms]
    order = rng.permutation(len(centers))
    links = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    if len(centers) > 2 and rng.random() < 0.5:  # an occasional extra corridor (loop)
        a, b = rng.choice(len(centers), 2, replace=False)
        links.append((a, b))
    for a, b in links:
        (r1, c1), (r2, c2) = centers[a], centers[b]
        if rng.random() < 0.5:
            w[min(r1, r2):max(r1, r2) + 1, c1] = 0
            w[r2, min(c1, c2):max(c1, c2) + 1] = 0
        else:
            w[r1, min(c1, c2):max(c1, c2) + 1] = 0
            w[min(r1, r2):max(r1, r2) + 1, c2] = 0
    return w


def _largest_component(w):
    S = w.shape[0]
    seen = np.zeros_like(w, dtype=bool)
    best = None
    for r in range(S):
        for c in range(S):
            if w[r, c] == 0 and not seen[r, c]:
                d = bfs_dist(w, r, c) >= 0
                seen |= d
                if best is None or d.sum() > best.sum():
                    best = d
    out = _blank(S)
    if best is not None:
        out[best] = 0
    return out


def gen_caves(rng, S=SIZE):
    while True:
        w = _blank(S)
        w[1:-1, 1:-1] = (rng.random((S - 2, S - 2)) < 0.47).astype(np.uint8)
        for _ in range(3):
            p = np.pad(w, 1, constant_values=1)
            n = sum(p[1 + dr:S + 1 + dr, 1 + dc:S + 1 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0))
            w = np.where((n >= 5) | ((w == 1) & (n >= 4)), 1, 0).astype(np.uint8)
            w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 1
        w = _largest_component(w)
        if (w == 0).sum() >= 40:
            return w


def gen_spiral(rng, S=SIZE):
    if rng.random() < 0.5:
        # concentric rings: wall rings at offset 2, 4 (and centre cell area), one gap each,
        # plus one radial blocker per corridor ring so each ring is traversed one way round.
        w = np.zeros((S, S), dtype=np.uint8)
        w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 1
        for k in (2, 4):
            lo, hi = k, S - 1 - k
            ring = [(lo, c) for c in range(lo, hi + 1)] + [(hi, c) for c in range(lo, hi + 1)] + \
                   [(r, lo) for r in range(lo + 1, hi)] + [(r, hi) for r in range(lo + 1, hi)]
            for r, c in ring:
                w[r, c] = 1
            odd = [(r, c) for r, c in ring if (r + c) % 2 == 1]  # non-corner cells
            r, c = odd[rng.integers(len(odd))]
            w[r, c] = 0
        for k in (1, 3):  # corridor rings at offset 1 and 3: add a radial blocker
            lo, hi = k, S - 1 - k
            ring = [(lo, c) for c in range(lo + 1, hi)] + [(hi, c) for c in range(lo + 1, hi)] + \
                   [(r, lo) for r in range(lo + 1, hi)] + [(r, hi) for r in range(lo + 1, hi)]
            for _ in range(20):
                r, c = ring[rng.integers(len(ring))]
                w2 = w.copy()
                w2[r, c] = 1
                if _n_components(w2) == 1:
                    w = w2
                    break
        return w
    # serpentine / zigzag
    w = np.zeros((S, S), dtype=np.uint8)
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 1
    for i, r in enumerate(range(2, S - 1, 2)):
        w[r, 1:S - 1] = 1
        span = rng.integers(1, 3)
        if (i + rng.integers(2)) % 2 == 0:
            c = rng.integers(1, 1 + span)
        else:
            c = rng.integers(S - 1 - span, S - 1)
        w[r, c] = 0
        if rng.random() < 0.3:  # rare shortcut
            w[r, rng.integers(1, S - 1)] = 0
    if rng.random() < 0.5:
        w = w.T.copy()
    return w


def _n_components(w):
    S = w.shape[0]
    seen = np.zeros_like(w, dtype=bool)
    n = 0
    for r in range(S):
        for c in range(S):
            if w[r, c] == 0 and not seen[r, c]:
                seen |= bfs_dist(w, r, c) >= 0
                n += 1
    return n


GENERATORS = dict(dfs=gen_dfs, braid=gen_braid, rooms=gen_rooms, caves=gen_caves, spiral=gen_spiral)


def place_start_goal(w, rng, min_frac=0.5, min_dist=6):
    """Random start; goal random among cells at least max(min_dist, min_frac*farthest) away."""
    floor = np.argwhere(w == 0)
    for _ in range(50):
        s = floor[rng.integers(len(floor))]
        d = bfs_dist(w, s[0], s[1])
        far = d.max()
        thr = max(min_dist, int(np.ceil(min_frac * far)))
        cand = np.argwhere(d >= thr)
        if len(cand):
            g = cand[rng.integers(len(cand))]
            return s.astype(np.int64), g.astype(np.int64)
    # tiny map fallback: farthest pair from a random start
    s = floor[rng.integers(len(floor))]
    d = bfs_dist(w, s[0], s[1])
    g = np.array(np.unravel_index(np.argmax(d), d.shape))
    return s.astype(np.int64), g.astype(np.int64)


def make_level(family, rng):
    w = GENERATORS[family](rng)
    s, g = place_start_goal(w, rng)
    assert is_valid(w, s, g), family
    return w, s, g


def make_levels(family, n, seed):
    rng = np.random.default_rng(seed)
    W = np.empty((n, SIZE, SIZE), np.uint8)
    St = np.empty((n, 2), np.int64)
    G = np.empty((n, 2), np.int64)
    for i in range(n):
        W[i], St[i], G[i] = make_level(family, rng)
    return dict(walls=W, start=St, goal=G)


def concat_levels(*sets):
    return {k: np.concatenate([s[k] for s in sets]) for k in ("walls", "start", "goal")}


def subset(levels, idx):
    return {k: v[idx] for k, v in levels.items()}


# ----------------------------------------------------------------------------- stats
def level_stats(w, s, g):
    """Structural statistics of one level."""
    floor = w == 0
    p = np.pad(floor, 1)
    deg = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    deg = np.where(floor, deg, 0)
    V = int(floor.sum())
    E = int((floor[1:, :] & floor[:-1, :]).sum() + (floor[:, 1:] & floor[:, :-1]).sum())
    C = _n_components(w)
    open2x2 = int((floor[1:, 1:] & floor[:-1, 1:] & floor[1:, :-1] & floor[:-1, :-1]).sum())
    d = bfs_dist(w, s[0], s[1])
    return dict(
        floor=V,
        dead_ends=int(((deg == 1) & floor).sum()),
        junctions=int(((deg >= 3) & floor).sum()),
        loops=E - V + C,             # cyclomatic number of the floor graph
        components=C,
        open2x2=open2x2,
        path_len=int(d[g[0], g[1]]),
        reachable=int((d >= 0).sum()),
    )


def levels_stats(levels):
    rows = [level_stats(w, s, g) for w, s, g in zip(levels["walls"], levels["start"], levels["goal"])]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
