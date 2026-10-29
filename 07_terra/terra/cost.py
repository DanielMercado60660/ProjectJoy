"""THE cost definition. Every planner (Python and the JS port in demo.html) uses exactly this.

Moving from cell a to an 8-neighbour b costs

    c(a, b) = d * m_bar * (1 + SLOPE_K * |z_b - z_a| / d)

  d       = 1 for orthogonal moves, sqrt(2) for diagonal moves
  m_bar   = (m_a + m_b) / 2, terrain multipliers: grass 1, forest 2, rock 3, bridge 1
  z       = elevation * HEIGHT_SCALE (cell units)
  water   = impassable (no edge into or out of a water cell); bridges are passable
  no corner cutting: a diagonal move is allowed only if both orthogonal cells it squeezes
  between are passable (otherwise one-cell-wide diagonal rivers would leak).

The cost is symmetric, c(a,b) == c(b,a), so predecessors == successors (simplifies D* Lite).
Lower bound: m >= 1 and slope factor >= 1, so c(a,b) >= d, and the octile distance is an
admissible, consistent heuristic (MIN_UNIT_COST = 1).
"""
import math
import numpy as np
from .terrain import GRASS, FOREST, ROCK, WATER, BRIDGE

MULT = {GRASS: 1.0, FOREST: 2.0, ROCK: 3.0, WATER: math.inf, BRIDGE: 1.0}
HEIGHT_SCALE = 24.0
SLOPE_K = 1.0
MIN_UNIT_COST = 1.0
SQRT2 = math.sqrt(2.0)

# direction k -> (dr, dc). Orthogonal first, then diagonal.
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def mult_map(kind):
    m = np.empty(kind.shape, np.float64)
    for k, v in MULT.items():
        m[kind == k] = v
    return m


def edge_costs(elev, kind, mult=None):
    """cost[k, r, c] = cost of moving from (r, c) in direction DIRS[k]; inf if not allowed.
    `mult` optionally overrides the per-cell multiplier (used only for *belief* maps in the
    partial-information scenario; the true world always uses mult_map(kind))."""
    h, w = kind.shape
    z = elev.astype(np.float64) * HEIGHT_SCALE
    m = mult_map(kind) if mult is None else np.asarray(mult, np.float64)
    passable = np.isfinite(m)
    out = np.full((8, h, w), np.inf)
    for k, (dr, dc) in enumerate(DIRS):
        d = SQRT2 if dr and dc else 1.0
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        a = (slice(r0, r1), slice(c0, c1))
        b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        ok = passable[a] & passable[b]
        if dr and dc:  # no corner cutting
            ok &= passable[r0 + dr:r1 + dr, c0:c1] & passable[r0:r1, c0 + dc:c1 + dc]
        mbar = 0.5 * (m[a] + m[b])
        c = d * mbar * (1.0 + SLOPE_K * np.abs(z[b] - z[a]) / d)
        out[k][a] = np.where(ok, c, np.inf)
    return out


def octile(r0, c0, r1, c1):
    dr, dc = abs(r0 - r1), abs(c0 - c1)
    return MIN_UNIT_COST * (max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc))


def octile_field(h, w, gr, gc):
    """Admissible heuristic to (gr, gc) for every cell, as an HxW array."""
    rr, cc = np.mgrid[0:h, 0:w]
    dr, dc = np.abs(rr - gr), np.abs(cc - gc)
    return MIN_UNIT_COST * (np.maximum(dr, dc) + (SQRT2 - 1.0) * np.minimum(dr, dc))


class Graph:
    """Grid graph with Python adjacency lists (fast enough for pure-Python planners).
    Node id = r * W + c. adj[u] = list of (v, cost) with finite cost."""

    def __init__(self, elev, kind, mult=None):
        self.elev = np.asarray(elev, np.float32).copy()
        self.kind = np.asarray(kind, np.int8).copy()
        self.mult = mult_map(self.kind) if mult is None else np.asarray(mult, np.float64).copy()
        self.h, self.w = self.kind.shape
        self.n = self.h * self.w
        self.adj = [None] * self.n
        self._rebuild_all()

    def _rebuild_all(self):
        cost = edge_costs(self.elev, self.kind, self.mult)
        h, w = self.h, self.w
        offs = [dr * w + dc for dr, dc in DIRS]
        cl = cost.reshape(8, -1)
        fin = np.isfinite(cl)
        cl_list = cl.tolist()
        fin_list = fin.tolist()
        for u in range(self.n):
            self.adj[u] = [(u + offs[k], cl_list[k][u]) for k in range(8) if fin_list[k][u]]

    def set_cells(self, cells, kind_value=None, elev_value=None, mult_values=None):
        """Change cells and rebuild adjacency locally. Returns the set of node ids whose
        outgoing (== incoming, symmetric) edges may have changed."""
        for i, (r, c) in enumerate(cells):
            if kind_value is not None:
                self.kind[r, c] = kind_value
                self.mult[r, c] = MULT[int(kind_value)]
            if elev_value is not None:
                self.elev[r, c] = elev_value
            if mult_values is not None:
                self.mult[r, c] = mult_values[i]
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        # any edge touched by a changed cell (as endpoint or as a corner) has both
        # endpoints within Chebyshev distance 1 of it -> rebuild a 1-cell margin
        r0, r1 = max(0, min(rs) - 2), min(self.h, max(rs) + 3)
        c0, c1 = max(0, min(cs) - 2), min(self.w, max(cs) + 3)
        sub = edge_costs(self.elev[r0:r1, c0:c1], self.kind[r0:r1, c0:c1], self.mult[r0:r1, c0:c1])
        affected = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        affected.add(rr * self.w + cc)
        w = self.w
        for u in affected:
            r, c = divmod(u, w)
            lst = []
            for k, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.h and 0 <= cc < self.w):
                    continue
                # use the sub-window cost if the edge is fully inside it (it always is
                # for affected cells thanks to the 2-cell margin)
                v = sub[k, r - r0, c - c0]
                if np.isfinite(v):
                    lst.append((rr * w + cc, float(v)))
            self.adj[u] = lst
        return affected

    def cost_array(self):
        return edge_costs(self.elev, self.kind, self.mult)

    def h_octile(self, u, v):
        r0, c0 = divmod(u, self.w)
        r1, c1 = divmod(v, self.w)
        return octile(r0, c0, r1, c1)


def path_cost(graph, path):
    """Sum of edge costs along a node path using the graph's current costs (inf if broken)."""
    tot = 0.0
    for a, b in zip(path[:-1], path[1:]):
        for v, c in graph.adj[a]:
            if v == b:
                tot += c
                break
        else:
            return math.inf
    return tot
