"""Synthetic road maps as vector graphs, rasterized to 64x64, plus rectangular holes.

Styles:
  grid     : perturbed grid (jittered nodes, ~20% edges dropped) + 1-2 curved arterials (train style)
  delaunay : random planar graph (Delaunay triangulation, long edges pruned)   (held-out style)
The vector graph (node coords + edge list + widths) is kept as ground truth next to the raster.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

S = 64  # raster size


def _grid_graph(rng):
    sp = rng.uniform(9, 14)
    off = rng.uniform(0, sp, 2) - sp
    n = int(S / sp) + 3
    ij = [(i, j) for i in range(n) for j in range(n)]
    idx = {p: k for k, p in enumerate(ij)}
    nodes = np.array([[off[0] + i * sp, off[1] + j * sp] for i, j in ij]) + rng.uniform(-2, 2, (len(ij), 2))
    edges = []
    for (i, j), k in idx.items():
        for di, dj in ((1, 0), (0, 1)):
            q = (i + di, j + dj)
            if q in idx and rng.random() > 0.2:
                edges.append((k, idx[q], 2))
    # arterials: gentle curved polylines edge to edge, width 3
    for _ in range(rng.integers(1, 3)):
        a = rng.uniform(0, np.pi)
        c = rng.uniform(16, 48, 2)
        d = np.array([np.cos(a), np.sin(a)])
        nrm = np.array([-d[1], d[0]])
        bend = rng.uniform(-8, 8)
        ts = np.linspace(-50, 50, 9)
        pts = c + ts[:, None] * d + (bend * (1 - (ts / 50) ** 2))[:, None] * nrm
        base = len(nodes)
        nodes = np.vstack([nodes, pts])
        edges += [(base + t, base + t + 1, 3) for t in range(len(pts) - 1)]
    return nodes, edges


def _delaunay_graph(rng):
    k = rng.integers(25, 45)
    pts = rng.uniform(-6, S + 6, (k, 2))
    tri = Delaunay(pts)
    es = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            es.add(tuple(sorted((s[a], s[b]))))
    edges = []
    for a, b in es:
        L = np.linalg.norm(pts[a] - pts[b])
        if L < 22 and rng.random() > 0.3:
            edges.append((a, b, 3 if rng.random() < 0.15 else 2))
    return pts, edges


def make_map(seed, style="grid"):
    rng = np.random.default_rng(seed)
    nodes, edges = (_grid_graph if style == "grid" else _delaunay_graph)(rng)
    return nodes, edges


def rasterize(nodes, edges):
    im = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(im)
    for a, b, w in edges:
        dr.line([tuple(nodes[a]), tuple(nodes[b])], fill=255, width=int(w))
    return (np.asarray(im) > 127).astype(np.uint8)


def make_hole(rng, lo, hi, margin=3):
    """Axis-aligned rectangle hole, fully inside the image with a margin so the boundary ring exists."""
    h, w = rng.integers(lo, hi + 1, 2)
    y = rng.integers(margin, S - margin - h + 1)
    x = rng.integers(margin, S - margin - w + 1)
    m = np.zeros((S, S), np.uint8)
    m[y:y + h, x:x + w] = 1
    return m


def build_set(seeds, style="grid", hole=(12, 24), hole_seed=None):
    """Fixed evaluation/training set: rasters, vector graphs, and (optionally) fixed holes."""
    ims, graphs, holes = [], [], []
    hr = np.random.default_rng(hole_seed) if hole_seed is not None else None
    for s in seeds:
        n, e = make_map(int(s), style)
        im = rasterize(n, e)
        if hr is not None:
            # require at least 2 road pixels on the hole ring so the task is non-trivial
            for _ in range(50):
                m = make_hole(hr, *hole)
                if (ring_of(m) & im).sum() >= 2:
                    break
            holes.append(m)
        ims.append(im)
        graphs.append((n, e))
    out = dict(ims=np.stack(ims), graphs=graphs)
    if holes:
        out["holes"] = np.stack(holes)
    return out


def ring_of(hole):
    """1-px ring just outside the hole (8-neighbourhood)."""
    from scipy.ndimage import binary_dilation
    return (binary_dilation(hole, np.ones((3, 3), bool)) & ~hole.astype(bool)).astype(np.uint8)


# Seed ranges (disjoint): train 0..99_999, val 1_000_000.., test 2_000_000..
TRAIN0, VAL0, TEST0 = 0, 1_000_000, 2_000_000
