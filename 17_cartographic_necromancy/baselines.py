"""Classical inpainting baselines (all hyperparameters tuned on the validation set in evaluate.py).

stub_lines : direction-aware straight-line reconnection. Each stub's direction is the principal axis of the
             known road pixels connected to it in a band outside the hole. Pairs scored by
             cost = dist/hole_size + w_ang * misalignment, greedily matched (min cost first, each stub
             once, cost < thr). Unmatched stubs are extended straight along their direction until they
             hit already-restored road or leave the hole (handles T-junctions).
diffusion  : Laplace (heat) fill of the hole from the known pixels, then threshold.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, binary_dilation, distance_transform_edt
from metrics import stubs, K8


def stub_geometry(known, hole, band=5):
    hole_b = hole.astype(bool)
    lab, n = stubs(known, hole)
    dist_out = distance_transform_edt(~hole_b)
    bandmask = known.astype(bool) & (dist_out <= band) & ~hole_b
    blab, _ = label(bandmask, K8)
    ys, xs = np.nonzero(hole_b)
    hc = np.array([ys.mean(), xs.mean()])
    geo = []
    for i in range(n):
        py, px = np.nonzero(lab == i + 1)
        c = np.array([py.mean(), px.mean()])
        comp_ids = np.unique(blab[lab == i + 1]); comp_ids = comp_ids[comp_ids > 0]
        m = np.isin(blab, comp_ids) & (dist_out <= band)
        qy, qx = np.nonzero(m)
        P = np.stack([qy, qx], 1).astype(float)
        # keep only pixels near the stub (avoid using a whole road running parallel to the hole)
        P = P[np.linalg.norm(P - c, axis=1) <= band + 1]
        d = None
        # direction = from the centroid of nearby known road pixels to the stub. For a straight road this is
        # the road axis; for a junction just outside the hole it still points junction -> stub (PCA did not).
        if len(P) >= 3:
            v = c - P.mean(0)
            if np.linalg.norm(v) > 1.0:
                d = v / np.linalg.norm(v)
        if d is None:  # fallback: inward normal of the nearest hole side (roads mostly cross sides squarely)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            gaps = [(abs(c[0] - (y0 - 1)), (1, 0)), (abs(c[0] - (y1 + 1)), (-1, 0)),
                    (abs(c[1] - (x0 - 1)), (0, 1)), (abs(c[1] - (x1 + 1)), (0, -1))]
            d = np.array(min(gaps)[1], float)
        width = 3 if len(py) >= 3 + 1 else 2
        geo.append((c, d, width))
    return geo


def _draw(canvas, p, q, w):
    canvas.line([(p[1], p[0]), (q[1], q[0])], fill=255, width=int(w))


def stub_lines(known, hole, w_ang=1.0, thr=1.5, ext=1.0, lw=2):
    hole_b = hole.astype(bool)
    ys, xs = np.nonzero(hole_b)
    hs = max(np.ptp(ys), np.ptp(xs)) + 1
    geo = stub_geometry(known, hole)
    n = len(geo)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, di, _ = geo[i]; cj, dj, _ = geo[j]
            v = cj - ci; L = np.linalg.norm(v)
            if L < 1.5: continue
            u = v / L
            mis = (1 - np.dot(di, u)) + (1 - np.dot(dj, -u))
            cost = L / hs + w_ang * mis
            if cost < thr: cands.append((cost, i, j))
    cands.sort()
    used = set(); pairs = []
    for c, i, j in cands:
        if i in used or j in used: continue
        used.add(i); used.add(j); pairs.append((i, j))
    im = Image.new("L", known.shape[::-1], 0); dr = ImageDraw.Draw(im)
    for i, j in pairs:
        _draw(dr, geo[i][0], geo[j][0], lw)
    out = np.asarray(im).copy() > 127
    # extend unmatched stubs straight until they hit restored road or leave the hole
    for i in range(n):
        if i in used or ext <= 0: continue
        c, d, _ = geo[i]
        p = c.copy(); maxlen = ext * hs; t = 0.0; end = c.copy()
        while t < maxlen:
            t += 0.5; p = c + t * d
            y, x = int(round(p[0])), int(round(p[1]))
            if not (0 <= y < hole.shape[0] and 0 <= x < hole.shape[1]) or (not hole_b[y, x] and t > 2):
                break
            end = p
            if t > 3 and out[y, x]:
                break
        im2 = Image.fromarray((out * 255).astype(np.uint8)); dr2 = ImageDraw.Draw(im2)
        _draw(dr2, c, end, lw)
        out = np.asarray(im2) > 127
    return (out & hole_b).astype(np.uint8)


def diffusion(known, hole, iters=200, thr=0.3):
    hole_b = hole.astype(bool)
    u = known.astype(np.float32) * (~hole_b)
    for _ in range(iters):
        nb = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1)) / 4
        u = np.where(hole_b, nb, u)
    return ((u > thr) & hole_b).astype(np.uint8)
