"""Learned heuristic: a small MLX U-Net that predicts the cost-to-go field to a goal.

Input  (per map + goal): terrain features + goal channels (see `features`).
Output: r(s) >= 0, a log-ratio over the admissible octile heuristic:
            h_learned(s) = h_octile(s) * exp(r(s))
Clipping r at 0 is free information: the true cost-to-go is never below octile.
The learned h is NOT admissible; A* with it trades path optimality for fewer expansions.
"""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from . import cost
from .terrain import WATER, BRIDGE, FOREST

N_STATIC = 6
N_IN = N_STATIC + 2


def static_features(elev, kind):
    """Map-only channels, (6, H, W) float32."""
    e = elev.astype(np.float32)
    gy, gx = np.gradient(e * cost.HEIGHT_SCALE)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    m = cost.mult_map(kind)
    water = (kind == WATER).astype(np.float32)
    return np.stack([
        e,
        np.tanh(slope / 2.0),
        np.where(np.isfinite(m), (m - 1.0) / 2.0, 0.0),   # 0 grass/bridge, .5 forest, 1 rock
        water,
        (kind == BRIDGE).astype(np.float32),
        (kind == FOREST).astype(np.float32),
    ]).astype(np.float32)


def goal_features(h, w, gr, gc):
    """Goal channels, (2, H, W): octile distance / size, and a goal blob."""
    oct_ = cost.octile_field(h, w, gr, gc)
    blob = np.exp(-(oct_ ** 2) / 8.0)
    return np.stack([oct_ / h, blob]).astype(np.float32)


def sparse_graph(elev, kind):
    """scipy CSR matrix of the cost graph (same costs as cost.Graph)."""
    c = cost.edge_costs(elev, kind)
    h, w = kind.shape
    rows, cols, vals = [], [], []
    idx = np.arange(h * w).reshape(h, w)
    for k, (dr, dc) in enumerate(cost.DIRS):
        fin = np.isfinite(c[k])
        r, cc = np.nonzero(fin)
        rows.append(idx[r, cc]); cols.append(idx[r + dr, cc + dc]); vals.append(c[k][r, cc])
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    return csr_matrix((vals, (rows, cols)), shape=(h * w, h * w))


def cost_to_go(elev, kind, goal_rc, M=None):
    if M is None:
        M = sparse_graph(elev, kind)
    h, w = kind.shape
    return sp_dijkstra(M, indices=goal_rc[0] * w + goal_rc[1]).reshape(h, w)


def target_logratio(ctg, gr, gc):
    h, w = ctg.shape
    o = cost.octile_field(h, w, gr, gc)
    mask = np.isfinite(ctg) & (o > 0)
    t = np.zeros_like(ctg, dtype=np.float32)
    t[mask] = np.log(np.maximum(ctg[mask] / o[mask], 1.0))
    return t, mask


# ---------------------------------------------------------------------------------------
# Model (NHWC, as MLX convs expect)
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout, pytorch_compatible=True)
        self.n2 = nn.GroupNorm(8, cout, pytorch_compatible=True)

    def __call__(self, x):
        x = nn.silu(self.n1(self.c1(x)))
        return nn.silu(self.n2(self.c2(x)))


class HNet(nn.Module):
    """U-Net, 4 downsamplings (96 -> 6) plus a global-context vector at the bottleneck."""

    def __init__(self, cin=N_IN, base=32, cout=1):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 4, base * 4]
        self.enc = [Block(cin, ch[0])] + [Block(ch[i], ch[i + 1]) for i in range(4)]
        self.glob = nn.Linear(ch[4], ch[4])
        self.dec = [Block(ch[i + 1] + ch[i], ch[i]) for i in range(4)]
        self.out = nn.Conv2d(ch[0], cout, 1)
        self.cout = cout

    def __call__(self, x):
        skips = []
        for i, b in enumerate(self.enc):
            if i > 0:
                x = _pool(x)
            x = b(x)
            skips.append(x)
        g = self.glob(mx.mean(x, axis=(1, 2)))
        x = x + g[:, None, None, :]
        for i in range(3, -1, -1):
            x = _up(x)
            x = self.dec[i](mx.concatenate([x, skips[i]], axis=-1))
        y = self.out(x)
        return y[..., 0] if self.cout == 1 else y


def _pool(x):
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def _up(x):
    b, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
    return x.reshape(b, h * 2, w * 2, c)


def predict_h(model, elev, kind, gr, gc, static=None):
    """Learned heuristic field (H, W) for one map+goal. Returns (h_learned, h_octile)."""
    h, w = kind.shape
    if static is None:
        static = static_features(elev, kind)
    x = np.concatenate([static, goal_features(h, w, gr, gc)])[None].transpose(0, 2, 3, 1)
    r = model(mx.array(x))
    mx.eval(r)
    r = np.maximum(np.array(r[0], dtype=np.float64), 0.0)
    o = cost.octile_field(h, w, gr, gc)
    return o * np.exp(r), o


def export_js(model, base):
    """Flatten weights to one float16 buffer (base64) + an index, for web/hnet.js."""
    import base64
    from mlx.utils import tree_flatten
    index, chunks, off = [], [], 0
    for name, v in tree_flatten(model.parameters()):
        a = np.array(v, dtype=np.float32).astype("<f2").ravel()
        index.append([name, list(v.shape), off])
        chunks.append(a); off += a.size
    data = np.concatenate(chunks)
    return {"base": base, "index": index, "data": base64.b64encode(data.tobytes()).decode()}
