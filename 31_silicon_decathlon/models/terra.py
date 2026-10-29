"""Terra learned A* heuristic (copied from 07_terra/terra/learn.py + the cost from terra/cost.py), in three
dialects: MLX (reference), NumPy (baseline) and torch (Core ML conversion frontend only).

HNet: U-Net, base 32, 4 max-pool downsamplings (96 -> 6) + a global-context Linear at the bottleneck,
GroupNorm(8) + SiLU blocks. Input (B, 96, 96, 8) NHWC: 6 static map channels + 2 goal channels.
Output r (B, 96, 96), a log-ratio over octile: h_learned = octile * exp(max(r, 0)).

Maps come from the sibling's generated validation set (../07_terra/data/val.npz: 150 maps, 600 goals,
static features stored as float16, exact cost-to-go log-ratio labels). Terrain kind and elevation for
A* are reconstructed from those features (elevation is the stored float16 value).
"""
import heapq
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = "../07_terra/checkpoints/hnet"        # README: hnet.* is the evaluated heuristic (base 32)
VAL = "../07_terra/data/val.npz"
SIZE, N_IN = 96, 8
GRASS, FOREST, ROCK, WATER, BRIDGE = 0, 1, 2, 3, 4
MULT = {GRASS: 1.0, FOREST: 2.0, ROCK: 3.0, WATER: math.inf, BRIDGE: 1.0}
HEIGHT_SCALE, SQRT2 = 24.0, math.sqrt(2.0)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------- data + features ----------
_val = {}


def val():
    if not _val:
        d = np.load(VAL)
        _val.update({k: d[k] for k in ("static", "map_index", "goal", "target", "mask")})
    return _val


def octile_field(h, w, gr, gc):
    rr, cc = np.mgrid[0:h, 0:w]
    dr, dc = np.abs(rr - gr), np.abs(cc - gc)
    return np.maximum(dr, dc) + (SQRT2 - 1.0) * np.minimum(dr, dc)


def goal_features(h, w, gr, gc):
    o = octile_field(h, w, gr, gc)
    return np.stack([o / h, np.exp(-(o ** 2) / 8.0)]).astype(np.float32)


def sample_inputs(idx):
    """(len(idx), 96, 96, 8) float32 NHWC network inputs for validation samples idx."""
    v = val()
    xs = []
    for i in idx:
        st = v["static"][v["map_index"][i]].astype(np.float32)
        gr, gc = v["goal"][i]
        xs.append(np.concatenate([st, goal_features(SIZE, SIZE, gr, gc)]))
    return np.stack(xs).transpose(0, 2, 3, 1).copy()


def map_of(i):
    """(elev, kind) reconstructed from the stored static features of sample i's map."""
    st = val()["static"][val()["map_index"][i]].astype(np.float32)
    kind = np.full(st.shape[1:], GRASS, np.int8)
    kind[st[5] > 0.5] = FOREST
    kind[st[2] > 0.75] = ROCK
    kind[st[4] > 0.5] = BRIDGE
    kind[st[3] > 0.5] = WATER
    return st[0], kind


# ---------- the cost (terra/cost.py) and A* ----------
def edge_costs(elev, kind):
    h, w = kind.shape
    z = elev.astype(np.float64) * HEIGHT_SCALE
    m = np.empty(kind.shape, np.float64)
    for k, v in MULT.items():
        m[kind == k] = v
    passable = np.isfinite(m)
    out = np.full((8, h, w), np.inf)
    for k, (dr, dc) in enumerate(DIRS):
        d = SQRT2 if dr and dc else 1.0
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        a = (slice(r0, r1), slice(c0, c1))
        b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        ok = passable[a] & passable[b]
        if dr and dc:
            ok &= passable[r0 + dr:r1 + dr, c0:c1] & passable[r0:r1, c0 + dc:c1 + dc]
        c = d * 0.5 * (m[a] + m[b]) * (1.0 + np.abs(z[b] - z[a]) / d)
        out[k][a] = np.where(ok, c, np.inf)
    return out


def adjacency(elev, kind):
    cost = edge_costs(elev, kind)
    h, w = kind.shape
    offs = [dr * w + dc for dr, dc in DIRS]
    cl, fin = cost.reshape(8, -1).tolist(), np.isfinite(cost.reshape(8, -1)).tolist()
    return [[(u + offs[k], cl[k][u]) for k in range(8) if fin[k][u]] for u in range(h * w)]


def astar(adj, s, g, hv):
    """A* as in terra/planners.astar (ties toward larger g). Returns (cost, expanded)."""
    n = len(adj)
    gs, closed = [math.inf] * n, bytearray(n)
    gs[s] = 0.0
    pq = [(hv[s], 0.0, s)]
    expanded = 0
    while pq:
        f, ng, u = heapq.heappop(pq)
        if closed[u]:
            continue
        closed[u] = 1
        expanded += 1
        if u == g:
            return gs[g], expanded
        gu = gs[u]
        for v, c in adj[u]:
            nv = gu + c
            if nv < gs[v]:
                gs[v] = nv
                heapq.heappush(pq, (nv + hv[v], -nv, v))
    return math.inf, expanded


# ---------- MLX (same module layout as the original, so weights load by name) ----------
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


def _pool(x):
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def _up(x):
    b, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c)).reshape(b, h * 2, w * 2, c)


class HNet(nn.Module):
    def __init__(self, cin=N_IN, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 4, base * 4]
        self.enc = [Block(cin, ch[0])] + [Block(ch[i], ch[i + 1]) for i in range(4)]
        self.glob = nn.Linear(ch[4], ch[4])
        self.dec = [Block(ch[i + 1] + ch[i], ch[i]) for i in range(4)]
        self.out = nn.Conv2d(ch[0], 1, 1)

    def __call__(self, x):
        skips = []
        for i, b in enumerate(self.enc):
            if i > 0:
                x = _pool(x)
            x = b(x)
            skips.append(x)
        x = x + self.glob(mx.mean(x, axis=(1, 2)))[:, None, None, :]
        for i in range(3, -1, -1):
            x = self.dec[i](mx.concatenate([_up(x), skips[i]], axis=-1))
        return self.out(x)[..., 0]


def config(prefix=CKPT):
    import json
    with open(prefix + ".json") as f:
        return json.load(f)


def load_mlx(prefix=CKPT, dtype=mx.float32):
    m = HNet(base=config(prefix)["base"])
    m.load_weights(prefix + ".safetensors")
    m.eval()
    if dtype != mx.float32:
        m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def weights_numpy(prefix=CKPT):
    return {k: np.array(v, dtype=np.float32) for k, v in mx.load(prefix + ".safetensors").items()}


# ---------- NumPy baseline ----------
def _conv3(x, k, b):
    """x (B,H,W,C), k (O,3,3,C) MLX layout, 'same' padding: im2col + one matmul."""
    B, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols = np.empty((B, H, W, 3, 3, C), np.float32)
    for dy in range(3):
        for dx in range(3):
            cols[:, :, :, dy, dx] = xp[:, dy:dy + H, dx:dx + W]
    return (cols.reshape(B * H * W, 9 * C) @ k.reshape(k.shape[0], -1).T + b).reshape(B, H, W, -1)


def _gn(x, g, b, groups=8, eps=1e-5):
    B, H, W, C = x.shape
    xg = x.reshape(B, H * W, groups, C // groups)
    mu = xg.mean(axis=(1, 3), keepdims=True)
    var = xg.var(axis=(1, 3), keepdims=True)
    return ((xg - mu) / np.sqrt(var + eps)).reshape(B, H, W, C) * g + b


def _silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyHNet:
    def __init__(self, prefix=CKPT):
        self.w = weights_numpy(prefix)

    def _block(self, x, p):
        w = self.w
        x = _silu(_gn(_conv3(x, w[p + "c1.weight"], w[p + "c1.bias"]), w[p + "n1.weight"], w[p + "n1.bias"]))
        return _silu(_gn(_conv3(x, w[p + "c2.weight"], w[p + "c2.bias"]), w[p + "n2.weight"], w[p + "n2.bias"]))

    def __call__(self, x):
        w = self.w
        skips = []
        for i in range(5):
            if i > 0:
                B, H, W, C = x.shape
                x = x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))
            x = self._block(x, f"enc.{i}.")
            skips.append(x)
        x = x + (x.mean(axis=(1, 2)) @ w["glob.weight"].T + w["glob.bias"])[:, None, None, :]
        for i in range(3, -1, -1):
            x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
            x = self._block(np.concatenate([x, skips[i]], axis=-1), f"dec.{i}.")
        B, H, W, C = x.shape
        y = x.reshape(-1, C) @ w["out.weight"][:, 0, 0, :].T + w["out.bias"]
        return y.reshape(B, H, W).astype(np.float32)


# ---------- torch (Core ML frontend only) ----------
def torch_net(prefix=CKPT):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    base = config(prefix)["base"]
    w = weights_numpy(prefix)
    ch = [base, base * 2, base * 4, base * 4, base * 4]

    class TBlock(tnn.Module):
        def __init__(s, cin, cout):
            super().__init__()
            s.c1, s.c2 = tnn.Conv2d(cin, cout, 3, padding=1), tnn.Conv2d(cout, cout, 3, padding=1)
            s.n1, s.n2 = tnn.GroupNorm(8, cout), tnn.GroupNorm(8, cout)

        def forward(s, x):
            return F.silu(s.n2(s.c2(F.silu(s.n1(s.c1(x))))))

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.enc = tnn.ModuleList([TBlock(N_IN, ch[0])] + [TBlock(ch[i], ch[i + 1]) for i in range(4)])
            s.glob = tnn.Linear(ch[4], ch[4])
            s.dec = tnn.ModuleList([TBlock(ch[i + 1] + ch[i], ch[i]) for i in range(4)])
            s.out = tnn.Conv2d(ch[0], 1, 1)

        def forward(s, x):  # NHWC in like MLX, (B, H, W) out
            x = x.permute(0, 3, 1, 2)
            skips = []
            for i, b in enumerate(s.enc):
                if i > 0:
                    x = F.max_pool2d(x, 2)
                x = b(x)
                skips.append(x)
            x = x + s.glob(x.mean(dim=(2, 3)))[:, :, None, None]
            for i in range(3, -1, -1):
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = s.dec[i](torch.cat([x, skips[i]], dim=1))
            return s.out(x)[:, 0]

    net = Net().eval()
    sd = net.state_dict()
    for k in sd:
        a = w[k]
        sd[k] = torch.from_numpy((a.transpose(0, 3, 1, 2) if a.ndim == 4 else a).copy())
    net.load_state_dict(sd)
    return net
