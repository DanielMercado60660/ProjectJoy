"""Object-permanence tracker (copied from 15_object_permanence/model.py + the renderer from sim.py),
as a single *frame step*: (frame, GRU hidden state) -> (Gaussian head output, new hidden state).

Three dialects: MLX (reference; same module layout as the original so weights load by name),
NumPy (baseline) and torch (Core ML conversion frontend only; GRU cell written with plain matmuls).

frame (B, 64, 64, 3) float32 in [0, 1]; h (B, 256).
out (B, K=3, 5): raw head per ball slot = mu_x, mu_y (normalized), log_sx, log_sy, rho_raw.
"""
import pickle

import mlx.core as mx
import mlx.nn as nn
import numpy as np

H = W = 64
K = 3
POS_SCALE = 32.0
LOG_S_MIN, LOG_S_MAX = -5.5, 1.0
RHO_MAX = 0.95
SRC = "../15_object_permanence"
CKPT = SRC + "/checkpoints/base/final"     # README/RESULTS headline model (base world)
DATA = SRC + "/data"


def coords():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([(xx - 31.5) / 32, (yy - 31.5) / 32], -1)   # (H, W, 2)


def load_cfg(prefix=CKPT):
    with open(prefix + ".state.pkl", "rb") as f:
        return pickle.load(f)["cfg"]


# ---------- MLX ----------
class Tracker(nn.Module):
    """Same layout and math as the original Tracker (1-layer GRU)."""

    def __init__(self, K=3, enc_dim=512, hid=256, layers=1):
        super().__init__()
        self.K = K
        self.conv1 = nn.Conv2d(5, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.fc = nn.Linear(8 * 8 * 64, enc_dim)
        self.grus = [nn.GRU(enc_dim if i == 0 else hid, hid) for i in range(layers)]
        self.head1 = nn.Linear(hid + enc_dim, 256)
        self.head2 = nn.Linear(256, K * 5)
        self._coords = mx.array(coords())

    def encode_frame(self, x):
        """x (B,H,W,3) -> (B,enc_dim)"""
        B = x.shape[0]
        x = mx.concatenate([x, mx.broadcast_to(self._coords.astype(x.dtype), (B, H, W, 2))], -1)
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))
        return nn.relu(self.fc(x.reshape(B, -1)))

    def step(self, frame, h):
        """One frame step: the GRU cell of mlx.nn.GRU written out (identical math)."""
        e = self.encode_frame(frame)
        g = self.grus[0]
        n_h = g.Wh.shape[1]
        x = e @ g.Wx.T + g.b
        hp = h @ g.Wh.T
        rz = mx.sigmoid(x[:, :2 * n_h] + hp[:, :2 * n_h])
        r, z = rz[:, :n_h], rz[:, n_h:]
        n = mx.tanh(x[:, 2 * n_h:] + r * (hp[:, 2 * n_h:] + g.bhn))
        h = (1 - z) * n + z * h
        out = self.head2(nn.relu(self.head1(mx.concatenate([h, e], -1))))
        return out.reshape(-1, self.K, 5), h

    def __call__(self, frames):
        """Whole sequence like the original: frames (B,T,H,W,3) -> raw (B,T,K,5)."""
        B, T = frames.shape[:2]
        e = self.encode_frame(frames.reshape(B * T, H, W, 3)).reshape(B, T, -1)
        h = e
        for g in self.grus:
            h = g(h)
        out = self.head2(nn.relu(self.head1(mx.concatenate([h, e], -1))))
        return out.reshape(B, T, self.K, 5)


def load_mlx(prefix=CKPT):
    cfg = load_cfg(prefix)
    assert cfg["layers"] == 1, "step() is written for the 1-layer GRU the README checkpoints use"
    m = Tracker(K=K, enc_dim=cfg["enc_dim"], hid=cfg["hid"], layers=cfg["layers"])
    m.load_weights(prefix + ".weights.npz")
    mx.eval(m.parameters())
    return m, cfg


def weights_numpy(prefix=CKPT):
    z = np.load(prefix + ".weights.npz")
    return {k: z[k].astype(np.float32) for k in z.files}


# ---------- NumPy baseline ----------
def _conv(x, k, b, stride, pad):
    """x (B,H,W,C), k (O,kh,kw,C) MLX layout -> (B,Ho,Wo,O)."""
    kh = k.shape[1]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kh), axis=(1, 2))[:, ::stride, ::stride]
    B, Ho, Wo = win.shape[:3]
    cols = win.transpose(0, 1, 2, 4, 5, 3).reshape(B * Ho * Wo, -1)
    return (cols @ k.reshape(k.shape[0], -1).T + b).reshape(B, Ho, Wo, -1)


class NumpyTracker:
    def __init__(self, prefix=CKPT):
        self.w = weights_numpy(prefix)
        self.coords = coords()
        self.hid = self.w["grus.0.Wh"].shape[1]

    def step(self, frame, h):
        w, relu = self.w, (lambda a: np.maximum(a, 0))
        B = frame.shape[0]
        x = np.concatenate([frame, np.broadcast_to(self.coords, (B, H, W, 2))], -1)
        for name, s, p in (("conv1", 2, 2), ("conv2", 2, 1), ("conv3", 2, 1), ("conv4", 1, 1)):
            x = relu(_conv(x, w[name + ".weight"], w[name + ".bias"], s, p))
        e = relu(x.reshape(B, -1) @ w["fc.weight"].T + w["fc.bias"])
        n_h = self.hid
        gx = e @ w["grus.0.Wx"].T + w["grus.0.b"]
        hp = h @ w["grus.0.Wh"].T
        rz = 1 / (1 + np.exp(-(gx[:, :2 * n_h] + hp[:, :2 * n_h])))
        r, z = rz[:, :n_h], rz[:, n_h:]
        n = np.tanh(gx[:, 2 * n_h:] + r * (hp[:, 2 * n_h:] + w["grus.0.bhn"]))
        h = (1 - z) * n + z * h
        o = relu(np.concatenate([h, e], -1) @ w["head1.weight"].T + w["head1.bias"])
        o = o @ w["head2.weight"].T + w["head2.bias"]
        return o.reshape(B, K, 5).astype(np.float32), h.astype(np.float32)


# ---------- torch (Core ML frontend only) ----------
def torch_step(batch, prefix=CKPT):
    """torch module (frame (B,64,64,3), h (B,hid)) -> (out (B,K*5), h_new). Static batch."""
    import torch
    import torch.nn as tnn
    w = weights_numpy(prefix)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))

    class Step(tnn.Module):
        def __init__(s):
            super().__init__()
            s.convs = tnn.ModuleList()
            for name, cin, cout, k, st, p in (("conv1", 5, 32, 5, 2, 2), ("conv2", 32, 64, 3, 2, 1),
                                              ("conv3", 64, 64, 3, 2, 1), ("conv4", 64, 64, 3, 1, 1)):
                c = tnn.Conv2d(cin, cout, k, stride=st, padding=p)
                c.weight.data = t(w[name + ".weight"].transpose(0, 3, 1, 2))
                c.bias.data = t(w[name + ".bias"])
                s.convs.append(c)
            s.register_buffer("coords", t(np.repeat(coords().transpose(2, 0, 1)[None], batch, 0)))
            for k in ("fc.weight", "fc.bias", "grus.0.Wx", "grus.0.Wh", "grus.0.b", "grus.0.bhn",
                      "head1.weight", "head1.bias", "head2.weight", "head2.bias"):
                s.register_buffer(k.replace(".", "_"), t(w[k]))
            s.n_h = w["grus.0.Wh"].shape[1]

        def forward(s, frame, h):
            x = torch.cat([frame.permute(0, 3, 1, 2), s.coords], 1)
            for c in s.convs:
                x = torch.relu(c(x))
            x = x.permute(0, 2, 3, 1).reshape(-1, 8 * 8 * 64)       # NHWC flatten, like MLX
            e = torch.relu(x @ s.fc_weight.T + s.fc_bias)
            n_h = s.n_h
            gx = e @ s.grus_0_Wx.T + s.grus_0_b
            hp = h @ s.grus_0_Wh.T
            rz = torch.sigmoid(gx[:, :2 * n_h] + hp[:, :2 * n_h])
            r, z = rz[:, :n_h], rz[:, n_h:]
            n = torch.tanh(gx[:, 2 * n_h:] + r * (hp[:, 2 * n_h:] + s.grus_0_bhn))
            h2 = (1 - z) * n + z * h
            o = torch.relu(torch.cat([h2, e], 1) @ s.head1_weight.T + s.head1_bias)
            return o @ s.head2_weight.T + s.head2_bias, h2

    return Step().eval()


# ---------- outputs in pixels ----------
def to_pixels(out):
    """raw (...,5) -> mean (...,2) px, sigma (...,2) px, rho (...)"""
    out = np.asarray(out, np.float64)
    mu = out[..., :2] * POS_SCALE + 32.0
    s = np.exp(np.clip(out[..., 2:4], LOG_S_MIN, LOG_S_MAX)) * POS_SCALE
    return mu, s, RHO_MAX * np.tanh(out[..., 4])


# ---------- recorded sequences (renderer copied from sim.py; state read from the sibling's data/) ----------
BG = np.array([0.05, 0.05, 0.07], dtype=np.float32)
R = 3.5
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5


def render(pos, present, occ, colors, noise=0.02, rng=None):
    B, Tn = pos.shape[:2]
    img = np.broadcast_to(BG, (B, Tn, H, W, 3)).copy()
    for k in range(pos.shape[2]):
        dx = _XX - pos[:, :, k, 0, None, None]
        dy = _YY - pos[:, :, k, 1, None, None]
        a = np.clip(R + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1)
        a *= present[:, None, k, None, None]
        img = img * (1 - a[..., None]) + a[..., None] * colors[:, None, None, None, k]
    for m in range(occ.shape[1]):
        x0, x1 = occ[:, m, 0], occ[:, m, 1]
        cover = np.clip(np.minimum(_XX[0] + 0.5, x1[:, None]) - np.maximum(_XX[0] - 0.5, x0[:, None]), 0, 1)
        cover *= (x0 >= 0)[:, None]
        cover = cover[:, None, None, :, None]
        img = img * (1 - cover) + cover * np.float32(0.45)
    if noise > 0:
        rng = rng or np.random.default_rng()
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


_split_cache = {}


def recorded(split, idx):
    """Frames (N,T,H,W,3) exactly as the original evaluation renders them, plus the split's state."""
    if split not in _split_cache:
        z = np.load(f"{DATA}/{split}.npz")
        _split_cache[split] = {k: z[k] for k in z.files}
    d = _split_cache[split]
    frames = []
    for i in idx:
        rng = np.random.default_rng(int(d["noise_seed"]) * 100003 + int(i))
        frames.append(render(d["pos"][i:i + 1], d["present"][i:i + 1], d["occ"][i:i + 1],
                             d["colors"][i:i + 1], 0.02, rng))
    sel = {k: d[k][list(idx)] for k in ("pos", "present", "vis")}
    return np.concatenate(frames), sel
