"""Round-5 NumPy baseline, copied from models/tracker.py (no MLX import).
Frozen-network inference only; no new training. Keep math identical to the exhibit.
"""
import numpy as np
H = W = 64
K = 3
CKPT = "checkpoints/r5_selected/tracker"
BG = np.array([0.05, 0.05, 0.07], dtype=np.float32)
R = 3.5
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5

def coords():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([(xx - 31.5) / 32, (yy - 31.5) / 32], -1)

def weights_numpy(prefix=CKPT):
    z = np.load(prefix + ".weights.npz")
    return {k: z[k].astype(np.float32) for k in z.files}

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

class NumpyCar:
    def __init__(self, path="checkpoints/r5_selected/car.npz"):
        with np.load(path, allow_pickle=False) as z:
            theta = z["theta"].astype(np.float32)
        assert theta.shape == (1570,)
        self.ps, offset = [], 0
        for shape in ((13,32),(32,),(32,32),(32,),(32,2),(2,)):
            n = int(np.prod(shape))
            self.ps.append(theta[offset:offset+n].reshape(shape))
            offset += n

    def __call__(self, obs):
        x = obs.astype(np.float32)
        for i in range(0, len(self.ps), 2):
            x = np.tanh(x @ self.ps[i] + self.ps[i+1])
        return x
