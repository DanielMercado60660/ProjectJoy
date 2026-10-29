"""Tiny AlphaZero policy/value net (copied from 09_tiny_alphazero/net.py), in three dialects:
MLX (reference), NumPy (baseline) and torch (Core ML conversion frontend only).

Input (B, 6, 7, 2) NHWC: plane 0 = stones of the player to move, plane 1 = opponent, row 0 = bottom.
Outputs: policy logits (B, 7), value (B,).
"""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

H, W = 6, 7
CKPT = "../09_tiny_alphazero/checkpoints/main/gen_191"   # README: gen_191 is the final net


# ---------- MLX (same module layout as the original, so weights load by name) ----------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm(ch)

    def __call__(self, x):
        y = nn.relu(self.b1(self.c1(x)))
        return nn.relu(x + self.b2(self.c2(y)))


class PVNet(nn.Module):
    def __init__(self, channels=64, blocks=5, value_hidden=64):
        super().__init__()
        self.stem = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm(channels)
        self.blocks = [ResBlock(channels) for _ in range(blocks)]
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm(2)
        self.p_fc = nn.Linear(2 * H * W, W)
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm(1)
        self.v_fc1 = nn.Linear(H * W, value_hidden)
        self.v_fc2 = nn.Linear(value_hidden, 1)

    def __call__(self, x):
        x = nn.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        p = nn.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.reshape(p.shape[0], -1))
        v = nn.relu(self.v_bn(self.v_conv(x)))
        v = nn.relu(self.v_fc1(v.reshape(v.shape[0], -1)))
        v = mx.tanh(self.v_fc2(v)).squeeze(-1)
        return p, v


def load_mlx(prefix=CKPT):
    with open(prefix + ".json") as f:
        cfg = json.load(f)["config"]
    m = PVNet(**cfg)
    m.load_weights(prefix + ".safetensors")
    m.eval()
    return m, cfg


def weights_numpy(prefix=CKPT):
    return {k: np.array(v, dtype=np.float32) for k, v in mx.load(prefix + ".safetensors").items()}


# ---------- NumPy baseline (BatchNorm folded into the convs) ----------
def _fold(w, p, name, eps=1e-5):
    scale = w[f"{name}.weight"] / np.sqrt(w[f"{name}.running_var"] + eps)
    shift = w[f"{name}.bias"] - w[f"{name}.running_mean"] * scale
    return scale, shift


def _conv(x, k):
    """x (B,H,W,C), k (O,kh,kw,C), 'same' padding, stride 1."""
    kh = k.shape[1]
    if kh == 1:
        return x @ k[:, 0, 0, :].T
    p = kh // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kh), axis=(1, 2))  # B,H,W,C,kh,kw
    B, Hh, Ww = x.shape[:3]
    cols = win.transpose(0, 1, 2, 4, 5, 3).reshape(B * Hh * Ww, -1)
    out = cols @ k.reshape(k.shape[0], -1).T
    return out.reshape(B, Hh, Ww, -1)


class NumpyNet:
    def __init__(self, prefix=CKPT, blocks=5):
        w = weights_numpy(prefix)
        self.layers = {}
        def conv_bn(conv, bn):
            s, b = _fold(w, None, bn)
            return (w[f"{conv}.weight"] * s[:, None, None, None]).astype(np.float32), b.astype(np.float32)
        self.stem = conv_bn("stem", "stem_bn")
        self.blocks = [(conv_bn(f"blocks.{i}.c1", f"blocks.{i}.b1"), conv_bn(f"blocks.{i}.c2", f"blocks.{i}.b2"))
                       for i in range(blocks)]
        self.p = conv_bn("p_conv", "p_bn")
        self.v = conv_bn("v_conv", "v_bn")
        self.p_fc = (w["p_fc.weight"], w["p_fc.bias"])
        self.v_fc1 = (w["v_fc1.weight"], w["v_fc1.bias"])
        self.v_fc2 = (w["v_fc2.weight"], w["v_fc2.bias"])

    def __call__(self, x):
        relu = lambda a: np.maximum(a, 0)
        x = relu(_conv(x, self.stem[0]) + self.stem[1])
        for (k1, b1), (k2, b2) in self.blocks:
            y = relu(_conv(x, k1) + b1)
            x = relu(x + _conv(y, k2) + b2)
        B = x.shape[0]
        p = relu(_conv(x, self.p[0]) + self.p[1]).reshape(B, -1)
        p = p @ self.p_fc[0].T + self.p_fc[1]
        v = relu(_conv(x, self.v[0]) + self.v[1]).reshape(B, -1)
        v = relu(v @ self.v_fc1[0].T + self.v_fc1[1])
        v = np.tanh(v @ self.v_fc2[0].T + self.v_fc2[1])[:, 0]
        return p, v


# ---------- torch (Core ML frontend only) ----------
def torch_net(prefix=CKPT):
    import torch
    import torch.nn as tnn
    w = weights_numpy(prefix)
    with open(prefix + ".json") as f:
        cfg = json.load(f)["config"]
    C, NB, VH = cfg["channels"], cfg["blocks"], cfg["value_hidden"]

    class Res(tnn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = tnn.Conv2d(C, C, 3, padding=1, bias=False); s.b1 = tnn.BatchNorm2d(C)
            s.c2 = tnn.Conv2d(C, C, 3, padding=1, bias=False); s.b2 = tnn.BatchNorm2d(C)

        def forward(s, x):
            return torch.relu(x + s.b2(s.c2(torch.relu(s.b1(s.c1(x))))))

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.stem = tnn.Conv2d(2, C, 3, padding=1, bias=False); s.stem_bn = tnn.BatchNorm2d(C)
            s.blocks = tnn.ModuleList([Res() for _ in range(NB)])
            s.p_conv = tnn.Conv2d(C, 2, 1, bias=False); s.p_bn = tnn.BatchNorm2d(2); s.p_fc = tnn.Linear(2 * H * W, W)
            s.v_conv = tnn.Conv2d(C, 1, 1, bias=False); s.v_bn = tnn.BatchNorm2d(1)
            s.v_fc1 = tnn.Linear(H * W, VH); s.v_fc2 = tnn.Linear(VH, 1)

        def forward(s, x):  # NHWC in, like MLX; reshape with -1 (a traced shape[0] breaks conversion)
            x = x.permute(0, 3, 1, 2)
            x = torch.relu(s.stem_bn(s.stem(x)))
            for b in s.blocks:
                x = b(x)
            p = torch.relu(s.p_bn(s.p_conv(x))).permute(0, 2, 3, 1).reshape(-1, 2 * H * W)
            v = torch.relu(s.v_bn(s.v_conv(x))).permute(0, 2, 3, 1).reshape(-1, H * W)
            return s.p_fc(p), torch.tanh(s.v_fc2(torch.relu(s.v_fc1(v)))).squeeze(-1)

    net = Net().eval()
    sd = net.state_dict()
    for k in sd:
        if k.endswith("num_batches_tracked"):
            continue
        a = w[k]
        sd[k] = torch.from_numpy((a.transpose(0, 3, 1, 2) if a.ndim == 4 else a).copy())
    net.load_state_dict(sd)
    return net


# ---------- realistic inputs: positions from random legal play ----------
def random_positions(n, seed=0, max_plies=30):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, H, W, 2), np.float32)
    for i in range(n):
        grid = np.zeros((H, W), np.int8)  # 0 empty, 1 first player, 2 second
        plies = int(rng.integers(0, max_plies))
        for t in range(plies):
            legal = [c for c in range(W) if grid[H - 1, c] == 0]
            c = int(rng.choice(legal))
            r = int(np.argmax(grid[:, c] == 0))
            grid[r, c] = 1 + (t % 2)
        me = 1 + (plies % 2)
        out[i, :, :, 0] = grid == me
        out[i, :, :, 1] = (grid != 0) & (grid != me)
    return out
