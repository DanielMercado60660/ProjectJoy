"""Tiny Grand Prix car policy (08_tiny_grand_prix/racing/policy.py): 13 -> 32 -> 32 -> 2, tanh on every
layer (hidden and output). Weights are one flat float32 vector theta (1570,), laid out as
[W0 (13,32), b0, W1 (32,32), b1, W2 (32,2), b2] and applied as h @ W + b.

Three dialects: NumPy (baseline; this is also the original project's own inference path),
MLX (reference), torch (Core ML conversion frontend only).
Input obs (B, 13): 10 wall-distance rays / 60 m, speed / 30, previous steer, previous pedal.
Output action (B, 2): steer, pedal in [-1, 1].
"""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = "../08_tiny_grand_prix/checkpoints/es_best.npz"   # README: *_best.npz is what eval/demos use
OBS_DIM, HIDDEN, ACT_DIM = 13, (32, 32), 2


def shapes():
    dims = (OBS_DIM,) + HIDDEN + (ACT_DIM,)
    out = []
    for a, b in zip(dims[:-1], dims[1:]):
        out += [(a, b), (b,)]
    return out


def load_theta(path=CKPT):
    d = np.load(path, allow_pickle=False)
    return d["theta"].astype(np.float32), json.loads(str(d["meta"]))


def unflatten(theta):
    out, o = [], 0
    for s in shapes():
        k = int(np.prod(s))
        out.append(theta[o:o + k].reshape(s))
        o += k
    return out


# ---------- NumPy ----------
class NumpyPolicy:
    def __init__(self, path=CKPT):
        self.ps = unflatten(load_theta(path)[0])

    def __call__(self, obs):
        h = obs.astype(np.float32)
        for i in range(0, len(self.ps), 2):
            h = np.tanh(h @ self.ps[i] + self.ps[i + 1])
        return h


# ---------- MLX ----------
class MLXPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        dims = (OBS_DIM,) + HIDDEN + (ACT_DIM,)
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def __call__(self, x):
        for l in self.layers:
            x = mx.tanh(l(x))
        return x


def load_mlx(path=CKPT):
    ps = unflatten(load_theta(path)[0])
    m = MLXPolicy()
    m.load_weights([(f"layers.{i // 2}.weight", mx.array(ps[i].T)) for i in range(0, len(ps), 2)]
                   + [(f"layers.{i // 2}.bias", mx.array(ps[i + 1])) for i in range(0, len(ps), 2)])
    m.eval()
    return m


# ---------- torch (Core ML frontend only) ----------
def torch_net(path=CKPT):
    import torch
    import torch.nn as tnn
    ps = unflatten(load_theta(path)[0])

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.l0, s.l1, s.l2 = tnn.Linear(13, 32), tnn.Linear(32, 32), tnn.Linear(32, 2)

        def forward(s, x):
            return torch.tanh(s.l2(torch.tanh(s.l1(torch.tanh(s.l0(x))))))

    net = Net().eval()
    with torch.no_grad():
        for i, l in enumerate((net.l0, net.l1, net.l2)):
            l.weight.copy_(torch.from_numpy(ps[2 * i].T.copy()))
            l.bias.copy_(torch.from_numpy(ps[2 * i + 1].copy()))
    return net
