"""Physics Detective amortized posterior network (11_physics_detective/sbi.py PosteriorNet): a Deep-Sets
network that reads a set of up to 15 push experiments (design + 20 camera frames each) and outputs an
8-component Gaussian-mixture posterior over z = logit((mu_k - 0.05) / 0.75), mu_k = hidden friction.

  feats (B, 15, 42) float32, mask (B, 15) float32 (1 = experiment present)
  -> logw (B, 8) normalised log mixture weights, loc (B, 8), scale (B, 8)

Encoder 42-192-192-128 (GELU), masked sum pooling / 4, concat log(n), head 129-192-192-24 (GELU).
Dialects: MLX (reference), NumPy (baseline; GELU via scipy.special.erf), torch (Core ML frontend only).
Also vendored: the analytic world-1 simulator + camera (physics.py) and featurize/sample_designs
(sbi.py), to generate realistic inputs, and the grid posterior summary (sbi.posterior_on_grid).
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.special import erf

CKPT = "../11_physics_detective/checkpoints/sbi"
MAXN = 15
G, SIGMA, VIEW = 9.81, 0.08, 1.0
TIMES = np.round(np.arange(1, 21) * 0.1, 10)
NT = len(TIMES)
FEAT = 2 + 2 * NT
MU_RATIO, MU_LO, MU_HI = 1.2, 0.05, 0.8
F_GRID = np.linspace(1.0, 15.0, 15)
T_GRID = np.round(np.linspace(0.1, 1.0, 10), 3)
DESIGNS = np.array([(f, t) for f in F_GRID for t in T_GRID])
GRID = np.linspace(MU_LO, MU_HI, 1500)   # inference.py GRID


# ---------- vendored data generation (physics.py + sbi.py) ----------
def positions(F, T, mu_k, mu_s, m=1.0, k=1.0, times=TIMES):
    F, T, mu_k, mu_s, m, k = [np.asarray(a, dtype=np.float64)[..., None] for a in (F, T, mu_k, mu_s, m, k)]
    t = np.asarray(times, dtype=np.float64)
    moves = k * F > mu_s * m * G
    a1 = np.where(moves, k * F / m - mu_k * G, 0.0)
    a2 = mu_k * G
    v1 = a1 * T
    x1 = 0.5 * a1 * T ** 2
    t_stop = v1 / a2
    tau = np.clip(t - T, 0.0, t_stop)
    x_push = 0.5 * a1 * np.minimum(t, T) ** 2
    x_after = x1 + v1 * tau - 0.5 * a2 * tau ** 2
    return np.where(t <= T, x_push, x_after)


def observe(x, rng):
    y = x + SIGMA * rng.standard_normal(np.shape(x))
    cens = y > VIEW
    return np.where(cens, np.nan, y), cens


def world1_positions(F, T, mu_k):
    return positions(F, T, mu_k, MU_RATIO * np.asarray(mu_k))


def featurize(designs, y, cens):
    yv = np.where(cens, 0.0, np.nan_to_num(y, nan=0.0))
    return np.concatenate([designs[..., :1] / 15.0, designs[..., 1:2], yv, cens.astype(np.float64)], -1).astype(np.float32)


def sample_designs(rng, B, N):
    disc = DESIGNS[rng.integers(len(DESIGNS), size=(B, N))]
    cont = np.stack([rng.uniform(1, 15, (B, N)), rng.uniform(0.1, 1.0, (B, N))], -1)
    d = np.where(rng.random((B, 1, 1)) < 0.5, disc, cont)
    rep = rng.random(B) < 0.3
    k = rng.integers(1, 4, size=B)
    idx = np.minimum(rng.integers(0, 3, size=(B, N)), k[:, None] - 1)
    d_rep = np.take_along_axis(d, idx[..., None].repeat(2, -1), 1)
    return np.where(rep[:, None, None], d_rep, d)


def fresh_datasets(B, seed):
    """Same recipe as eval_sbi.py's 'fresh random-design datasets' (mu ~ U(0.05, 0.8), 1-15 pushes)."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.8, B)
    d = sample_designs(rng, B, MAXN)
    y, c = observe(world1_positions(d[..., 0], d[..., 1], mu[:, None]), rng)
    n = rng.integers(1, MAXN + 1, B)
    mask = (np.arange(MAXN)[None] < n[:, None]).astype(np.float32)
    return featurize(d, y, c), mask, mu


def grid_summary(logw, loc, scale, grid=GRID):
    """sbi.posterior_on_grid on host (float64), then posterior mean and sd of mu_k."""
    logw, loc, scale = (np.asarray(a, np.float64) for a in (logw, loc, scale))
    p = np.clip((grid - MU_LO) / (MU_HI - MU_LO), 1e-6, 1 - 1e-6)
    z = np.log(p) - np.log1p(-p)
    log_jac = -np.log((MU_HI - MU_LO) * p * (1 - p))
    zz = (z[None, None, :] - loc[..., None]) / scale[..., None]
    logpdf = np.logaddexp.reduce(logw[..., None] - 0.5 * zz ** 2 - np.log(scale[..., None]), axis=1) + log_jac
    w = np.exp(logpdf - logpdf.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    mean = w @ grid
    sd = np.sqrt(np.maximum(w @ grid ** 2 - mean ** 2, 0))
    return mean, sd


# ---------- MLX ----------
class PosteriorNet(nn.Module):
    def __init__(self, hidden=192, embed=128, comps=8, **_):
        super().__init__()
        self.enc = [nn.Linear(FEAT, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, embed)]
        self.head = [nn.Linear(embed + 1, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 3 * comps)]
        self.comps = comps

    def __call__(self, feats, mask):
        h = feats
        for i, l in enumerate(self.enc):
            h = l(h)
            if i < len(self.enc) - 1:
                h = nn.gelu(h)
        n = mask.sum(1, keepdims=True)
        pooled = (h * mask[..., None]).sum(1) / 4.0
        h = mx.concatenate([pooled, mx.log(n)], -1)
        for i, l in enumerate(self.head):
            h = l(h)
            if i < len(self.head) - 1:
                h = nn.gelu(h)
        logit_w, loc, raw_s = mx.split(h, 3, axis=-1)
        scale = nn.softplus(raw_s - 2.0) + 1e-3
        return logit_w - mx.logsumexp(logit_w, axis=-1, keepdims=True), loc, scale


def config(ckpt=CKPT):
    return json.load(open(os.path.join(ckpt, "config.json")))


def load_mlx(ckpt=CKPT):
    m = PosteriorNet(**config(ckpt))
    m.load_weights(os.path.join(ckpt, "model.safetensors"))
    m.eval()
    return m


def weights_numpy(ckpt=CKPT):
    return {k: np.array(v, np.float32) for k, v in mx.load(os.path.join(ckpt, "model.safetensors")).items()}


# ---------- NumPy ----------
def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)).astype(np.float32))


class NumpyNet:
    def __init__(self, ckpt=CKPT):
        w = weights_numpy(ckpt)
        self.enc = [(w[f"enc.{i}.weight"].T.copy(), w[f"enc.{i}.bias"]) for i in range(3)]
        self.head = [(w[f"head.{i}.weight"].T.copy(), w[f"head.{i}.bias"]) for i in range(3)]
        self.comps = config(ckpt)["comps"]

    def __call__(self, feats, mask):
        B, N, F = feats.shape
        h = feats.reshape(B * N, F)
        for i, (W, b) in enumerate(self.enc):
            h = h @ W + b
            if i < 2:
                h = _gelu(h)
        h = h.reshape(B, N, -1)
        n = mask.sum(1, keepdims=True)
        pooled = (h * mask[..., None]).sum(1) / np.float32(4.0)
        h = np.concatenate([pooled, np.log(n)], -1)
        for i, (W, b) in enumerate(self.head):
            h = h @ W + b
            if i < 2:
                h = _gelu(h)
        c = self.comps
        lw, loc, raw = h[:, :c], h[:, c:2 * c], h[:, 2 * c:]
        mx_ = lw.max(-1, keepdims=True)
        lse = mx_ + np.log(np.exp(lw - mx_).sum(-1, keepdims=True))
        scale = np.logaddexp(0, raw - 2.0) + np.float32(1e-3)
        return (lw - lse).astype(np.float32), loc, scale.astype(np.float32)


# ---------- torch (Core ML frontend only) ----------
def torch_net(ckpt=CKPT):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as Fn
    cfg = config(ckpt)
    H, E, C = cfg["hidden"], cfg["embed"], cfg["comps"]
    w = weights_numpy(ckpt)

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.enc = tnn.ModuleList([tnn.Linear(FEAT, H), tnn.Linear(H, H), tnn.Linear(H, E)])
            s.head = tnn.ModuleList([tnn.Linear(E + 1, H), tnn.Linear(H, H), tnn.Linear(H, 3 * C)])

        def forward(s, feats, mask):
            h = feats
            for i, l in enumerate(s.enc):
                h = l(h)
                if i < 2:
                    h = Fn.gelu(h)
            n = mask.sum(1, keepdim=True)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / 4.0
            h = torch.cat([pooled, torch.log(n)], -1)
            for i, l in enumerate(s.head):
                h = l(h)
                if i < 2:
                    h = Fn.gelu(h)
            lw, loc, raw = h[:, :C], h[:, C:2 * C], h[:, 2 * C:]
            return lw - torch.logsumexp(lw, -1, keepdim=True), loc, Fn.softplus(raw - 2.0) + 1e-3

    net = Net().eval()
    sd = {k: torch.from_numpy(v.copy()) for k, v in w.items()}
    net.load_state_dict(sd)
    return net
