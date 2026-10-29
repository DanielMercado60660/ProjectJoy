"""Neural Orchestra drum VAE decoder + style classifier (copied from 26_neural_orchestra/model.py),
in three dialects: MLX (reference), NumPy (baseline) and torch (Core ML conversion frontend only).

Decoder: [z (16) | style one-hot (8)] -> Linear 256 -> GELU -> Linear 512 -> GELU -> Linear 864,
split into hit logits, velocity (sigmoid) and microtiming offset (0.5 tanh), 288 cells each
(32 sixteenth steps x 9 drums). GELU is the exact erf form (MLX nn.gelu, torch nn.GELU()).
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = "../26_neural_orchestra/checkpoints/vae_cond_b1.0/best"   # README: the demo model
CLF = "../26_neural_orchestra/checkpoints/classifier"
STEPS, N_INST, N_STYLES = 32, 9, 8
CELLS = STEPS * N_INST


def load_config(path=CKPT):
    with open(os.path.join(path, "vae_config.json")) as f:
        return json.load(f)


# ---------- MLX (decoder only; same parameter names as the original VAE) ----------
class Decoder(nn.Module):
    def __init__(self, latent=16, hidden=512, **_):
        super().__init__()
        self.dec = [nn.Linear(latent + N_STYLES, hidden // 2), nn.Linear(hidden // 2, hidden)]
        self.out = nn.Linear(hidden, 3 * CELLS)

    def __call__(self, z, s):
        h = mx.concatenate([z, s], 1)
        for l in self.dec:
            h = nn.gelu(l(h))
        o = self.out(h)
        return mx.sigmoid(o[:, :CELLS]), mx.sigmoid(o[:, CELLS:2 * CELLS]), 0.5 * mx.tanh(o[:, 2 * CELLS:])


def weights(path=CKPT):
    return {k: np.array(v, dtype=np.float32) for k, v in mx.load(os.path.join(path, "vae.safetensors")).items()}


def load_mlx(dtype=mx.float32, path=CKPT):
    cfg = load_config(path)
    assert cfg["kind"] == "mlp" and cfg["cond"]
    m = Decoder(cfg["latent"], cfg["hidden"])
    w = {k: mx.array(v) for k, v in weights(path).items() if k.startswith(("dec.", "out."))}
    m.load_weights(list(w.items()))
    if dtype != mx.float32:
        m.set_dtype(dtype)
    m.eval()
    mx.eval(m.parameters())
    return m


# ---------- NumPy ----------
def _gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)).astype(np.float32))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyDecoder:
    def __init__(self, path=CKPT):
        w = weights(path)
        self.l = [(np.ascontiguousarray(w[f"{n}.weight"].T), w[f"{n}.bias"]) for n in ("dec.0", "dec.1", "out")]

    def __call__(self, z, s):
        h = np.concatenate([z, s], 1)
        for W, b in self.l[:2]:
            h = _gelu(h @ W + b)
        o = h @ self.l[2][0] + self.l[2][1]
        return _sigmoid(o[:, :CELLS]), _sigmoid(o[:, CELLS:2 * CELLS]), 0.5 * np.tanh(o[:, 2 * CELLS:])


class NumpyClassifier:
    """The project's separate style classifier (eval mode: dropout off), fp32 numpy. Used for the task check."""

    def __init__(self, path=CLF):
        w = {k: np.array(v, np.float32) for k, v in mx.load(os.path.join(path, "classifier.safetensors")).items()}
        self.l = [(np.ascontiguousarray(w[f"{n}.weight"].T), w[f"{n}.bias"]) for n in ("l1", "l2", "l3")]

    def __call__(self, hits, vel, off):
        B = hits.shape[0]
        x = np.concatenate([hits.reshape(B, -1), vel.reshape(B, -1), 2 * off.reshape(B, -1)], 1).astype(np.float32)
        for W, b in self.l[:2]:
            x = _gelu(x @ W + b)
        return x @ self.l[2][0] + self.l[2][1]


def decode_fp64(z, s, path=CKPT):
    """Exact-ish truth: the same decoder in float64 numpy (for judging the fp32 engines themselves)."""
    from scipy.special import erf
    w = weights(path)
    h = np.concatenate([z, s], 1).astype(np.float64)
    for n in ("dec.0", "dec.1"):
        h = h @ w[n + ".weight"].T.astype(np.float64) + w[n + ".bias"]
        h = 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))
    o = h @ w["out.weight"].T.astype(np.float64) + w["out.bias"]
    return 1 / (1 + np.exp(-o[:, :CELLS])), 1 / (1 + np.exp(-o[:, CELLS:2 * CELLS])), 0.5 * np.tanh(o[:, 2 * CELLS:])


# ---------- torch (Core ML frontend only) ----------
def torch_decoder(path=CKPT):
    import torch
    import torch.nn as tnn
    w = weights(path)
    cfg = load_config(path)
    L, H = cfg["latent"], cfg["hidden"]

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.d0 = tnn.Linear(L + N_STYLES, H // 2)
            s.d1 = tnn.Linear(H // 2, H)
            s.out = tnn.Linear(H, 3 * CELLS)
            s.act = tnn.GELU()

        def forward(s, z, style):
            o = s.out(s.act(s.d1(s.act(s.d0(torch.cat([z, style], 1))))))
            return torch.sigmoid(o[:, :CELLS]), torch.sigmoid(o[:, CELLS:2 * CELLS]), 0.5 * torch.tanh(o[:, 2 * CELLS:])

    net = Net().eval()
    with torch.no_grad():
        for t, n in ((net.d0, "dec.0"), (net.d1, "dec.1"), (net.out, "out")):
            t.weight.copy_(torch.from_numpy(w[f"{n}.weight"]))
            t.bias.copy_(torch.from_numpy(w[f"{n}.bias"]))
    return net


# ---------- inputs and the project's pattern rule ----------
def random_latents(n, seed=0):
    """z ~ N(0, I) like make_demo.vae_samples, styles cycled through all 8 (one-hot float32)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, load_config()["latent"])).astype(np.float32)
    style = np.arange(n) % N_STYLES
    return z, np.eye(N_STYLES, dtype=np.float32)[style], style


def to_pattern(p, vel, off, mode, seed):
    """decode_numpy's rule: hits = p > 0.5 ('threshold') or Bernoulli(p) with default_rng(seed) ('sample',
    the demo mode); velocity and offset are zeroed where there is no hit. Same uniforms for every athlete."""
    p = np.asarray(p, np.float32)
    if mode == "threshold":
        h = (p > 0.5).astype(np.uint8)
    else:
        h = (np.random.default_rng(seed).random(p.shape) < p).astype(np.uint8)
    B = h.shape[0]
    h = h.reshape(B, STEPS, N_INST)
    return h, np.asarray(vel).reshape(B, STEPS, N_INST) * h, np.asarray(off).reshape(B, STEPS, N_INST) * h
