"""MLX models: a style-conditional MLP VAE over drum grids, and a separate style classifier."""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

STEPS, N_INST, N_STYLES = 32, 9, 8
CELLS = STEPS * N_INST


def to_input(hits, vel, off):
    """[B,32,9] x3 -> [B, 864] float32 (hits, velocity, offset*2 so it spans [-1, 1])."""
    B = hits.shape[0]
    return np.concatenate([hits.reshape(B, -1), vel.reshape(B, -1), 2 * off.reshape(B, -1)], 1).astype(np.float32)


def one_hot(style, n=N_STYLES):
    return np.eye(n, dtype=np.float32)[style]


class VAE(nn.Module):
    kind = "mlp"

    def __init__(self, latent=16, hidden=512, cond=True):
        super().__init__()
        self.latent, self.hidden, self.cond = latent, hidden, cond
        c = N_STYLES if cond else 0
        self.enc = [nn.Linear(3 * CELLS + c, hidden), nn.Linear(hidden, hidden // 2)]
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.dec = [nn.Linear(latent + c, hidden // 2), nn.Linear(hidden // 2, hidden)]
        self.out = nn.Linear(hidden, 3 * CELLS)

    def config(self):
        return dict(kind="mlp", latent=self.latent, hidden=self.hidden, cond=self.cond)

    def encode(self, x, s=None):
        h = mx.concatenate([x, s], 1) if self.cond else x
        for l in self.enc:
            h = nn.gelu(l(h))
        return self.mu(h), mx.clip(self.logvar(h), -8, 4)

    def decode(self, z, s=None):
        """Returns (hit logits, velocity in (0,1), offset in (-0.5, 0.5)), each [B, 288]."""
        h = mx.concatenate([z, s], 1) if self.cond else z
        for l in self.dec:
            h = nn.gelu(l(h))
        o = self.out(h)
        return o[:, :CELLS], mx.sigmoid(o[:, CELLS:2 * CELLS]), 0.5 * mx.tanh(o[:, 2 * CELLS:])

    def decode_tf(self, z, s, x):
        return self.decode(z, s)  # factorised decoder: no dependence on x

    def __call__(self, x, s=None, key=None):
        mu, logvar = self.encode(x, s)
        eps = mx.random.normal(mu.shape, key=key)
        z = mu + mx.exp(0.5 * logvar) * eps
        return self.decode(z, s), mu, logvar


class ARVAE(nn.Module):
    """Same MLP encoder; autoregressive GRU decoder that emits one 16th-note column (9 drums) per step.

    Each step sees the previous column (hits, vel, offset), z and the style, so it can model
    "bar 2 repeats bar 1" and fills, which the factorised MLP decoder cannot when sampling.
    """
    kind = "ar"

    def __init__(self, latent=16, hidden=512, cond=True, rnn=256):
        super().__init__()
        self.latent, self.hidden, self.cond, self.rnn_size = latent, hidden, cond, rnn
        c = N_STYLES if cond else 0
        self.enc = [nn.Linear(3 * CELLS + c, hidden), nn.Linear(hidden, hidden // 2)]
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.init_h = nn.Linear(latent + c, rnn)
        self.rnn = nn.GRU(3 * N_INST + latent + c, rnn)
        self.out = nn.Linear(rnn, 3 * N_INST)

    def config(self):
        return dict(kind="ar", latent=self.latent, hidden=self.hidden, cond=self.cond, rnn=self.rnn_size)

    encode = VAE.encode

    def _cond(self, z, s):
        return mx.concatenate([z, s], 1) if self.cond else z

    def decode_tf(self, z, s, x):
        """Teacher-forced decode. x: [B, 864] model input -> per-cell (logit, vel, off), each [B, 288]."""
        B = x.shape[0]
        cols = mx.concatenate([x[:, :CELLS].reshape(B, STEPS, N_INST), x[:, CELLS:2 * CELLS].reshape(B, STEPS, N_INST),
                               x[:, 2 * CELLS:].reshape(B, STEPS, N_INST)], 2)  # [B, 32, 27], offset already x2
        prev = mx.concatenate([mx.zeros((B, 1, 3 * N_INST)), cols[:, :-1]], 1)
        zc = self._cond(z, s)
        inp = mx.concatenate([prev, mx.broadcast_to(zc[:, None, :], (B, STEPS, zc.shape[1]))], 2)
        h = self.rnn(inp, mx.tanh(self.init_h(zc)))
        o = self.out(h)  # [B, 32, 27]
        logit = o[..., :N_INST].reshape(B, CELLS)
        vel = mx.sigmoid(o[..., N_INST:2 * N_INST]).reshape(B, CELLS)
        off = 0.5 * mx.tanh(o[..., 2 * N_INST:]).reshape(B, CELLS)
        return logit, vel, off

    def __call__(self, x, s=None, key=None):
        mu, logvar = self.encode(x, s)
        eps = mx.random.normal(mu.shape, key=key)
        z = mu + mx.exp(0.5 * logvar) * eps
        return self.decode_tf(z, s, x), mu, logvar

    def generate(self, z, s, mode, rng, temperature=1.0):
        """Step-by-step sampling. Returns numpy hits/vel/off [B,32,9] and the per-cell hit probabilities."""
        B = z.shape[0]
        zc = self._cond(z, s)
        hid = mx.tanh(self.init_h(zc))
        prev = mx.zeros((B, 3 * N_INST))
        H = np.zeros((B, STEPS, N_INST), np.uint8)
        V = np.zeros((B, STEPS, N_INST), np.float32)
        O = np.zeros((B, STEPS, N_INST), np.float32)
        P = np.zeros((B, STEPS, N_INST), np.float32)
        for t in range(STEPS):
            hid = self.rnn(mx.concatenate([prev, zc], 1)[:, None, :], hid)[:, 0]
            o = self.out(hid)
            p = np.array(mx.sigmoid(o[:, :N_INST] / temperature))
            h = (p > 0.5) if mode == "threshold" else (rng.random(p.shape) < p)
            v = np.array(mx.sigmoid(o[:, N_INST:2 * N_INST])) * h
            of = np.array(0.5 * mx.tanh(o[:, 2 * N_INST:])) * h
            H[:, t], V[:, t], O[:, t], P[:, t] = h, v, of, p
            prev = mx.array(np.concatenate([h, v, 2 * of], 1).astype(np.float32))
        return H, V, O, P


def recon_terms(model, x, s, key=None):
    """Per-example (hit BCE, masked vel MSE, masked offset MSE, KL)."""
    (logit, vel, off), mu, logvar = model(x, s, key)
    hits = x[:, :CELLS]
    bce = nn.losses.binary_cross_entropy(logit, hits, with_logits=True, reduction="none").sum(1)
    vmse = (hits * (vel - x[:, CELLS:2 * CELLS]) ** 2).sum(1)
    omse = (hits * (2 * off - x[:, 2 * CELLS:]) ** 2).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - mx.exp(logvar)).sum(1)
    return bce, vmse, omse, kl


def decode_numpy(model, z, style=None, mode="threshold", rng=None, temperature=1.0):
    """Decode latent vectors to (hits, vel, off) numpy grids.

    style may be int labels or a [B, 8] soft style vector (used for interpolation).
    mode: 'threshold' (p > 0.5) or 'sample' (Bernoulli draw, optional temperature on logits).
    """
    z = mx.array(np.asarray(z, np.float32))
    s = None
    if model.cond:
        style = np.asarray(style)
        s = mx.array(one_hot(style) if style.ndim == 1 else style.astype(np.float32))
    if model.kind == "ar":
        return model.generate(z, s, mode, rng or np.random.default_rng(0), temperature)
    logit, vel, off = model.decode(z, s)
    p = np.array(mx.sigmoid(logit / temperature))
    if mode == "threshold":
        h = (p > 0.5).astype(np.uint8)
    else:
        rng = rng or np.random.default_rng(0)
        h = (rng.random(p.shape) < p).astype(np.uint8)
    B = h.shape[0]
    h = h.reshape(B, STEPS, N_INST)
    v = np.array(vel).reshape(B, STEPS, N_INST) * h
    o = np.array(off).reshape(B, STEPS, N_INST) * h
    return h, v.astype(np.float32), o.astype(np.float32), p.reshape(B, STEPS, N_INST)


def encode_numpy(model, hits, vel, off, style=None):
    x = mx.array(to_input(hits, vel, off))
    s = mx.array(one_hot(np.asarray(style))) if model.cond else None
    mu, logvar = model.encode(x, s)
    return np.array(mu), np.array(logvar)


class StyleClassifier(nn.Module):
    """Separate MLP that guesses the style of a grid (hits + velocity + offsets)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(3 * CELLS, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, N_STYLES)
        self.drop = nn.Dropout(0.3)

    def __call__(self, x):
        h = self.drop(nn.gelu(self.l1(x)))
        h = self.drop(nn.gelu(self.l2(h)))
        return self.l3(h)


def save_vae(model, path_dir, extra=None):
    os.makedirs(path_dir, exist_ok=True)
    model.save_weights(os.path.join(path_dir, "vae.safetensors"))
    with open(os.path.join(path_dir, "vae_config.json"), "w") as f:
        json.dump({**model.config(), **(extra or {})}, f, indent=1)


def load_vae(path_dir):
    with open(os.path.join(path_dir, "vae_config.json")) as f:
        cfg = json.load(f)
    if cfg.get("kind", "mlp") == "ar":
        m = ARVAE(cfg["latent"], cfg["hidden"], cfg["cond"], cfg["rnn"])
    else:
        m = VAE(cfg["latent"], cfg["hidden"], cfg["cond"])
    m.load_weights(os.path.join(path_dir, "vae.safetensors"))
    m.eval()
    return m, cfg


def load_classifier(path_dir):
    with open(os.path.join(path_dir, "classifier_config.json")) as f:
        cfg = json.load(f)
    m = StyleClassifier(cfg["hidden"])
    m.load_weights(os.path.join(path_dir, "classifier.safetensors"))
    m.eval()
    return m


def classify_numpy(clf, hits, vel, off, batch=2048):
    probs = []
    for i in range(0, len(hits), batch):
        x = mx.array(to_input(hits[i:i + batch], vel[i:i + batch], off[i:i + batch]))
        probs.append(np.array(mx.softmax(clf(x), axis=1)))
    return np.concatenate(probs)
