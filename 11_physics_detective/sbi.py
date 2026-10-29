"""Amortized simulation-based inference (neural posterior estimation) in MLX.

A permutation-invariant network reads a *set* of experiments (design + camera trajectory) and
outputs a mixture-of-Gaussians posterior over z = logit((mu_k - MU_LO) / (MU_HI - MU_LO)).
It is trained only on simulated worlds, so after training inference is one forward pass.

    ../.venv/bin/python sbi.py --steps 30000 --ckpt checkpoints/sbi        # train (resumable)
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from physics import DESIGNS, MU_HI, MU_LO, NT, observe, world1_positions

MAXN = 15
FEAT = 2 + 2 * NT
CONFIG = dict(hidden=192, embed=128, comps=8, maxn=MAXN, batch=512, lr=1e-3, steps=30000)


# ------------------------------------------------------------------ data
def featurize(designs, y, cens):
    """designs (..., 2), y/cens (..., NT) -> features (..., FEAT) as float32."""
    yv = np.where(cens, 0.0, np.nan_to_num(y, nan=0.0))
    return np.concatenate([designs[..., :1] / 15.0, designs[..., 1:2], yv, cens.astype(np.float64)], -1).astype(np.float32)


def sample_designs(rng, B, N):
    """Training design distribution: half from the discrete menu, half continuous, and 30% of
    sets reuse only 1-3 distinct designs (like an active learner that keeps repeating its
    favourite push)."""
    disc = DESIGNS[rng.integers(len(DESIGNS), size=(B, N))]
    cont = np.stack([rng.uniform(1, 15, (B, N)), rng.uniform(0.1, 1.0, (B, N))], -1)
    d = np.where(rng.random((B, 1, 1)) < 0.5, disc, cont)
    rep = rng.random(B) < 0.3
    k = rng.integers(1, 4, size=B)
    idx = np.minimum(rng.integers(0, 3, size=(B, N)), k[:, None] - 1)
    d_rep = np.take_along_axis(d, idx[..., None].repeat(2, -1), 1)
    return np.where(rep[:, None, None], d_rep, d)


def simulate_batch(rng, B, N=MAXN):
    mu = rng.uniform(MU_LO, MU_HI, B)
    d = sample_designs(rng, B, N)
    x = world1_positions(d[..., 0], d[..., 1], mu[:, None])
    y, c = observe(x, rng)
    n = rng.integers(1, N + 1, B)
    mask = (np.arange(N)[None] < n[:, None]).astype(np.float32)
    return featurize(d, y, c), mask, mu


def mu_to_z(mu):
    p = (mu - MU_LO) / (MU_HI - MU_LO)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def z_to_mu(z):
    return MU_LO + (MU_HI - MU_LO) / (1 + np.exp(-z))


# ------------------------------------------------------------------ model
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
        pooled = (h * mask[..., None]).sum(1) / 4.0              # sum pooling: evidence adds up
        h = mx.concatenate([pooled, mx.log(n)], -1)
        for i, l in enumerate(self.head):
            h = l(h)
            if i < len(self.head) - 1:
                h = nn.gelu(h)
        logit_w, loc, raw_s = mx.split(h, 3, axis=-1)
        scale = nn.softplus(raw_s - 2.0) + 1e-3
        return logit_w - mx.logsumexp(logit_w, axis=-1, keepdims=True), loc, scale


def mixture_logpdf(logw, loc, scale, z):
    zz = (z[:, None] - loc) / scale
    return mx.logsumexp(logw - 0.5 * zz ** 2 - mx.log(scale) - 0.5 * np.log(2 * np.pi), axis=-1)


def loss_fn(model, feats, mask, z):
    logw, loc, scale = model(feats, mask)
    return -mixture_logpdf(logw, loc, scale, z).mean()


# ------------------------------------------------------------------ posterior utilities
def posterior_on_grid(model, feats, mask, grid):
    """Mixture density converted to a normalised density over mu on `grid`. Returns (B, len(grid))."""
    logw, loc, scale = model(mx.array(feats), mx.array(mask))
    logw, loc, scale = (np.array(a, dtype=np.float64) for a in (logw, loc, scale))
    z = mu_to_z(grid)
    p = np.clip((grid - MU_LO) / (MU_HI - MU_LO), 1e-6, 1 - 1e-6)
    log_jac = -np.log((MU_HI - MU_LO) * p * (1 - p))                 # dz/dmu
    zz = (z[None, None, :] - loc[..., None]) / scale[..., None]
    logpdf = np.logaddexp.reduce(logw[..., None] - 0.5 * zz ** 2 - np.log(scale[..., None]), axis=1) + log_jac
    w = np.exp(logpdf - logpdf.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


def load(ckpt_dir):
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    model = PosteriorNet(**cfg)
    model.load_weights(os.path.join(ckpt_dir, "model.safetensors"))
    model.eval()
    return model, cfg


# ------------------------------------------------------------------ training
def save_ckpt(d, model, opt, step, rng, cfg, log):
    os.makedirs(d, exist_ok=True)
    model.save_weights(os.path.join(d, "model.safetensors"))
    mx.save_safetensors(os.path.join(d, "opt.safetensors"), dict(tree_flatten(opt.state)))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    json.dump({"step": step, "rng": rng.bit_generator.state, "log": log}, open(os.path.join(d, "state.json"), "w"))


def train(cfg, ckpt, seed=0, stop_at=None):
    mx.random.seed(seed)
    model = PosteriorNet(**cfg)
    sched = optim.cosine_decay(cfg["lr"], cfg["steps"], 1e-5)
    opt = optim.AdamW(learning_rate=sched, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    step, log = 0, []
    if os.path.exists(os.path.join(ckpt, "state.json")):
        st = json.load(open(os.path.join(ckpt, "state.json")))
        model.load_weights(os.path.join(ckpt, "model.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(os.path.join(ckpt, "opt.safetensors")).items()))
        rng.bit_generator.state = st["rng"]
        step, log = st["step"], st["log"]
        print(f"resumed from step {step}", flush=True)
    vg = nn.value_and_grad(model, loss_fn)

    t0, run = time.time(), []
    while step < cfg["steps"]:
        f, m, mu = simulate_batch(rng, cfg["batch"], cfg["maxn"])
        loss, g = vg(model, mx.array(f), mx.array(m), mx.array(mu_to_z(mu).astype(np.float32)))
        g, _ = optim.clip_grad_norm(g, 5.0)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)
        run.append(loss.item())
        step += 1
        if step % 500 == 0:
            log.append([step, float(np.mean(run))])
            print(f"step {step} nll_z {np.mean(run):.3f}  {time.time()-t0:.0f}s", flush=True)
            run = []
        if step % 2000 == 0 or step == cfg["steps"]:
            save_ckpt(ckpt, model, opt, step, rng, cfg, log)
        if stop_at and step >= stop_at:
            save_ckpt(ckpt, model, opt, step, rng, cfg, log)
            print(f"stopping early at {step} (stop_at)", flush=True)
            return model
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CONFIG["steps"])
    ap.add_argument("--ckpt", default="checkpoints/sbi")
    ap.add_argument("--stop_at", type=int, default=None, help="stop early (for resume testing)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = dict(CONFIG, steps=a.steps)
    train(cfg, a.ckpt, a.seed, a.stop_at)
