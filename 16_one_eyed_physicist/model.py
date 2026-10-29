"""Frames -> (mean, log-variance) of restitution.  Per-frame CNN (with coordinate channels)
followed by a temporal 1-D conv stack and pooled MLP head (Gaussian NLL)."""
import json
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class OneEyed(nn.Module):
    def __init__(self, width=32, feat=128, H=64, W=64):
        super().__init__()
        c = width
        self.convs = [nn.Conv2d(3, c // 2, 3, stride=2, padding=1),
                      nn.Conv2d(c // 2, c, 3, stride=2, padding=1),
                      nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
                      nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)]
        self.proj = nn.Linear(2 * c * (H // 16) * (W // 16), feat)
        self.tconvs = [nn.Conv1d(feat, feat, 5, padding=2) for _ in range(3)]
        self.head = nn.Sequential(nn.Linear(2 * feat, feat), nn.GELU(), nn.Linear(feat, 2))
        ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
        self._coords = mx.array(np.stack([xs, ys], -1).astype(np.float32))

    def __call__(self, frames):
        B, T, H, W = frames.shape
        x = frames.reshape(B * T, H, W, 1) - 0.1
        x = mx.concatenate([x, mx.broadcast_to(self._coords, (B * T, H, W, 2))], -1)
        for cv in self.convs:
            x = nn.gelu(cv(x))
        x = self.proj(x.reshape(B * T, -1)).reshape(B, T, -1)
        for tc in self.tconvs:
            x = x + nn.gelu(tc(x))
        z = mx.concatenate([x.mean(1), x.max(1)], -1)
        out = self.head(z)
        mu = out[:, 0] + 0.575
        logvar = mx.clip(out[:, 1] - 4.0, -14.0, 2.0)
        return mu, logvar


def load(ckpt_dir):
    cfg = json.load(open(f"{ckpt_dir}/config.json"))
    m = OneEyed(width=cfg["width"], feat=cfg["feat"])
    m.load_weights(f"{ckpt_dir}/model.safetensors")
    return m, cfg


def predict(model, frames_u8, bs=128):
    """frames_u8: (n,T,H,W) uint8 -> (mu, sigma) numpy arrays."""
    mus, sds = [], []
    for i in range(0, len(frames_u8), bs):
        x = mx.array(frames_u8[i:i + bs].astype(np.float32) / 255.0)
        mu, lv = model(x)
        mx.eval(mu, lv)
        mus.append(np.array(mu)); sds.append(np.exp(0.5 * np.array(lv)))
    return np.concatenate(mus), np.concatenate(sds)
