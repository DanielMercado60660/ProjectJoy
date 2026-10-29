"""CNN encoder + GRU tracker that outputs a 2D Gaussian per ball slot per frame (MLX)."""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

H = W = 64
POS_SCALE = 32.0          # normalized coords: (p - 32) / 32
LOG_S_MIN, LOG_S_MAX = -5.5, 1.0   # std floor 0.13 px
RHO_MAX = 0.95


class Tracker(nn.Module):
    def __init__(self, K=3, enc_dim=512, hid=256, layers=1):
        super().__init__()
        self.K = K
        self.conv1 = nn.Conv2d(5, 32, 5, stride=2, padding=2)    # 32x32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # 16x16
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)   # 8x8
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)   # 8x8
        self.fc = nn.Linear(8 * 8 * 64, enc_dim)
        self.grus = [nn.GRU(enc_dim if i == 0 else hid, hid) for i in range(layers)]
        self.head1 = nn.Linear(hid + enc_dim, 256)
        self.head2 = nn.Linear(256, K * 5)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self._coords = mx.array(np.stack([(xx - 31.5) / 32, (yy - 31.5) / 32], -1))

    def encode(self, frames):
        """frames (B,T,H,W,3) in [0,1] -> (B,T,enc_dim)"""
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, H, W, 3)
        x = mx.concatenate([x, mx.broadcast_to(self._coords, (B * T, H, W, 2))], -1)
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = nn.relu(self.conv4(x))
        x = nn.relu(self.fc(x.reshape(B * T, -1)))
        return x.reshape(B, T, -1)

    def __call__(self, frames):
        """Returns raw head output (B,T,K,5): mu_x, mu_y, log_sx, log_sy, rho_raw (normalized)."""
        e = self.encode(frames)
        h = e
        for g in self.grus:
            h = g(h)
        z = nn.relu(self.head1(mx.concatenate([h, e], -1)))
        out = self.head2(z)
        B, T = out.shape[:2]
        return out.reshape(B, T, self.K, 5)


def gaussian_params(out):
    """raw (…,5) -> mu (…,2) normalized, sx, sy, rho"""
    mu = out[..., :2]
    ls = mx.clip(out[..., 2:4], LOG_S_MIN, LOG_S_MAX)
    rho = RHO_MAX * mx.tanh(out[..., 4])
    return mu, ls, rho


def gaussian_nll(out, target):
    """Per-element 2D Gaussian NLL in normalized coordinates. target (…,2) normalized."""
    mu, ls, rho = gaussian_params(out)
    s = mx.exp(ls)
    d = (target - mu) / s
    one_m = 1 - rho * rho
    z = (d[..., 0] ** 2 + d[..., 1] ** 2 - 2 * rho * d[..., 0] * d[..., 1]) / one_m
    return math.log(2 * math.pi) + ls[..., 0] + ls[..., 1] + 0.5 * mx.log(one_m) + 0.5 * z


def to_pixels(out):
    """raw head output (numpy or mx) -> mean (…,2) pixels, cov (…,2,2) pixels^2 (numpy)."""
    out = np.asarray(out, dtype=np.float64)
    mu = out[..., :2] * POS_SCALE + 32.0
    s = np.exp(np.clip(out[..., 2:4], LOG_S_MIN, LOG_S_MAX)) * POS_SCALE
    rho = RHO_MAX * np.tanh(out[..., 4])
    cov = np.empty(out.shape[:-1] + (2, 2))
    cov[..., 0, 0] = s[..., 0] ** 2
    cov[..., 1, 1] = s[..., 1] ** 2
    cov[..., 0, 1] = cov[..., 1, 0] = rho * s[..., 0] * s[..., 1]
    return mu, cov
