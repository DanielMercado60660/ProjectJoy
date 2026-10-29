"""Damage masks. Each function returns a boolean (H, W) mask: True = cell is erased (all 16 channels set to 0).

Training damage (in-distribution): `circle`.
Held-out damage (never seen in training): `half`, `stripes`, `blobs`.
Coordinates follow the Distill paper: the grid spans [-1, 1] in both axes.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def _grid(h, w):
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    return y, x


def circle(h, w, rng):
    """Random disc: centre in [-0.5, 0.5]^2, radius in [0.1, 0.4] (paper's training damage)."""
    y, x = _grid(h, w)
    cy, cx = rng.uniform(-0.5, 0.5, 2)
    r = rng.uniform(0.1, 0.4)
    return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2


def half(h, w, rng):
    """Remove the left/right/top/bottom half of the grid (cut through the centre)."""
    side = rng.integers(4)
    m = np.zeros((h, w), bool)
    if side == 0:
        m[:, : w // 2] = True
    elif side == 1:
        m[:, w // 2:] = True
    elif side == 2:
        m[: h // 2] = True
    else:
        m[h // 2:] = True
    return m


def stripes(h, w, rng):
    """Parallel stripes, 3 cells wide every 8 cells, random orientation and phase (~37% of cells erased)."""
    width, period = 3, 8
    phase = rng.integers(period)
    m = np.zeros((h, w), bool)
    idx = (np.arange(max(h, w)) + phase) % period < width
    if rng.integers(2) == 0:
        m[idx[:h], :] = True
    else:
        m[:, idx[:w]] = True
    return m


def blobs(h, w, rng, frac=0.35, inner=0.6):
    """Irregular random blobs: smoothed noise thresholded so ~`frac` of the central region is erased."""
    noise = gaussian_filter(rng.normal(size=(h, w)), sigma=4)
    y, x = _grid(h, w)
    centre = (np.abs(x) < inner) & (np.abs(y) < inner)
    thr = np.quantile(noise[centre], 1 - frac)
    return (noise > thr) & centre


TRAIN_DAMAGE = {"circle": circle}
HELDOUT_DAMAGE = {"half": half, "stripes": stripes, "blobs": blobs}
ALL_DAMAGE = {**TRAIN_DAMAGE, **HELDOUT_DAMAGE}


def apply(x, mask):
    """x: (..., H, W, C) numpy array; zero erased cells (returns a copy)."""
    x = x.copy()
    x[..., mask, :] = 0.0
    return x
