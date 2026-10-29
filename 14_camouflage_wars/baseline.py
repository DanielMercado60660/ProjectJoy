"""Classical baselines (no learning). Both use the same local texture features:
smoothed colour, local colour std and oriented gradient energy, pooled with a Gaussian window.

  kmeans  : per-image 2-means on standardised features; creature cluster = the one touching the
            image border least.
  outlier : per-image robust background model (median / MAD of every feature), per-pixel
            outlier score, Otsu threshold on the score.

The window size (and which variant) is tuned on the seen-family val split only.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_opening, label


def features(img, sigma):
    g = img.mean(-1)
    gx, gy = np.gradient(gaussian_filter(g, 0.7))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) % np.pi
    fs = []
    mean = np.stack([gaussian_filter(img[..., c], sigma) for c in range(3)], -1)
    fs += [mean[..., c] for c in range(3)]
    for c in range(3):
        fs.append(np.sqrt(np.maximum(gaussian_filter(img[..., c] ** 2, sigma) - mean[..., c] ** 2, 0)))
    for a in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        w = np.abs(((ang - a + np.pi / 2) % np.pi) - np.pi / 2) < np.pi / 8
        fs.append(gaussian_filter(mag * w, sigma))
    return np.stack(fs, -1)


def _otsu(v, bins=64):
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def clean(m, min_area=30):
    """Opening + drop connected components smaller than min_area pixels."""
    m = binary_opening(m, iterations=1)
    comp, n = label(m)
    if n:
        areas = np.bincount(comp.ravel())
        keep = areas >= min_area
        keep[0] = False
        m = keep[comp]
    return m


def seg_outlier(img, sigma):
    F = features(img, sigma).reshape(-1, 10)
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-3
    s = np.sqrt((((F - med) / mad) ** 2).mean(1))
    s = np.log1p(s)
    m = (s > _otsu(s)).reshape(img.shape[:2])
    return clean(m)


def seg_kmeans(img, sigma, iters=15, seed=0):
    F = features(img, sigma).reshape(-1, 10)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    C = F[rng.choice(len(F), 2, replace=False)]
    for _ in range(iters):
        lab = ((F[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(2):
            if (lab == k).any():
                C[k] = F[lab == k].mean(0)
    lab = lab.reshape(img.shape[:2])
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    fg = 1 if (border == 1).mean() < (border == 0).mean() else 0
    return clean(lab == fg)


def run(X, method, sigma):
    f = seg_outlier if method == "outlier" else seg_kmeans
    return np.stack([f(x.astype(np.float32) / 255, sigma) for x in X])
