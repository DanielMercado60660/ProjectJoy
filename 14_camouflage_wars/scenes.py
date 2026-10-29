"""Procedural camouflage scenes: textured backgrounds with textured blob "creatures".

Every texture is a scalar field t(x, y) in [0, 1] from one of nine families, coloured by a
two-colour palette. Six families are used for training; three are held out (never seen in
training, neither as background nor as creature skin).

Camouflage knob: a creature's skin is either
  * "same family": the background's own family with its parameters perturbed by `delta`
    in [0, 1] (delta=0: identical parameters, only a fresh phase/seed, i.e. near-perfect
    camouflage; delta=1: colours, scale and orientation re-drawn far from the background), or
  * "other family" (25% of creatures): an independent texture from another family of the split.

Scenes are 96x96 RGB float32 in [0, 1]. Instance map: 0 = background, k = creature k.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

H = W = 96
TRAIN_FAMILIES = ["stripes", "checker", "spots", "fbm", "cells", "plaid"]
HELDOUT_FAMILIES = ["marble", "ripples", "streaks"]
ALL_FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES

_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return _xx * c + _yy * s, -_xx * s + _yy * c


def _noise(rng, cells, h=H, w=W):
    """Smooth value noise: random grid upsampled with cubic interpolation, normalised."""
    g = rng.random((cells + 3, cells + 3)).astype(np.float32)
    up = zoom(g, (h + 2 * h // cells) / g.shape[0], order=3)
    oy, ox = rng.integers(0, up.shape[0] - h + 1), rng.integers(0, up.shape[1] - w + 1)
    n = up[oy:oy + h, ox:ox + w]
    return (n - n.mean()) / (n.std() + 1e-6)


def _fbm(rng, base_cells, octaves=4):
    tot, amp = np.zeros((H, W), np.float32), 1.0
    for o in range(octaves):
        tot += amp * _noise(rng, max(2, int(base_cells * 2 ** o)))
        amp *= 0.5
    return tot


def _squash(f, sharp):
    return 1.0 / (1.0 + np.exp(-sharp * f))


def sample_params(family, rng):
    """Draw parameters for one texture of a family. `scale` is a period in pixels."""
    p = dict(family=family,
             scale=float(np.exp(rng.uniform(np.log(5), np.log(18)))),
             theta=float(rng.uniform(0, np.pi)),
             sharp=float(rng.uniform(1.5, 8.0)),
             c1=_rand_colour(rng), c2=_rand_colour(rng))
    return p


def _rand_colour(rng):
    # HSV with moderate saturation so palettes are natural-ish but varied.
    h, s, v = rng.random(), rng.uniform(0.15, 0.85), rng.uniform(0.2, 0.95)
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p_, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p_), (q, v, p_), (p_, v, t), (p_, q, v), (t, p_, v), (v, p_, q)][i], np.float32)


def render_texture(p, rng):
    """Render a full HxWx3 texture field for params p (rng supplies phase / seeds)."""
    fam, sc, th, sh = p["family"], p["scale"], p["theta"], p["sharp"]
    u, v = _rot(th)
    ph = rng.uniform(0, 2 * np.pi, size=4)
    if fam == "stripes":
        f = np.sin(2 * np.pi * u / sc + ph[0] + 0.6 * np.sin(2 * np.pi * v / (4 * sc) + ph[1]))
    elif fam == "checker":
        f = np.sin(2 * np.pi * u / sc + ph[0]) * np.sin(2 * np.pi * v / sc + ph[1]) * 3
    elif fam == "spots":
        # jittered lattice of dots
        gu, gv = np.floor(u / sc + ph[0]), np.floor(v / sc + ph[1])
        jit = np.sin(gu * 12.9898 + gv * 78.233 + ph[2]) * 43758.5453
        jit = (jit - np.floor(jit)) - 0.5
        fu = (u / sc + ph[0]) - gu - 0.5 - 0.3 * jit
        fv = (v / sc + ph[1]) - gv - 0.5 + 0.3 * jit
        f = (0.32 - np.sqrt(fu ** 2 + fv ** 2)) * 8
    elif fam == "fbm":
        f = _fbm(rng, max(2, int(W / (2 * sc))))
    elif fam == "cells":
        n = max(4, int((W / sc) ** 2 * 0.6))
        pts = rng.uniform(-8, W + 8, size=(n, 2)).astype(np.float32)
        d = (_xx[..., None] - pts[:, 0]) ** 2 + (_yy[..., None] - pts[:, 1]) ** 2
        d.sort(axis=-1)
        f = (np.sqrt(d[..., 1]) - np.sqrt(d[..., 0]) - 1.5) / 2  # dark cell borders
    elif fam == "plaid":
        f = (np.sin(2 * np.pi * u / sc + ph[0]) + np.sin(2 * np.pi * v / (1.7 * sc) + ph[1])) * 1.5
    elif fam == "marble":
        turb = _fbm(rng, max(2, int(W / (3 * sc))), octaves=3)
        f = np.sin(2 * np.pi * u / sc + 2.5 * turb + ph[0]) * 2
    elif fam == "ripples":
        cx, cy = rng.uniform(-W, 2 * W, size=2)
        cx2, cy2 = rng.uniform(-W, 2 * W, size=2)
        r1 = np.sqrt((_xx - cx) ** 2 + (_yy - cy) ** 2)
        r2 = np.sqrt((_xx - cx2) ** 2 + (_yy - cy2) ** 2)
        f = (np.sin(2 * np.pi * r1 / sc + ph[0]) + np.sin(2 * np.pi * r2 / (1.3 * sc) + ph[1])) * 1.5
    elif fam == "streaks":
        # anisotropic noise: stretched along theta
        g = rng.standard_normal((H * 2, W * 2)).astype(np.float32)
        g = gaussian_filter(g, sigma=(sc * 0.12, sc * 0.9))
        from scipy.ndimage import rotate
        g = rotate(g, np.degrees(th), reshape=False, order=1, mode="wrap")
        f = g[H // 2:H // 2 + H, W // 2:W // 2 + W]
        f = (f - f.mean()) / (f.std() + 1e-6) * 1.5
    else:
        raise ValueError(fam)
    t = _squash(f, sh / 3)[..., None]
    return (1 - t) * p["c1"] + t * p["c2"]


def perturb_params(p, delta, rng):
    """Same family, parameters moved away from p by `delta` in [0, 1]."""
    q = dict(p)
    q["scale"] = float(np.clip(p["scale"] * np.exp(delta * rng.normal(0, 0.6)), 3.5, 26))
    q["theta"] = float(p["theta"] + delta * rng.normal(0, np.pi / 2.5))
    q["sharp"] = float(np.clip(p["sharp"] * np.exp(delta * rng.normal(0, 0.5)), 1.0, 10))
    q["c1"] = ((1 - delta) * p["c1"] + delta * _rand_colour(rng)).astype(np.float32)
    q["c2"] = ((1 - delta) * p["c2"] + delta * _rand_colour(rng)).astype(np.float32)
    return q


def blob_mask(rng, cx, cy, R):
    """Radial-Fourier blob."""
    ang = np.arctan2(_yy - cy, _xx - cx)
    rad = np.sqrt((_xx - cx) ** 2 + (_yy - cy) ** 2)
    r = np.ones_like(ang)
    for k in range(2, 6):
        r += rng.uniform(0, 0.35 / k ** 0.8) * np.cos(k * ang + rng.uniform(0, 2 * np.pi))
    return rad < R * r


def texture_stats(img):
    """Model-independent texture descriptor used only to bin difficulty (not by any detector)."""
    g = img.mean(-1)
    gx, gy = np.gradient(gaussian_filter(g, 0.7))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) % np.pi
    ori = [float((mag * (np.abs(((ang - a + np.pi / 2) % np.pi) - np.pi / 2) < np.pi / 8)).mean()) for a in
           (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)]
    coarse = gaussian_filter(g, 3.0)
    return np.array(list(img.reshape(-1, 3).mean(0)) + list(img.reshape(-1, 3).std(0)) + ori +
                    [float(mag.mean()), float(np.abs(g - coarse).mean())], np.float32)


def make_scene(rng, families, p_other=0.25):
    """Returns img (H,W,3), inst (H,W) uint8, and per-creature metadata list."""
    bg_p = sample_params(families[rng.integers(len(families))], rng)
    bg = render_texture(bg_p, rng)
    img = bg.copy()
    inst = np.zeros((H, W), np.uint8)
    meta = []
    n = int(rng.integers(1, 4))
    for k in range(1, n + 1):
        R = rng.uniform(8, 18)
        cx, cy = rng.uniform(R * 0.6, W - R * 0.6), rng.uniform(R * 0.6, H - R * 0.6)
        m = blob_mask(rng, cx, cy, R)
        if rng.random() < p_other:
            others = [f for f in families if f != bg_p["family"]]
            cp = sample_params(others[rng.integers(len(others))], rng)
            delta, kind = float("nan"), "other"
        else:
            delta = float(rng.random())
            cp, kind = perturb_params(bg_p, delta, rng), "same"
        skin = render_texture(cp, rng)
        img[m] = skin[m]
        inst[m] = k
        meta.append(dict(kind=kind, delta=delta, family=cp["family"],
                         stat_dist_raw=(texture_stats(skin) - texture_stats(bg)).tolist()))
    # global nuisance: illumination gradient, slight blur, sensor noise
    a = rng.uniform(-0.25, 0.25, size=2)
    illum = 1 + a[0] * (_xx / W - 0.5) + a[1] * (_yy / H - 0.5)
    img = img * illum[..., None]
    img = gaussian_filter(img, sigma=(0.5, 0.5, 0))
    img = img + rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, 1).astype(np.float32)
    # creatures fully occluded by later ones drop out of metadata (keep ids consistent)
    present = [k for k in range(1, n + 1) if (inst == k).sum() >= 20]
    return img, inst, meta, present, bg_p["family"]


def make_split(seed, n, families, p_other=0.25):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, H, W, 3), np.uint8)
    Y = np.zeros((n, H, W), np.uint8)
    metas = []
    for i in range(n):
        img, inst, meta, present, bgf = make_scene(rng, families, p_other)
        X[i] = np.round(img * 255).astype(np.uint8)
        Y[i] = inst
        metas.append(dict(bg_family=bgf, creatures={k: meta[k - 1] for k in present}))
    return X, Y, metas


def _worker(args):
    return make_split(*args)


def make_split_parallel(seed, n, families, workers=4, chunk=500):
    """Deterministic: chunk i uses seed*1000+i regardless of worker count."""
    from multiprocessing import get_context
    jobs = [(seed * 1000 + i, min(chunk, n - i * chunk), families) for i in range((n + chunk - 1) // chunk)]
    with get_context("spawn").Pool(workers) as pool:
        outs = pool.map(_worker, jobs)
    X = np.concatenate([o[0] for o in outs])
    Y = np.concatenate([o[1] for o in outs])
    metas = [m for o in outs for m in o[2]]
    return X, Y, metas
