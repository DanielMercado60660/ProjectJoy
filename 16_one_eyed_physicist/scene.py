"""Synthetic bouncing-ball scene: physics (analytic, multi-bounce) and a camera renderer.

Known scene geometry (shared by every method):
  * side view, orthographic camera, 64x64 px, SCALE = 32 px per metre
  * ground surface at image row GROUND_ROW (pixel rows below it are the floor)
  * ball radius R = 0.07 m, gravity G = 9.81 m/s^2, FPS = 30, T = 32 frames (1.07 s)
Per-clip nuisances: initial position/velocity, exposure (motion blur), pixel noise,
background and ball brightness.  Hidden parameter of interest: restitution e.
"""
import numpy as np

H = W = 64
SCALE = 32.0          # px / m
GROUND_ROW = 60.0     # image row coordinate of the ground surface
R = 0.07              # ball radius (m)
G = 9.81
FPS = 30.0
T = 32
K_SUB = 5             # sub-frame samples for motion blur
E_RANGE = (0.2, 0.95)
NOISE_RANGE = (0.0, 0.12)
EXPO_RANGE = (0.0, 0.8)   # exposure as a fraction of the frame interval
MAX_BOUNCES = 40


def sample_params(rng, n, e_range=E_RANGE, noise_range=NOISE_RANGE, expo_range=EXPO_RANGE):
    """Rejection-sample n clips whose first impact lands between 0.2 s and 0.55 s."""
    out = {k: [] for k in ("x0", "vx", "y0", "vy0", "e", "noise", "expo", "bg", "fg")}
    got = 0
    while got < n:
        m = 2 * (n - got) + 16
        y0 = rng.uniform(0.5, 1.5, m); vy0 = rng.uniform(-1.0, 1.0, m)
        t1 = (vy0 + np.sqrt(vy0 ** 2 + 2 * G * (y0 - R))) / G
        ok = (t1 > 0.2) & (t1 < 0.55)
        k = min(int(ok.sum()), n - got)
        idx = np.nonzero(ok)[0][:k]
        out["y0"].append(y0[idx]); out["vy0"].append(vy0[idx])
        out["x0"].append(rng.uniform(0.15, 0.5, k)); out["vx"].append(rng.uniform(0.2, 1.2, k))
        out["e"].append(rng.uniform(*e_range, k))
        out["noise"].append(rng.uniform(*noise_range, k)); out["expo"].append(rng.uniform(*expo_range, k))
        out["bg"].append(rng.uniform(0.0, 0.15, k)); out["fg"].append(rng.uniform(0.7, 1.0, k))
        got += k
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def bounce_table(y0, vy0, e):
    """Segment start times / heights / velocities for each clip (padded with +inf)."""
    n = len(y0)
    tb = np.full((n, MAX_BOUNCES + 1), np.inf); ys = np.full_like(tb, R); vs = np.zeros_like(tb)
    tb[:, 0] = 0.0; ys[:, 0] = y0; vs[:, 0] = vy0
    t1 = (vy0 + np.sqrt(vy0 ** 2 + 2 * G * (y0 - R))) / G
    vin = vy0 - G * t1                    # negative
    t = t1.copy(); v = -e * vin
    for k in range(1, MAX_BOUNCES + 1):
        tb[:, k] = t; vs[:, k] = v
        t = t + 2 * v / G; v = e * v      # next contact; after ~40 bounces v is ~0 (ball at rest)
    return tb, ys, vs


def heights(p, t):
    """Ball centre height y(t) (m). p: param dict (n clips); t: (n, ...) times in s."""
    tb, ys, vs = bounce_table(p["y0"], p["vy0"], p["e"])
    n = len(tb); shp = t.shape
    tf = t.reshape(n, -1)
    seg = (tf[:, :, None] >= tb[:, None, :]).sum(-1) - 1           # (n, m)
    rows = np.arange(n)[:, None]
    dt = tf - tb[rows, seg]
    y = ys[rows, seg] + vs[rows, seg] * dt - 0.5 * G * dt ** 2
    return np.maximum(y, R).reshape(shp)


def positions(p, t):
    x = p["x0"][:, None] + p["vx"][:, None] * t.reshape(len(t), -1)
    return x.reshape(t.shape), heights(p, t)


def frame_times(n, expo, t_frames=T):
    """(n, T, K) sample times inside each frame's exposure window."""
    f = np.arange(t_frames) / FPS
    s = np.linspace(0.0, 1.0, K_SUB)
    return f[None, :, None] + expo[:, None, None] * s[None, None, :] / FPS + 0 * expo[:, None, None]


def render(p, key_seed=0, use_mlx=True):
    """Render clips -> float32 array (n, T, H, W) in [0, 1], quantised to 8 bits."""
    n = len(p["e"])
    tt = frame_times(n, p["expo"])
    x, y = positions(p, tt)
    px = (x * SCALE).astype(np.float32)                 # pixel-space centre (col coordinate)
    py = (GROUND_ROW - y * SCALE).astype(np.float32)    # row coordinate
    if use_mlx:
        import mlx.core as mx
        xx = mx.arange(W, dtype=mx.float32) + 0.5
        yy = mx.arange(H, dtype=mx.float32) + 0.5
        pxm = mx.array(px)[..., None, None]; pym = mx.array(py)[..., None, None]
        d = mx.sqrt((xx[None, None, None, None, :] - pxm) ** 2 + (yy[None, None, None, :, None] - pym) ** 2)
        a = mx.mean(mx.clip(R * SCALE + 0.5 - d, 0.0, 1.0), axis=2)      # (n,T,H,W) coverage
        bg = mx.array(p["bg"].astype(np.float32))[:, None, None, None]
        fg = mx.array(p["fg"].astype(np.float32))[:, None, None, None]
        img = bg * (1 - a) + fg * a
        floor = (yy >= GROUND_ROW).astype(mx.float32)[None, None, :, None]
        img = img * (1 - floor) + 0.35 * floor
        key = mx.random.key(key_seed)
        sig = mx.array(p["noise"].astype(np.float32))[:, None, None, None]
        img = img + sig * mx.random.normal(img.shape, key=key)
        img = mx.round(mx.clip(img, 0.0, 1.0) * 255.0) / 255.0
        return img
    xx = np.arange(W, dtype=np.float32) + 0.5; yy = np.arange(H, dtype=np.float32) + 0.5
    d = np.sqrt((xx - px[..., None, None]) ** 2 + (yy[:, None] - py[..., None, None]) ** 2)
    a = np.clip(R * SCALE + 0.5 - d, 0, 1).mean(2)
    img = p["bg"][:, None, None, None] * (1 - a) + p["fg"][:, None, None, None] * a
    img[:, :, int(GROUND_ROW):, :] = 0.35
    rng = np.random.default_rng(key_seed)
    img = img + p["noise"][:, None, None, None] * rng.standard_normal(img.shape)
    return (np.round(np.clip(img, 0, 1) * 255) / 255).astype(np.float32)


def make_set(seed, n, **kw):
    """Deterministic dataset: (uint8 frames (n,T,H,W), params dict)."""
    rng = np.random.default_rng(seed)
    p = sample_params(rng, n, **kw)
    chunks = []
    for i in range(0, n, 128):
        sub = {k: v[i:i + 128] for k, v in p.items()}
        img = render(sub, key_seed=seed * 100003 + i)
        chunks.append((np.array(img) * 255 + 0.5).astype(np.uint8))
    return np.concatenate(chunks), p
