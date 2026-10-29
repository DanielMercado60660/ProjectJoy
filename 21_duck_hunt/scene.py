"""Arcade duck-hunt scene renderer (numpy, 2x supersampled, 80x80 output).

Each scene: a static sky + hills + trees background, drifting clouds, moving distractor
birds/objects, and the duck flying a waypoint course (trajectory from the flight sim with
a fixed, untuned PD controller). Conditions:
  train      day-blue skies, green hills, crows + gulls, normal lighting
  iid        same distribution as train, new seeds
  bg         HELD-OUT backgrounds: sunset skies + desert/snow terrain, pink clouds
  light      HELD-OUT lighting: dusk (gain 0.35-0.6, blue tint) + more sensor noise
  distract   HELD-OUT distractors: red/yellow balloons and brown hawks (duck-coloured body)
  all        all three held-out shifts at once
"""
import numpy as np
import flight as F

H = W = 80
SS = 2
HS, WS = H * SS, W * SS
N_FRAMES = 40
FRAME_EVERY = 3          # sim steps per frame (0.15 s)
CONDS = ["train", "iid", "bg", "light", "distract", "all"]
COND_SEED = dict(train=10_000, iid=20_000, bg=30_000, light=40_000, distract=50_000, all=60_000, val=70_000)

YY, XX = np.mgrid[0:HS, 0:WS].astype(np.float32)


def world_to_img(p):
    """flight-world metres -> output-pixel coords (x right, y down)."""
    x = 4 + (p[..., 0] + 3) / 36.0 * 72
    y = 74 - p[..., 1] / 17.0 * 66
    return np.stack([x, y], -1)


def ellipse(img, cx, cy, rx, ry, col, alpha=1.0):
    x0, x1 = int(max(cx - rx - 1, 0)), int(min(cx + rx + 2, WS))
    y0, y1 = int(max(cy - ry - 1, 0)), int(min(cy + ry + 2, HS))
    if x0 >= x1 or y0 >= y1:
        return
    m = ((XX[y0:y1, x0:x1] - cx) / rx) ** 2 + ((YY[y0:y1, x0:x1] - cy) / ry) ** 2 <= 1
    sub = img[y0:y1, x0:x1]
    sub[m] = sub[m] * (1 - alpha) + np.asarray(col, np.float32) * alpha


def draw_bird(img, cx, cy, s, phase, facing, body, wing, head=None, beak=None, ring=False):
    ellipse(img, cx, cy, 7 * s, 4.2 * s, body)
    wy = cy - 3.5 * s * np.sin(phase)            # wing flaps up/down
    ellipse(img, cx - 1.0 * s * facing, wy, 5 * s, 2.0 * s, wing)
    hx, hy = cx + 6.5 * s * facing, cy - 3.5 * s
    if ring:
        ellipse(img, cx + 5.0 * s * facing, cy - 2.0 * s, 2.2 * s, 1.6 * s, (0.95, 0.95, 0.92))
    ellipse(img, hx, hy, 3.0 * s, 2.8 * s, head if head is not None else body)
    if beak is not None:
        ellipse(img, hx + 3.3 * s * facing, hy + 0.6 * s, 2.0 * s, 1.0 * s, beak)


def draw_duck(img, cx, cy, s, phase, facing):
    draw_bird(img, cx, cy, s, phase, facing, body=(0.52, 0.34, 0.18), wing=(0.36, 0.26, 0.16),
              head=(0.10, 0.45, 0.22), beak=(0.98, 0.62, 0.10), ring=True)


def sample_scene(rng, cond):
    sc = {}
    bg_shift = cond in ("bg", "all")
    if not bg_shift:
        sc["sky_top"] = rng.uniform([0.25, 0.50, 0.80], [0.45, 0.70, 0.98])
        sc["sky_bot"] = rng.uniform([0.65, 0.80, 0.92], [0.85, 0.92, 1.00])
        sc["ground"] = rng.uniform([0.18, 0.45, 0.12], [0.35, 0.65, 0.28])
        sc["tree"] = rng.uniform([0.05, 0.25, 0.08], [0.15, 0.40, 0.15])
        sc["cloud"] = rng.uniform([0.92, 0.92, 0.94], [1.0, 1.0, 1.0])
    else:
        sc["sky_top"] = rng.uniform([0.30, 0.15, 0.40], [0.50, 0.30, 0.60])
        sc["sky_bot"] = rng.uniform([0.95, 0.50, 0.25], [1.00, 0.75, 0.45])
        if rng.random() < 0.5:   # desert
            sc["ground"] = rng.uniform([0.75, 0.60, 0.35], [0.88, 0.72, 0.48])
            sc["tree"] = rng.uniform([0.25, 0.45, 0.20], [0.35, 0.55, 0.28])
        else:                    # snow
            sc["ground"] = rng.uniform([0.88, 0.90, 0.94], [0.96, 0.97, 1.0])
            sc["tree"] = rng.uniform([0.10, 0.20, 0.18], [0.18, 0.30, 0.25])
        sc["cloud"] = rng.uniform([0.95, 0.70, 0.75], [1.0, 0.85, 0.88])
    if cond in ("light", "all"):
        sc["gain"] = rng.uniform(0.35, 0.6)
        sc["tint"] = np.array([0.85, 0.9, 1.15]) * rng.uniform(0.95, 1.05, 3)
        sc["noise"] = rng.uniform(0.03, 0.06)
    else:
        sc["gain"] = rng.uniform(0.85, 1.15)
        sc["tint"] = rng.uniform(0.95, 1.05, 3)
        sc["noise"] = rng.uniform(0.005, 0.025)
    sc["hills"] = [(rng.uniform(4, 12), rng.uniform(0.02, 0.08), rng.uniform(0, 6.28)) for _ in range(3)]
    sc["horizon"] = rng.uniform(128, 146)
    sc["trees"] = [(rng.uniform(0, WS), rng.uniform(8, 16)) for _ in range(rng.integers(1, 6))]
    sc["clouds"] = [(rng.uniform(-20, WS), rng.uniform(5, 100), rng.uniform(8, 22), rng.uniform(4, 9),
                     rng.uniform(0.3, 1.5) * rng.choice([-1, 1])) for _ in range(rng.integers(1, 5))]
    nd = rng.integers(0, 4) if cond not in ("distract", "all") else rng.integers(2, 5)
    ds = []
    for _ in range(nd):
        if cond in ("distract", "all"):
            kind = rng.choice(["balloon", "hawk"])
        else:
            kind = rng.choice(["crow", "gull"])
        ds.append(dict(kind=kind, x=rng.uniform(0, WS), y=rng.uniform(10, 120),
                       vx=rng.uniform(1.5, 5) * rng.choice([-1, 1]) if kind != "balloon" else rng.uniform(-0.6, 0.6),
                       vy=rng.uniform(-1, 1) if kind != "balloon" else -rng.uniform(0.5, 1.5),
                       s=rng.uniform(0.8, 1.2), ph=rng.uniform(0, 6.28),
                       col=(rng.uniform(0.85, 1.0), rng.uniform(0.0, 0.2), rng.uniform(0.0, 0.2)) if rng.random() < 0.6
                       else (0.95, 0.85, 0.1)))
    sc["distractors"] = ds
    sc["duck_s"] = rng.uniform(0.8, 1.2)
    return sc


def background(sc):
    t = (YY / HS)[..., None]
    sky = sc["sky_top"] * (1 - t) + sc["sky_bot"] * t
    ground = np.zeros((HS, WS), bool)
    hy = sc["horizon"] + sum(a * np.sin(f * XX[0] + p) for a, f, p in sc["hills"])
    ground = YY >= hy[None, :]
    fg = np.zeros((HS, WS, 3), np.float32); fga = np.zeros((HS, WS), np.float32)
    fg[ground] = sc["ground"] * (1 - 0.15 * (YY[ground] - 120) / 40)[:, None]; fga[ground] = 1
    for tx, th in sc["trees"]:
        base = sc["horizon"] + 4
        ellipse(fg, tx, base - th, th * 0.45, th * 0.8, sc["tree"])
        m = ((XX - tx) / (th * 0.45)) ** 2 + ((YY - (base - th)) / (th * 0.8)) ** 2 <= 1
        fga[m] = 1
    return sky.astype(np.float32), fg, fga


def duck_track(rng):
    course = F.make_courses(rng, 1)
    off = int(rng.integers(0, 40))
    r = F.run_episodes(course, lambda e, o: F.pd_action(e, F.PD_DEFAULT), seed=int(rng.integers(1 << 30)),
                       record=True)
    tr = r["traj"][0]
    tr = tr[~np.isnan(tr[:, 0])]
    need = off + N_FRAMES * FRAME_EVERY + 1
    if len(tr) < need:
        tr = np.concatenate([tr, np.repeat(tr[-1:], need - len(tr), 0)])
    idx = off + np.arange(N_FRAMES + 1) * FRAME_EVERY
    return world_to_img(tr[idx])  # (N+1, 2) in output pixels


def render_scene(seed, cond, n_frames=N_FRAMES, pts=None):
    """pts: optional (n_frames+1, 2) duck track in output pixels (e.g. from the learned flyer)."""
    rng = np.random.default_rng(seed)
    sc = sample_scene(rng, cond)
    if pts is None:
        pts = duck_track(rng)
    sky, fg, fga = background(sc)
    frames = np.zeros((n_frames, H, W, 3), np.uint8)
    gt = pts[:n_frames].copy()
    for f in range(n_frames):
        img = sky.copy()
        for (cx, cy, rx, ry, vx) in sc["clouds"]:
            x = (cx + vx * f) % (WS + 60) - 30
            ellipse(img, x, cy, rx, ry, sc["cloud"], 0.9)
            ellipse(img, x + rx * 0.6, cy + 2, rx * 0.7, ry * 0.8, sc["cloud"], 0.9)
        img = img * (1 - fga[..., None]) + fg * fga[..., None]
        for d in sc["distractors"]:
            x = d["x"] + d["vx"] * f; y = d["y"] + d["vy"] * f
            x = (x + 30) % (WS + 60) - 30
            if d["kind"] == "crow":
                draw_bird(img, x, y, d["s"], d["ph"] + 1.3 * f, np.sign(d["vx"]), (0.08, 0.08, 0.10), (0.15, 0.15, 0.17))
            elif d["kind"] == "gull":
                draw_bird(img, x, y, d["s"], d["ph"] + 1.1 * f, np.sign(d["vx"]), (0.93, 0.93, 0.95), (0.6, 0.62, 0.66),
                          beak=(0.9, 0.8, 0.2))
            elif d["kind"] == "hawk":
                draw_bird(img, x, y, d["s"] * 1.1, d["ph"] + 0.9 * f, np.sign(d["vx"]), (0.55, 0.36, 0.20), (0.42, 0.28, 0.16),
                          head=(0.80, 0.75, 0.62), beak=(0.3, 0.3, 0.3))
            else:  # balloon
                ellipse(img, x, y, 6 * d["s"], 7.5 * d["s"], d["col"])
                img[int(np.clip(y + 7.5 * d["s"], 0, HS - 1)):int(np.clip(y + 18 * d["s"], 0, HS)), int(np.clip(x, 0, WS - 1))] = 0.3
        cx, cy = gt[f] * SS
        vx = pts[f + 1, 0] - pts[f, 0]
        draw_duck(img, cx, cy, sc["duck_s"], 1.6 * f, 1.0 if vx >= 0 else -1.0)
        small = img.reshape(H, SS, W, SS, 3).mean((1, 3))
        small = small * sc["gain"] * sc["tint"] + rng.normal(0, sc["noise"], small.shape)
        frames[f] = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    # pixel-centre convention: output pixel i covers [i, i+1); gt in same units
    gt = gt - 0.25  # 2x2 mean pooling: output pixel i is centred on 2x coordinate 2i+0.5
    vis = (gt[:, 0] >= 0) & (gt[:, 0] < W) & (gt[:, 1] >= 0) & (gt[:, 1] < H)
    return frames, gt.astype(np.float32), vis


def make_set(cond, n_scenes, base=None, n_frames=N_FRAMES):
    base = COND_SEED[cond] if base is None else base
    rcond = "train" if cond == "val" else cond
    X, Y, V = [], [], []
    for i in range(n_scenes):
        fr, gt, vis = render_scene(base + i, rcond, n_frames)
        X.append(fr); Y.append(gt); V.append(vis)
    return np.stack(X), np.stack(Y), np.stack(V)


def photometric_jitter(frames, rng):
    """Generic photometric augmentation for the '_aug' follow-up (applied identically to every
    frame passed in): gain 0.3-1.6, per-channel tint 0.75-1.25, offset +-0.08, noise 0-0.07.
    NOTE: these ranges cover the held-out 'light' condition, so 'light' is not a strict held-out
    test for the augmented variants; 'bg' and 'distract' still are."""
    x = frames.astype(np.float32) / 255.0
    x = x * rng.uniform(0.3, 1.6) * rng.uniform(0.75, 1.25, 3) + rng.uniform(-0.08, 0.08)
    x = x + rng.normal(0, rng.uniform(0, 0.07), x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def jitter_set(X, seed):
    rng = np.random.default_rng(seed)
    return np.stack([photometric_jitter(x, rng) for x in X])
