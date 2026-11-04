"""R6 tracker scenes: domain-randomized (DR) training scenes + cached copies of the existing
held-out sets. scene.py is left untouched; this reuses its drawing helpers and duck trajectories.

DR ("dr" condition) randomizes, independently per scene:
  palette   sky top/bottom, ground, trees, clouds: random HSV colours (any hue)
  texture   smooth low-frequency value noise on sky and ground (random amplitude and cell size)
  lighting  global gain 0.3-1.4, per-channel tint 0.7-1.3, offset +-0.08, sensor noise 0-0.07
  clutter   0-5 distractors: crows, gulls, random-colour birds (body/wing/head/beak each random),
            duck-brown-bodied birds with random head/beak colours (so the duck must be told apart
            by its head/ring/beak conjunction, not by "brown blob"), random-colour ellipse blobs;
            straight-line motion with an optional sinusoidal wobble
Not in DR, by design (they are the pre-registered fresh held-out families, see STATUS.md):
rectangles/polygons, buildings, kites, drones, spatially varying illumination, haze; and no
hard-coded sunset/snow/desert palettes, balloons or hawks.

Seeds: dr train 100000+, dr val 110000+, fresh held-out 300000+ (generated only after freeze).
"""
import colorsys, hashlib, os
from multiprocessing import Pool
import numpy as np
import scene as S

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "checkpoints", "r6_tracker_cache")
DR_SEED = dict(dr=100_000, drval=110_000)
OLD_CONDS = ["iid", "bg", "light", "distract", "all"]
DUCK_BODY, DUCK_WING = np.array([0.52, 0.34, 0.18]), np.array([0.36, 0.26, 0.16])


def flight_sha():
    return hashlib.sha1(open(os.path.join(HERE, "flight.py"), "rb").read()).hexdigest()


def rand_rgb(rng, s=(0.0, 1.0), v=(0.0, 1.0)):
    return np.array(colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(*s), rng.uniform(*v)), np.float32)


def value_noise(rng, cells, amp):
    """smooth (HS,WS) noise in [-amp, amp]: bilinear upsampling of a random (cells+1)^2 grid."""
    g = rng.uniform(-1, 1, (cells + 1, cells + 1)).astype(np.float32)
    u = np.linspace(0, cells, S.HS, dtype=np.float32)
    i = np.minimum(u.astype(int), cells - 1); t = u - i
    rows = g[i] * (1 - t)[:, None] + g[i + 1] * t[:, None]          # (HS, cells+1)
    out = rows[:, i] * (1 - t)[None] + rows[:, i + 1] * t[None]       # (HS, WS)
    return amp * out


def sample_dr(rng, distract=True):
    """distract=False: ablation with only the ordinary crows/gulls as clutter."""
    sc = dict(kind="dr")
    sc["sky_top"] = rand_rgb(rng, (0, 0.9), (0.25, 1.0)); sc["sky_bot"] = rand_rgb(rng, (0, 0.9), (0.25, 1.0))
    sc["ground"] = rand_rgb(rng, (0, 1), (0.15, 1.0)); sc["tree"] = rand_rgb(rng, (0, 1), (0.05, 0.9))
    sc["cloud"] = rand_rgb(rng, (0, 0.5), (0.6, 1.0))
    sc["sky_tex"] = (int(rng.integers(3, 13)), float(rng.uniform(0, 0.12)), int(rng.integers(1 << 30)))
    sc["ground_tex"] = (int(rng.integers(4, 21)), float(rng.uniform(0, 0.2)), int(rng.integers(1 << 30)))
    sc["gain"] = rng.uniform(0.3, 1.4); sc["tint"] = rng.uniform(0.7, 1.3, 3)
    sc["offset"] = rng.uniform(-0.08, 0.08); sc["noise"] = rng.uniform(0.0, 0.07)
    sc["hills"] = [(rng.uniform(4, 14), rng.uniform(0.02, 0.1), rng.uniform(0, 6.28)) for _ in range(3)]
    sc["horizon"] = rng.uniform(110, 150)
    sc["trees"] = [(rng.uniform(0, S.WS), rng.uniform(6, 18)) for _ in range(rng.integers(0, 7))]
    sc["clouds"] = [(rng.uniform(-20, S.WS), rng.uniform(5, 110), rng.uniform(6, 24), rng.uniform(3, 10),
                     rng.uniform(0.2, 2.0) * rng.choice([-1, 1])) for _ in range(rng.integers(0, 6))]
    ds = []
    for _ in range(rng.integers(0, 6)):
        kind = str(rng.choice(["crow", "gull", "bird_rand", "bird_brown", "blob"], p=[.15, .15, .3, .15, .25]))
        if not distract:
            kind = str(rng.choice(["crow", "gull"]))
        d = dict(kind=kind, x=rng.uniform(0, S.WS), y=rng.uniform(8, 150),
                 vx=rng.uniform(0.3, 5) * rng.choice([-1, 1]), vy=rng.uniform(-1.5, 1.5),
                 wob=rng.uniform(0, 6) if rng.random() < 0.3 else 0.0, wf=rng.uniform(0.1, 0.6),
                 s=rng.uniform(0.7, 1.3), ph=rng.uniform(0, 6.28), fr=rng.uniform(0.8, 1.6))
        if kind == "bird_rand":
            d["cols"] = [rand_rgb(rng) for _ in range(4)]
        elif kind == "bird_brown":
            d["cols"] = [np.clip(DUCK_BODY + rng.normal(0, 0.06, 3), 0, 1), np.clip(DUCK_WING + rng.normal(0, 0.06, 3), 0, 1),
                         rand_rgb(rng), rand_rgb(rng)]
        elif kind == "blob":
            d["cols"] = [rand_rgb(rng, (0, 1), (0.1, 1))]; d["rx"] = rng.uniform(3, 9); d["ry"] = rng.uniform(3, 9)
        ds.append(d)
    sc["distractors"] = ds
    sc["duck_s"] = rng.uniform(0.8, 1.2)
    return sc


def dr_background(sc):
    t = (S.YY / S.HS)[..., None]
    sky = sc["sky_top"] * (1 - t) + sc["sky_bot"] * t
    c, a, sd = sc["sky_tex"]; sky = sky + value_noise(np.random.default_rng(sd), c, a)[..., None]
    hy = sc["horizon"] + sum(a * np.sin(f * S.XX[0] + p) for a, f, p in sc["hills"])
    ground = S.YY >= hy[None, :]
    c, a, sd = sc["ground_tex"]; gt = value_noise(np.random.default_rng(sd), c, a)
    fg = np.zeros((S.HS, S.WS, 3), np.float32); fga = np.zeros((S.HS, S.WS), np.float32)
    shade = (1 - 0.15 * (S.YY - 120) / 40 + gt)
    fg[ground] = sc["ground"] * shade[ground][:, None]; fga[ground] = 1
    for tx, th in sc["trees"]:
        base = sc["horizon"] + 4
        S.ellipse(fg, tx, base - th, th * 0.45, th * 0.8, sc["tree"])
        fga[((S.XX - tx) / (th * 0.45)) ** 2 + ((S.YY - (base - th)) / (th * 0.8)) ** 2 <= 1] = 1
    return np.clip(sky, 0, 1).astype(np.float32), fg, fga


def draw_distractor(img, d, f):
    x = d["x"] + d["vx"] * f; y = d["y"] + d["vy"] * f + d["wob"] * np.sin(d["wf"] * f + d["ph"])
    x = (x + 30) % (S.WS + 60) - 30
    k, face, ph = d["kind"], np.sign(d["vx"]), d["ph"] + d["fr"] * f
    if k == "crow":
        S.draw_bird(img, x, y, d["s"], ph, face, (0.08, 0.08, 0.10), (0.15, 0.15, 0.17))
    elif k == "gull":
        S.draw_bird(img, x, y, d["s"], ph, face, (0.93, 0.93, 0.95), (0.6, 0.62, 0.66), beak=(0.9, 0.8, 0.2))
    elif k in ("bird_rand", "bird_brown"):
        b, w, h, bk = d["cols"]; S.draw_bird(img, x, y, d["s"], ph, face, b, w, head=h, beak=bk)
    elif k == "blob":
        S.ellipse(img, x, y, d["rx"], d["ry"], d["cols"][0])
    return x, y


def render_from_sc(sc, pts, rng, n_frames=S.N_FRAMES, bg_fn=dr_background, draw_fn=draw_distractor, light_fn=None):
    """Generalised scene.render_scene for a sampled scene dict; same gt convention."""
    sky, fg, fga = bg_fn(sc)
    frames = np.zeros((n_frames, S.H, S.W, 3), np.uint8)
    gt = pts[:n_frames].copy()
    for f in range(n_frames):
        img = sky.copy()
        for (cx, cy, rx, ry, vx) in sc["clouds"]:
            x = (cx + vx * f) % (S.WS + 60) - 30
            S.ellipse(img, x, cy, rx, ry, sc["cloud"], 0.9)
            S.ellipse(img, x + rx * 0.6, cy + 2, rx * 0.7, ry * 0.8, sc["cloud"], 0.9)
        img = img * (1 - fga[..., None]) + fg * fga[..., None]
        for d in sc["distractors"]:
            draw_fn(img, d, f)
        cx, cy = gt[f] * S.SS
        S.draw_duck(img, cx, cy, sc["duck_s"], 1.6 * f, 1.0 if pts[f + 1, 0] - pts[f, 0] >= 0 else -1.0)
        small = img.reshape(S.H, S.SS, S.W, S.SS, 3).mean((1, 3))
        if light_fn is not None:
            small = light_fn(sc, small)
        small = small * sc["gain"] * sc["tint"] + sc.get("offset", 0.0) + rng.normal(0, sc["noise"], small.shape)
        frames[f] = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    gt = gt - 0.25
    vis = (gt[:, 0] >= 0) & (gt[:, 0] < S.W) & (gt[:, 1] >= 0) & (gt[:, 1] < S.H)
    return frames, gt.astype(np.float32), vis


def render_dr(seed, n_frames=S.N_FRAMES):
    rng = np.random.default_rng(seed)
    sc = sample_dr(rng)
    pts = S.duck_track(rng)
    return render_from_sc(sc, pts, rng, n_frames)


def render_dr_nodist(seed, n_frames=S.N_FRAMES):
    rng = np.random.default_rng(seed)
    sc = sample_dr(rng, distract=False)
    return render_from_sc(sc, S.duck_track(rng), rng, n_frames)


# ---- Round 2: DR v2 = the same DR scene + a global haze blend (contrast compression) --------------
# Diagnosis of round 1 (all2): losses track haze, which DR v1 could not produce. The haze draw uses
# its own RNG (seed + 7e6), so a DR v2 scene is exactly the DR v1 scene of the same seed plus haze.
# With prob 0.5: frame <- frame*(1-h) + h*c, h ~ U(0, 0.55), c = random colour with value 0.4-1.
DR_SEED.update(dr2val=110_000, dr2sel=130_000)


def dr_haze(sc, small):
    if "haze" not in sc:
        return small
    h, c = sc["haze"]
    return small * (1 - h) + h * c


def render_dr2(seed, n_frames=S.N_FRAMES):
    rng = np.random.default_rng(seed)
    sc = sample_dr(rng)
    pts = S.duck_track(rng)
    hr = np.random.default_rng(seed + 7_000_000)
    if hr.random() < 0.5:
        sc["haze"] = (float(hr.uniform(0, 0.55)), rand_rgb(hr, (0, 1), (0.4, 1.0)))
    return render_from_sc(sc, pts, rng, n_frames, light_fn=dr_haze)


def dr2_set(split, n):
    return cached(f"{split}_{n}", lambda: make_set_dr(n, DR_SEED[split], fn=render_dr2))


def _render(args):
    fn, seed = args
    return fn(seed)


def make_set_dr(n, base, workers=4, fn=render_dr):
    with Pool(workers) as p:
        out = p.map(_render, [(fn, base + i) for i in range(n)], chunksize=16)
    return tuple(np.stack(z) for z in zip(*out))


def cached(name, maker):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        z = np.load(path); return z["X"], z["GT"], z["VIS"]
    X, GT, VIS = maker()
    np.savez(path, X=X, GT=GT, VIS=VIS, flight_sha=flight_sha())
    return X, GT, VIS


def heldout(cond, n=60):
    """the existing held-out sets (same seeds as eval_tracker.py), cached once to guard against
    flight.py edits by other agents changing the duck trajectories."""
    return cached(f"old_{cond}_{n}", lambda: S.make_set(cond, n))


def dr_set(split, n):
    return cached(f"{split}_{n}", lambda: make_set_dr(n, DR_SEED[split]))


# ======================================================================================
# Fresh held-out families (pre-registered in STATUS.md 2026-11-04 15:30). Rendering is gated on
# checkpoints/r6_tracker_FROZEN.json (hashes of the final checkpoints/params), so these scenes are
# produced only after both R6 trackers are frozen. Seeds 300000+.
#   bg2       city skyline (rectangular buildings, window grids) or autumn canopy (dense
#             orange/red/brown/yellow foliage), day sky
#   light2    spatially varying illumination (soft diagonal sun/shadow boundary, gain 0.35-1.3
#             across the frame) + haze (blend toward pale grey, 0.2-0.45)
#   distract2 2-4 kites (two-colour diamond with a bow tail, wobbling) or drones (grey cross with
#             four dark rotors, hovering random walk)
#   all2      all three
#   bg_f, light_f, distract_f, all_f   the OLD scene.py generators at fresh seeds (330000+ etc.)
# ======================================================================================
FRESH_CONDS = ["bg2", "light2", "distract2", "all2", "bg_f", "light_f", "distract_f", "all_f"]
FRESH_SEED = dict(bg2=300_000, light2=310_000, distract2=320_000, all2=325_000,
                  bg_f=330_000, light_f=340_000, distract_f=350_000, all_f=360_000)
FROZEN = os.path.join(HERE, "checkpoints", "r6_tracker_FROZEN.json")


def poly(img, pts, col):
    """fill a convex polygon (pts (n,2) in 2x coords, any winding) on the 2x canvas."""
    pts = np.asarray(pts, np.float32)
    x0, x1 = int(max(pts[:, 0].min() - 1, 0)), int(min(pts[:, 0].max() + 2, S.WS))
    y0, y1 = int(max(pts[:, 1].min() - 1, 0)), int(min(pts[:, 1].max() + 2, S.HS))
    if x0 >= x1 or y0 >= y1:
        return
    X, Y = S.XX[y0:y1, x0:x1], S.YY[y0:y1, x0:x1]
    sgn = []
    for a, b in zip(pts, np.roll(pts, -1, 0)):
        sgn.append((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]))
    sgn = np.stack(sgn)
    m = (sgn >= 0).all(0) | (sgn <= 0).all(0)
    img[y0:y1, x0:x1][m] = np.asarray(col, np.float32)


def sample_fresh(rng, cond):
    sc = dict(kind=cond)
    sc["sky_top"] = rng.uniform([0.25, 0.50, 0.80], [0.45, 0.70, 0.98])     # day sky as in scene.py 'train'
    sc["sky_bot"] = rng.uniform([0.65, 0.80, 0.92], [0.85, 0.92, 1.00])
    sc["ground"] = rng.uniform([0.18, 0.45, 0.12], [0.35, 0.65, 0.28])
    sc["tree"] = rng.uniform([0.05, 0.25, 0.08], [0.15, 0.40, 0.15])
    sc["cloud"] = rng.uniform([0.92, 0.92, 0.94], [1.0, 1.0, 1.0])
    sc["hills"] = [(rng.uniform(4, 12), rng.uniform(0.02, 0.08), rng.uniform(0, 6.28)) for _ in range(3)]
    sc["horizon"] = rng.uniform(128, 146)
    sc["trees"] = [(rng.uniform(0, S.WS), rng.uniform(8, 16)) for _ in range(rng.integers(1, 6))]
    sc["clouds"] = [(rng.uniform(-20, S.WS), rng.uniform(5, 100), rng.uniform(8, 22), rng.uniform(4, 9),
                     rng.uniform(0.3, 1.5) * rng.choice([-1, 1])) for _ in range(rng.integers(1, 5))]
    sc["gain"] = rng.uniform(0.85, 1.15); sc["tint"] = rng.uniform(0.95, 1.05, 3); sc["noise"] = rng.uniform(0.005, 0.025)
    if cond in ("bg2", "all2"):
        sc["bg2"] = str(rng.choice(["city", "autumn"]))
        if sc["bg2"] == "city":
            b, x = [], -10.0
            while x < S.WS + 10:
                w = rng.uniform(10, 30); h = rng.uniform(30, 100)
                b.append((x, w, h, rng.uniform(0.25, 0.75, 3) * rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.6)))
                x += w + rng.uniform(-4, 6)
            sc["buildings"] = b
        else:
            sc["leaves"] = [(rng.uniform(-10, S.WS + 10), rng.uniform(70, 160), rng.uniform(6, 16), rng.uniform(5, 12),
                             np.array([rng.uniform(0.55, 0.95), rng.uniform(0.15, 0.6), rng.uniform(0.02, 0.15)]))
                            for _ in range(rng.integers(40, 70))]
    if cond in ("light2", "all2"):
        ang = rng.uniform(0, 2 * np.pi)
        sc["illum"] = (np.cos(ang), np.sin(ang), rng.uniform(-20, 20), rng.uniform(4, 15),
                       rng.uniform(0.35, 0.6), rng.uniform(0.95, 1.3))
        sc["haze"] = (rng.uniform(0.2, 0.45), rng.uniform(0.75, 0.9, 3))
    ds = []
    if cond in ("distract2", "all2"):
        for _ in range(rng.integers(2, 5)):
            kind = str(rng.choice(["kite", "drone"]))
            ds.append(dict(kind=kind, x=rng.uniform(0, S.WS), y=rng.uniform(10, 120), vx=rng.uniform(-1.5, 1.5),
                           vy=rng.uniform(-0.6, 0.6), s=rng.uniform(0.8, 1.3), ph=rng.uniform(0, 6.28),
                           cols=[rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)], seed=int(rng.integers(1 << 30))))
    else:
        for _ in range(rng.integers(0, 4)):     # ordinary crows/gulls as in 'train'
            kind = str(rng.choice(["crow", "gull"]))
            ds.append(dict(kind=kind, x=rng.uniform(0, S.WS), y=rng.uniform(10, 120),
                           vx=rng.uniform(1.5, 5) * rng.choice([-1, 1]), vy=rng.uniform(-1, 1), wob=0.0, wf=0.0,
                           s=rng.uniform(0.8, 1.2), ph=rng.uniform(0, 6.28), fr=1.3 if kind == "crow" else 1.1))
    sc["distractors"] = ds
    sc["duck_s"] = rng.uniform(0.8, 1.2)
    return sc


def fresh_background(sc):
    sky, fg, fga = S.background(sc)
    if sc.get("bg2") == "city":
        for (x, w, h, col, lit) in sc["buildings"]:
            top = S.HS - h
            poly(fg, [(x, top), (x + w, top), (x + w, S.HS), (x, S.HS)], col)
            fga[int(max(top, 0)):, int(max(x, 0)):int(min(x + w, S.WS))] = 1
            for wy in np.arange(top + 3, S.HS - 3, 6):
                for wx in np.arange(x + 3, x + w - 3, 5):
                    wc = (0.95, 0.85, 0.45) if (hash((int(wx), int(wy))) % 100) / 100 < lit else col * 0.45
                    poly(fg, [(wx, wy), (wx + 2.5, wy), (wx + 2.5, wy + 3), (wx, wy + 3)], wc)
    elif sc.get("bg2") == "autumn":
        for (x, y, rx, ry, col) in sc["leaves"]:
            S.ellipse(fg, x, y, rx, ry, col)
            fga[((S.XX - x) / rx) ** 2 + ((S.YY - y) / ry) ** 2 <= 1] = 1
    return sky, fg, fga


def draw_fresh_distractor(img, d, f):
    if d["kind"] in ("crow", "gull"):
        return draw_distractor(img, d, f)
    r = np.random.default_rng(d["seed"] + f)
    if d["kind"] == "kite":
        x = (d["x"] + d["vx"] * f + 30) % (S.WS + 60) - 30
        y = d["y"] + d["vy"] * f + 4 * np.sin(0.5 * f + d["ph"])
        s = d["s"]; tilt = 0.25 * np.sin(0.7 * f + d["ph"])
        c, sn = np.cos(tilt), np.sin(tilt)
        P = lambda u, v: (x + s * (u * c - v * sn), y + s * (u * sn + v * c))
        poly(img, [P(0, -9), P(6, 0), P(0, 0)], d["cols"][0]); poly(img, [P(0, -9), P(-6, 0), P(0, 0)], d["cols"][1])
        poly(img, [P(0, 0), P(6, 0), P(0, 12)], d["cols"][1]); poly(img, [P(0, 0), P(-6, 0), P(0, 12)], d["cols"][0])
        for k in range(1, 4):   # tail bows
            bx, by = P(2 * np.sin(0.8 * f + k), 12 + 6 * k)
            poly(img, [(bx - 2, by - 1.5), (bx + 2, by + 1.5), (bx + 2, by - 1.5), (bx - 2, by + 1.5)], d["cols"][k % 2])
    else:  # drone: hovering random walk around its anchor
        d.setdefault("pos", [d["x"], d["y"]]); d.setdefault("f", -1)
        if f != d["f"]:
            d["pos"] = [d["pos"][0] + r.normal(0, 1.2) + 0.3 * d["vx"], d["pos"][1] + r.normal(0, 1.0)]; d["f"] = f
        x, y = d["pos"]; s = d["s"]
        poly(img, [(x - 8 * s, y - 1), (x + 8 * s, y - 1), (x + 8 * s, y + 1), (x - 8 * s, y + 1)], (0.35, 0.35, 0.38))
        poly(img, [(x - 3 * s, y - 2.5 * s), (x + 3 * s, y - 2.5 * s), (x + 3 * s, y + 2.5 * s), (x - 3 * s, y + 2.5 * s)],
             (0.55, 0.56, 0.6))
        for ox in (-8, 8):
            S.ellipse(img, x + ox * s, y - 2 * s, 3.5 * s, 1.0 * s, (0.12, 0.12, 0.14))
    return None


def fresh_light(sc, small):
    if "illum" not in sc:
        return small
    cx, cy, off, width, g_dark, g_bright = sc["illum"]
    yy, xx = np.mgrid[0:S.H, 0:S.W].astype(np.float32)
    t = 1 / (1 + np.exp(-(((xx - S.W / 2) * cx + (yy - S.H / 2) * cy) - off) / width))
    small = small * (g_dark + (g_bright - g_dark) * t)[..., None]
    h, col = sc["haze"]
    return small * (1 - h) + h * col


def render_fresh(seed, cond):
    rng = np.random.default_rng(seed)
    if cond.endswith("_f"):
        return S.render_scene(seed, cond[:-2])
    sc = sample_fresh(rng, cond)
    pts = S.duck_track(rng)
    return render_from_sc(sc, pts, rng, bg_fn=fresh_background, draw_fn=draw_fresh_distractor, light_fn=fresh_light)


def _render_fresh(args):
    return render_fresh(*args)


def fresh_set(cond, n=60):
    assert os.path.exists(FROZEN), "fresh held-out sets are rendered only after the trackers are frozen"
    def mk():
        with Pool(4) as p:
            out = p.map(_render_fresh, [(FRESH_SEED[cond] + i, cond) for i in range(n)], chunksize=8)
        return tuple(np.stack(z) for z in zip(*out))
    return cached(f"fresh_{cond}_{n}", mk)


# ======================================================================================
# Round-2 held-out families (pre-registered in STATUS.md 2026-11-04 16:31), seeds 400000+.
# Rendering is gated on checkpoints/r6_tracker_FROZEN_r2.json.
#   bg3       ocean (water band, horizontal wave stripes, white caps, sandy shore) or mountain
#             ranges (layered jagged ridges with snow caps)
#   light3    sun glare: additive radial glow at a random point (0.3-0.6) + 2-3 faint flare discs
#   distract3 paper planes (white/pastel gliding triangles) and bats (dark flapping wings, erratic)
#   all3      all three
#   all2_f2   the (now dev-visible) all2 generator at fresh seeds 390000+
# ======================================================================================
FRESH3_CONDS = ["bg3", "light3", "distract3", "all3", "all2_f2"]
FRESH3_SEED = dict(bg3=400_000, light3=410_000, distract3=420_000, all3=430_000, all2_f2=390_000)
FROZEN_R2 = os.path.join(HERE, "checkpoints", "r6_tracker_FROZEN_r2.json")


def sample_fresh3(rng, cond):
    sc = sample_fresh(rng, "base")          # day palette, crows/gulls, normal lighting
    sc["kind"] = cond
    if cond in ("bg3", "all3"):
        sc["bg3"] = str(rng.choice(["ocean", "mountains"]))
        if sc["bg3"] == "ocean":
            sc["sea_y"] = rng.uniform(70, 110); sc["sea"] = rng.uniform([0.05, 0.3, 0.4], [0.2, 0.5, 0.7])
            sc["wave"] = (rng.uniform(0.3, 0.8), rng.uniform(0, 6.28), rng.uniform(0.08, 0.2))
            sc["caps"] = [(rng.uniform(0, S.WS), rng.uniform(0, 1), rng.uniform(2, 5)) for _ in range(rng.integers(10, 25))]
            sc["sand"] = rng.uniform([0.85, 0.75, 0.5], [0.95, 0.85, 0.65])
        else:
            sc["ridges"] = []
            for k in range(int(rng.integers(2, 4))):
                xs = np.sort(np.r_[0, rng.uniform(0, S.WS, 8), S.WS])
                base = 60 + 30 * k
                sc["ridges"].append((xs, base + rng.uniform(-25, 25, len(xs)),
                                     np.array([0.35, 0.4, 0.5]) * (1 - 0.2 * k) + rng.uniform(-0.05, 0.05, 3)))
    if cond in ("light3", "all3"):
        sc["glare"] = (rng.uniform(0, S.W), rng.uniform(0, S.H * 0.7), rng.uniform(10, 25), rng.uniform(0.3, 0.6),
                       np.array([1.0, rng.uniform(0.85, 0.97), rng.uniform(0.6, 0.85)]),
                       [(rng.uniform(0.3, 1.6), rng.uniform(3, 8), rng.uniform(0.08, 0.2)) for _ in range(rng.integers(2, 4))])
    if cond in ("distract3", "all3"):
        ds = []
        for _ in range(rng.integers(2, 5)):
            kind = str(rng.choice(["plane", "bat"]))
            ds.append(dict(kind=kind, x=rng.uniform(0, S.WS), y=rng.uniform(10, 120), vx=rng.uniform(1, 4) * rng.choice([-1, 1]),
                           vy=rng.uniform(-0.3, 0.6), s=rng.uniform(0.8, 1.3), ph=rng.uniform(0, 6.28),
                           col=(rng.uniform(0.85, 1, 3) if rng.random() < 0.5 else rng.uniform(0.6, 1, 3)),
                           seed=int(rng.integers(1 << 30))))
        sc["distractors"] = ds
    return sc


def fresh3_background(sc):
    sky, fg, fga = S.background(sc)
    if sc.get("bg3") == "ocean":
        top = sc["sea_y"]; m = S.YY >= top
        amp, ph, fr = sc["wave"]
        shade = 1 + amp * 0.35 * np.sin(fr * (S.YY - top) * (1 + (S.YY - top) / 40) + 0.05 * S.XX + ph)
        fg[:] = 0; fga[:] = 0
        fg[m] = (sc["sea"] * shade[m][:, None]); fga[m] = 1
        for (x, t, r) in sc["caps"]:
            y = top + t * (S.HS - 18 - top)
            S.ellipse(fg, x, y, r * 1.6, 0.8, (0.95, 0.97, 1.0))
        sand = S.YY >= S.HS - 14 + 3 * np.sin(0.04 * S.XX)
        fg[sand] = sc["sand"]; fga[sand] = 1
    elif sc.get("bg3") == "mountains":
        fg[:] = 0; fga[:] = 0
        for xs, ys, col in sc["ridges"]:           # far ridges first; jagged ridge = piecewise-linear top
            top = np.interp(np.arange(S.WS), xs, ys)
            m = S.YY >= top[None, :]
            fg[m] = col; fga[m] = 1
            snow = m & (S.YY < top[None, :] + 5) & (top[None, :] < ys.min() + 12)
            fg[snow] = (0.93, 0.94, 0.97)
        g = S.YY >= sc["horizon"] + 10
        fg[g] = sc["ground"]; fga[g] = 1
    return sky, fg, fga


def draw_fresh3_distractor(img, d, f):
    if d["kind"] in ("crow", "gull"):
        return draw_distractor(img, d, f)
    if d["kind"] == "plane":
        x = (d["x"] + d["vx"] * f + 30) % (S.WS + 60) - 30; y = d["y"] + d["vy"] * f + 2 * np.sin(0.3 * f + d["ph"])
        s, face = d["s"], np.sign(d["vx"])
        poly(img, [(x + 9 * s * face, y), (x - 7 * s * face, y - 4 * s), (x - 4 * s * face, y + 1)], d["col"])
        poly(img, [(x + 9 * s * face, y), (x - 4 * s * face, y + 1), (x - 7 * s * face, y + 3 * s)], np.asarray(d["col"]) * 0.8)
    else:  # bat: erratic velocity, fast flapping dark wings
        d.setdefault("pos", [d["x"], d["y"]]); d.setdefault("v", [d["vx"], d["vy"]]); d.setdefault("f", -1)
        if f != d["f"]:
            r = np.random.default_rng(d["seed"] + f)
            d["v"] = [np.clip(d["v"][0] + r.normal(0, 1.5), -5, 5), np.clip(d["v"][1] + r.normal(0, 1.5), -4, 4)]
            d["pos"] = [(d["pos"][0] + d["v"][0] + 30) % (S.WS + 60) - 30, float(np.clip(d["pos"][1] + d["v"][1], 5, 140))]
            d["f"] = f
        x, y = d["pos"]; s = d["s"]; tip = 6 * s * np.sin(2.2 * f + d["ph"])
        col = (0.10, 0.08, 0.11)
        S.ellipse(img, x, y, 2.5 * s, 3 * s, col)
        poly(img, [(x - 1, y - 1), (x - 10 * s, y - tip), (x - 6 * s, y + 2)], col)
        poly(img, [(x + 1, y - 1), (x + 10 * s, y - tip), (x + 6 * s, y + 2)], col)
    return None


def fresh3_light(sc, small):
    if "glare" not in sc:
        return small
    gx, gy, sig, a, col, flares = sc["glare"]
    yy, xx = np.mgrid[0:S.H, 0:S.W].astype(np.float32)
    small = small + a * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sig ** 2))[..., None] * col
    for (t, r, b) in flares:   # discs along the line from the glare centre through the image centre
        fx, fy = gx + t * (S.W / 2 - gx), gy + t * (S.H / 2 - gy)
        small = small + b * (((xx - fx) ** 2 + (yy - fy) ** 2) <= r * r)[..., None] * col
    return small


def render_fresh3(seed, cond):
    if cond == "all2_f2":
        return render_fresh(seed, "all2")
    rng = np.random.default_rng(seed)
    sc = sample_fresh3(rng, cond)
    pts = S.duck_track(rng)
    return render_from_sc(sc, pts, rng, bg_fn=fresh3_background, draw_fn=draw_fresh3_distractor, light_fn=fresh3_light)


def _render_fresh3(args):
    return render_fresh3(*args)


def fresh3_set(cond, n=60):
    assert os.path.exists(FROZEN_R2), "round-2 held-out sets are rendered only after the round-2 freeze"
    def mk():
        with Pool(4) as p:
            out = p.map(_render_fresh3, [(FRESH3_SEED[cond] + i, cond) for i in range(n)], chunksize=8)
        return tuple(np.stack(z) for z in zip(*out))
    return cached(f"fresh3_{cond}_{n}", mk)
