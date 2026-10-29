"""Procedural labeled monster dataset (PIL + numpy).

Each monster has 6 discrete attributes plus continuous nuisance jitter
(position, scale, rotation, color shade, background tint, spot placement).
Rendered at 128x128 with anti-aliasing, then downsampled to 32x32 and 64x64.

    ../.venv/bin/python data.py              # 40k train / 8k test -> data/monsters.npz
    ../.venv/bin/python data.py --grid       # also writes media/real_grid.png
"""
import argparse, math, os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

ATTRS = {
    "shape":   ["blob", "square", "triangle", "star"],
    "color":   ["red", "orange", "yellow", "green", "blue", "purple"],
    "eyes":    ["1", "2", "3"],
    "horns":   ["none", "horns"],
    "mouth":   ["smile", "fangs", "open"],
    "pattern": ["plain", "spots", "stripes"],
}
ATTR_NAMES = list(ATTRS)
N_CLASSES = [len(v) for v in ATTRS.values()]
N_COMBOS = int(np.prod(N_CLASSES))  # 1296

BASE_RGB = {
    "red": (215, 45, 45), "orange": (240, 135, 25), "yellow": (235, 205, 30),
    "green": (55, 170, 60), "blue": (45, 95, 220), "purple": (140, 60, 190),
}
R = 128  # render resolution


def _shade(rgb, f):
    return tuple(int(max(0, min(255, c * f))) for c in rgb)


def _body_polygon(shape, cx, cy, s, rng):
    """Return polygon points for the body (s = half-size in px)."""
    if shape == "blob":
        n = 48
        k = rng.integers(3, 6)
        ph = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(0.04, 0.09)
        pts = []
        for i in range(n):
            t = 2 * math.pi * i / n
            r = s * (1 + amp * math.sin(k * t + ph))
            pts.append((cx + r * math.cos(t), cy + 0.92 * r * math.sin(t)))
        return pts
    if shape == "square":
        r = 0.18 * s  # rounded corners via polygon
        pts = []
        for (qx, qy, a0) in [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]:
            ccx, ccy = cx + qx * (s - r) * 0.92, cy + qy * (s - r) * 0.92
            for j in range(7):
                a = math.radians(a0 + 15 * j)
                pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
        return pts
    if shape == "triangle":
        return [(cx, cy - 1.15 * s), (cx + 1.15 * s, cy + 0.85 * s), (cx - 1.15 * s, cy + 0.85 * s)]
    if shape == "star":
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            r = s * (1.12 if i % 2 == 0 else 0.68)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts
    raise ValueError(shape)


def render(attr_idx, seed):
    """attr_idx: tuple of 6 ints. Returns uint8 RGB array (R, R, 3)."""
    rng = np.random.default_rng(seed)
    shape, color, eyes, horns, mouth, pattern = [ATTRS[n][i] for n, i in zip(ATTR_NAMES, attr_idx)]
    bg = tuple(int(c) for c in rng.integers(222, 250, size=3))
    img = Image.new("RGB", (R, R), bg)
    layer = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    s = R * 0.30 * rng.uniform(0.88, 1.05)
    cx, cy = R / 2, R / 2 + 0.04 * R
    shade = rng.uniform(0.85, 1.12)
    body = _shade(BASE_RGB[color], shade)
    dark = _shade(body, 0.62)

    # face anchor per shape
    fy = {"blob": cy - 0.05 * s, "square": cy - 0.05 * s, "triangle": cy + 0.18 * s, "star": cy}[shape]
    fw = {"blob": 1.0, "square": 1.0, "triangle": 0.85, "star": 0.85}[shape]

    # horns (drawn first so the body overlaps their base)
    if horns == "horns":
        hc = (240, 228, 185)
        top = {"blob": cy - 0.85 * s, "square": cy - 0.85 * s, "triangle": cy - 0.45 * s, "star": cy - 0.60 * s}[shape]
        hx = {"blob": 0.48, "square": 0.55, "triangle": 0.42, "star": 0.40}[shape] * s
        for sgn in (-1, 1):
            bx = cx + sgn * hx
            d.polygon([(bx - 0.16 * s, top + 0.15 * s), (bx + 0.16 * s, top + 0.15 * s),
                       (bx + sgn * 0.22 * s, top - 0.55 * s)], fill=hc, outline=(70, 55, 40), width=3)

    poly = _body_polygon(shape, cx, cy, s, rng)
    d.polygon(poly, fill=body)

    # pattern, clipped to the body mask
    if pattern != "plain":
        pat = Image.new("RGBA", (R, R), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pat)
        if pattern == "spots":
            for _ in range(int(rng.integers(6, 10))):
                px, py = cx + rng.uniform(-1, 1) * s, cy + rng.uniform(-1, 1) * s
                if abs(py - fy) < 0.45 * s and abs(px - cx) < 0.7 * s:
                    continue  # keep the face clear
                pr = s * rng.uniform(0.13, 0.19)
                pd.ellipse([px - pr, py - pr, px + pr, py + pr], fill=dark)
        else:  # stripes
            w = 0.13 * s
            off = rng.uniform(0, 0.4 * s)
            y = cy - 1.3 * s + off
            while y < cy + 1.3 * s:
                pd.rectangle([cx - 1.4 * s, y, cx + 1.4 * s, y + w], fill=dark)
                y += 0.38 * s
        mask = Image.new("L", (R, R), 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        layer.paste(pat, (0, 0), Image.composite(pat, Image.new("RGBA", (R, R)), mask).split()[3])
        d = ImageDraw.Draw(layer)

    # eyes
    ne = int(eyes)
    er = 0.17 * s * (1.0 if ne < 3 else 0.85)
    xs = {1: [0.0], 2: [-0.36, 0.36], 3: [-0.48, 0.0, 0.48]}[ne]
    ey = fy - 0.18 * s
    for i, ex in enumerate(xs):
        ex = cx + ex * s * fw
        eyy = ey - (0.12 * s if (ne == 3 and i == 1) else 0)
        rr = er * (1.25 if ne == 1 else 1.0)
        d.ellipse([ex - rr, eyy - rr, ex + rr, eyy + rr], fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        pr = rr * 0.5
        jx, jy = rng.uniform(-0.25, 0.25) * rr, rng.uniform(-0.2, 0.25) * rr
        d.ellipse([ex + jx - pr, eyy + jy - pr, ex + jx + pr, eyy + jy + pr], fill=(15, 15, 15))

    # mouth
    my = fy + 0.33 * s
    mw = 0.42 * s * fw
    if mouth == "smile":
        d.arc([cx - mw, my - 0.35 * s, cx + mw, my + 0.12 * s], 20, 160, fill=(20, 20, 20), width=max(3, int(0.07 * s)))
    elif mouth == "fangs":
        d.rectangle([cx - mw, my - 0.03 * s, cx + mw, my + 0.05 * s], fill=(20, 20, 20))
        for sgn in (-1, 1):
            fx = cx + sgn * 0.22 * s * fw
            d.polygon([(fx - 0.09 * s, my + 0.04 * s), (fx + 0.09 * s, my + 0.04 * s), (fx, my + 0.26 * s)],
                      fill=(255, 255, 255), outline=(20, 20, 20))
    else:  # open "O" mouth
        d.ellipse([cx - 0.17 * s, my - 0.1 * s, cx + 0.17 * s, my + 0.2 * s], fill=(70, 10, 20), outline=(20, 20, 20), width=2)

    # pose jitter: rotation + translation
    ang = rng.uniform(-12, 12)
    layer = layer.rotate(ang, resample=Image.BICUBIC, center=(cx, cy),
                         translate=(rng.uniform(-0.07, 0.07) * R, rng.uniform(-0.06, 0.06) * R))
    img.paste(layer, (0, 0), layer)
    return np.asarray(img)


def _work(args):
    attr_idx, seed = args
    big = Image.fromarray(render(attr_idx, seed))
    return (np.asarray(big.resize((32, 32), Image.LANCZOS)), np.asarray(big.resize((64, 64), Image.LANCZOS)))


def sample_attrs(n, rng):
    return np.stack([rng.integers(0, k, size=n) for k in N_CLASSES], axis=1)


def combo_id(labels):
    """Map (N,6) attribute labels to a single combo index in [0, 1296)."""
    labels = np.asarray(labels)
    idx = np.zeros(len(labels), dtype=np.int64)
    for j, k in enumerate(N_CLASSES):
        idx = idx * k + labels[:, j]
    return idx


def build(n_train=40000, n_test=8000, seed=0, out="data/monsters.npz", workers=4):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    labels = sample_attrs(n, rng)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    with Pool(workers) as p:
        res = p.map(_work, [(tuple(l), int(s)) for l, s in zip(labels, seeds)], chunksize=256)
    x32 = np.stack([r[0] for r in res]); x64 = np.stack([r[1] for r in res])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, x32_train=x32[:n_train], x32_test=x32[n_train:], x64_train=x64[:n_train],
                        x64_test=x64[n_train:], y_train=labels[:n_train], y_test=labels[n_train:],
                        seeds=seeds)
    return out


def load(res=32, path=None):
    path = path or os.environ.get("MONSTER_DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monsters.npz")
    z = np.load(path)
    return z[f"x{res}_train"], z["y_train"], z[f"x{res}_test"], z["y_test"]


def save_grid(imgs, path, nrow=None, scale=3, pad=1):
    """imgs: (N,H,W,3) uint8. Writes an upscaled PNG grid."""
    imgs = np.asarray(imgs)
    n, h, w, _ = imgs.shape
    nrow = nrow or int(math.ceil(math.sqrt(n)))
    ncol = int(math.ceil(n / nrow))
    grid = np.full((ncol * (h + pad) + pad, nrow * (w + pad) + pad, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = imgs[i]
    im = Image.fromarray(grid)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_test", type=int, default=8000)
    ap.add_argument("--out", default="data/monsters.npz")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    out = build(a.n_train, a.n_test, out=a.out)
    print("wrote", out)
    if a.grid:
        xtr, ytr, _, _ = load(32, a.out)
        save_grid(xtr[:64], "media/real_grid_32.png", nrow=8, scale=4)
        save_grid(load(64, a.out)[0][:64], "media/real_grid_64.png", nrow=8, scale=2)
