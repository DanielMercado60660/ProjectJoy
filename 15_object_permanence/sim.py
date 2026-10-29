"""Synthetic 'object permanence' videos: colored balls bouncing behind opaque vertical walls.

Coordinates are continuous pixels: pixel (i, j) covers x in [j, j+1), y in [i, i+1).
A ball's center lives in [R, 64 - R] on both axes and bounces elastically off the frame edges.

We store only the *state* of each sequence (positions, occluders, colors); frames are rendered
on the fly (cheap, vectorized numpy), which keeps the fixed splits tiny on disk.

Variants (flags):
  collisions   - balls bounce off each other (equal-mass elastic)
  same_color   - all balls look identical; slot k = k-th ball from the left at frame 0, so the
                 identity must be tracked by motion
  hidden_turn  - when a ball becomes fully hidden, with prob p_turn its velocity is redrawn
                 (it may come back out the way it went in): uncertain hidden motion
"""
import argparse
import os

import numpy as np

H = W = 64
R = 3.5          # ball radius in pixels
T = 40           # frames per sequence
K = 3            # ball slots (2-3 balls present)
PALETTE = np.array([[0.92, 0.18, 0.15],   # red
                    [0.20, 0.80, 0.25],   # green
                    [0.25, 0.40, 1.00]],  # blue
                   dtype=np.float32)
SAME = np.array([0.95, 0.85, 0.20], dtype=np.float32)  # yellow, used for every ball in same_color
BG = np.array([0.05, 0.05, 0.07], dtype=np.float32)

# split configs -------------------------------------------------------------------------------
BASE = dict(n_occ=(1, 2), occ_w=(10, 16), n_balls=(2, 3), speed=(0.6, 2.2), collisions=False,
            same_color=False, hidden_turn=0.0, min_long_occ=0)
SPLITS = {
    # name: (n_seq, seed, overrides)
    "train":     (40000, 1, {}),
    "val":       (1000, 2, {}),
    "test":      (1000, 3, {}),
    "test_long": (1000, 4, dict(n_occ=(1, 1), occ_w=(18, 26), min_long_occ=8)),
    # control experiment: training walls that cover test_long's widths (is the overconfidence on
    # long occlusions just missing training data?)
    "train_wide": (40000, 5, dict(occ_w=(10, 26))),
}
# go-deeper variant: collisions + identical-looking balls + uncertain hidden motion
HARD = dict(collisions=True, same_color=True, hidden_turn=0.3)
SPLITS_HARD = {
    "hard_train":     (40000, 11, HARD),
    "hard_val":       (1000, 12, HARD),
    "hard_test":      (1000, 13, HARD),
    "hard_test_long": (1000, 14, dict(HARD, n_occ=(1, 1), occ_w=(18, 26), min_long_occ=8)),
}
# factor ablation of the hard world (test-only; all keep identical-looking balls so the hard model
# can be applied): which ingredient makes tracking hard?
SPLITS_ABL = {
    "abl_ident":      (1000, 21, dict(same_color=True)),
    "abl_ident_coll": (1000, 22, dict(same_color=True, collisions=True)),
    "abl_ident_turn": (1000, 23, dict(same_color=True, hidden_turn=0.3)),
}
ALL_SPLITS = {**SPLITS, **SPLITS_HARD, **SPLITS_ABL}


def occ_visible_frac(x, occ):
    """Fraction of a ball's horizontal extent that is not covered by occluders (vertical bars span
    the full height, so visibility only depends on x).  Approximated on a fine grid across the
    disk weighted by chord length."""
    xs = x[..., None] + R * np.linspace(-0.98, 0.98, 25)           # sample across diameter
    wts = np.sqrt(np.clip(1 - ((xs - x[..., None]) / R) ** 2, 0, None))
    hidden = np.zeros(xs.shape, bool)
    for (a, b) in occ:
        hidden |= (xs >= a) & (xs < b)
    return (wts * ~hidden).sum(-1) / wts.sum(-1)


def simulate_one(rng, cfg):
    """Returns pos (T,K,2), present (K,), occ (2,2) [x0,x1] (unused rows = -1), colors (K,3)."""
    for _attempt in range(1000):
        n_occ = rng.integers(cfg["n_occ"][0], cfg["n_occ"][1] + 1)
        occ = []
        for _ in range(n_occ):
            for _try in range(100):
                w = rng.uniform(*cfg["occ_w"])
                a = rng.uniform(10, W - 10 - w)
                if all(a + w + 8 < o[0] or o[1] + 8 < a for o in occ):
                    occ.append((a, a + w))
                    break
        occ.sort()
        nb = rng.integers(cfg["n_balls"][0], cfg["n_balls"][1] + 1)
        present = np.zeros(K, bool)
        present[:nb] = True
        p = rng.uniform(R, W - R, size=(K, 2))
        v = rng.uniform(*cfg["speed"], size=(K, 2)) * rng.choice([-1, 1], size=(K, 2))
        pos = np.zeros((T, K, 2))
        was_hidden = np.zeros(K, bool)
        turned = np.zeros((T, K), bool)   # turned[t,k]: velocity redrawn after frame t (hidden)
        for t in range(T):
            pos[t] = p
            p = p + v
            for d in range(2):
                lo, hi = p[:, d] < R, p[:, d] > W - R
                p[lo, d] = 2 * R - p[lo, d]
                p[hi, d] = 2 * (W - R) - p[hi, d]
                v[lo | hi, d] *= -1
            if cfg["collisions"]:
                for i in range(nb):
                    for j in range(i + 1, nb):
                        dvec = p[j] - p[i]
                        dist = np.linalg.norm(dvec)
                        if 1e-6 < dist < 2 * R:
                            n = dvec / dist
                            rel = np.dot(v[i] - v[j], n)
                            if rel > 0:  # approaching
                                v[i] -= rel * n
                                v[j] += rel * n
            if cfg["hidden_turn"] > 0:
                hid = occ_visible_frac(p[:, 0], occ) == 0
                newly = hid & ~was_hidden
                for k in np.where(newly[:nb])[0]:
                    if rng.random() < cfg["hidden_turn"]:
                        turned[t, k] = True
                        sp = rng.uniform(*cfg["speed"], size=2) * rng.choice([-1, 1], size=2)
                        v[k] = sp
                was_hidden = hid
        vis = occ_visible_frac(pos[..., 0], occ)                # (T,K)
        # accept only if every present ball is fully visible in frames 0 and 1 and balls don't
        # start overlapping
        if (vis[:2, :nb] < 0.999).any():
            continue
        d0 = np.linalg.norm(pos[0, :nb, None] - pos[0, None, :nb], axis=-1) + np.eye(nb) * 99
        if d0.min() < 2 * R + 1:
            continue
        if cfg["min_long_occ"] > 0:
            runs = hidden_runs(vis[:, :nb] == 0)
            if not any(L >= cfg["min_long_occ"] and end < T for (_, _, L, end) in runs):
                continue
        if cfg["same_color"]:
            # slot order = left-to-right at frame 0, so identity is defined without color
            order = np.argsort(pos[0, :nb, 0])
            pos[:, :nb] = pos[:, order]
            turned[:, :nb] = turned[:, order]
            colors = np.tile(SAME, (K, 1))
        else:
            colors = PALETTE.copy()
        occ_arr = -np.ones((2, 2))
        occ_arr[:len(occ)] = occ
        return pos.astype(np.float32), present, occ_arr.astype(np.float32), colors, turned
    raise RuntimeError("rejection sampling failed")


def hidden_runs(hidden):
    """hidden: (T,K) bool.  Returns list of (k, start, length, end) for maximal fully-hidden runs;
    end == T means the ball never reappeared within the clip."""
    out = []
    Tn, Kn = hidden.shape
    for k in range(Kn):
        t = 0
        while t < Tn:
            if hidden[t, k]:
                s = t
                while t < Tn and hidden[t, k]:
                    t += 1
                out.append((k, s, t - s, t))
            else:
                t += 1
    return out


def make_split(name):
    n, seed, over = ALL_SPLITS[name]
    cfg = dict(BASE, **over)
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, T, K, 2), np.float32)
    present = np.zeros((n, K), bool)
    occ = np.zeros((n, 2, 2), np.float32)
    colors = np.zeros((n, K, 3), np.float32)
    turned = np.zeros((n, T, K), bool)
    for i in range(n):
        pos[i], present[i], occ[i], colors[i], turned[i] = simulate_one(rng, cfg)
    vis = np.stack([occ_visible_frac(pos[i, ..., 0], [o for o in occ[i] if o[0] >= 0])
                    for i in range(n)]).astype(np.float32)
    vis[~np.broadcast_to(present[:, None, :], vis.shape)] = 0
    return dict(pos=pos, present=present, occ=occ, colors=colors, vis=vis, turned=turned,
                noise_seed=np.int64(seed * 7919))


def load_split(name, data_dir=None):
    data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    path = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(path):
        d = make_split(name)
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(path, **d)
    z = np.load(path)
    return {k: z[k] for k in z.files}


# rendering -----------------------------------------------------------------------------------
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5


def render(pos, present, occ, colors, noise=0.02, rng=None):
    """pos (B,T,K,2), present (B,K), occ (B,2,2), colors (B,K,3) -> frames (B,T,H,W,3) float32.
    Balls are drawn anti-aliased in slot order (later slots on top), then the opaque walls."""
    B, Tn = pos.shape[:2]
    img = np.broadcast_to(BG, (B, Tn, H, W, 3)).copy()
    for k in range(pos.shape[2]):
        dx = _XX - pos[:, :, k, 0, None, None]
        dy = _YY - pos[:, :, k, 1, None, None]
        a = np.clip(R + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1)
        a *= present[:, None, k, None, None]
        img = img * (1 - a[..., None]) + a[..., None] * colors[:, None, None, None, k]
    for m in range(occ.shape[1]):
        x0, x1 = occ[:, m, 0], occ[:, m, 1]
        cover = np.clip(np.minimum(_XX[0] + 0.5, x1[:, None]) - np.maximum(_XX[0] - 0.5, x0[:, None]), 0, 1)
        cover *= (x0 >= 0)[:, None]                                          # (B,W)
        cover = cover[:, None, None, :, None]
        img = img * (1 - cover) + cover * np.float32(0.45)
    if noise > 0:
        rng = rng or np.random.default_rng()
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def render_split(d, idx, noise=0.02):
    """Deterministic frames for evaluation (noise seeded by split + index)."""
    out = []
    for i in idx:
        rng = np.random.default_rng(int(d["noise_seed"]) * 100003 + int(i))
        out.append(render(d["pos"][i:i + 1], d["present"][i:i + 1], d["occ"][i:i + 1],
                          d["colors"][i:i + 1], noise, rng))
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args()
    for s in args.splits:
        d = load_split(s)
        hid = d["vis"] == 0
        runs = []
        for i in range(len(d["pos"])):
            runs += [r for r in hidden_runs(hid[i][:, d["present"][i]]) if r[3] < T]
        L = np.array([r[2] for r in runs]) if runs else np.zeros(1)
        print(f"{s}: n={len(d['pos'])} hidden-frac={hid[d['present'][:, None, :].repeat(T, 1)].mean():.3f} "
              f"completed-occlusions={len(runs)} len mean={L.mean():.1f} p50={np.median(L):.0f} "
              f"p90={np.percentile(L, 90):.0f} max={L.max():.0f}")
