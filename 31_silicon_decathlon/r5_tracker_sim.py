"""Round-5 fresh sequences. Vendored 2026-10-24 from 15_object_permanence/sim.py.
Only constants and pure simulation helpers; no sibling imports or data writes.
"""
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
