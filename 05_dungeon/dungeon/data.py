"""Fixed, seeded level sets (cached to data/*.npz).

  train pool : TRAIN_FAMILIES only, seeds 1000+i      (what the solver trains on)
  test sets  : every family, seeds 9000+i              (train-family test levels are new
                                                         samples; caves/spiral are never trained on)
"""
import os
import numpy as np

from .mazes import ALL_FAMILIES, TRAIN_FAMILIES, make_levels, concat_levels

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _cached(name, fn):
    path = os.path.join(ROOT, name + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = fn()
    os.makedirs(ROOT, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


def train_pool(n_per_family=20000):
    def build():
        sets = []
        for i, f in enumerate(TRAIN_FAMILIES):
            L = make_levels(f, n_per_family, seed=1000 + i)
            L["family"] = np.full(n_per_family, i, np.int64)
            sets.append(L)
        return {k: np.concatenate([s[k] for s in sets]) for k in sets[0]}
    return _cached(f"train_pool_{n_per_family}", build)


def test_set(family, n=500):
    i = ALL_FAMILIES.index(family)
    return _cached(f"test_{family}_{n}", lambda: make_levels(family, n, seed=9000 + i))


def pool_sampler(pool):
    n = len(pool["walls"])

    def sample(k, rng):
        idx = rng.integers(n, size=k)
        return {kk: pool[kk][idx] for kk in ("walls", "start", "goal")}
    return sample


def mixed_sampler(samplers, probs):
    probs = np.asarray(probs, float) / np.sum(probs)

    def sample(k, rng):
        which = rng.choice(len(samplers), size=k, p=probs)
        out = {kk: np.empty((k,) + shp, np.int64 if kk != "walls" else np.uint8)
               for kk, shp in (("walls", (13, 13)), ("start", (2,)), ("goal", (2,)))}
        for j, s in enumerate(samplers):
            m = which == j
            if m.any():
                lv = s(int(m.sum()), rng)
                for kk in out:
                    out[kk][m] = lv[kk]
        return out
    return sample


def exit_in_view_probe(n_per_family=500, min_dist=6):
    """Diagnostic set: the family test levels, but with the start moved so the exit is already inside
    the 5x5 view at spawn (Chebyshev distance <= 2) while the BFS path is still >= min_dist steps.
    This situation is rare in training (~3% of the train pool). Levels where no such start exists
    are skipped."""
    from .mazes import bfs_dist

    def build():
        rng = np.random.default_rng(777)
        W, St, G = [], [], []
        for f in ALL_FAMILIES:
            L = test_set(f, n_per_family)
            for w, g in zip(L["walls"], L["goal"]):
                d = bfs_dist(w, g[0], g[1])
                cand = [(r, c) for r in range(g[0] - 2, g[0] + 3) for c in range(g[1] - 2, g[1] + 3)
                        if 0 <= r < w.shape[0] and 0 <= c < w.shape[1] and d[r, c] >= min_dist]
                if cand:
                    W.append(w); G.append(g); St.append(np.array(cand[rng.integers(len(cand))], np.int64))
        return dict(walls=np.array(W, np.uint8), start=np.array(St, np.int64), goal=np.array(G, np.int64))
    return _cached(f"probe_exit_in_view_{n_per_family}", build)
