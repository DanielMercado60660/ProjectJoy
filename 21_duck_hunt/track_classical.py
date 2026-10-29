"""Classical duck tracker: causal background subtraction (running median of the last K frames)
+ duck-colour score in chromaticity space + blob detection + constant-velocity Kalman filter
with gating. Also holds the shared Kalman tracker and metrics used for the CNN.

Tuning (random search on TRAINING-distribution scenes only):
  ../.venv/bin/python track_classical.py --tune
"""
import argparse, json, time
import numpy as np
from scipy import ndimage
import scene as S

LOST_PX = 5.0     # a frame counts as "lost" if the estimate is more than 5 px from the duck
DUCK_COLS = np.array([[0.52, 0.34, 0.18], [0.10, 0.45, 0.22], [0.36, 0.26, 0.16]])  # body, head, wing (renderer prior)
DUCK_CHROMA = (DUCK_COLS / DUCK_COLS.sum(1, keepdims=True))[:, :2]


# ---------------- metrics ----------------
def metrics(est, gt, vis):
    err = np.linalg.norm(est - gt, axis=-1)[vis]
    return dict(mean_err_px=float(err.mean()), median_err_px=float(np.median(err)),
                lost_rate=float((err > LOST_PX).mean()), clipped_err=float(np.minimum(err, 20).mean()),
                n_frames=int(vis.sum()))


# ---------------- Kalman (constant velocity) ----------------
class Kalman:
    def __init__(self, q, r):
        self.q, self.r = q, r
        self.x = None

    def init(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0]); self.P = np.diag([self.r, self.r, 25.0, 25.0])

    def predict(self):
        Fm = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.x = Fm @ self.x
        Q = self.q * np.array([[.25, 0, .5, 0], [0, .25, 0, .5], [.5, 0, 1, 0], [0, .5, 0, 1]])
        self.P = Fm @ self.P @ Fm.T + Q
        return self.x[:2]

    def update(self, z):
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
        Sm = Hm @ self.P @ Hm.T + self.r * np.eye(2)
        K = self.P @ Hm.T @ np.linalg.inv(Sm)
        self.x = self.x + K @ (z - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P


def run_filter(cands, prm):
    """Generic gated tracker. cands[f] = list of (x, y, score) detections for frame f.
    Picks the candidate maximising score * proximity to the prediction; reinitialises on the
    best-scoring candidate after `reinit` consecutive misses. Returns (F,2) estimates."""
    kf = Kalman(prm["q"], prm["r"])
    out = np.zeros((len(cands), 2)); misses = 0; last = np.array([S.W / 2, S.H / 2])
    for f, cs in enumerate(cands):
        pred = kf.predict() if kf.x is not None else None
        best = None
        if cs:
            arr = np.array(cs)
            if pred is None or misses >= prm["reinit"]:
                best = arr[np.argmax(arr[:, 2])]
                kf.init(best[:2]); misses = 0; pred = None
            else:
                d = np.linalg.norm(arr[:, :2] - pred, axis=1)
                w = arr[:, 2] * np.exp(-0.5 * (d / prm["gate"]) ** 2)
                i = int(np.argmax(w))
                if d[i] < 2.5 * prm["gate"]:
                    best = arr[i]
        if best is not None and pred is not None:
            if prm.get("use_kf", True):
                kf.update(best[:2]); misses = 0
                out[f] = kf.x[:2]
            else:
                kf.init(best[:2]); out[f] = best[:2]
        elif best is not None:
            out[f] = best[:2]
        else:
            misses += 1
            out[f] = pred if pred is not None else last
        last = out[f]
    return out


# ---------------- classical detector ----------------
def precompute(frames, Ks=(3, 6, 10)):
    """frames (F,H,W,3) uint8 of one scene -> chroma distance map and bg-diff maps per K."""
    x = frames.astype(np.float32) / 255.0
    ch = x[..., :2] / (x.sum(-1, keepdims=True) + 1e-3)
    dcol = np.min(np.linalg.norm(ch[:, :, :, None, :] - DUCK_CHROMA[None, None, None], axis=-1), axis=-1)
    dark = x.sum(-1) / 3  # very dark/bright pixels have unreliable chroma
    fgs = {}
    for K in Ks:
        fg = np.ones(x.shape[:3], np.float32)
        for f in range(1, len(x)):
            bg = np.median(x[max(0, f - K):f], axis=0)
            fg[f] = np.abs(x[f] - bg).max(-1)
        fgs[K] = fg
    return dict(dcol=dcol.astype(np.float32), lum=dark.astype(np.float32), fg=fgs)


def detect(pre, f, prm):
    col = np.exp(-(pre["dcol"][f] / prm["sigma"]) ** 2) * (pre["lum"][f] > prm["min_lum"])
    fg = np.clip(pre["fg"][prm["K"]][f] / prm["fg_scale"], 0, 1)
    if prm["mode"] == "product":
        sc = col * fg ** prm["fg_pow"]
    else:
        sc = prm["cw"] * col + (1 - prm["cw"]) * fg
    m = sc > prm["thr"]
    lab, n = ndimage.label(m)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    area = ndimage.sum(m, lab, idx)
    ssum = ndimage.sum(sc, lab, idx)
    cy, cx = np.array(ndimage.center_of_mass(sc, lab, idx)).T.reshape(2, -1)
    ok = (area >= prm["amin"]) & (area <= prm["amax"])
    return [(cx[i], cy[i], ssum[i]) for i in np.nonzero(ok)[0]]


def track_scene(pre, prm):
    nf = pre["dcol"].shape[0]
    return run_filter([detect(pre, f, prm) for f in range(nf)], prm)


RANGES = dict(sigma=(0.02, 0.25), min_lum=(0.0, 0.3), fg_scale=(0.03, 0.4), fg_pow=(0.0, 2.0), cw=(0.0, 1.0),
              thr=(0.05, 0.9), amin=(1, 8), amax=(12, 90), q=(0.05, 8), r=(0.1, 8), gate=(2, 15))


def sample(rng):
    p = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    p["K"] = int(rng.choice([3, 6, 10])); p["mode"] = str(rng.choice(["sum", "product"]))
    p["reinit"] = int(rng.integers(1, 8)); p["use_kf"] = bool(rng.random() < 0.7)
    return p


def perturb(rng, p, scale):
    q = dict(p)
    for k, (lo, hi) in RANGES.items():
        q[k] = float(np.clip(p[k] + rng.normal() * scale * (hi - lo), lo, hi))
    if rng.random() < 0.15: q["K"] = int(rng.choice([3, 6, 10]))
    if rng.random() < 0.1: q["reinit"] = int(rng.integers(1, 8))
    return q


def eval_set(pres, GT, VIS, prm):
    est = np.stack([track_scene(p, prm) for p in pres])
    return metrics(est, GT, VIS), est


def tune(n_scenes=40, n_random=250, n_local=150, seed=0, aug=False):
    t0 = time.time()
    X, GT, VIS = S.make_set("train", n_scenes)
    if aug:
        X = S.jitter_set(X, 4241)   # per-scene photometric jitter (follow-up variant)
    pres = [precompute(x) for x in X]
    print(f"precompute {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(seed)
    best = (1e9, None)
    for i in range(n_random):
        p = sample(rng)
        m, _ = eval_set(pres, GT, VIS, p)
        if m["clipped_err"] < best[0]:
            best = (m["clipped_err"], p); print(i, round(best[0], 3), m, flush=True)
    cur_s, cur = best
    for i in range(n_local):
        p = perturb(rng, cur, 0.12 * (1 - i / n_local) + 0.02)
        m, _ = eval_set(pres, GT, VIS, p)
        if m["clipped_err"] < cur_s:
            cur_s, cur = m["clipped_err"], p; print("local", i, round(cur_s, 3), m, flush=True)
    m, _ = eval_set(pres, GT, VIS, cur)
    out = dict(params=cur, train_metrics=m, n_configs=n_random + n_local, n_scenes=n_scenes,
               objective="mean(min(err,20px)) over visible frames", wall_s=time.time() - t0)
    sfx = "_aug" if aug else ""
    out["aug"] = aug
    json.dump(cur, open(f"checkpoints/classical_tracker_params{sfx}.json", "w"), indent=1)
    json.dump(out, open(f"results/classical_tracker_tuning{sfx}.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--n_random", type=int, default=250)
    ap.add_argument("--n_local", type=int, default=150)
    a = ap.parse_args()
    if a.quick:
        X, GT, VIS = S.make_set("train", 2)
        pres = [precompute(x) for x in X]
        rng = np.random.default_rng(0)
        for _ in range(3):
            print(eval_set(pres, GT, VIS, sample(rng))[0])
    elif a.tune:
        tune(n_random=a.n_random, n_local=a.n_local, aug=a.aug)
