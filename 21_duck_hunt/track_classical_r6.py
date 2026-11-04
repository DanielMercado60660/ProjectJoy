"""R6 classical tracker: the original bg-sub + colour + blob + Kalman tracker given the same kind of
help the learned tracker gets in R6.

Cues (each precomputed per scene, combined as a product of powers so tuning can switch any off):
  fg    background subtraction: max-channel |frame - median of previous K frames|, computed on
        grey-world-normalised frames (each channel divided by its frame mean) when norm='gw'
  col   duck-colour score: exp(-(chroma distance to the sprite colours / sigma)^2), optional
        luminance gate (raw or grey-world chroma)
  tpl_ch  masked per-channel normalised cross-correlation with the duck sprite (2 facings x 4 wing
        phases, rendered by the scene renderer at scale 1). Per-channel NCC is invariant to per-
        channel gain and offset, i.e. to global lighting/tint, and only looks at duck pixels, so the
        background colour does not enter. Mean over channels, max over templates.
  tpl_jl (added in v3 after the held-out diagnosis that 'ch' is colour-blind and fires on
        balloon edges): one joint NCC over the three channels of log(frame + 0.03), invariant to
        per-channel gain/tint but sensitive to the duck's colour contrasts.
  score = col^a_col * clip(fg/fg_scale,0,1)^a_fg * clip((tpl_ch-t0)/(1-t0),0,1)^a_tpl
          * clip((tpl_jl-t0j)/(1-t0j),0,1)^a_tplj
Pre-denoise (v4, blur=True): the motion and template cues are computed on Gaussian-blurred
(sigma 0.6 px) frames, with the templates blurred identically (motivated by the DR-val diagnosis:
failures concentrate in noisy scenes).
Detection: thresholded blobs (area limits, score-weighted centroid, as before) or, det='peaks', the
top-3 local maxima of the 3x3-smoothed score (added after DR-val diagnosis: in noisy scenes the duck
blob merges with noise blobs and the area cap rejects it).
Filter: run_filter_r6 = the original gated constant-velocity Kalman (track_classical.run_filter)
plus a re-detection rule: if the best candidate anywhere scores `switch` times more than the
candidate chosen near the prediction (or there is none), re-initialise on it. The learned tracker
uses the same filter (eval_tracker_r6.py).

Tuning: random search + local refinement on the first N DR training scenes (the same scenes the CNN
trains on), objective mean(min(err, 20 px)) as before.
  ../bin/gpu_run ../.venv/bin/python track_classical_r6.py --tune [--cues old] [--n_random 500 --n_local 300]
"""
import argparse, json, time
from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import scene as S
import scene_r6 as R
import track_classical as TC

TH, TW = 4, 7                    # template half-height/half-width in output pixels (9 x 15)
DUCK_CHROMA = TC.DUCK_CHROMA


# ---------------- duck templates ----------------
def make_templates(s=1.0):
    """render the duck at output pixel (40, 40) on black and on white backgrounds -> colour + alpha."""
    out = []
    for facing in (1.0, -1.0):
        for phase in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            ims = []
            for bgv in (0.0, 1.0):
                img = np.full((S.HS, S.WS, 3), bgv, np.float32)
                S.draw_duck(img, 80.5, 80.5, s, phase, facing)
                ims.append(img.reshape(S.H, S.SS, S.W, S.SS, 3).mean((1, 3))[40 - TH:40 + TH + 1, 40 - TW:40 + TW + 1])
            alpha = 1 - (ims[1] - ims[0]).mean(-1)
            m = (alpha > 0.5).astype(np.float32)
            col = ims[0] / np.maximum(alpha, 1e-3)[..., None]
            out.append((col, m))
    return out


TEMPLATES = make_templates()


def ncc_map(x, templates=TEMPLATES, joint=False):
    """x (F,H,W,3) float -> (F,H,W) max over templates of the masked NCC.
    joint=False: per-channel NCC, mean over channels (invariant to per-channel gain and offset,
                 but colour-blind: only the spatial pattern within each channel counts).
    joint=True : one NCC over all three channels (per-channel mean removed, joint normalisation),
                 meant for log intensities: per-channel gain/tint becomes an offset and cancels,
                 while the relative size of the contrasts across channels (the duck's colours) counts."""
    best = np.full(x.shape[:3], -1.0, np.float32)
    corr = lambda a, k: fftconvolve(a, k[None, ::-1, ::-1], mode="same", axes=(1, 2))
    for col, m in templates:
        n = m.sum(); acc = 0; num_j = 0; var_j = 0; tn_j = 0
        for c in range(3):
            S1 = corr(x[..., c], m)
            t0 = (col[..., c] - (col[..., c] * m).sum() / n) * m
            num = corr(x[..., c], t0)
            var = np.maximum(corr(x[..., c] ** 2, m) - S1 ** 2 / n, 1e-4 if not joint else 0)
            if joint:
                num_j = num_j + num; var_j = var_j + var; tn_j += (t0 ** 2).sum()
            else:
                acc = acc + num / (np.sqrt(var) * np.sqrt((t0 ** 2).sum()) + 1e-6)
        r = num_j / (np.sqrt(np.maximum(var_j, 1e-3)) * np.sqrt(tn_j) + 1e-6) if joint else acc / 3
        best = np.maximum(best, r.astype(np.float32))
    return best


LOG_EPS = 0.03
LOG_TEMPLATES = [(np.log(col + LOG_EPS), m) for col, m in TEMPLATES]
BLUR = 0.6     # v4 option: Gaussian pre-denoise of the frames; the templates are blurred the same way


def _blur_tpl(col, m):
    b = lambda a: ndimage.gaussian_filter(a, BLUR)
    return np.stack([b(col[..., c] * m) / np.maximum(b(m), 1e-3) for c in range(3)], -1), m


TEMPLATES_B = [_blur_tpl(c, m) for c, m in TEMPLATES]
LOG_TEMPLATES_B = [(np.log(col + LOG_EPS), m) for col, m in TEMPLATES_B]


def greyworld(x):
    return np.clip(x / (x.mean((1, 2), keepdims=True) + 1e-3) * 0.5, 0, 2)


def chroma_dist(x):
    ch = x[..., :2] / (x.sum(-1, keepdims=True) + 1e-3)
    return np.min(np.linalg.norm(ch[:, :, :, None, :] - DUCK_CHROMA[None, None, None], axis=-1), axis=-1).astype(np.float32)


def bg_diff(x, K):
    fg = np.ones(x.shape[:3], np.float32)
    for f in range(1, len(x)):
        fg[f] = np.abs(x[f] - np.median(x[max(0, f - K):f], axis=0)).max(-1)
    return fg


def precompute(frames, Ks=(3, 6, 10)):
    x = frames.astype(np.float32) / 255.0
    g = greyworld(x)
    xb = ndimage.gaussian_filter(x, (0, BLUR, BLUR, 0))
    pre = dict(tpl_ch=ncc_map(x), tpl_jl=ncc_map(np.log(x + LOG_EPS), LOG_TEMPLATES, joint=True),
               tpl_ch_b=ncc_map(xb, TEMPLATES_B), tpl_jl_b=ncc_map(np.log(xb + LOG_EPS), LOG_TEMPLATES_B, joint=True))
    for nm, z, zb in (("raw", x, xb), ("gw", g, greyworld(xb))):
        pre["dcol_" + nm] = chroma_dist(z)
        pre["lum_" + nm] = z.mean(-1).astype(np.float32)
        for K in Ks:
            pre[f"fg_{nm}_{K}"] = bg_diff(z, K)
            pre[f"fg_{nm}_{K}_b"] = bg_diff(zb, K)
    return pre


# ---------------- detection + filter ----------------
def score_map(pre, f, prm):
    nm = prm["norm"]; b = "_b" if prm.get("blur", False) else ""
    sc = np.ones(pre["tpl_ch"].shape[1:], np.float32)
    if prm["a_col"] > 0:
        col = np.exp(-(pre["dcol_" + nm][f] / prm["sigma"]) ** 2) * (pre["lum_" + nm][f] > prm["min_lum"])
        sc = sc * col ** prm["a_col"]
    if prm["a_fg"] > 0:
        sc = sc * np.clip(pre[f"fg_{nm}_{prm['K']}{b}"][f] / prm["fg_scale"], 0, 1) ** prm["a_fg"]
    if prm["a_tpl"] > 0:
        sc = sc * np.clip((pre["tpl_ch" + b][f] - prm["t0"]) / (1 - prm["t0"]), 0, 1) ** prm["a_tpl"]
    if prm.get("a_tplj", 0) > 0:
        sc = sc * np.clip((pre["tpl_jl" + b][f] - prm["t0j"]) / (1 - prm["t0j"]), 0, 1) ** prm["a_tplj"]
    return sc


def detect(pre, f, prm):
    sc = score_map(pre, f, prm)
    if prm.get("det", "blob") == "peaks":      # top-3 local maxima of the 3x3-smoothed score (noise-robust)
        sm = ndimage.uniform_filter(sc, 3)
        pk = (sm == ndimage.maximum_filter(sm, 7)) & (sm > prm["thr"])
        ys, xs = np.nonzero(pk)
        order = np.argsort(-sm[ys, xs])[:3]
        out = []
        for i in order:   # sub-pixel centroid of the score in a 5x5 window
            y0, x0 = max(ys[i] - 2, 0), max(xs[i] - 2, 0)
            w = sc[y0:ys[i] + 3, x0:xs[i] + 3]
            cy, cx = ndimage.center_of_mass(w + 1e-9)
            out.append((x0 + cx, y0 + cy, float(sm[ys[i], xs[i]])))
        return out
    m = sc > prm["thr"]
    lab, n = ndimage.label(m)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    area = ndimage.sum(m, lab, idx); ssum = ndimage.sum(sc, lab, idx)
    cy, cx = np.array(ndimage.center_of_mass(sc, lab, idx)).T.reshape(2, -1)
    ok = (area >= prm["amin"]) & (area <= prm["amax"])
    return [(cx[i], cy[i], ssum[i]) for i in np.nonzero(ok)[0]]


def run_filter_r6(cands, prm):
    """track_classical.run_filter + re-detection: switch to the globally best candidate when it
    beats the gated choice by a factor prm['switch'] (inf = original behaviour)."""
    kf = TC.Kalman(prm["q"], prm["r"])
    out = np.zeros((len(cands), 2)); misses = 0; last = np.array([S.W / 2, S.H / 2])
    sw = prm.get("switch", np.inf)
    for f, cs in enumerate(cands):
        pred = kf.predict() if kf.x is not None else None
        best = None
        if cs:
            arr = np.array(cs); top = arr[np.argmax(arr[:, 2])]
            if pred is None or misses >= prm["reinit"]:
                best = top; kf.init(best[:2]); misses = 0; pred = None
            else:
                d = np.linalg.norm(arr[:, :2] - pred, axis=1)
                w = arr[:, 2] * np.exp(-0.5 * (d / prm["gate"]) ** 2)
                i = int(np.argmax(w))
                if d[i] < 2.5 * prm["gate"]:
                    best = arr[i]
                ref = best[2] if best is not None else 0.0
                if top[2] > sw * max(ref, 1e-9) and (best is None or np.linalg.norm(top[:2] - best[:2]) > prm["gate"]):
                    best = top; kf.init(best[:2]); misses = 0; pred = None
        if best is not None and pred is not None:
            if prm.get("use_kf", True):
                kf.update(best[:2]); misses = 0; out[f] = kf.x[:2]
            else:
                kf.init(best[:2]); out[f] = best[:2]
        elif best is not None:
            out[f] = best[:2]
        else:
            misses += 1
            out[f] = pred if pred is not None else last
        last = out[f]
    return out


def track_scene(pre, prm):
    return run_filter_r6([detect(pre, f, prm) for f in range(pre["tpl_ch"].shape[0])], prm)


# ---------------- tuning ----------------
RANGES = dict(sigma=(0.02, 0.3), min_lum=(0.0, 0.3), fg_scale=(0.02, 0.5), a_col=(0.0, 2.0), a_fg=(0.0, 2.0),
              a_tpl=(0.0, 3.0), t0=(-0.2, 0.7), a_tplj=(0.0, 3.0), t0j=(-0.2, 0.7), thr=(0.01, 0.95), amin=(1, 8), amax=(12, 90), q=(0.05, 8),
              r=(0.1, 8), gate=(2, 15), log_switch=(0.2, 3.5))
OLD_CUES = dict(a_tpl=0.0, a_tplj=0.0, blur=False, norm="raw", log_switch=99.0, det="blob")   # ablation: the original cue set, just retuned on DR


def finish(p, cues):
    if cues == "old":
        p.update(OLD_CUES)
    p["switch"] = float(np.exp(p["log_switch"])) if p["log_switch"] < 50 else float("inf")
    return p


def sample(rng, cues="all"):
    p = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    for k in ("a_col", "a_fg", "a_tpl", "a_tplj"):      # each cue switched off with prob 0.2
        if rng.random() < 0.2: p[k] = 0.0
    if rng.random() < 0.25: p["log_switch"] = 99.0
    p["K"] = int(rng.choice([3, 6, 10])); p["norm"] = str(rng.choice(["raw", "gw"]))
    p["reinit"] = int(rng.integers(1, 8)); p["use_kf"] = bool(rng.random() < 0.7)
    p["det"] = str(rng.choice(["blob", "peaks"])); p["blur"] = bool(rng.random() < 0.5)
    return finish(p, cues)


def perturb(rng, p, scale, cues="all"):
    q = dict(p)
    for k, (lo, hi) in RANGES.items():
        if k == "log_switch" and p[k] > 50:
            if rng.random() < 0.1: q[k] = float(rng.uniform(lo, hi))
            continue
        if k.startswith("a_") and p[k] == 0.0 and rng.random() > 0.1:
            continue
        q[k] = float(np.clip(p[k] + rng.normal() * scale * (hi - lo), lo, hi))
    if rng.random() < 0.15: q["K"] = int(rng.choice([3, 6, 10]))
    if rng.random() < 0.1: q["reinit"] = int(rng.integers(1, 8))
    if rng.random() < 0.1: q["norm"] = "gw" if p["norm"] == "raw" else "raw"
    if rng.random() < 0.05: q["use_kf"] = not p["use_kf"]
    if rng.random() < 0.05: q["blur"] = not p.get("blur", False)
    if rng.random() < 0.05: q["det"] = "peaks" if p.get("det", "blob") == "blob" else "blob"
    return finish(q, cues)


_PRES = None


def _init(pres):
    global _PRES
    _PRES = pres


def _eval_cfg(prm):
    return np.stack([track_scene(p, prm) for p in _PRES])


def eval_many(pool, cfgs, GT, VIS):
    return [TC.metrics(e, GT, VIS) for e in pool.map(_eval_cfg, cfgs)]


def tune(n_scenes=60, n_random=500, n_local=300, seed=0, cues="all", workers=4, out="r6_tracker_classical", dr="full"):
    t0 = time.time()
    X, GT, VIS = R.make_set_dr(n_scenes, R.DR_SEED["dr"], fn=R.render_dr2 if dr == "haze" else R.render_dr)  # = CNN's first scenes
    with Pool(workers) as p:
        pres = p.map(precompute, list(X))
    print(f"precompute {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(seed)
    log, best = [], (1e9, None)
    with Pool(workers, initializer=_init, initargs=(pres,)) as pool:
        cfgs = [sample(rng, cues) for _ in range(n_random)]
        for i in range(0, n_random, 20):
            for p, m in zip(cfgs[i:i + 20], eval_many(pool, cfgs[i:i + 20], GT, VIS)):
                log.append(m["clipped_err"])
                if m["clipped_err"] < best[0]:
                    best = (m["clipped_err"], p); print(len(log), round(best[0], 3), m, flush=True)
        cur_s, cur = best
        for i in range(0, n_local, 4):      # 4 perturbations per round (one per worker), keep the best
            cand = [perturb(rng, cur, 0.12 * (1 - i / n_local) + 0.02, cues) for _ in range(4)]
            ms = eval_many(pool, cand, GT, VIS)
            log += [m["clipped_err"] for m in ms]
            j = int(np.argmin([m["clipped_err"] for m in ms]))
            if ms[j]["clipped_err"] < cur_s:
                cur_s, cur = ms[j]["clipped_err"], cand[j]; print("local", i, round(cur_s, 3), ms[j], flush=True)
        m = eval_many(pool, [cur], GT, VIS)[0]
    res = dict(params=cur, train_metrics=m, n_configs=len(log), n_scenes=n_scenes, cues=cues, dr=dr, scenes="dr train seeds 100000+",
               objective="mean(min(err,20px)) over visible frames", wall_s=time.time() - t0, flight_sha=R.flight_sha(),
               clipped_err_log=log)
    json.dump(cur, open(f"checkpoints/{out}.json", "w"), indent=1)
    json.dump(res, open(f"results/{out}_tuning.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "clipped_err_log"}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--cues", default="all", choices=["all", "old"])
    ap.add_argument("--n_scenes", type=int, default=60)
    ap.add_argument("--n_random", type=int, default=500)
    ap.add_argument("--n_local", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dr", default="full", choices=["full", "haze"], help="haze: round-2 DR v2 scenes")
    a = ap.parse_args()
    if a.tune:
        tune(a.n_scenes, a.n_random, a.n_local, cues=a.cues, dr=a.dr,
             out=a.out or ("r6_tracker_classical" if a.cues == "all" else "r6_tracker_classical_oldcues"))
