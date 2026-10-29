"""Evaluate generators with the attribute classifier (trained once on real data).

Metrics (all on N generated samples vs the held-out real test set):
  valid       fraction of samples where all 6 attribute heads are >0.9 confident ("recognizable monster")
  attr_tv     mean total-variation distance between predicted attribute marginals and the real marginals
  coverage    distinct attribute combos (of 1296) among valid samples; real test subset of same N for reference
  cfd         classifier-feature Frechet distance (FID-like, but in OUR classifier's 256-d feature space - NOT FID)
  precision / recall   k-NN manifold precision/recall (Kynkaanniemi et al. 2019) in classifier-feature space, k=3
  div_feat / div_pix   mean pairwise L2 distance among samples (feature / pixel space)
  mem_rate    fraction of samples whose pixel-space nearest training image is closer than the 1st percentile
              of real-test-to-train NN distances (i.e. closer than 99% of fresh real monsters get)
  nn_pix_med / nn_feat_med   median NN distance to train, relative to the same median for real test images

    ../bin/gpu_run ../.venv/bin/python evaluate.py main runs/flow runs/ddpm runs/vae runs/gan
    ../bin/gpu_run ../.venv/bin/python evaluate.py sweep runs/flow runs/ddpm
    ../bin/gpu_run ../.venv/bin/python evaluate.py stability runs/flow runs/ddpm runs/vae runs/gan
"""
import json, os, sys, time

import numpy as np
import mlx.core as mx
from scipy import linalg

import data as D
import train_gen as TG
from train_classifier import load_classifier, predict

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def real_stats(res=32):
    if res in _CACHE:
        return _CACHE[res]
    xtr, ytr, xte, yte = D.load(res)
    clf = load_classifier(res)
    _, ftr = predict(clf, xtr)
    pte, fte = predict(clf, xte)
    xtr_f = xtr.reshape(len(xtr), -1).astype(np.float32) / 255
    xte_f = xte.reshape(len(xte), -1).astype(np.float32) / 255
    d_te_pix = nn_dist(xte_f, xtr_f)
    d_te_feat = nn_dist(fte, ftr)
    s = dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte, clf=clf, ftr=ftr, fte=fte, pte=pte, xtr_f=xtr_f,
             d_te_pix=d_te_pix, d_te_feat=d_te_feat, mem_thresh=float(np.percentile(d_te_pix, 1)),
             marg=[np.bincount(ytr[:, j], minlength=k) / len(ytr) for j, k in enumerate(D.N_CLASSES)])
    _CACHE[res] = s
    return s


def nn_dist(q, ref, chunk=1000, return_idx=False, cand=16):
    """L2 nearest-neighbour distance from each row of q to rows of ref.
    GPU matmul finds `cand` candidates (fast but reduced precision on large-norm vectors), then the exact
    distance is recomputed in float64 on CPU for those candidates."""
    mu = ref.mean(0, keepdims=True)
    refc = (ref - mu).astype(np.float32)
    R = mx.array(refc); rn = (R * R).sum(1)
    ds, ids = [], []
    for i in range(0, len(q), chunk):
        qc = (q[i:i + chunk] - mu).astype(np.float32)
        Q = mx.array(qc)
        d2 = rn[None] - 2 * Q @ R.T
        c = np.asarray(mx.argpartition(d2, cand, axis=1)[:, :cand])
        diff = refc[c].astype(np.float64) - qc[:, None, :].astype(np.float64)
        ex = np.sqrt((diff ** 2).sum(-1))
        j = ex.argmin(1)
        ds.append(ex[np.arange(len(j)), j]); ids.append(c[np.arange(len(j)), j])
    d = np.concatenate(ds)
    return (d, np.concatenate(ids)) if return_idx else d


def frechet(f1, f2):
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    cs = linalg.sqrtm(c1 @ c2)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs.real))


def prec_recall(real, fake, k=3):
    def radii(x):
        X = mx.array(x); n = (X * X).sum(1)
        d = mx.sqrt(mx.maximum(n[:, None] + n[None] - 2 * X @ X.T, 0))
        return np.sort(np.asarray(d), axis=1)[:, k]

    def inside(q, ref, r):
        Q, Rr = mx.array(q), mx.array(ref)
        d = mx.sqrt(mx.maximum((Q * Q).sum(1)[:, None] + (Rr * Rr).sum(1)[None] - 2 * Q @ Rr.T, 0))
        return float(np.mean(np.asarray((d <= mx.array(r)[None]).any(1))))
    return inside(fake, real, radii(real)), inside(real, fake, radii(fake))


def mean_pairwise(x, n=1000):
    x = x[:n]
    X = mx.array(x); s = (X * X).sum(1)
    d = np.sqrt(np.maximum(np.asarray(s[:, None] + s[None] - 2 * X @ X.T), 0))
    return float(d[np.triu_indices(len(x), 1)].mean())


def metrics(imgs, res=32, n_ref=None):
    """imgs: (N,H,W,3) uint8. Returns dict of metrics."""
    S = real_stats(res)
    n = len(imgs)
    probs, f = predict(S["clf"], imgs)
    pred = np.stack([p.argmax(1) for p in probs], 1)
    conf = np.stack([p.max(1) for p in probs], 1)
    valid = (conf > 0.9).all(1)
    tv = [0.5 * np.abs(np.bincount(pred[:, j], minlength=k) / n - S["marg"][j]).sum() for j, k in enumerate(D.N_CLASSES)]
    cov = len(np.unique(D.combo_id(pred[valid])))
    fte = S["fte"][: (n_ref or n)]
    prec, rec = prec_recall(fte[:min(len(fte), 5000)], f[:5000])
    xf = imgs.reshape(n, -1).astype(np.float32) / 255
    d_pix, idx = nn_dist(xf, S["xtr_f"], return_idx=True)
    d_feat = nn_dist(f, S["ftr"])
    return {
        "n": n, "valid": float(valid.mean()), "attr_tv": float(np.mean(tv)),
        "attr_tv_per": dict(zip(D.ATTR_NAMES, [float(t) for t in tv])),
        "coverage": cov, "cfd": frechet(f, fte), "precision": prec, "recall": rec,
        "div_feat": mean_pairwise(f), "div_pix": mean_pairwise(xf),
        "mem_rate": float((d_pix < S["mem_thresh"]).mean()),
        "nn_pix_med_rel": float(np.median(d_pix) / np.median(S["d_te_pix"])),
        "nn_feat_med_rel": float(np.median(d_feat) / np.median(S["d_te_feat"])),
        "_nn_idx": idx, "_nn_d": d_pix, "_pred": pred, "_valid": valid,
    }


def real_reference(n, res=32):
    """Same metrics for a real (train) subset of size n vs the test set: the 'perfect generator' floor."""
    S = real_stats(res)
    rng = np.random.default_rng(0)
    sub = S["xtr"][rng.choice(len(S["xtr"]), n, replace=False)]
    m = metrics(sub, res)
    m["mem_rate"] = None  # train images are trivially memorized; use test->train instead
    m["nn_pix_med_rel"] = m["nn_feat_med_rel"] = None  # 1.0 by definition for fresh real (test) images
    S2 = S["d_te_pix"]
    m["mem_rate_test_images"] = float((S2 < S["mem_thresh"]).mean())
    return m


def gen_samples(run, n, steps=None, method=None, ckpt="final", seed=0, use_ema=True):
    model, g, cfg = TG.load_generator(run, ckpt, use_ema=use_ema)
    steps = steps or {"ddpm": 50, "flow": 50}.get(model, 1)
    t0 = time.time()
    x = TG.sample(model, g, n, cfg, mx.random.key(seed), steps=steps, method=method)
    mx.eval(x)
    dt = time.time() - t0
    return TG.to_uint8(x), dt, model, g, cfg


def nfe(model, steps, method):
    if model in ("vae", "gan"):
        return 1
    if model == "flow" and method == "heun":
        return 2 * steps - 1
    return steps


def _meta_path(run):
    """runs/<m>/final/meta.json, or the copy exported to checkpoints/<m>/meta.json (runs/ is gitignored)."""
    p = os.path.join(run, "final", "meta.json")
    return p if os.path.exists(p) else os.path.join(HERE, "checkpoints", os.path.basename(run.rstrip("/")), "meta.json")


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def pca_gaussian_samples(n, k=64, seed=0):
    """Classical baseline: Gaussian in the top-k PCA subspace of training pixels (+ mean)."""
    S = real_stats(32)
    X = S["xtr_f"].astype(np.float64)
    mu = X.mean(0)
    U, sv, Vt = np.linalg.svd(X[:20000] - mu, full_matrices=False)
    std = sv[:k] / np.sqrt(20000 - 1)
    z = np.random.default_rng(seed).normal(size=(n, k)) * std
    x = mu + z @ Vt[:k]
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8).reshape(n, 32, 32, 3)


def cmd_main(runs, n=5000, out_path="results/main_eval.json", baselines=True):
    out = {"real_reference": strip(real_reference(n))}
    if not baselines:
        return _gen_rows(runs, n, out, out_path)
    S = real_stats(32)
    copies = S["xtr"][np.random.default_rng(1).choice(len(S["xtr"]), n, replace=False)]
    m = strip(metrics(copies)); m.update(model="copy_train", sampler="memorizer: returns a training image")
    out["copy_train"] = m
    x = pca_gaussian_samples(n)
    m = metrics(x)
    np.savez_compressed("results/samples_pca.npz", x=x, nn_idx=m["_nn_idx"], nn_d=m["_nn_d"])
    m = strip(m); m.update(model="pca", sampler="PCA-64 Gaussian (no training loop)")
    out["pca"] = m
    for k in ("copy_train", "pca"):
        print(k, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out[k].items() if kk != "attr_tv_per"}), flush=True)
    return _gen_rows(runs, n, out, out_path)


def _gen_rows(runs, n, out, out_path):
    for run in runs:
        x, dt, model, g, cfg = gen_samples(run, n)
        m = metrics(x)
        meta = json.load(open(_meta_path(run)))
        fwd = TG.sampler_flops(TG.build(model, cfg), model, cfg)
        steps = {"ddpm": 50, "flow": 50}.get(model, 1)
        m.update(model=model, sampler={"ddpm": "ddim-50", "flow": "euler-50"}.get(model, "1 pass"),
                 sample_sec_per_1k=dt / n * 1000, sample_gflop_per_img=fwd * nfe(model, steps, None) / 1e9,
                 train_steps=meta["step"], train_images=meta["images"], train_pflop=meta["flops"] / 1e15,
                 train_min=meta["train_sec"] / 60, stop_reason=meta["stop_reason"],
                 params=int(sum(v.size for _, v in __import__("mlx.utils").utils.tree_flatten(g.parameters()))))
        os.makedirs(run, exist_ok=True)  # clean clone: runs/ is not committed
        np.savez_compressed(os.path.join(run, "samples_final.npz"), x=x, nn_idx=m["_nn_idx"], nn_d=m["_nn_d"],
                            pred=m["_pred"], valid=m["_valid"])
        out[os.path.basename(run.rstrip("/"))] = strip(m)
        print(os.path.basename(run), json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in strip(m).items() if k != "attr_tv_per"}), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)


def cmd_sweep(runs, n=1000, out_path="results/sweep.json", grid=None):
    rows = [{"model": "real", **strip(real_reference(n))}]
    grid = grid or {"ddpm": [("ddim", s) for s in (5, 10, 20, 50, 100, 250)] + [("ddpm", s) for s in (20, 50, 250)],
            "flow": [("euler", s) for s in (1, 2, 4, 8, 16, 32, 64, 128)] + [("heun", s) for s in (2, 4, 8, 16, 32)],
            "vae": [("decode", 1)], "gan": [("generate", 1)]}
    for run in runs:
        name = os.path.basename(run.rstrip("/"))
        model = TG.load_generator(run)[0]
        for method, steps in grid[model]:
            x, dt, model, g, cfg = gen_samples(run, n, steps=steps, method=method if model in ("ddpm", "flow") else None)
            fwd = TG.sampler_flops(TG.build(model, cfg), model, cfg)
            m = strip(metrics(x))
            m.update(model=name, method=method, steps=steps, nfe=nfe(model, steps, method),
                     sec_per_1k=dt / n * 1000, gflop_per_img=fwd * nfe(model, steps, method) / 1e9)
            rows.append(m)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "attr_tv_per"}), flush=True)
            json.dump(rows, open(out_path, "w"), indent=1)


def cmd_stability(runs, n=1000):
    out = {}
    for run in runs:
        name = os.path.basename(run.rstrip("/"))
        meta = json.load(open(os.path.join(run, "final", "meta.json")))
        rows = []
        for c in meta["ckpts"]:
            x, dt, model, g, cfg = gen_samples(run, n, ckpt=c["name"])
            m = strip(metrics(x))
            m.update(c)
            rows.append(m)
            print(name, c["name"], round(m["cfd"], 3), round(m["valid"], 3), m["coverage"], flush=True)
        out[name] = rows
        if name == "gan":  # GAN EMA can average over oscillating modes: also track the raw generator
            raw = []
            for c in meta["ckpts"]:
                x, dt, model, g, cfg = gen_samples(run, n, ckpt=c["name"], use_ema=False)
                m = strip(metrics(x)); m.update(c); raw.append(m)
                print(name, "raw", c["name"], round(m["cfd"], 3), round(m["valid"], 3), m["coverage"], flush=True)
            out["gan_raw"] = raw
        json.dump(out, open("results/stability.json", "w"), indent=1)


if __name__ == "__main__":
    os.chdir(HERE)
    cmd, runs = sys.argv[1], sys.argv[2:]
    if cmd == "sweep_x2":
        cmd_sweep(runs, out_path="results/sweep_x2.json",
                  grid={"flow": [("euler", s) for s in (4, 8, 16, 50)] + [("heun", 4)],
                        "ddpm": [("ddim", 20), ("ddim", 50), ("ddpm", 50), ("ddpm", 250)]})
    elif cmd == "main_x2":
        cmd_main(runs, out_path="results/main_eval_x2.json", baselines=False)
    else:
        {"main": cmd_main, "sweep": cmd_sweep, "stability": cmd_stability}[cmd](runs)
