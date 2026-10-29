"""Follow-up (round 3): one round of the camouflage arms race, small.

Painter = best-of-K random search *inside the generator's own skin distribution*. For each scene
(held-out families only: marble / ripples / streaks, fresh seeds outside every existing split), the
painter draws K candidate skins with `perturb_params(bg, delta, rng_k)` + `render_texture` at a FIXED
camouflage knob delta. Background, blob mask and the nuisances (illumination, blur, sensor noise) are
identical across the K candidates, so candidates differ only in the skin. Each candidate is scored by
every detector:

  unet0     frozen seed-0 U-Net  (checkpoints/unet)        -- the attacker the painter optimises against
  unet1     frozen seed-1 U-Net  (checkpoints/unet_seed1)  -- not used in optimisation (transfer)
  base      shipped classical baseline (outlier, sigma from baseline_choice.json)
  base_er   reviewer's eroded baseline (outlier, sigma 3, erode 3, min area 30)

"random skin" = expectation over the K candidates (and candidate 0 as a single-draw check);
"painted vs D" = the candidate minimising detector D's IoU (soft IoU for U-Nets, hard IoU for baselines).
Because the painter never leaves the delta-conditional distribution it cannot delete the creature,
change its shape/opacity, or move colours off the generator's palette; at matched delta the only
lever is *which* draw. All scores are saved, so the full painter x detector transfer matrix is post hoc.

Retraining round: `painter.py --families train --no_baselines` makes painted TRAINING-family scenes,
`painter_finetune.py` fine-tunes U-Net0 on them, and `painter.py --ckpt0 <fine-tuned>` re-attacks.
"""
import argparse, json, os, sys, time, subprocess
import numpy as np
from multiprocessing import get_context
from scipy.ndimage import gaussian_filter
import scenes as S

DELTAS = (0.25, 0.5, 0.75, 1.0)
SEED_BASE = 9_000_000  # existing splits use chunk seeds 1000..4999 -> no collision


def render_scene(args):
    seed, delta, K, families = args
    rng = np.random.default_rng(seed)
    bg_p = S.sample_params(families[rng.integers(len(families))], rng)
    bg = S.render_texture(bg_p, rng)
    R = rng.uniform(10, 18)
    cx, cy = rng.uniform(R * 1.2, S.W - R * 1.2), rng.uniform(R * 1.2, S.H - R * 1.2)
    m = S.blob_mask(rng, cx, cy, R)
    a = rng.uniform(-0.25, 0.25, size=2)
    illum = 1 + a[0] * (S._xx / S.W - 0.5) + a[1] * (S._yy / S.H - 0.5)
    noise = rng.normal(0, 0.02, (S.H, S.W, 3))
    st_bg = S.texture_stats(bg)
    X = np.zeros((K, S.H, S.W, 3), np.uint8)
    raw = np.zeros((K, 12), np.float32)
    for k in range(K):
        rk = np.random.default_rng([seed, k])
        cp = S.perturb_params(bg_p, delta, rk)
        skin = S.render_texture(cp, rk)
        img = bg.copy()
        img[m] = skin[m]
        img = gaussian_filter(img * illum[..., None], sigma=(0.5, 0.5, 0)) + noise
        X[k] = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
        raw[k] = S.texture_stats(skin) - st_bg
    return X, m, raw, bg_p["family"]


def _base_worker(args):
    X, kind, sigma = args
    sys.path.insert(0, "review")
    import baseline as B
    if kind == "base":
        return B.run(X, "outlier", sigma)
    import baseline_erode as E
    return E.run(X, sigma=3.0, erode=3, min_area=30)


def unet_scores(ckpt, Xflat, Mflat, bs=256):
    import mlx.core as mx
    import model as M
    net = M.load(ckpt)
    soft, hard = [], []
    for i in range(0, len(Xflat), bs):
        p = np.array(mx.sigmoid(net(mx.array(Xflat[i:i + bs].astype(np.float32) / 255))))
        g = Mflat[i:i + bs]
        inter = (p * g).sum((1, 2)); soft.append(inter / (p.sum((1, 2)) + g.sum((1, 2)) - inter + 1e-6))
        b = p > 0.5
        hard.append((b & g).sum((1, 2)) / np.maximum((b | g).sum((1, 2)), 1))
    return np.concatenate(soft), np.concatenate(hard)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_delta", type=int, default=75)
    ap.add_argument("--k", type=int, default=48, help="painter budget: candidate skins per scene")
    ap.add_argument("--out", default="results/painter")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt0", default="checkpoints/unet")
    ap.add_argument("--ckpt1", default="checkpoints/unet_seed1")
    ap.add_argument("--families", default="heldout", choices=["heldout", "train"])
    ap.add_argument("--seed_base", type=int, default=SEED_BASE)
    ap.add_argument("--no_baselines", action="store_true", help="score U-Nets only (for making painted training data)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fams = S.HELDOUT_FAMILIES if a.families == "heldout" else S.TRAIN_FAMILIES
    t0 = time.time()
    jobs = [(a.seed_base + 1000 * di + i, d, a.k, fams) for di, d in enumerate(DELTAS) for i in range(a.n_per_delta)]
    with get_context("spawn").Pool(a.workers) as pool:
        outs = pool.map(render_scene, jobs, chunksize=4)
    X = np.stack([o[0] for o in outs])                       # (N, K, H, W, 3)
    Mk = np.stack([o[1] for o in outs])                      # (N, H, W)
    raw = np.stack([o[2] for o in outs])                     # (N, K, 12)
    fam = [o[3] for o in outs]
    delta = np.array([j[1] for j in jobs])
    N, K = X.shape[:2]
    t_render = time.time() - t0
    print(f"rendered {N} scenes x {K} skins in {t_render:.0f}s", flush=True)
    Xf = X.reshape(N * K, S.H, S.W, 3)
    Mf = np.repeat(Mk, K, axis=0)
    scores = {}
    for name, ck in (("unet0", a.ckpt0), ("unet1", a.ckpt1)):
        s, h = unet_scores(ck, Xf, Mf)
        scores[name + "_soft"], scores[name] = s.reshape(N, K), h.reshape(N, K)
    print(f"unet scored {time.time() - t0:.0f}s", flush=True)
    sigma = json.load(open("baseline_choice.json"))["best"]["sigma"]
    chunks = np.array_split(np.arange(N * K), a.workers * 4)
    for kind in (() if a.no_baselines else ("base", "base_er")):
        with get_context("spawn").Pool(a.workers) as pool:
            preds = pool.map(_base_worker, [(Xf[c], kind, sigma) for c in chunks])
        P = np.concatenate(preds)
        scores[kind] = ((P & Mf).sum((1, 2)) / np.maximum((P | Mf).sum((1, 2)), 1)).reshape(N, K)
        print(f"{kind} scored {time.time() - t0:.0f}s", flush=True)
    # texture-stat distance (scale from the train split, as evaluate.py)
    import pickle
    from metrics import stat_scale
    scale = stat_scale([pickle.load(open("data/train_meta.pkl", "rb"))])
    sdist = np.sqrt(((raw / scale) ** 2).mean(-1))           # (N, K)
    np.savez_compressed(os.path.join(a.out, "scores.npz"), delta=delta, family=np.array(fam), stat_dist=sdist,
                        mask=Mk, **scores)

    # ---- summary: painter (selects with detector P) x evaluated detector D ----
    dets = ["unet0", "unet1"] if a.no_baselines else ["unet0", "unet1", "base", "base_er"]
    sel_key = {"unet0": "unet0_soft", "unet1": "unet1_soft", "base": "base", "base_er": "base_er"}
    pick = {p: scores[sel_key[p]].argmin(1) for p in dets}
    rows = np.arange(N)
    # keep only the images needed for demos: random draw 0 and each painter's pick
    np.savez_compressed(os.path.join(a.out, "images.npz"), random0=X[:, 0],
                        **{f"painted_by_{p}": X[rows, pick[p]] for p in dets}, **{f"pick_{p}": pick[p] for p in dets})
    bins = [("all", np.ones(N, bool))] + [(f"delta={d}", delta == d) for d in DELTAS]
    summ = dict(N=int(N), K=int(K), families=fams, render_s=t_render, minutes=(time.time() - t0) / 60,
                uptime=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), table={})
    for bn, sel in bins:
        t = {}
        for D in dets:
            sc = scores[D]
            t[f"{D}/random_mean"] = float(sc[sel].mean())
            t[f"{D}/random_cand0"] = float(sc[sel, 0].mean())
            t[f"{D}/random_det"] = float((sc[sel] >= 0.5).mean())
            for P in dets:
                v = sc[rows, pick[P]][sel]
                t[f"{D}/painted_by_{P}"] = float(v.mean())
                t[f"{D}/painted_by_{P}_det"] = float((v >= 0.5).mean())
        t["stat_dist/random_mean"] = float(sdist[sel].mean())
        for P in dets:
            t[f"stat_dist/painted_by_{P}"] = float(sdist[rows, pick[P]][sel].mean())
        # oracle: painter that minimises the stat distance itself (a model-free "closest draw" painter)
        po = sdist.argmin(1)
        for D in dets:
            t[f"{D}/painted_by_statdist"] = float(scores[D][rows, po][sel].mean())
        summ["table"][bn] = t
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    # markdown
    L = [f"# Painter vs detectors (N={N} scenes, K={K} skins each, families={fams})\n",
         f"{summ['uptime']}\n",
         "Rows: evaluated detector. Columns: mean IoU on random skins (mean over K draws) and on the skin "
         "picked by each painter (the detector it minimised). Det = fraction with IoU >= 0.5.\n"]
    for bn, _ in bins:
        t = summ["table"][bn]
        L.append(f"\n## {bn}\n\n| detector | random | " + " | ".join(f"painted by {P}" for P in dets) + " | by stat-dist |\n|" + "---|" * (len(dets) + 3))
        for D in dets:
            L.append(f"| {D} | {t[f'{D}/random_mean']:.3f} | " + " | ".join(f"{t[f'{D}/painted_by_{P}']:.3f}" for P in dets)
                     + f" | {t[f'{D}/painted_by_statdist']:.3f} |")
        L.append(f"| stat_dist (skin vs bg) | {t['stat_dist/random_mean']:.3f} | " +
                 " | ".join(f"{t[f'stat_dist/painted_by_{P}']:.3f}" for P in dets) + " | (min) |")
        L.append("\ndet rate (IoU>=0.5): " + ", ".join(
            f"{D}: random {t[f'{D}/random_det']:.2f} / by unet0 {t[f'{D}/painted_by_unet0_det']:.2f}" for D in dets))
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"done in {(time.time() - t0) / 60:.1f} min")
