"""Evaluate VAE(s) against the baselines under matched conditions. Writes results/eval.json.

  ../.venv/bin/python eval.py --vaes checkpoints/vae_cond_b1.0 checkpoints/vae_cond_b0.5 ...

All generators are fit on the train split; every generator produces the same number of patterns per
style with fixed seeds. The main VAE is the conditional one with the best validation NLL (IWAE).
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

import baselines as B
import data
import metrics as M
from model import load_vae, load_classifier, classify_numpy, decode_numpy, encode_numpy, to_input, one_hot, CELLS

STYLES = data.STYLES


def run_name(path):
    parts = path.rstrip("/").split("/")
    return parts[-2] if parts[-1] == "best" and len(parts) > 1 else parts[-1]


def iwae_nll(model, sp, K=200, batch=64, seed=0):
    """Importance-weighted estimate of -log p(hits) in nats (velocity/offset not scored)."""
    key = mx.random.key(seed)
    out = []
    for i in range(0, len(sp["hits"]), batch):
        h, v, o, s = (sp[k][i:i + batch] for k in ("hits", "vel", "off", "style"))
        x = mx.array(to_input(h, v, o))
        sv = mx.array(one_hot(s)) if model.cond else None
        mu, logvar = model.encode(x, sv)
        Bn, L = mu.shape
        key, sub = mx.random.split(key)
        eps = mx.random.normal((K, Bn, L), key=sub)
        z = mu[None] + mx.exp(0.5 * logvar)[None] * eps
        zf = z.reshape(K * Bn, L)
        sK = mx.repeat(sv[None], K, 0).reshape(K * Bn, -1) if model.cond else None
        xK = mx.repeat(x[None], K, 0).reshape(K * Bn, -1)
        logit, _, _ = model.decode_tf(zf, sK, xK)
        hx = xK[:, :CELLS]
        logp_h = -(mx.logaddexp(0, logit) - hx * logit).sum(1).reshape(K, Bn)
        logp_z = (-0.5 * (z ** 2 + np.log(2 * np.pi))).sum(-1)
        logq = (-0.5 * (eps ** 2 + np.log(2 * np.pi)) - 0.5 * logvar[None]).sum(-1)
        lw = logp_h + logp_z - logq
        ll = mx.logsumexp(lw, axis=0) - np.log(K)
        out.append(-np.array(ll))
    return float(np.concatenate(out).mean())


def recon_f1(model, sp):
    mu, _ = encode_numpy(model, sp["hits"], sp["vel"], sp["off"], sp["style"])
    h, _, _, _ = decode_numpy(model, mu, sp["style"])
    t = sp["hits"].astype(bool)
    hb = h.astype(bool)
    tp = (t & hb).sum()
    return {"hit_f1": float(2 * tp / (t.sum() + hb.sum())), "exact_recon_pct": float(100 * (h == sp["hits"]).all((1, 2)).mean())}


def c2st(h, style, tr, seed=0, epochs=30):
    """Classifier two-sample test: held-out accuracy of an MLP telling generated from real train grids.

    Input is the hit grid plus the style one-hot. 0.5 = indistinguishable, 1.0 = trivially fake.
    """
    import mlx.nn as nn
    import mlx.optimizers as optim
    rng = np.random.default_rng(seed)
    n = len(h)
    ridx = rng.choice(len(tr["hits"]), n, replace=False)
    X = np.concatenate([np.concatenate([M.flat(h), one_hot(style)], 1),
                        np.concatenate([M.flat(tr["hits"][ridx]), one_hot(tr["style"][ridx])], 1)])
    y = np.concatenate([np.ones(n, np.int32), np.zeros(n, np.int32)])
    perm = rng.permutation(2 * n)
    X, y = X[perm], y[perm]
    k = int(0.7 * len(X))
    mx.random.seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, 2))
    opt = optim.Adam(learning_rate=1e-3)
    lg = nn.value_and_grad(net, lambda m, a, b: nn.losses.cross_entropy(m(a), b).mean())
    for _ in range(epochs):
        p = rng.permutation(k)
        for i in range(0, k, 128):
            idx = p[i:i + 128]
            _, g = lg(net, mx.array(X[idx]), mx.array(y[idx]))
            opt.update(net, g)
            mx.eval(net.parameters(), opt.state)
    pred = np.array(net(mx.array(X[k:]))).argmax(1)
    return float((pred == y[k:]).mean())


def score_set(name, h, v, o, style, tr, te, clf):
    res = {"n": int(len(h))}
    accs = [c2st(h, style, tr, seed=s) for s in range(3)]  # 3 seeds: different real subsample, split and init
    res["c2st_acc"], res["c2st_seeds"] = float(np.mean(accs)), accs
    res.update(M.novelty(h, tr["hits"]))
    res.update(M.diversity(h, style))
    p = classify_numpy(clf, h, v, o)
    res["style_acc"] = float((p.argmax(1) == style).mean())
    per = {}
    for si, sname in enumerate(STYLES):
        m = style == si
        if not m.any():
            continue
        r = M.pulse(h[m])
        r["profile_l1_vs_test"] = M.profile_l1(h[m], te["hits"][te["style"] == si])
        r["swing"] = M.swing(h[m], o[m])
        r["style_acc"] = float((p[m].argmax(1) == si).mean())
        per[sname] = r
    res["per_style"] = per
    for k in ("beat_offbeat_density_ratio", "kick_on_beat_frac", "bar_repeat_hamming", "profile_l1_vs_test",
              "empty_or_sparse_pct", "hits_mean"):
        res[k] = float(np.mean([per[s][k] for s in per]))
    print(f"  {name:16s} c2st {res['c2st_acc']:.3f} acc {res['style_acc']:.3f} nn {res['nn_hamming_mean']:.2f} copy% {res['exact_copy_pct']:.1f} "
          f"div {res['pairwise_hamming_within_style']:.1f} barrep {res['bar_repeat_hamming']:.2f} "
          f"profL1 {res['profile_l1_vs_test']:.4f}", flush=True)
    return res


def transfer_eval(model, te, clf, n=200, seed=0):
    """Style swap: encode a real test pattern with its own label, decode with every other style label.

    Reports classifier accuracy for the *target* style and how many cells change (content kept vs replaced).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(te["hits"]), min(n, len(te["hits"])), replace=False)
    mu, _ = encode_numpy(model, te["hits"][idx], te["vel"][idx], te["off"][idx], te["style"][idx])
    acc, ham = [], []
    for t in range(len(STYLES)):
        keep = te["style"][idx] != t
        h, v, o, _ = decode_numpy(model, mu[keep], np.full(keep.sum(), t))
        pred = classify_numpy(clf, h, v, o).argmax(1)
        acc.append((pred == t).mean())
        ham.append(np.abs(h.astype(int) - te["hits"][idx][keep]).sum((1, 2)).mean())
    return {"target_style_acc": float(np.mean(acc)), "per_target_acc": dict(zip(STYLES, map(float, acc))),
            "mean_cells_changed": float(np.mean(ham))}


def crossfade(a, b, alpha, u):
    """Data-space interpolation: each cell switches from A to B once alpha passes its random threshold u."""
    take_b = u < alpha
    return tuple(np.where(take_b, y, x) for x, y in zip(a, b))


def interp_eval(model, te, tr, clf, n_pairs=64, K=9, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_pairs:
        i, j = rng.choice(len(te["hits"]), 2, replace=False)
        if te["style"][i] != te["style"][j]:
            pairs.append((i, j))
    alphas = np.linspace(0, 1, K)
    out = {}
    for method in ("vae", "crossfade"):
        adj, ratio, nn_mid, rep_mid, mono, end_err, sparse = [], [], [], [], [], [], []
        for i, j in pairs:
            A = tuple(te[k][i] for k in ("hits", "vel", "off"))
            Bp = tuple(te[k][j] for k in ("hits", "vel", "off"))
            sa, sb = te["style"][i], te["style"][j]
            if method == "vae":
                mu, _ = encode_numpy(model, np.stack([A[0], Bp[0]]), np.stack([A[1], Bp[1]]), np.stack([A[2], Bp[2]]),
                                     np.array([sa, sb]))
                z = (1 - alphas)[:, None] * mu[0] + alphas[:, None] * mu[1]
                sv = (1 - alphas)[:, None] * one_hot(np.array([sa]))[0] + alphas[:, None] * one_hot(np.array([sb]))[0]
                h, v, o, _ = decode_numpy(model, z, sv if model.cond else None)
            else:
                u = rng.random(A[0].shape)
                path = [crossfade(A, Bp, a, u) for a in alphas]
                h, v, o = (np.stack([p[k] for p in path]) for k in range(3))
            d = np.abs(np.diff(h.astype(np.int32), axis=0)).sum((1, 2))
            adj.append(d.mean())
            ratio.append(d.max() / max(d.mean(), 1e-6))
            mid = h[1:-1]
            nn_mid.append(M.nn_hamming(mid, tr["hits"]).mean())
            rep_mid.append(np.abs(mid[:, :16].astype(int) - mid[:, 16:]).sum((1, 2)).mean())
            pb = classify_numpy(clf, h, v, o)[:, sb]
            mono.append(float((np.diff(pb) >= -0.05).mean()))
            end_err.append(0.5 * (np.abs(h[0].astype(int) - A[0]).sum() + np.abs(h[-1].astype(int) - Bp[0]).sum()))
            sparse.append((h.sum((1, 2)) < 4).mean())
        out[method] = {"adjacent_hamming_mean": float(np.mean(adj)), "max_over_mean_step_ratio": float(np.mean(ratio)),
                       "midpoints_nn_hamming_to_train": float(np.mean(nn_mid)),
                       "midpoints_bar_repeat_hamming": float(np.mean(rep_mid)),
                       "p_targetstyle_monotone_frac": float(np.mean(mono)),
                       "endpoint_hamming_error": float(np.mean(end_err)), "sparse_point_pct": float(100 * np.mean(sparse))}
        print(f"  interp {method}: {out[method]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/drums.npz")
    ap.add_argument("--vaes", nargs="+", required=True)
    ap.add_argument("--clf", default="checkpoints/classifier")
    ap.add_argument("--n_per_style", type=int, default=250)
    ap.add_argument("--iwae_k", type=int, default=200)
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo_vae", default=None, help="checkpoint used for the demo (default: the selected one)")
    a = ap.parse_args()
    t0 = time.time()
    d = data.load(a.data)
    tr, va, te = (data.subset(d, k) for k in ("train", "val", "test"))
    clf = load_classifier(a.clf)
    styles = np.repeat(np.arange(len(STYLES)), a.n_per_style)
    res = {"n_per_style": a.n_per_style, "data": {"train": len(tr["hits"]), "val": len(va["hits"]), "test": len(te["hits"])},
           "classifier_real_test_acc": json.load(open(os.path.join(a.clf, "classifier_config.json")))["test_acc"],
           "sets": {}, "nll_test": {}, "nll_val": {}, "vaes": {}}

    print("baselines", flush=True)
    bern, markov, copy = B.BernoulliModel().fit(tr), B.MarkovModel().fit(tr), B.CopyModel().fit(tr)
    markov_bar = B.BarMarkovModel().fit(tr)
    for m in (bern, markov, markov_bar):
        res["nll_test"][m.name] = float(m.nll(te["hits"], te["style"]).mean())
        res["nll_val"][m.name] = float(m.nll(va["hits"], va["style"]).mean())
    # references: the real held-out patterns themselves (val is scored against test for the profile metric)
    res["sets"]["real_test"] = score_set("real_test", te["hits"], te["vel"], te["off"], te["style"], tr, te, clf)
    res["sets"]["real_val"] = score_set("real_val", va["hits"], va["vel"], va["off"], va["style"], tr, te, clf)
    for m in (bern, markov, markov_bar, copy):
        h, v, o = m.sample(styles, np.random.default_rng(a.seed))
        res["sets"][m.name] = score_set(m.name, h, v, o, styles, tr, te, clf)

    best, best_nll = None, np.inf
    for path in a.vaes:
        name = run_name(path)
        model, cfg = load_vae(path)
        print(name, cfg, flush=True)
        info = {"config": cfg, "nll_val_iwae": iwae_nll(model, va, a.iwae_k), "nll_test_iwae": iwae_nll(model, te, a.iwae_k),
                "recon_test": recon_f1(model, te)}
        print(f"  NLL val {info['nll_val_iwae']:.2f} test {info['nll_test_iwae']:.2f} recon {info['recon_test']}", flush=True)
        res["vaes"][name] = info
        rng = np.random.default_rng(a.seed)
        z = rng.standard_normal((len(styles), cfg["latent"])).astype(np.float32)
        sty = styles if cfg["cond"] else None
        def score_decoded(tag, h, v, o):
            lab = styles
            if not cfg["cond"]:  # unconditional: label each sample with the classifier's guess (style acc then n/a)
                lab = classify_numpy(clf, h, v, o).argmax(1)
            r = score_set(f"{name}:{tag}", h, v, o, lab, tr, te, clf)
            if not cfg["cond"]:
                r["style_acc"] = None
                r["style_counts"] = np.bincount(lab, minlength=len(STYLES)).tolist()
            res["sets"][f"{name}:{tag}"] = r

        h, v, o, _ = decode_numpy(model, z, sty, mode="threshold")
        score_decoded("threshold", h, v, o)
        for T in (1.0, 0.7, 0.5):
            h, v, o, _ = decode_numpy(model, z, sty, mode="sample", rng=np.random.default_rng(a.seed + 1), temperature=T)
            score_decoded("sample" if T == 1.0 else f"sample_T{T}", h, v, o)
        if cfg["cond"]:
            info["interpolation"] = interp_eval(model, te, tr, clf)
            info["style_transfer"] = transfer_eval(model, te, clf)
            print("  style transfer", info["style_transfer"]["target_style_acc"], flush=True)
            if info["nll_val_iwae"] < best_nll:
                best, best_nll = path, info["nll_val_iwae"]
    res["selected_vae"] = best  # lowest validation NLL among conditional models (the pre-declared rule)
    res["iwae_k"] = a.iwae_k
    demo = a.demo_vae or best
    res["demo_vae"] = demo
    # demo decode mode: the demo model's decode mode that is hardest to tell from real data (C2ST vs train split)
    bname = run_name(demo)
    modes = {k.split(":", 1)[1]: v["c2st_acc"] for k, v in res["sets"].items() if k.startswith(bname + ":")}
    res["demo_mode"] = min(modes, key=modes.get)
    print("selected (val NLL)", best, "| demo", demo, "mode", res["demo_mode"], flush=True)
    res["interpolation"] = res["vaes"][run_name(best)]["interpolation"]
    res["style_transfer"] = res["vaes"][run_name(best)]["style_transfer"]
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out} in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
