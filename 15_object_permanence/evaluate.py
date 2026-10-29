"""Evaluate the learned tracker and the Kalman baselines on identical rendered test frames.

  ../.venv/bin/python evaluate.py --variant base --ckpt checkpoints/base/final
  ../.venv/bin/python evaluate.py --variant hard --ckpt checkpoints/hard/final --extra_ckpt base=checkpoints/base/final

Writes results/eval_<variant>.json and media/curves_<variant>.png.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

import kalman
import metrics
import sim
from model import to_pixels
from train import load_model

MAXS = 24


def run_model(model, frames, bs=50):
    outs = []
    for i in range(0, len(frames), bs):
        o = model(mx.array(frames[i:i + bs]))
        mx.eval(o)
        outs.append(np.array(o))
    return to_pixels(np.concatenate(outs))


def since_hidden_curves(mean, cov, d, idx):
    """Error / sigma / coverage as a function of frames since the ball became fully hidden."""
    gt = d["pos"][idx].astype(np.float64)
    vis, present = d["vis"][idx], d["present"][idx]
    err, d2, _ = metrics.per_frame(mean, cov, gt)
    sig = np.linalg.det(cov) ** 0.25
    acc = {s: [] for s in range(1, MAXS + 1)}
    for n in range(len(idx)):
        for (k, s0, L, end) in sim.hidden_runs((vis[n] == 0) & present[n][None]):
            if s0 == 0:
                continue
            for t in range(s0, min(end, s0 + MAXS)):
                acc[t - s0 + 1].append((err[n, t, k], sig[n, t, k], d2[n, t, k]))
    out = dict(s=[], n=[], err=[], sigma=[], cov1=[], cov2=[])
    for s, v in acc.items():
        if len(v) < 20:
            continue
        v = np.array(v)
        out["s"].append(s)
        out["n"].append(len(v))
        out["err"].append(float(v[:, 0].mean()))
        out["sigma"].append(float(v[:, 1].mean()))
        out["cov1"].append(float((v[:, 2] <= 1).mean()))
        out["cov2"].append(float((v[:, 2] <= 4).mean()))
    return out


def reliability(mean, cov, d, idx, hidden_only):
    gt = d["pos"][idx].astype(np.float64)
    vis, present = d["vis"][idx], d["present"][idx]
    sel = np.broadcast_to(present[:, None], vis.shape) & ((vis == 0) if hidden_only else True)
    _, d2, _ = metrics.per_frame(mean, cov, gt)
    ps = np.linspace(0.05, 0.95, 19)
    return dict(p=ps.tolist(), observed=[float((d2[sel] <= -2 * np.log(1 - p)).mean()) for p in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "hard"])
    ap.add_argument("--ckpt", default="checkpoints/base/final")
    ap.add_argument("--extra_ckpt", nargs="*", default=[], help="name=path of more models to compare")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", nargs="*", default=None, help="override evaluation splits")
    args = ap.parse_args()
    pre = "" if args.variant == "base" else "hard_"
    splits = args.splits or [pre + "test", pre + "test_long"]
    with open(f"results/kalman_tuning_{args.variant}.json") as f:
        tj = json.load(f)
    tun = tj["best"]
    kf_cfgs = {"kalman_bounce": tun["bounce=True"], "kalman_nobounce": tun["bounce=False"]}
    # second operating point: the bounce filter tuned for hidden-frame *error* instead of NLL
    err_best = min([g for g in tj["grid"] if g["bounce"]], key=lambda g: g["err_hidden"])
    if (err_best["q"], err_best["r"]) != (tun["bounce=True"]["q"], tun["bounce=True"]["r"]):
        kf_cfgs["kalman_bounce_errtuned"] = err_best
    models = {"gru": load_model(args.ckpt)[0]}
    for e in args.extra_ckpt:
        name, path = e.split("=")
        models[name] = load_model(path)[0]
    results = dict(ckpt=args.ckpt, kalman=kf_cfgs, splits={})
    curves = {}
    for split in splits:
        d = sim.load_split(split)
        idx = np.arange(min(args.n, len(d["pos"])))
        frames = sim.render_split(d, idx)
        preds = {}
        for name, m in models.items():
            preds[name] = run_model(m, frames)
        for name, c in kf_cfgs.items():
            preds[name] = kalman.track_split(d, idx, frames=frames, q=c["q"], r=c["r"], bounce=c["bounce"])
        results["splits"][split] = {}
        curves[split] = {}
        for name, (mean, cov) in preds.items():
            r = metrics.evaluate(mean, cov, d, idx)
            r["since_hidden"] = since_hidden_curves(mean, cov, d, idx)
            r["reliability_all"] = reliability(mean, cov, d, idx, False)
            r["reliability_hidden"] = reliability(mean, cov, d, idx, True)
            results["splits"][split][name] = r
            curves[split][name] = r
            h, re, lh = r["hidden"], r["reappear"], r["last_hidden"]
            print(f"{split:16s} {name:16s} vis_err {r['visible']['err']:.2f}  hidden_err {h['err']:.2f} "
                  f"hidden_nll {h['nll']:.2f} cov1/2 {h['cov1']:.2f}/{h['cov2']:.2f}  last_hidden_err {lh['err']:.2f} "
                  f"reappear_err {re['err']:.2f}  idsw/seq {r['identity']['switches_per_seq']:.3f}", flush=True)
            for tg in ["turned", "straight"]:
                if f"runs_{tg}" in r:
                    q = r[f"runs_{tg}"]
                    print(f"{'':16s}   {tg:9s} runs={q['runs']} hidden_err {q['hidden']['err']:.2f} nll {q['hidden']['nll']:.2f} "
                          f"cov1/2 {q['hidden']['cov1']:.2f}/{q['hidden']['cov2']:.2f} reappear_err {q['reappear']['err']:.2f}", flush=True)
    os.makedirs("results", exist_ok=True)
    tag = args.tag or args.variant
    with open(f"results/eval_{tag}.json", "w") as f:
        json.dump(results, f, indent=1)
    plot(curves, f"media/curves_{tag}.png")


COL = {"gru": "#d1495b", "kalman_bounce": "#00798c", "kalman_nobounce": "#8d99ae", "base_model": "#edae49",
       "kalman_bounce_errtuned": "#30638e"}


def plot(curves, path):
    splits = list(curves)
    fig, ax = plt.subplots(len(splits), 4, figsize=(18, 4.2 * len(splits)))
    ax = np.atleast_2d(ax)
    for i, sp in enumerate(splits):
        for name, r in curves[sp].items():
            c = r["since_hidden"]
            col = COL.get(name, None)
            ax[i, 0].plot(c["s"], c["err"], "-o", ms=3, label=name, color=col)
            ax[i, 1].plot(c["s"], c["sigma"], "-o", ms=3, label=name, color=col)
            ax[i, 2].plot(c["s"], c["cov2"], "-o", ms=3, label=name + " 2σ", color=col)
            ax[i, 2].plot(c["s"], c["cov1"], "--", label=name + " 1σ", color=col)
            rl = r["reliability_hidden"]
            ax[i, 3].plot(rl["p"], rl["observed"], "-o", ms=3, label=name, color=col)
        ax[i, 2].axhline(metrics.EXP2, color="k", lw=0.8)
        ax[i, 2].axhline(metrics.EXP1, color="k", lw=0.8, ls="--")
        ax[i, 3].plot([0, 1], [0, 1], "k", lw=0.8)
        ax[i, 0].set(title=f"{sp}: position error while hidden", xlabel="frames since fully hidden", ylabel="px")
        ax[i, 1].set(title="predicted σ (det^¼ of cov)", xlabel="frames since fully hidden", ylabel="px")
        ax[i, 2].set(title="ellipse coverage (black = ideal)", xlabel="frames since fully hidden", ylim=(0, 1.02))
        ax[i, 3].set(title="calibration on hidden frames", xlabel="nominal coverage", ylabel="observed")
        for a in ax[i]:
            a.grid(alpha=0.3)
        ax[i, 0].legend(fontsize=8)
        ax[i, 2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
