"""Render all demo media from saved runs + results/*.json into media/.

    ../.venv/bin/python make_figures.py            # everything that has inputs available
"""
import glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
MODELS = ["vae", "gan", "ddpm", "flow"]
NICE = {"vae": "VAE", "gan": "DCGAN", "ddpm": "DDPM (U-Net)", "flow": "Flow matching (U-Net)",
        "flow_cond": "Cond. flow matching", "real": "Real data",
        "copy_train": "Copy train image", "gan_raw": "DCGAN (raw G, no EMA)", "pca": "PCA-Gaussian"}
COL = {"vae": "#8a6d3b", "gan": "#c0392b", "ddpm": "#2e86c1", "flow": "#27ae60", "real": "#555555", "flow_cond": "#8e44ad"}


def font(sz=14):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def grid_img(imgs, nrow=8, scale=3, pad=1):
    n, h, w, _ = imgs.shape
    rows = int(np.ceil(n / nrow))
    g = np.full((rows * (h + pad) + pad, nrow * (w + pad) + pad, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        g[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = imgs[i]
    im = Image.fromarray(g)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def titled(im, title, sz=18):
    out = Image.new("RGB", (im.width, im.height + sz + 10), "white")
    out.paste(im, (0, sz + 10))
    ImageDraw.Draw(out).text((6, 4), title, fill="black", font=font(sz))
    return out


def hcat(ims, gap=12):
    w = sum(i.width for i in ims) + gap * (len(ims) - 1); h = max(i.height for i in ims)
    out = Image.new("RGB", (w, h), "white"); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + gap
    return out


def vcat(ims, gap=12):
    w = max(i.width for i in ims); h = sum(i.height for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (w, h), "white"); y = 0
    for i in ims:
        out.paste(i, (0, y)); y += i.height + gap
    return out


def avail(models=MODELS):
    return [m for m in models if os.path.exists(f"runs/{m}/samples_final.npz")]


def sample_grids():
    xtr = D.load(32)[0]
    panels = [titled(grid_img(xtr[:64], 8, 3), "Real training data")]
    for m in avail(MODELS + ["flow_cond"]):
        x = np.load(f"runs/{m}/samples_final.npz")["x"]
        grid_img(x[:64], 8, 4).save(f"media/samples_{m}.png")
        panels.append(titled(grid_img(x[:64], 8, 3), NICE[m]))
    if os.path.exists("runs/flow_cond/combos.npz"):
        z = np.load("runs/flow_cond/combos.npz")
        idx = np.random.default_rng(0).choice(len(z["x"]), 64, replace=False)
        panels.append(titled(grid_img(z["x"][idx], 8, 3), "Cond. flow (random recipes, guidance w=3)"))
    if len(panels) > 1:
        top = hcat(panels[:3]); rows = [top]
        if len(panels) > 3:
            rows.append(hcat(panels[3:]))
        vcat(rows).save("media/sample_grids.png")
        print("media/sample_grids.png")


def training_gif():
    ms = [m for m in MODELS if os.path.exists(f"runs/{m}/final/meta.json")]
    if not ms:
        return
    snaps = {m: json.load(open(f"runs/{m}/final/meta.json"))["snaps"] for m in ms}
    n = min(len(v) for v in snaps.values())
    frames = []
    for i in range(n):
        ims = []
        for m in ms:
            s = snaps[m][i]
            im = Image.open(os.path.join(f"runs/{m}/snaps", os.path.basename(s["path"]))).convert("RGB")
            cell = im.width // 8
            im = im.crop((0, 0, cell * 5, cell * 4))  # 5x4 of the 64 fixed-noise samples keeps the GIF small
            ims.append(titled(im, f"{NICE[m]}  step {s['step']}", 13))
        fr = hcat(ims)
        fr = titled(fr, f"Training progress: {100 * min(1, snaps[ms[0]][i]['frac']):.0f}% of matched budget", 18)
        frames.append(fr)
    frames += [frames[-1]] * 6
    frames[0].save("media/training_progress.gif", save_all=True, append_images=frames[1:], duration=350, loop=0)
    print("media/training_progress.gif", len(frames))


def loss_curves():
    ms = [m for m in MODELS if os.path.exists(f"runs/{m}/final/meta.json")]
    if not ms:
        return
    fig, axs = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.4))
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, ms):
        log = json.load(open(f"runs/{m}/final/meta.json"))["log"]
        st = [d["step"] for d in log]
        if m == "gan":
            ax.plot(st, [d["d_loss"] for d in log], label="D loss", color="#c0392b")
            ax.plot(st, [d["g_loss"] for d in log], label="G loss", color="#2c3e50")
            ax.legend(frameon=False)
        elif m == "vae":
            ax.plot(st, [d["rec"] for d in log], label="recon (nats)", color="#8a6d3b")
            ax.plot(st, [d["kl"] for d in log], label="KL (nats)", color="#2c3e50")
            ax.set_yscale("log"); ax.legend(frameon=False)
        else:
            ax.plot(st, [d["loss"] for d in log], color=COL[m]); ax.set_yscale("log")
        ax.set_title(NICE[m]); ax.set_xlabel("step"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("media/loss_curves.png", dpi=110); plt.close(fig)
    print("media/loss_curves.png")


def comparison_table():
    if not os.path.exists("results/main_eval.json"):
        return
    R = json.load(open("results/main_eval.json"))
    cols = ["model", "valid ↑", "attr TV ↓", "coverage ↑", "CFD ↓", "prec ↑", "recall ↑", "div(feat)", "mem rate ↓",
            "NN dist (rel)", "train steps", "train PFLOP", "train min", "ms/sample", "GFLOP/sample"]
    rows = []
    for k in ["real_reference"] + [m for m in ["copy_train", "pca"] + MODELS + ["flow_cond"] if m in R]:
        r = R[k]
        smp = r.get("sampler", "") if k in MODELS + ["flow_cond"] else ("memorizer" if k == "copy_train" else "classical")
        rows.append([("Real (train subset)" if k == "real_reference" else NICE[k] + "\n" + smp),
                     f"{r['valid']:.3f}", f"{r['attr_tv']:.3f}", f"{r['coverage']}", f"{r['cfd']:.2f}",
                     f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['div_feat']:.2f}",
                     "-" if r.get("mem_rate") is None else f"{r['mem_rate']:.3f}",
                     "1 (def.)" if r.get("nn_pix_med_rel") is None else f"{r['nn_pix_med_rel']:.2f}",
                     str(r.get("train_steps", "-")),
                     f"{r['train_pflop']:.2f}" if "train_pflop" in r else "-",
                     f"{r['train_min']:.1f}" if "train_min" in r else "-",
                     f"{r['sample_sec_per_1k']:.2f}" if "sample_sec_per_1k" in r else "-",
                     f"{r['sample_gflop_per_img']:.2f}" if "sample_gflop_per_img" in r else "-"])
    fig, ax = plt.subplots(figsize=(19, 1.0 + 0.62 * len(rows)))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=[0.14] + [0.86 / (len(cols) - 1)] * (len(cols) - 1))
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 2.4)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor("#ecf0f1")
    ax.set_title(f"Monster Factory: {R['real_reference']['n']} samples per model, 32x32, "
                 "matched training budget (CFD = Frechet distance in our attribute-classifier feature space, not FID)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig("media/comparison_table.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("media/comparison_table.png")


def quality_latency():
    if not os.path.exists("results/sweep.json"):
        return
    rows = json.load(open("results/sweep.json"))
    real = [r for r in rows if r["model"] == "real"][0]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for name in sorted({r["model"] for r in rows} - {"real"}):
        for meth in sorted({r["method"] for r in rows if r["model"] == name}):
            rr = sorted([r for r in rows if r["model"] == name and r["method"] == meth], key=lambda r: r["nfe"])
            mk = "o" if meth in ("euler", "ddim", "decode", "generate") else "s"
            ls = "-" if meth in ("euler", "ddim") else "--"
            lab = f"{NICE.get(name, name)} [{meth}]"
            c = COL.get(name.split("_")[0], "k")
            axs[0].plot([r["nfe"] for r in rr], [r["cfd"] for r in rr], ls, marker=mk, color=c, label=lab)
            axs[1].plot([r["sec_per_1k"] for r in rr], [r["cfd"] for r in rr], ls, marker=mk, color=c, label=lab)
            axs[2].plot([r["gflop_per_img"] for r in rr], [r["valid"] for r in rr], ls, marker=mk, color=c, label=lab)
            for r in rr:
                if name in ("flow", "ddpm"):
                    axs[1].annotate(str(r["steps"]), (r["sec_per_1k"], r["cfd"]), fontsize=7, color=c,
                                    xytext=(3, 3), textcoords="offset points")
    for ax in axs[:2]:
        ax.axhline(real["cfd"], color="#555", ls=":", label="real vs real (floor)")
        ax.set_yscale("log"); ax.set_xscale("log"); ax.set_ylabel("CFD (classifier-feature FD) ↓"); ax.grid(alpha=.3, which="both")
    axs[2].axhline(real["valid"], color="#555", ls=":", label="real data")
    axs[0].set_xlabel("network evaluations (NFE)")
    axs[1].set_xlabel("wall-clock seconds per 1000 samples (shared GPU)")
    axs[2].set_xlabel("sampling GFLOP per image"); axs[2].set_xscale("log"); axs[2].set_ylabel("valid (all attrs >0.9 conf) ↑")
    axs[2].grid(alpha=.3, which="both")
    axs[0].legend(fontsize=7.5, frameon=False)
    axs[0].set_title("Quality vs steps"); axs[1].set_title("Quality vs latency (numbers = sampler steps)")
    axs[2].set_title("Classifier-valid rate vs sampling compute (crispness proxy)")
    fig.tight_layout(); fig.savefig("media/quality_latency.png", dpi=110); plt.close(fig)
    print("media/quality_latency.png")


def stability():
    if not os.path.exists("results/stability.json"):
        return
    S = json.load(open("results/stability.json"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for m, rows in S.items():
        x = [r["frac"] * 100 for r in rows]
        c = COL.get(m.split("_")[0], "k")
        ls = "--" if m.endswith("_raw") else "-"
        axs[0].plot(x, [r["cfd"] for r in rows], ls + "o", color=c, label=NICE.get(m, m))
        axs[1].plot(x, [r["valid"] for r in rows], ls + "o", color=c)
        axs[2].plot(x, [r["coverage"] for r in rows], ls + "o", color=c)
    axs[0].set_yscale("log"); axs[0].set_ylabel("CFD ↓"); axs[1].set_ylabel("valid ↑"); axs[2].set_ylabel("distinct combos (of 1296) ↑")
    for ax in axs:
        ax.set_xlabel("% of training budget"); ax.grid(alpha=.3)
    axs[0].legend(frameon=False)
    fig.suptitle("Quality over training (1000 EMA samples per checkpoint; U-Nets sampled with 50 steps)")
    fig.tight_layout(); fig.savefig("media/stability.png", dpi=110); plt.close(fig)
    print("media/stability.png")


def memorization():
    ms = avail(MODELS + ["flow_cond"])
    if not ms:
        return
    xtr = D.load(32)[0]
    panels = []
    for m in ms:
        z = np.load(f"runs/{m}/samples_final.npz")
        order = np.argsort(z["nn_d"])[:8]  # the 8 most train-like samples
        pairs = np.concatenate([z["x"][order], xtr[z["nn_idx"][order]]])
        panels.append(titled(grid_img(pairs, 8, 3), f"{NICE[m]}: 8 closest-to-train samples (top) and their train NN (bottom)", 13))
    vcat(panels).save("media/memorization_pairs.png")
    print("media/memorization_pairs.png")


def trajectory_gif():
    import mlx.core as mx
    import train_gen as TG
    frames_by = {}
    for m, steps, method in [("ddpm", 50, "ddim"), ("flow", 50, "euler")]:
        if not os.path.exists(f"checkpoints/{m}/ema.safetensors"):
            continue
        model, g, cfg = TG.load_generator(f"checkpoints/{m}")
        x = mx.random.normal((16, 32, 32, 3), key=mx.random.key(5))
        frames = [x]
        if model == "flow":
            ts = np.linspace(0, 1, steps + 1)
            for i in range(steps):
                x = x + (ts[i + 1] - ts[i]) * g(x, mx.full((16,), ts[i] * 1000.0)); mx.eval(x); frames.append(x)
        else:
            abar = TG.get_abar(); tsd = np.linspace(999, 0, steps).round().astype(int)
            for i, t in enumerate(tsd):
                a_t = abar[int(t)]; a_p = abar[int(tsd[i + 1])] if i + 1 < steps else mx.array(1.0)
                eps = g(x, mx.full((16,), float(t)))
                x0 = mx.clip((x - mx.sqrt(1 - a_t) * eps) / mx.sqrt(a_t), -1, 1)
                eps = (x - mx.sqrt(a_t) * x0) / mx.sqrt(1 - a_t)
                x = mx.sqrt(a_p) * x0 + mx.sqrt(1 - a_p) * eps; mx.eval(x); frames.append(x)
        frames_by[m] = [TG.to_uint8(f) for f in frames]
    if not frames_by:
        return
    n = min(len(v) for v in frames_by.values())
    out = []
    for i in range(n):
        if i % 2 and i != n - 1:
            continue
        ims = [titled(grid_img(frames_by[m][i], 4, 3), f"{NICE[m]} step {i}/{n - 1}", 13) for m in frames_by]
        out.append(hcat(ims, 24))
    out += [out[-1]] * 10
    out[0].save("media/trajectory.gif", save_all=True, append_images=out[1:], duration=150, loop=0)
    print("media/trajectory.gif")


if __name__ == "__main__":
    what = sys.argv[1:] or ["grids", "gif", "loss", "table", "ql", "stab", "mem", "traj"]
    fns = {"grids": sample_grids, "gif": training_gif, "loss": loss_curves, "table": comparison_table,
           "ql": quality_latency, "stab": stability, "mem": memorization, "traj": trajectory_gif}
    for w in what:
        if w in fns:
            fns[w]()


def x2_figures():
    """1x vs 2x budget: sample grids and a metric chart."""
    if not os.path.exists("results/main_eval_x2.json"):
        return
    A, B = json.load(open("results/main_eval.json")), json.load(open("results/main_eval_x2.json"))
    rows = []
    for m in MODELS:
        p1, p2 = f"runs/{m}/samples_final.npz", f"runs/{m}_x2/samples_final.npz"
        if os.path.exists(p1) and os.path.exists(p2):
            a = titled(grid_img(np.load(p1)["x"][:32], 8, 3), f"{NICE[m]} @ 1x budget  (CFD {A[m]['cfd']:.1f})", 14)
            b = titled(grid_img(np.load(p2)["x"][:32], 8, 3), f"{NICE[m]} @ 2x budget  (CFD {B[m + '_x2']['cfd']:.1f})", 14)
            rows.append(hcat([a, b], 24))
    if rows:
        vcat(rows).save("media/samples_1x_vs_2x.png"); print("media/samples_1x_vs_2x.png")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (key, lab, log) in zip(axs, [("cfd", "CFD ↓", True), ("coverage", "combos covered (of 1296) ↑", False),
                                          ("recall", "recall ↑", False)]):
        for m in MODELS:
            if m + "_x2" not in B:
                continue
            ax.plot([1, 2], [A[m][key], B[m + "_x2"][key]], "-o", color=COL[m], label=NICE[m], lw=2)
        ax.axhline(A["real_reference"][key], color="#555", ls=":", label="real data")
        ax.set_xticks([1, 2]); ax.set_xticklabels(["1x budget\n(2.5 PFLOP / 25 min)", "2x budget\n(5 PFLOP / 50 min)"])
        ax.set_ylabel(lab); ax.grid(alpha=.3)
        if log:
            ax.set_yscale("log")
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Doubling the matched training budget (5000 samples each; U-Nets FLOP-capped, VAE/GAN wall-capped)")
    fig.tight_layout(); fig.savefig("media/budget_scaling.png", dpi=110); plt.close(fig)
    print("media/budget_scaling.png")


if __name__ == "__main__" and (not sys.argv[1:] or "x2" in sys.argv[1:]):
    x2_figures()
