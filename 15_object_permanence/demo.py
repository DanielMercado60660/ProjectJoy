"""Render demo GIFs (learned GRU tracker vs Kalman baseline, side by side) and export the data for
media/demo.html.

  ../.venv/bin/python demo.py --variant base --ckpt checkpoints/base/final
  ../.venv/bin/python demo.py --variant hard --ckpt checkpoints/hard/final

Each panel: the rendered frame (what the trackers see), the true position of hidden balls drawn
faintly (dotted ring), and the tracker's predicted mean (+) with 1σ / 2σ ellipses.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import kalman
import metrics
import sim
from evaluate import run_model
from train import load_model

S = 6  # upscale
DRAW_COL = [(255, 90, 80), (90, 230, 110), (110, 150, 255)]
DRAW_COL_SAME = [(255, 120, 200), (80, 230, 255), (255, 200, 60)]


def ellipse_pts(m, c, k, n=40):
    w, v = np.linalg.eigh(c)
    w = np.sqrt(np.maximum(w, 1e-9)) * k
    a = np.linspace(0, 2 * np.pi, n)
    pts = m[:, None] + (v @ (w[:, None] * np.stack([np.cos(a), np.sin(a)])))
    return [(float(x * S), float(y * S)) for x, y in pts.T]


def panel(frame, gt, vis, present, mean, cov, title, same_color, font):
    img = Image.fromarray((frame * 255).astype(np.uint8)).resize((64 * S, 64 * S), Image.NEAREST)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    pal = DRAW_COL_SAME if same_color else DRAW_COL
    for k in range(len(present)):
        if not present[k]:
            continue
        col = pal[k]
        if vis[k] < 0.999:  # faint truth for (partly) hidden balls
            x, y = gt[k] * S
            r = sim.R * S
            for a in range(0, 360, 30):
                dr.arc([x - r, y - r, x + r, y + r], a, a + 15, fill=col + (200,), width=2)
        m, c = mean[k], cov[k]
        if np.trace(c) < 4000:
            dr.polygon(ellipse_pts(m, c, 2.0), outline=col + (170,), fill=col + (30,), width=2)
            dr.polygon(ellipse_pts(m, c, 1.0), outline=col + (255,), fill=col + (60,), width=2)
        x, y = m * S
        dr.line([x - 6, y, x + 6, y], fill=col + (255,), width=2)
        dr.line([x, y - 6, x, y + 6], fill=col + (255,), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), ov)
    top = Image.new("RGBA", (img.width, 26), (20, 20, 24, 255))
    ImageDraw.Draw(top).text((6, 5), title, fill=(235, 235, 235), font=font)
    out = Image.new("RGBA", (img.width, img.height + 26))
    out.paste(top, (0, 0))
    out.paste(img, (0, 26))
    return out


def make_gif(path, frames, d, i, preds, names, font):
    gt, vis, present = d["pos"][i], d["vis"][i], d["present"][i]
    same = bool(np.allclose(d["colors"][i][0], d["colors"][i][1]))
    ims = []
    for t in range(frames.shape[0]):
        ps = []
        for name in names:
            mean, cov = preds[name]
            e = np.linalg.norm(mean[t] - gt[t], axis=-1)[present]
            ps.append(panel(frames[t], gt[t], vis[t], present, mean[t], cov[t],
                            f"{name}  t={t:02d}  err={e.mean():.1f}px", same, font))
        W = sum(p.width for p in ps) + 8 * (len(ps) - 1)
        canvas = Image.new("RGB", (W, ps[0].height), (40, 40, 44))
        x = 0
        for p in ps:
            canvas.paste(p.convert("RGB"), (x, 0))
            x += p.width + 8
        ims.append(canvas.quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=160, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", choices=["base", "hard"])
    ap.add_argument("--ckpt", default="checkpoints/base/final")
    ap.add_argument("--n_pool", type=int, default=300)
    ap.add_argument("--n_gifs", type=int, default=2)
    ap.add_argument("--out", default="media")
    ap.add_argument("--data_out", default="results", help="where demo_data_<variant>.json goes")
    ap.add_argument("--split", default="", help="override split (default: <variant> test_long)")
    ap.add_argument("--crossing_swap", action="store_true",
                    help="instead pick the first sequence where two balls touch and the GRU ends swapped "
                         "but the Kalman filter does not (used for the identity-failure clip)")
    args = ap.parse_args()
    pre = "" if args.variant == "base" else "hard_"
    split = args.split or (pre + "test_long")
    d = sim.load_split(split)
    idx = np.arange(args.n_pool)
    frames = sim.render_split(d, idx)
    model = load_model(args.ckpt)[0]
    with open(f"results/kalman_tuning_{args.variant}.json") as f:
        kc = json.load(f)["best"]["bounce=True"]
    g = run_model(model, frames)
    kf = kalman.track_split(d, idx, frames=frames, q=kc["q"], r=kc["r"], bounce=True)
    preds_all = {"GRU tracker": g, "Kalman+bounce": kf}
    # pick sequences: #0 (unselected, for honesty) + the one where the trackers' hidden-frame
    # errors differ most in each direction
    gt = d["pos"][idx]
    hid = (d["vis"][idx] == 0) & d["present"][idx][:, None]
    eg = np.where(hid, np.linalg.norm(g[0] - gt, axis=-1), 0).sum((1, 2)) / np.maximum(hid.sum((1, 2)), 1)
    ek = np.where(hid, np.linalg.norm(kf[0] - gt, axis=-1), 0).sum((1, 2)) / np.maximum(hid.sum((1, 2)), 1)
    picks = {"random": 0, "gru_better": int(np.argmax(ek - eg)), "kalman_better": int(np.argmax(eg - ek))}
    if args.crossing_swap:
        import itertools
        def swapped_end(m, n):
            ks = np.where(d["present"][n])[0]
            base = np.linalg.norm(m[n][-1, ks] - gt[n][-1, ks], axis=-1).sum()
            return any(np.linalg.norm(m[n][-1, list(p)] - gt[n][-1, ks], axis=-1).sum() < base - 2
                       for p in itertools.permutations(ks) if list(p) != list(ks))
        def touching(n):
            ks = np.where(d["present"][n])[0]
            return min(np.linalg.norm(gt[n][:, a] - gt[n][:, b], axis=-1).min()
                       for a in ks for b in ks if a < b) < 2 * sim.R
        cand = [n for n in range(len(idx)) if touching(n) and swapped_end(g[0], n) and not swapped_end(kf[0], n)]
        picks = {f"crossing_swap_{args.split}": cand[0]}
    font = ImageFont.load_default(size=15)
    os.makedirs(args.out, exist_ok=True)
    # display frames are the same state rendered without sensor noise (the trackers saw the noisy
    # version); noise makes GIF palettes swallow the thin overlay lines
    clean = sim.render(d["pos"][idx], d["present"][idx], d["occ"][idx], d["colors"][idx], noise=0)
    for tag, i in picks.items():
        preds = {n: (p[0][i], p[1][i]) for n, p in preds_all.items()}
        path = os.path.join(args.out, f"demo_{args.variant}_{tag}.gif")
        make_gif(path, clean[i], d, i, preds, list(preds), font)
        print(f"wrote {path} (seq {i}: hidden err gru {eg[i]:.2f}px kalman {ek[i]:.2f}px)")
    if args.crossing_swap:
        return
    # export a handful of sequences for demo.html
    exp = []
    order = [0] + [int(j) for j in np.argsort(-(hid.sum((1, 2))))[:11] if j != 0][:11]
    for i in order:
        exp.append(dict(
            split=split, index=int(i), pos=np.round(d["pos"][i], 2).tolist(), vis=np.round(d["vis"][i], 3).tolist(),
            present=d["present"][i].tolist(), occ=np.round(d["occ"][i], 2).tolist(),
            colors=np.round(d["colors"][i], 3).tolist(),
            gru=dict(mean=np.round(g[0][i], 2).tolist(), cov=np.round(g[1][i], 3).tolist()),
            kalman=dict(mean=np.round(kf[0][i], 2).tolist(), cov=np.round(np.minimum(kf[1][i], 1e4), 3).tolist())))
    with open(os.path.join(args.data_out, f"demo_data_{args.variant}.json"), "w") as f:
        json.dump(exp, f)
    print("exported", len(exp), "sequences for demo.html")


if __name__ == "__main__":
    main()
