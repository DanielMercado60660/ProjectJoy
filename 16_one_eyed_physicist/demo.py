"""Demo GIF: held-out clips with estimates overlaid, then a 1 s forecast re-simulated from each estimate.

  ../.venv/bin/python demo.py --ckpt checkpoints/main      -> media/demo.gif, media/demo_frame.png
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw

import scene, baseline
from model import load, predict

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/main")
ap.add_argument("--out", default="media/demo.gif")
args = ap.parse_args()

UP = 6
X, p = scene.make_set(1001, 60)             # held-out clips (test seed; training never sees seed 1001)
targets = [0.3, 0.6, 0.9]
idx = [int(np.argmin(np.abs(p["e"] - t) + 10 * (p["noise"] < 0.05))) for t in targets]
model, _ = load(args.ckpt)
mu, sd = predict(model, X[idx])
cl = [baseline.estimate(X[i]) for i in idx]
FUT = 30
tt_all = np.arange(scene.T + FUT) / scene.FPS


def to_px(x, y):
    return x * scene.SCALE * UP, (scene.GROUND_ROW - y * scene.SCALE) * UP


panels = []
for j, i in enumerate(idx):
    q = {k: v[i:i + 1] for k, v in p.items()}
    t = tt_all[None]
    xt, yt = scene.positions(q, t)
    qn = dict(q); qn["e"] = np.array([np.clip(mu[j], 0, 0.999)])
    qc = dict(q); qc["e"] = np.array([np.clip(cl[j]["e"], 0, 0.999)])
    _, yn = scene.positions(qn, t); _, yc = scene.positions(qc, t)
    frames = []
    for f in range(scene.T + FUT):
        if f < scene.T:
            im = X[i, f]
        else:
            # render the true future frame (noise-free) at time f/FPS
            xx, yy = xt[0, f], yt[0, f]
            im = np.zeros((scene.H, scene.W), np.float32) + q["bg"][0]
            cc, rr = np.meshgrid(np.arange(scene.W) + 0.5, np.arange(scene.H) + 0.5)
            d = np.hypot(cc - xx * scene.SCALE, rr - (scene.GROUND_ROW - yy * scene.SCALE))
            a = np.clip(scene.R * scene.SCALE + 0.5 - d, 0, 1)
            im = im * (1 - a) + q["fg"][0] * a
            im[int(scene.GROUND_ROW):] = 0.35
            im = (im * 255).astype(np.uint8)
        img = Image.fromarray(im).resize((scene.W * UP, scene.H * UP), Image.NEAREST).convert("RGB")
        dr = ImageDraw.Draw(img)
        # classical track so far (orange dots) during the clip
        tr = cl[j]["track"]
        for k in range(min(f + 1, scene.T)):
            if np.isfinite(tr[k, 0]):
                u, v = to_px(*tr[k]); dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(224, 123, 57))
        if f >= scene.T:
            rr_ = scene.R * scene.SCALE * UP
            u, v = to_px(xt[0, f], yn[0, f]); dr.ellipse([u - rr_, v - rr_, u + rr_, v + rr_], outline=(42, 111, 219), width=3)
            u, v = to_px(xt[0, f], yc[0, f]); dr.ellipse([u - rr_ - 3, v - rr_ - 3, u + rr_ + 3, v + rr_ + 3], outline=(224, 123, 57), width=2)
        dr.rectangle([0, 0, scene.W * UP, 58], fill=(0, 0, 0))
        dr.text((6, 4), f"true e = {q['e'][0]:.3f}   noise {q['noise'][0]:.2f}  exposure {q['expo'][0]:.2f}", fill=(255, 255, 255))
        dr.text((6, 20), f"network   e = {mu[j]:.3f} +/- {sd[j]:.3f}", fill=(110, 160, 255))
        dr.text((6, 36), f"classical e = {cl[j]['e']:.3f} +/- {cl[j]['sd']:.3f}", fill=(240, 150, 90))
        tag = "observed clip" if f < scene.T else "forecast (+%.2f s): rings = re-simulated from estimates" % ((f - scene.T + 1) / scene.FPS)
        dr.text((6, scene.H * UP - 16), tag, fill=(255, 255, 0))
        frames.append(img)
    panels.append(frames)

out = []
for f in range(scene.T + FUT):
    row = Image.new("RGB", (len(panels) * scene.W * UP + (len(panels) - 1) * 6, scene.H * UP), (40, 40, 40))
    for j, fr in enumerate(panels):
        row.paste(fr[f], (j * (scene.W * UP + 6), 0))
    out.append(row)
out[0].save(args.out, save_all=True, append_images=out[1:], duration=66, loop=0)
out[scene.T + 10].save(args.out.replace(".gif", "_frame.png"))
print("wrote", args.out, "clips", idx, "e", p["e"][idx])
