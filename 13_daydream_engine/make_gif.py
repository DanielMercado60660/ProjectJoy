"""Demo GIF: real (simulator) vs imagined (model, open loop) for held-out scenes.

Each row is one held-out test scene. Left: the true simulator. Middle: the
model's open-loop imagination from the same first frame (thin outlines = where
the real balls are). Right: the no-collision ballistic baseline, for contrast.
Reloads the checkpoint, so it reproduces the demo from saved weights.
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import sim
from model import load, imagine

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/dynamics")
ap.add_argument("--data", default="data")
ap.add_argument("--scenes", default="3,7,11")
ap.add_argument("--frames", type=int, default=100)
ap.add_argument("--out", default="media/real_vs_imagined.gif")
ap.add_argument("--px", type=int, default=200)
a = ap.parse_args()

ids = [int(s) for s in a.scenes.split(",")]
S = np.load(os.path.join(a.data, "test.npz"))["states"][ids]
n = min(a.frames, S.shape[1] - 1)
model, cfg = load(a.ckpt)
real = S[:, : n + 1]
imag = imagine(model, S[:, 0], n)
ball = sim.ballistic(S[:, 0], n)
COLORS = [(230, 80, 60), (60, 140, 230), (240, 180, 40)]
P, PAD, TOP = a.px, 10, 22


def to_px(x, y):
    return PAD + x * P, TOP + (1 - y) * P


def draw_panel(canvas, ox, oy, st, ghost=None, label=""):
    """Draw one panel on its own tile, then paste it, so balls outside the box are cropped
    at the panel edge instead of bleeding into the neighbouring panel."""
    tile = Image.new("RGB", (P + 2 * PAD, P + TOP + PAD), (250, 250, 247))
    d = ImageDraw.Draw(tile)
    d.rectangle([PAD, TOP, PAD + P, TOP + P], outline=(40, 40, 40), width=2)
    d.text((PAD, 5), label, fill=(30, 30, 30))
    r = sim.R * P
    for b in range(sim.N_BALLS):
        if ghost is not None:
            gx, gy = to_px(ghost[4 * b], ghost[4 * b + 1])
            d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=COLORS[b], width=1)
        x, y = to_px(st[4 * b], st[4 * b + 1])
        if -0.5 < st[4 * b] < 1.5 and -0.5 < st[4 * b + 1] < 1.5:
            d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[b])
    canvas.paste(tile, (ox, oy))


W = 3 * (P + 2 * PAD)
Hh = len(ids) * (P + TOP + PAD) + 24
frames = []
for t in range(n + 1):
    im = Image.new("RGB", (W, Hh), (250, 250, 247))
    d = ImageDraw.Draw(im)
    for row in range(len(ids)):
        oy = row * (P + TOP + PAD)
        e_m = np.linalg.norm((imag[row, t] - real[row, t]).reshape(-1, 4)[:, :2], axis=1).mean()
        e_b = np.linalg.norm((ball[row, t] - real[row, t]).reshape(-1, 4)[:, :2], axis=1).mean()
        draw_panel(im, 0, oy, real[row, t], label=f"REAL  scene {ids[row]}")
        draw_panel(im, P + 2 * PAD, oy, imag[row, t], ghost=real[row, t], label=f"IMAGINED (model) err {e_m:.3f}")
        draw_panel(im, 2 * (P + 2 * PAD), oy, ball[row, t], ghost=real[row, t], label=f"no-collision baseline {e_b:.2f}")
    d = ImageDraw.Draw(im)
    note = f"t = {t * sim.FRAME_DT:.2f} s  (frame {t}/{n}; model trained on {cfg['args']['unroll']}-frame unrolls)  0.5x speed"
    d.text((PAD, Hh - 18), note, fill=(200, 30, 30) if t > cfg["args"]["unroll"] else (30, 30, 30))
    frames.append(np.array(im))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
imageio.mimsave(a.out, frames, duration=40, loop=0)
# also a still strip of a few frames for quick viewing
strip = np.concatenate([frames[t] for t in (0, n // 4, n // 2, n)], axis=1)
Image.fromarray(strip).save(a.out.replace(".gif", "_strip.png"))
print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")
