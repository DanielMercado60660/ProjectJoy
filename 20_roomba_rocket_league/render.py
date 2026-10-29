"""PIL renderer for Arena snapshots -> GIF / MP4."""
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from env import W, H, GOAL_HALF, R_ROBOT, R_BALL

S = 220  # pixels per meter
PAD = 24


def to_px(x, y):
    return PAD + (x + W) * S, PAD + (H - y) * S


def draw_frame(snap, caption="", score=None):
    w, h = int(2 * W * S + 2 * PAD), int(2 * H * S + 2 * PAD + 22)
    img = Image.new("RGB", (w, h), (34, 110, 60))
    d = ImageDraw.Draw(img)
    x0, y0 = to_px(-W, H)
    x1, y1 = to_px(W, -H)
    d.rectangle([x0, y0, x1, y1], outline=(235, 235, 235), width=3)
    cx, _ = to_px(0, 0)
    d.line([cx, y0, cx, y1], fill=(200, 230, 200), width=1)
    for sgn, col in ((-1, (70, 140, 255)), (1, (255, 90, 70))):
        gx, gy0 = to_px(sgn * W, GOAL_HALF)
        _, gy1 = to_px(sgn * W, -GOAL_HALF)
        d.rectangle([gx - 6 if sgn > 0 else gx - 6, gy0, gx + 6, gy1], fill=col)
    cols = [(70, 140, 255), (255, 90, 70)]
    for i in range(2):
        px, py = to_px(*snap["pos"][i])
        r = R_ROBOT * S
        d.ellipse([px - r, py - r, px + r, py + r], fill=cols[i], outline=(20, 20, 20), width=2)
        hx = px + np.cos(snap["th"][i]) * r
        hy = py - np.sin(snap["th"][i]) * r
        d.line([px, py, hx, hy], fill=(255, 255, 255), width=3)
    bx, by = to_px(*snap["ball"])
    rb = R_BALL * S
    d.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=(255, 220, 40), outline=(40, 40, 40), width=1)
    d.rectangle([0, h - 22, w, h], fill=(20, 20, 20))
    d.text((8, h - 18), f"{caption}  t={snap['t'] * 0.05:4.1f}s", fill=(240, 240, 240))
    return img


def save_clip(frames, path, fps=20):
    arr = [np.asarray(f) for f in frames]
    if path.endswith(".gif"):
        imageio.mimsave(path, arr, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(path, arr, fps=fps, macro_block_size=1)
