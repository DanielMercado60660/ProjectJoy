"""Draw terrariums to RGB frames (numpy -> PIL), write MP4/GIF."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import sim

SOIL = np.array([28, 22, 18], np.float32)
ROCK = np.array([110, 110, 118], np.uint8)
FOOD = np.array([255, 200, 40], np.uint8)
NEST = np.array([150, 80, 40], np.uint8)
HOME_C = np.array([60, 120, 255], np.float32)   # home pheromone: blue
FOOD_C = np.array([60, 230, 90], np.float32)    # food pheromone: green


def frame(world, b, scale=6, ants=True):
    p = world.pher[b]
    ph = np.clip(np.log1p(p[0]) / np.log1p(sim.P_MAX), 0, 1)[..., None]
    pf = np.clip(np.log1p(p[1]) / np.log1p(sim.P_MAX), 0, 1)[..., None]
    img = SOIL + ph * HOME_C * 0.7 + pf * FOOD_C * 1.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    nm = world.nest_mask(b)
    img[nm] = NEST
    f = world.food[b] > 0
    img[f] = FOOD
    img[world.obst[b] > 0] = ROCK
    img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    if ants:
        for i in range(world.N):
            if not world.alive[b, i]:
                continue
            y, x = world.pos[b, i]
            c = (255, 70, 60) if world.carry[b, i] else (240, 240, 240)
            cy, cx = y * scale + scale // 2, x * scale + scale // 2
            r = max(1, scale // 3)
            img[cy - r:cy + r + 1, cx - r:cx + r + 1] = c
    return img


_FONT = None


def font(size=13):
    global _FONT
    if _FONT is None:
        for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
            try:
                _FONT = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT


def label(img, lines, h=36):
    """Put a caption bar above an image."""
    H_, W_ = img.shape[:2]
    canvas = Image.new("RGB", (W_, H_ + h), (12, 12, 12))
    canvas.paste(Image.fromarray(img), (0, h))
    d = ImageDraw.Draw(canvas)
    for j, line in enumerate(lines):
        d.text((6, 3 + 16 * j), line, fill=(235, 235, 235), font=font())
    return np.asarray(canvas)


def hstack(imgs, gap=6):
    Hm = max(i.shape[0] for i in imgs)
    out = []
    for k, im in enumerate(imgs):
        if im.shape[0] < Hm:
            im = np.pad(im, ((0, Hm - im.shape[0]), (0, 0), (0, 0)))
        out.append(im)
        if k < len(imgs) - 1:
            out.append(np.full((Hm, gap, 3), 12, np.uint8))
    return np.concatenate(out, 1)


def write_video(frames, path, fps=30):
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    else:
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8,
                               ffmpeg_params=["-pix_fmt", "yuv420p"])
        for f in frames:
            w.append_data(f)
        w.close()
