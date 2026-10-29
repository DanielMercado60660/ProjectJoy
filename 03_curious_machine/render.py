"""Demo GIF: a trained explorer vs the random walk, same start, same 512-action budget.

Each panel shows the full map (the agent itself only ever sees its 7x7 window, drawn as a frame),
an accumulating visitation heatmap, and running counters for rooms / interactions / cells.

  ../.venv/bin/python render.py --ckpt checkpoints/rnd_s0.pkl --out media/explorer_vs_random.gif
  ../.venv/bin/python render.py --ckpt checkpoints/rnd_tv_s0.pkl --noisy_tv --out media/noisy_tv.gif
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import env as E
from evaluate import run_eval

CELL = 16


def heat_overlay(img, counts, cell=CELL):
    """Blend a magenta-ish heat layer onto floor cells by log visit count."""
    if counts.max() == 0:
        return img
    h = np.log1p(counts) / np.log1p(max(counts.max(), 8))
    h = np.kron(h, np.ones((cell, cell)))[..., None]
    hot = np.array([255, 70, 20], float)
    out = img.astype(float) * (1 - 0.65 * h) + hot * 0.65 * h
    return out.astype(np.uint8)


def panel(rec, t, title, counts, tv=False, cell=CELL):
    g = rec["grid"][t]
    y, x = rec["pos"][t]
    img = E.render_rgb(g, ROOM, None, rec["tv"][t] if tv else None, cell)
    img = heat_overlay(img, counts, cell)
    # agent
    c0 = cell // 5
    img[y * cell + c0:(y + 1) * cell - c0, x * cell + c0:(x + 1) * cell - c0] = (20, 20, 20)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    v = E.HALF
    d.rectangle([(x - v) * cell, (y - v) * cell, (x + v + 1) * cell - 1, (y + v + 1) * cell - 1],
                outline=(255, 255, 255), width=2)
    # header
    W_ = im.width
    canvas = Image.new("RGB", (W_, im.height + 58), (250, 250, 248))
    canvas.paste(im, (0, 58))
    d = ImageDraw.Draw(canvas)
    events = [k for (s, k) in rec["events"] if s < t]
    rooms = len({int(ROOM[p[0], p[1]]) for p in rec["pos"][:t + 1] if ROOM[p[0], p[1]] >= 0})
    cells = int((counts > 0).sum())
    tvs = ""
    if tv:
        tvs = f"  TV-in-view {int(rec['tv_seen'][:t + 1].sum())}"
    d.text((6, 4), title, fill=(20, 20, 20), font=FONT_B)
    d.text((6, 24), f"step {t:3d}/512  rooms {rooms}/9  interactions {len(events)}/8", fill=(40, 40, 40), font=FONT)
    d.text((6, 40), f"cells {cells}/{N_FLOOR}{tvs}", fill=(40, 40, 40), font=FONT)
    return canvas


def tv_seen(rec, tv_cells):
    if len(tv_cells) == 0:
        return np.zeros(len(rec["pos"]), bool)
    p = rec["pos"]
    return np.array([np.any((np.abs(tv_cells[:, 0] - y) <= E.HALF) & (np.abs(tv_cells[:, 1] - x) <= E.HALF))
                     for y, x in p])


def make_gif(recs, titles, out, noisy_tv=False, stride=2, fps=20):
    counts = [np.zeros((E.H, E.W)) for _ in recs]
    frames = []
    T = min(len(r["pos"]) for r in recs)
    for t in range(T):
        for r, c in zip(recs, counts):
            c[r["pos"][t][0], r["pos"][t][1]] += 1
        if t % stride == 0 or t == T - 1:
            ps = [panel(r, t, ti, c, noisy_tv) for r, ti, c in zip(recs, titles, counts)]
            w = sum(p.width for p in ps) + 10 * (len(ps) - 1)
            fr = Image.new("RGB", (w, ps[0].height), (255, 255, 255))
            xo = 0
            for p in ps:
                fr.paste(p, (xo, 0)); xo += p.width + 10
            frames.append(fr)
    frames += [frames[-1]] * fps * 2  # hold the final heatmap
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                   optimize=True)
    frames[-1].save(out.rsplit(".", 1)[0] + "_final.png")
    print("wrote", out, len(frames), "frames")


def _font(size, bold=False):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT, FONT_B = _font(13), _font(15, True)
_g, ROOM, _s, _m, _tv = E.build_layout(False)
N_FLOOR = int((_g != E.WALL).sum())


def pick_episode(ex, noisy_tv, seed, n=16, choose="median"):
    """Run n recorded episodes and pick a representative one (median #interactions, then coverage)."""
    eps, recs = run_eval(ex, n_episodes=n, noisy_tv=noisy_tv, seed=seed, record=n)
    score = np.array([e["interactions"] * 1000 + e["coverage"] for e in eps])
    i = int(np.argsort(score)[len(score) // 2]) if choose == "median" else int(np.argmax(score))
    return recs[i], eps[i], i


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints (one panel each)")
    p.add_argument("--label", nargs="+", default=None)
    p.add_argument("--no_random", action="store_true", help="omit the random-walk panel")
    p.add_argument("--noisy_tv", action="store_true")
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from train import Explorer
    tvc = E.build_layout(a.noisy_tv)[4]
    labels = a.label or [f"explorer {i}" for i in range(len(a.ckpt))]
    recs, titles = [], []
    if not a.no_random:
        r0, e0, i0 = pick_episode(None, a.noisy_tv, a.seed)
        print("random episode", i0, {k: e0[k] for k in E.EP_FIELDS})
        recs.append(r0); titles.append("Random walk")
    for ck, lab in zip(a.ckpt, labels):
        ex, _ = Explorer.load(ck)
        r1, e1, i1 = pick_episode(ex, a.noisy_tv, a.seed)
        print(lab, "episode", i1, {k: e1[k] for k in E.EP_FIELDS})
        recs.append(r1); titles.append(lab)
    for r in recs:
        r["tv_seen"] = tv_seen(r, tvc)
    make_gif(recs, titles, a.out, a.noisy_tv)
