"""Render matches to MP4/GIF from the saved checkpoints.

  ../.venv/bin/python render.py                                     # default 2x2 demo
  ../.venv/bin/python render.py --tiles final/final it40/final --ticks 600 --name early_vs_final

A tile spec is PURSUER/EVADER where each side is: final | it<N> (archive snapshot) | a scripted
kind (random, greedy, lead for pursuers; random, flee, smartflee for evaders).
"""
import argparse, os
import numpy as np
import imageio.v2 as imageio
import mlx.core as mx
from PIL import Image, ImageDraw, ImageFont

import sim
from policy import load_pair
from evaluate import Net, Script, load_archive

TILE = 360
PAD = 22


def ctrl(spec, role, final, arch, seed):
    if spec == "final":
        return Net(final[0 if role == "p" else 1], role, "learned", seed=seed)
    if spec.startswith("it"):
        it = int(spec[2:])
        return Net(arch[it]["P" if role == "p" else "E"], role, f"learned@{it}", seed=seed)
    return Script(spec, role, seed)


def to_px(xy):
    s = (TILE - 2 * 10) / (2 * sim.L)
    return 10 + (xy[0] + sim.L) * s, PAD + 10 + (sim.L - xy[1]) * s


def draw_tile(env, k, trail, title, score, flash):
    im = Image.new("RGB", (TILE, TILE + PAD), (250, 249, 246))
    d = ImageDraw.Draw(im)
    s = (TILE - 20) / (2 * sim.L)
    d.rectangle([10, PAD + 10, TILE - 10, PAD + TILE - 10], outline=(60, 60, 60), width=2,
                fill=(255, 220, 220) if flash else (255, 255, 255))
    for ox, oy, r in sim.OBSTACLES:
        cx, cy = to_px((ox, oy))
        d.ellipse([cx - r * s, cy - r * s, cx + r * s, cy + r * s], fill=(150, 150, 150))
    cols = [(214, 39, 40), (31, 119, 180)]
    for ag in (0, 1):
        pts = [to_px(p[ag]) for p in trail]
        if len(pts) > 1:
            d.line(pts, fill=tuple(int(c * 0.5 + 255 * 0.5) for c in cols[ag]), width=2)
        x, y = to_px(env.pos[k, ag])
        rr = sim.CAPTURE_R / 2 * s
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=cols[ag])
        # stamina bar
        bx = 14 + ag * 90
        d.rectangle([bx, PAD + TILE - 30, bx + 80, PAD + TILE - 24], outline=cols[ag])
        d.rectangle([bx, PAD + TILE - 30, bx + 80 * env.stam[k, ag], PAD + TILE - 24], fill=cols[ag])
    d.text((6, 4), title, fill=(20, 20, 20))
    d.text((TILE - 150, 4), f"caught {score[0]}  escaped {score[1]}", fill=(20, 20, 20))
    d.text((16, PAD + 14), f"t={env.t[k] * sim.DT:4.1f}s", fill=(90, 90, 90))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--tiles", nargs="+", default=["final/final", "final/smartflee", "lead/final", "EARLY/final"])
    ap.add_argument("--ticks", type=int, default=900)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--name", default="matches")
    ap.add_argument("--gif", action="store_true", help="also write a half-rate GIF")
    ap.add_argument("--media", default="media")
    a = ap.parse_args()
    mx.set_default_device(mx.cpu)
    os.makedirs(a.media, exist_ok=True)
    pf, ef, _ = load_pair(os.path.join(a.ckpt, "final.npz"))
    arch = load_archive(os.path.join(a.ckpt, "archive.npz"))
    its = list(arch)
    early = its[max(1, len(its) // 10)] if len(its) > 1 else its[0]
    tiles = []
    for i, t in enumerate(a.tiles):
        ps, es = t.replace("EARLY", f"it{early}").split("/")
        P = ctrl(ps, "p", (pf, ef), arch, a.seed + 10 * i)
        E = ctrl(es, "e", (pf, ef), arch, a.seed + 10 * i + 1)
        tiles.append({"env": sim.Arena(1, a.seed + i), "P": P, "E": E, "trail": [], "score": [0, 0],
                      "flash": 0, "title": f"{P.name} (red) vs {E.name} (blue)"})
    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    frames = []
    for t in range(a.ticks):
        imgs = []
        for tl in tiles:
            env = tl["env"]
            tl["trail"].append(env.pos[0].copy())
            tl["trail"] = tl["trail"][-40:]
            _, _, done, cap, _ = env.step(tl["P"].act(env), tl["E"].act(env))
            if done[0]:
                tl["score"][0 if cap[0] else 1] += 1
                tl["flash"] = 8 if cap[0] else 0
                tl["trail"] = []
            imgs.append(draw_tile(env, 0, tl["trail"], tl["title"], tl["score"], tl["flash"] > 0))
            tl["flash"] = max(0, tl["flash"] - 1)
        canvas = Image.new("RGB", (cols * TILE, rows * (TILE + PAD)), (255, 255, 255))
        for i, im in enumerate(imgs):
            canvas.paste(im, ((i % cols) * TILE, (i // cols) * (TILE + PAD)))
        frames.append(np.asarray(canvas))
    mp4 = os.path.join(a.media, f"{a.name}.mp4")
    imageio.mimwrite(mp4, frames, fps=int(1 / sim.DT), quality=7, macro_block_size=1)
    print("wrote", mp4, len(frames), "frames")
    if a.gif:
        small = [np.asarray(Image.fromarray(f).resize((f.shape[1] * 2 // 3, f.shape[0] * 2 // 3))) for f in frames[::2]]
        gif = os.path.join(a.media, f"{a.name}.gif")
        imageio.mimwrite(gif, small, duration=2 * sim.DT * 1000, loop=0)
        print("wrote", gif)
    for tl in tiles:
        print(f"{tl['title']}: caught {tl['score'][0]} escaped {tl['score'][1]}")


if __name__ == "__main__":
    main()
