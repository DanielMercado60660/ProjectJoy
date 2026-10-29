"""Render races on tracks as MP4/GIF (PIL frames -> imageio/ffmpeg).

All controllers drive the same track at the same time as non-interacting ghosts.
The first controller's distance rays are drawn. Cars that touch a wall freeze
with a red X; cars that finish freeze on the line with their lap time.

    ../.venv/bin/python render.py --split heldout --tracks 0 1 2 3 \
        --controllers es ray --out media/heldout_es_vs_ray.mp4
"""
import argparse
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racing.controllers import LABELS, get_controller
from racing.evaluate import run_lap
from racing.sim import DT, RAY_ANGLES, RAY_LEN, _cast
from racing.track import make_tracks, pack

COLORS = {"es": (230, 60, 60), "ppo": (40, 120, 230), "ray": (250, 190, 30), "ray_default": (200, 140, 30),
          "pursuit": (150, 150, 150), "es_dr": (220, 70, 200), "ppo_dr": (40, 190, 190),
          "es_soft": (255, 120, 20)}
ASPHALT, GRASS, WALL = (70, 72, 78), (46, 110, 58), (235, 235, 235)


def font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def signed_area(p):
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


class Cell:
    def __init__(self, track, size, pad=24):
        self.track, self.size = track, size
        pts = np.vstack([track.left, track.right])
        lo, hi = pts.min(0), pts.max(0)
        self.scale = (size - 2 * pad) / (hi - lo).max()
        self.off = (size - (hi - lo) * self.scale) / 2 - lo * self.scale
        self.bg = Image.new("RGB", (size, size), GRASS)
        d = ImageDraw.Draw(self.bg)
        outer, inner = (track.left, track.right)
        if abs(signed_area(outer)) < abs(signed_area(inner)):
            outer, inner = inner, outer
        d.polygon([tuple(q) for q in self.px(outer)], fill=ASPHALT)
        d.polygon([tuple(q) for q in self.px(inner)], fill=GRASS)
        for w in (outer, inner):
            q = [tuple(x) for x in self.px(np.vstack([w, w[:1]]))]
            d.line(q, fill=WALL, width=2)
        a, b = self.px(track.left[:1])[0], self.px(track.right[:1])[0]
        d.line([tuple(a), tuple(b)], fill=(255, 255, 255), width=4)  # start/finish line
        c, t = track.center[0], track.tangent[0]
        tip = self.px((c + 14 * t)[None])[0]
        d.line([tuple(self.px(c[None])[0]), tuple(tip)], fill=(255, 255, 255), width=2)

    def px(self, p):
        q = p * self.scale + self.off
        q[:, 1] = self.size - q[:, 1]
        return q

    def car(self, d, x, y, yaw, color, enlarge=2.2):
        Lh, Wh = 2.3 * enlarge, 1.0 * enlarge
        c, s = math.cos(yaw), math.sin(yaw)
        corners = np.array([[Lh, Wh], [Lh, -Wh], [-Lh, -Wh], [-Lh, Wh]])
        pts = np.stack([x + corners[:, 0] * c - corners[:, 1] * s, y + corners[:, 0] * s + corners[:, 1] * c], 1)
        d.polygon([tuple(q) for q in self.px(pts)], fill=color, outline=(0, 0, 0))


def simulate(names, split, tids, grip, mode="strict"):
    bank = pack(make_tracks(split, max(tids) + 1))
    runs = {}
    for n in names:
        ctrl = get_controller(n, bank)
        runs[n] = run_lap(ctrl, bank, grip=grip, mode=mode, record=True, track_ids=tids)
    return bank, runs


def render(names, split, tids, grip, out, cell=480, fps=20, every=1, max_frames=None, title=None, mode="strict"):
    bank, runs = simulate(names, split, tids, grip, mode)
    cols = 2 if len(tids) > 1 else 1
    rows = int(math.ceil(len(tids) / cols))
    cells = [Cell(bank.tracks[t], cell) for t in tids]
    n_steps = max(r["traj"].shape[0] for r in runs.values())
    if max_frames:
        n_steps = min(n_steps, max_frames * every)
    f_small, f_big = font(13), font(15)
    legend_h = 32
    frames = []
    for k in range(0, n_steps, every):
        img = Image.new("RGB", (cols * cell, rows * cell + legend_h), (20, 20, 24))
        for ci, t in enumerate(tids):
            im = cells[ci].bg.copy()
            d = ImageDraw.Draw(im)
            lines = []
            for ni, n in enumerate(names):
                r = runs[n]
                kk = min(k, r["traj"].shape[0] - 1)
                x, y, yaw = r["traj"][kk, ci, :3]
                if ni == 0:  # rays of the lead controller
                    tid = t
                    for ang in RAY_ANGLES:
                        dist = _cast(x, y, yaw + ang, bank.seg_a[tid], bank.seg_b[tid], bank.n_seg[tid], RAY_LEN)
                        end = np.array([[x + dist * math.cos(yaw + ang), y + dist * math.sin(yaw + ang)]])
                        p0, p1 = cells[ci].px(np.array([[x, y]]))[0], cells[ci].px(end)[0]
                        d.line([tuple(p0), tuple(p1)], fill=(255, 255, 160), width=1)
                        d.ellipse([p1[0] - 2, p1[1] - 2, p1[0] + 2, p1[1] + 2], fill=(255, 255, 160))
                cells[ci].car(d, x, y, yaw, COLORS.get(n, (255, 255, 255)))
                lt = r["lap_time"][ci]
                if np.isfinite(lt) and k * DT >= lt - 1e-9:
                    status = f"finished {lt:.1f}s"
                elif mode == "soft":
                    status = f"{r['traj'][kk, ci, 3] * 3.6:.0f} km/h" + (
                        "  SCRAPING WALL" if r["lat_hist"][kk, ci] >= 0.999 else "")
                elif kk == r["traj"].shape[0] - 1 and r["collisions"][ci] > 0:
                    status = "CRASHED"
                    p = cells[ci].px(np.array([[x, y]]))[0]
                    d.line([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=(255, 0, 0), width=3)
                    d.line([p[0] - 7, p[1] + 7, p[0] + 7, p[1] - 7], fill=(255, 0, 0), width=3)
                else:
                    status = f"{r['traj'][kk, ci, 3] * 3.6:.0f} km/h"
                lines.append((n, status))
            d.text((8, 6), f"{split} track #{t}   t={k * DT:5.1f}s   grip {grip}" + ("   SOFT WALLS" if mode == "soft" else ""), fill=(255, 255, 255), font=f_big)
            for li, (n, st) in enumerate(lines):
                d.rectangle([8, 28 + li * 17, 18, 38 + li * 17], fill=COLORS.get(n, (255, 255, 255)))
                d.text((24, 25 + li * 17), f"{n}: {st}", fill=(255, 255, 255), font=f_small)
            img.paste(im, ((ci % cols) * cell, (ci // cols) * cell))
        dd = ImageDraw.Draw(img)
        leg = "   ".join(f"{LABELS.get(n, n)}" for n in names)
        dd.text((8, rows * cell + 5), (title + "   " if title else "") + leg + "   (rays: " + names[0] + ")",
                fill=(220, 220, 220), font=f_small)
        frames.append(np.asarray(img))
    for _ in range(fps):  # hold the last frame for a second
        frames.append(frames[-1])
    if out.endswith(".gif"):
        imageio.mimsave(out, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimsave(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    print(f"wrote {out}: {len(frames)} frames")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--tracks", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--controllers", nargs="+", default=["es", "ray"])
    ap.add_argument("--grip", type=float, default=1.0)
    ap.add_argument("--out", default="media/heldout_race.mp4")
    ap.add_argument("--cell", type=int, default=480)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--mode", default="strict", choices=["strict", "soft"])
    a = ap.parse_args()
    render(a.controllers, a.split, a.tracks, a.grip, a.out, a.cell, a.fps, a.every, a.max_frames, a.title, a.mode)


if __name__ == "__main__":
    main()
