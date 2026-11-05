"""Round 8 Gate B demo: the camera streams side by side -> media/r8_stabcam.mp4 (25 fps, real time).

    ./r8_job.sh stabcam-video ../.venv/bin/python r8_stabcam_video.py        # ~5-8 min, one process

Segments (the measurement's own deterministic episodes, re-run):
  1. train-0020, walk 0.3 m/s straight (10 s) then stand (3 s)
  2. train-0020, walk 0.3 m/s steering +0.4 (10 s) then stand
  3. a reference-bot set_lamp episode (train-0040, k=8), its first 30 s
Tiles (each stream at its real 128 px, shown 1.5x): head | reflex | trunk | ORACLE / eis | eis+reflex | pl_head | pl_eis.
The reflex tiles come from the paired reflex run of the same segment (its own physics); the bot segment has none.
Phase-locked tiles hold the frame of the last left-foot touchdown.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402

TILE = 192
LABEL_H = 18
FPS = 25
NAMES = [["head (as now)", "reflex (gaze.py)", "trunk camera (sim)", "ORACLE (sim truth)"],
         ["eis (sensor-only)", "eis + reflex", "phase-locked head", "phase-locked eis"]]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


FONT, FONT_BIG = _font(12), _font(14)


def collect(run_fn, mode, every=2, max_steps=None):
    """Run an episode with a hook that renders this mode's streams every `every` 50 Hz steps."""
    frames = []
    state = {"pl_head": None, "pl_eis": None}

    def hook(pr, x):
        if pr.k % every or (max_steps is not None and pr.k >= max_steps):
            return
        rig = pr.rig
        wide = rig.render_pose(x["hp"], x["Rh"], S.WIDE_PX, wide=True)
        eis, _ = S.warp(wide, x["Rhat"], x["R_tgt"])
        head = rig.render_head()
        f = {"head": head, "eis": eis}
        if mode == "off":
            f["trunk"] = rig.render_trunk()
            f["oracle"] = rig.render_pose(x["hp"], x["R_or"])
            if pr.events and pr.events[-1][1] == "L" and pr.events[-1][0] >= pr.k - every + 1 and x["walking"]:
                state["pl_head"], state["pl_eis"] = head, eis
            f["pl_head"], f["pl_eis"] = state["pl_head"], state["pl_eis"]
        frames.append(f)

    run_fn(mode, hook)
    return frames


def tile(img, label):
    out = Image.new("RGB", (TILE, TILE + LABEL_H), (18, 18, 18))
    if img is not None:
        out.paste(Image.fromarray(img).resize((TILE, TILE), Image.Resampling.BILINEAR), (0, LABEL_H))
    else:
        ImageDraw.Draw(out).text((8, LABEL_H + TILE // 2 - 6), "-", fill=(120, 120, 120), font=FONT)
    ImageDraw.Draw(out).text((4, 2), label, fill=(235, 235, 235), font=FONT)
    return out


def compose(off, on, title, t):
    W, H = 4 * TILE, 2 * (TILE + LABEL_H) + 22
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(canvas).text((6, 3), f"{title}   t = {t:5.2f} s", fill=(255, 220, 120), font=FONT_BIG)
    grid = [[off.get("head"), on.get("head") if on else None, off.get("trunk"), off.get("oracle")],
            [off.get("eis"), on.get("eis") if on else None, off.get("pl_head"), off.get("pl_eis")]]
    for r in range(2):
        for c in range(4):
            label = NAMES[r][c] if not (c == 1 and on is None) else NAMES[r][c] + ": n/a here"
            canvas.paste(tile(grid[r][c], label), (c * TILE, 22 + r * (TILE + LABEL_H)))
    return np.asarray(canvas)


def scripted(cfg, vx, wz):
    def run(mode, hook):
        S.run_scripted(cfg, vx, wz, mode, "none_but_rig", hooks=[hook])
    return run


def main():
    cfg = {c["id"]: c for c in S.TRAIN}
    out = HERE / "media" / "r8_stabcam.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=2)
    segs = [("train-0020: walk 0.3 m/s, then stand", scripted(cfg["train-0020"], 0.3, 0.0), True, None),
            ("train-0020: walk 0.3 m/s steering +0.4, then stand", scripted(cfg["train-0020"], 0.3, 0.4), True, None)]

    def bot(mode, hook):
        S.run_bot(cfg["train-0040"], 8, mode, "none_but_rig", hooks=[hook])
    segs.append(("bot set_lamp, train-0040 k=8 (first 30 s)", bot, False, 1500))
    for title, fn, with_reflex, max_steps in segs:
        off = collect(fn, "off", max_steps=max_steps)
        on = collect(fn, "on", max_steps=max_steps) if with_reflex else None
        for i, f in enumerate(off):
            w.append_data(compose(f, on[i] if on and i < len(on) else ({} if on else None), title, i * 2 * S.SUB_DT))
        print(title, len(off), "frames", flush=True)
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
