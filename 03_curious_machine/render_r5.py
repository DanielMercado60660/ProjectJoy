"""Headless, fixed-first-episode demo from evaluate_r5.py's replay receipt."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from env import render_rgb


def render(receipt, out):
    receipt = Path(receipt); result = json.loads(receipt.read_text())
    data = np.load(receipt.with_suffix(".npz"), allow_pickle=False)
    frames = []
    seed = int(data["seed"])
    row = next(r for r in result["tasks"] if r["seed"] == seed and r["task"] == ["reach", 6])
    for phase in ("explore", "task"):
        length = len(data[phase + "_grid"])
        for index in sorted(set(range(0, length, 4)) | {length - 1}):
            rgb = render_rgb(data[phase + "_grid"][index], data["room"], data[phase + "_pos"][index],
                             data[phase + "_tv"][index], cell=12)
            canvas = Image.new("RGB", (340, 292), "#faf8ef"); canvas.paste(Image.fromarray(rgb), (56, 61))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f'R5 {result["method"]} | {result["world"]} | seed {seed}', fill="black")
            text = f"Reward-free exploration {index}/{length - 1}" if phase == "explore" else f"Surprise: reach R20 | step {index}/{length - 1}"
            draw.text((8, 23), text, fill="black")
            verdict = f'planned={row["planned"]}, success={row["success"]}, capped steps={row["steps"]}'
            draw.text((8, 40), "Fixed first trial (not selected for success)" if phase == "explore" else verdict, fill="black")
            frames.append(canvas)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0)
    print(f"DEMO {out} frames={len(frames)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--receipt", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); render(a.receipt, a.out)
