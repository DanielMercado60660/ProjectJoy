"""M1 media:
  media/m1_families.png   overhead view + the duck's head-camera view at the start, 2 configs per family
  media/m1_button_lamp.mp4 the duck walks over the button toward the lamp; overhead, chase and head views
  media/m1_lamp_pair.png  head-camera frames just before and after the press (lamp off / on)
"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs import hand_config  # noqa: E402
from m0_gate import goto_command  # noqa: E402
from room import World  # noqa: E402

MEDIA = HERE / "media"
S = 200   # tile size


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def top_view(world, r):
    r.update_scene(world.duck.data, camera="top")
    return r.render()


def families_png():
    load = lambda n: json.loads((HERE / "splits" / n).read_text())  # noqa: E731
    train = load("train.json")["configs"]
    picks = [("train", train[0]), ("train", train[1])]
    for f in ("instances", "arrangements", "appearances", "distractor"):
        d = load(f"{f}.json")
        for s in ("dev", "test"):
            picks.append((f"{f} {s}", d[s][0]))
    rows = []
    for name, cfg in picks:
        w = World(cfg)
        for _ in range(10):   # 0.2 s so the distractor shows a colour
            w.step()
        top = top_view(w, mujoco.Renderer(w.model, S, S))
        head = np.asarray(Image.fromarray(w.duck.head_image(mujoco.Renderer(w.model, 128, 128))).resize((S, S)))
        rows.append(np.concatenate([label(top, f"{name}: {cfg['id']}"), label(head, "head camera")], 1))
    grid = [np.concatenate(rows[i:i + 2], 1) for i in range(0, len(rows), 2)]
    imageio.imwrite(MEDIA / "m1_families.png", np.concatenate(grid, 0))
    return len(picks)


def button_clip():
    cfg = hand_config(button_pos=(0.6, 0.0), empty=True, start=((-0.4, 0.0), 0.0))
    w = World(cfg)
    rt, rc, rh = mujoco.Renderer(w.model, 240, 240), mujoco.Renderer(w.model, 240, 320), mujoco.Renderer(w.model, 240, 240)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 1.1, -20, 215
    goal = np.array([1.05, 0.0])
    frames, before, after, was_on = [], None, None, w.lamp_on
    for k in range(int(25 / 0.02)):
        near = np.linalg.norm(w.duck.pos()[:2] - goal) < 0.1
        w.duck.set_command(*((0, 0, 0) if near else goto_command(w.duck, goal)))
        head_now = None
        if k % 2 == 0 or w.lamp_on != was_on:
            head_now = w.duck.head_image(rh)
        if w.lamp_on and not was_on and after is None:
            after = head_now
        if not w.lamp_on and head_now is not None:
            before = head_now
        was_on = w.lamp_on
        w.step()
        if k % 2 == 0:
            cam.lookat[:] = w.duck.pos() + np.array([0, 0, -0.03])
            rc.update_scene(w.duck.data, camera=cam)
            txt = f"t={w.duck.time:4.1f}s  button {1000 * w.button_depth():.1f} mm  presses {w.press_count}  lamp {'ON' if w.lamp_on else 'off'}"
            row = np.concatenate([top_view(w, rt), rc.render(), w.duck.head_image(rh)], 1)
            frames.append(label(row, txt))
        if near and w.press_count and k > 50 and after is not None and len(frames) > 60 + 25:
            break
    imageio.mimsave(MEDIA / "m1_button_lamp.mp4", frames, fps=25, macro_block_size=8)
    if before is not None and after is not None:
        imageio.imwrite(MEDIA / "m1_lamp_pair.png",
                        np.concatenate([label(before, "lamp off"), label(after, "lamp on")], 1))
    return len(frames), w.press_count, w.lamp_on


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    print("families", families_png())
    print("clip frames, presses, lamp", button_clip())
