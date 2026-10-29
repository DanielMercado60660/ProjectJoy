"""M2 media: one ORACLE episode per task on a training config (overhead view | head camera), with the
instruction and what the judge sees. media/m2_<task>.mp4."""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interface import Interface  # noqa: E402
from oracle import controller_for  # noqa: E402
from room import World  # noqa: E402
from tasks import Judge, TASKS, ball_visibility, make_episode, start_episode  # noqa: E402

MEDIA = HERE / "media"
CONFIG_INDEX = {"go_and_return": 3, "set_lamp": 2, "find_ball": 0}


def label(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 13 * len(lines) + 2], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((3, 1 + 13 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def render_task(task):
    cfg = json.loads((HERE / "splits" / "train.json").read_text())["configs"][CONFIG_INDEX[task]]
    if task == "set_lamp":   # pick an episode that needs a press, so the demo shows one
        k = next(k for k in range(20) if make_episode(cfg, task, k)["instruction"]["target"] !=
                 ("on" if cfg["layout"]["lamp_initial_on"] else "off"))
    else:
        k = 0
    ep = make_episode(cfg, task, k)
    w = World(cfg)
    start_episode(w, ep)
    it = Interface(w)
    judge = Judge(w, ep)
    rt, rh = mujoco.Renderer(w.model, 300, 300), mujoco.Renderer(w.model, 300, 300)
    gen = controller_for(w, task)(it, ep["instruction"])
    frames = []

    def frame():
        rt.update_scene(w.duck.data, camera="top")
        info = f"lamp {'ON' if w.lamp_on else 'off'}, presses {w.press_count}" if task == "set_lamp" else (
            f"ball visible: {ball_visibility(w)[0]}" if task == "find_ball" else
            f"pad reached: {judge.reached_pad_tick is not None}")
        row = np.concatenate([rt.render(), w.duck.head_image(rh)], 1)
        frames.append(label(row, [f"ORACLE  {json.dumps(ep['instruction'])}  {cfg['id']}",
                                  f"t={w.duck.time:5.1f}s  judge: {info}"]))

    frame()
    cmd = next(gen)
    while not it.done and it.tick_count < ep["budget_ticks"]:
        it.tick(cmd)
        judge.update(it)
        if it.tick_count % 2 == 0:
            frame()
        if it.done:
            break
        try:
            cmd = gen.send(None)
        except StopIteration:
            break
    res = judge.result(it)
    for _ in range(10):
        frames.append(frames[-1])
    imageio.mimsave(MEDIA / f"m2_{task}.mp4", frames, fps=10, macro_block_size=4)
    return {"task": task, "config": cfg["id"], "k": k, "success": res["success"], "frames": len(frames),
            "sim_time": res["sim_time"]}


if __name__ == "__main__":
    for t in TASKS:
        print(render_task(t))
