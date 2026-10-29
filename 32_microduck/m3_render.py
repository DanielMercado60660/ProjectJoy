"""M3 media: one REFERENCE BOT episode per task on a dev config (overhead view | head camera with the
bot's detections marked | the bot's own occupancy map with its pose and remembered objects).
Captions show the bot's latest event and its memory. media/m3_<task>.mp4"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import interface  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, TASKS, Judge, make_episode, run_episode  # noqa: E402

MEDIA = HERE / "media"
# set_lamp was arrangements-dev-0002 until the M3 review: a no-press episode in which the bot never saw
# the lamp and said done after 75 s (right only because the lamp was already off). arrangements-dev-0001
# needs a press: the bot checks the lamp, finds the button, presses once and sees the lamp change.
PICK = {"go_and_return": ("instances.json", "dev", 1), "set_lamp": ("arrangements.json", "dev", 1),
        "find_ball": ("distractor.json", "dev", 1)}
COL = {"pad_red": (255, 60, 60), "pad_yellow": (255, 230, 0), "pad_green": (60, 220, 60), "pad_blue": (60, 120, 255),
       "button": (0, 255, 255), "ball": (255, 0, 255), "mat": (255, 255, 255), "lamp_on": (255, 255, 160),
       "lamp_off": (90, 90, 90)}


def map_image(bot, size=240):
    L = bot.grid.L
    img = np.zeros(L.shape + (3,), np.uint8) + 110
    img[L > 0.5] = 0
    img[L < -0.5] = 210
    im = Image.fromarray(img[::-1]).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(im)
    from botmap import CELL, HALF
    sc = size / L.shape[0]

    def px(p):
        return (p[0] + HALF) / CELL * sc, size - (p[1] + HALF) / CELL * sc
    for t in bot.mem.tracks:
        if t["n"] >= 3 and t["cls"] in COL:
            x, y = px(t["xy"])
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COL[t["cls"]])
    x, y, yaw = bot.pose()
    cx, cy = px((x, y))
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(255, 140, 0), width=2)
    d.line([cx, cy, cx + 12 * np.cos(yaw), cy - 12 * np.sin(yaw)], fill=(255, 140, 0), width=2)
    return np.asarray(im)


def head_with_dets(frame, dets, size=240):
    im = Image.fromarray(frame).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(im)
    k = size / frame.shape[0]
    for det in dets:
        i, j = det["px"]
        r = max(4, np.sqrt(det["area"]) * k / 2)
        d.rectangle([j * k - r, i * k - r, j * k + r, i * k + r], outline=COL.get(det["cls"], (255, 255, 255)), width=2)
    return np.asarray(im)


def caption(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 13 * len(lines) + 2], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((3, 1 + 13 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def render(task):
    f, part, i = PICK[task]
    cfg = json.loads((HERE / "splits" / f).read_text())[part][i]
    ep = make_episode(cfg, task, 0)
    w = World(cfg)
    fac, bot = controller(task, BUDGET_TICKS[task])
    top = mujoco.Renderer(w.model, 240, 240)
    frames = []
    orig = interface.Interface.tick

    pending = []      # (overhead, head frame, sim time, tick) of the last even tick, drawn once the bot has sensed it

    def flush():
        # the bot senses a tick's observation after Interface.tick returns, so its detections, map and
        # memory for tick N are only ready at the next call (M3 review: boxes used to be one tick stale)
        if not pending:
            return
        topimg, fr, t, k = pending.pop()
        dets = bot.dets if bot.tick == k else []
        row = np.concatenate([topimg, head_with_dets(fr, dets), map_image(bot)], 1)
        ev = bot.log[-1] if bot.log else {"event": "-"}
        evs = {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in ev.items() if kk not in ("tick", "direction", "last")}
        conf = sorted({tr["cls"] for tr in bot.mem.tracks if tr["n"] >= 3})
        frames.append(caption(row, [f"REFERENCE BOT  {json.dumps(ep['instruction'])}  {cfg['id']}  t={t:5.1f}s",
                                    f"last event: {json.dumps(evs)[:95]}",
                                    f"remembers: {', '.join(conf)[:100]}"]))

    def tick(self, cmd):
        flush()
        o = orig(self, cmd)
        if self.tick_count % 2 == 0:
            top.update_scene(w.duck.data, camera="top")
            pending.append((top.render(), o.frame(), w.duck.time, self.tick_count))
        return o
    interface.Interface.tick = tick
    try:
        res, _, _ = run_episode(cfg, ep, fac, world=w)
    finally:
        interface.Interface.tick = orig
    flush()
    for _ in range(10):
        frames.append(frames[-1])
    imageio.mimsave(MEDIA / f"m3_{task}.mp4", frames, fps=10, macro_block_size=4)
    return {"task": task, "config": cfg["id"], "success": res["success"], "sim_time": res["sim_time"], "frames": len(frames)}


if __name__ == "__main__":
    for t in (sys.argv[1:] or TASKS):
        print(render(t))
