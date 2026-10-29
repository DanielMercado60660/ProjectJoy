"""Debug picture of one bot episode: the bot's occupancy map (grey unknown, light free, black
obstacle), the true obstacles (red; dark red where the bot also has them), the duck's true path
(blue), its odometry path (yellow), the targets (green pad / cyan button / magenta ball) and its
remembered tracks (small dots). Truth is drawn for debugging only."""
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from botmap import CELL, HALF  # noqa: E402
from configs import footprints  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, make_episode, run_episode, start_episode  # noqa: E402

SPLIT = {"train": ("train.json", "configs"), "instances-dev": ("instances.json", "dev"),
         "arrangements-dev": ("arrangements.json", "dev"), "appearances-dev": ("appearances.json", "dev"),
         "distractor-dev": ("distractor.json", "dev")}


def debug(cid, task, out):
    fam = cid.rsplit("-", 1)[0]
    f, part = SPLIT[fam]
    cfg = next(c for c in json.loads((HERE / "splits" / f).read_text())[part] if c["id"] == cid)
    ep = make_episode(cfg, task, 0)
    w = World(cfg)
    fac, bot = controller(task, BUDGET_TICKS[task])
    st = ep["start"] or cfg["layout"]["start"]
    x0, y0 = st["pos"]
    yaw0 = st["yaw"]
    true_path, odo_path = [], []
    import interface
    orig = interface.Interface.tick

    def tick(self, c):
        o = orig(self, c)
        p = w.duck.pos()[:2]
        cc, ss = np.cos(-yaw0), np.sin(-yaw0)
        true_path.append([cc * (p[0] - x0) - ss * (p[1] - y0), ss * (p[0] - x0) + cc * (p[1] - y0)])
        odo_path.append(list(bot.pose()[:2]))
        return o
    interface.Interface.tick = tick
    try:
        res, w, it = run_episode(cfg, ep, fac, world=w)
    finally:
        interface.Interface.tick = orig

    def to_odo(p):
        cc, ss = np.cos(-yaw0), np.sin(-yaw0)
        return np.array([cc * (p[0] - x0) - ss * (p[1] - y0), ss * (p[0] - x0) + cc * (p[1] - y0)])
    L = bot.grid.L
    img = np.zeros(L.shape + (3,), np.uint8) + 128
    img[L > 0.5] = [0, 0, 0]
    img[L < -0.5] = [220, 220, 220]
    _, _, ob = footprints(cfg["layout"], pad=0.0)
    n = L.shape[0]
    cx = (np.arange(n) + 0.5) * CELL - HALF
    GX, GY = np.meshgrid(cx, cx)
    cc, ss = np.cos(yaw0), np.sin(yaw0)
    WX, WY = x0 + cc * GX - ss * GY, y0 + ss * GX + cc * GY
    col = np.clip(((WX + 1.5) / 0.02).astype(int), 0, ob.shape[1] - 1)
    row = np.clip(((WY + 1.5) / 0.02).astype(int), 0, ob.shape[0] - 1)
    true = np.where((abs(WX) < 1.5) & (abs(WY) < 1.5), ob[row, col], True)
    img[true & ~(L > 0.5)] = [255, 120, 120]
    img[true & (L > 0.5)] = [160, 0, 0]
    big = np.kron(img, np.ones((6, 6, 1), dtype=np.uint8))

    def px(p):
        return int((p[1] + HALF) / CELL * 6), int((p[0] + HALF) / CELL * 6)

    def dot(p, colr, r=2):
        i, j = px(p)
        big[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1] = colr
    for p in odo_path[::2]:
        dot(p, [230, 200, 0], 0)
    for p in true_path[::2]:
        dot(p, [0, 0, 255], 0)
    lay = cfg["layout"]
    if task == "go_and_return":
        tgt = next(p for p in lay["pads"] if p["color"] == ep["instruction"]["target"])["pos"]
        dot(to_odo(tgt), [0, 200, 0], 5)
    if task == "set_lamp":
        dot(to_odo(lay["button"]["pos"]), [0, 255, 255], 5)
    if task == "find_ball":
        dot(to_odo(w.ball_pos()[:2]), [255, 0, 255], 4)
    for t in bot.mem.tracks:
        if t["n"] >= 3:
            dot(t["xy"], [255, 128, 0], 1)
    iio.imwrite(out, big[::-1])
    return res, bot.log


if __name__ == "__main__":
    res, log = debug(sys.argv[1], sys.argv[2], sys.argv[3])
    print({k: res[k] for k in ("success", "ticks", "declared_done") if k in res})
    for e in log:
        print(e)
