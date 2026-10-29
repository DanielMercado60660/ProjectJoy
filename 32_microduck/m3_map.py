"""M3 stage 4: score the bot's floor model and occupancy map against the true obstacles (truth scores
only). Per config: the duck starts on its mat, fits the floor model from a head-down full turn, then
maps from the start and two more free spots (full scans with the head level and pitched down), using
the TRUE pose (to measure the map alone). Scores, on cells within 2 m of a viewpoint:
  obstacle recall: fraction of true partition/occluder/wall surface cells with a detected obstacle
                   within 10 cm
  false obstacles: detected obstacle cells more than 15 cm from any true obstacle (per config)
Writes results/m3/map_<tag>.json."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bodymodel import BodyModel  # noqa: E402
from botmap import CELL, HALF, FloorModel, OccupancyGrid, walkable_mask  # noqa: E402
from configs import _point_free, footprints  # noqa: E402
from interface import Command, Interface  # noqa: E402
from perception import CameraPose  # noqa: E402
from room import World  # noqa: E402

OUT = HERE / "results" / "m3"


def to_start_frame(xy, x0, y0, yaw0):
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    dx, dy = xy[..., 0] - x0, xy[..., 1] - y0
    return np.stack([c * dx - s * dy, s * dx + c * dy], -1)


def run(args):
    cfg, seed, fparams = args
    rng = np.random.default_rng(seed)
    w = World(cfg)
    bm = BodyModel()
    x0, y0, _ = w.duck.pos()
    yaw0 = w.duck.yaw()
    floor, grid = FloorModel(fparams), OccupancyGrid()

    def pose():
        p = to_start_frame(np.array(w.duck.pos()[:2]), x0, y0, yaw0)
        return float(p[0]), float(p[1]), float((w.duck.yaw() - yaw0 + np.pi) % (2 * np.pi) - np.pi)

    it = Interface(w)

    def full_scan(learn):
        for k in range(30):          # ~3 s per third of a turn; 3 x 120 deg with head yaw sweep
            for hy in (-0.9, 0.0, 0.9):
                for pitch in ((0.6,) if learn else (0.0, 0.35)):
                    for _ in range(3):
                        obs = it.tick(Command(head=(0.0, pitch, hy, 0.0)))
                    cam = CameraPose(bm, obs, pose())
                    fr = obs.frame()
                    if learn:
                        floor.add_samples(fr, cam, walkable_mask(fr))
                    else:
                        grid.update(fr, cam, floor)
            # turn ~120 deg
            for _ in range(8):
                it.tick(Command(wz=0.55))
            if k >= 2:
                break

    full_scan(learn=True)
    floor.fit()
    views = [np.array(w.duck.pos()[:2])]
    full_scan(learn=False)
    X, Y, obst = footprints(cfg["layout"], pad=0.25)
    for _ in range(2):
        for _ in range(100):
            p = rng.uniform(-1.2, 1.2, 2)
            if _point_free(X, Y, obst, p):
                break
        w.reset(start={"pos": p.round(3).tolist(), "yaw": float(rng.uniform(-np.pi, np.pi))})
        it = Interface(w)
        views.append(np.array(w.duck.pos()[:2]))
        full_scan(learn=False)
    # truth in the start frame
    n = grid.n
    cx = (np.arange(n) + 0.5) * CELL - HALF
    GX, GY = np.meshgrid(cx, cx, indexing="xy")
    c, s = np.cos(yaw0), np.sin(yaw0)
    WX, WY = x0 + c * GX - s * GY, y0 + s * GX + c * GY            # cell centres in world
    lay = cfg["layout"]
    _, _, ob0 = footprints(lay, pad=0.0)
    from configs import GRID
    def world_occ(wx, wy):
        col = np.clip(((wx + 1.5) / GRID).astype(int), 0, ob0.shape[1] - 1)
        row = np.clip(((wy + 1.5) / GRID).astype(int), 0, ob0.shape[0] - 1)
        inside = (np.abs(wx) < 1.5) & (np.abs(wy) < 1.5)
        return np.where(inside, ob0[row, col], True)
    true_occ = world_occ(WX, WY)
    surface = true_occ & ~ndimage.binary_erosion(true_occ, iterations=1)     # obstacle cells facing free space
    near_view = np.zeros_like(true_occ)
    for v in views:
        vs = to_start_frame(v, x0, y0, yaw0)
        near_view |= np.hypot(GX - vs[0], GY - vs[1]) < 2.0
    det = grid.occupied()
    det_near = ndimage.binary_dilation(det, iterations=2)                    # within 10 cm
    surf = surface & near_view
    recall = float((surf & det_near).sum() / max(surf.sum(), 1))
    true_near = ndimage.binary_dilation(true_occ, iterations=3)              # within 15 cm
    false_cells = int((det & ~true_near & near_view).sum())
    # planning check: from each viewpoint to 8 random free goals, plan on the bot's map; does the path
    # cross a true obstacle (5 cm inflation)? is a path missing where the true map has one?
    from skills import plan_path
    crossings = missing = plans = 0
    _, _, ob5 = footprints(lay, pad=0.05)
    Xg, Yg, obg = footprints(lay, pad=0.15)
    for v in views:
        for _ in range(4):
            for _ in range(50):
                gw = rng.uniform(-1.3, 1.3, 2)
                if _point_free(Xg, Yg, obg, gw):
                    break
            if plan_path(lay, v, gw) is None:
                continue
            plans += 1
            vs, gs = to_start_frame(v, x0, y0, yaw0), to_start_frame(gw, x0, y0, yaw0)
            path = grid.plan(vs, gs)
            if path is None:
                missing += 1
                continue
            pts = [vs] + [np.array(q) for q in path]
            hit = False
            for a, b in zip(pts[:-1], pts[1:]):
                for tt in np.linspace(0, 1, int(np.linalg.norm(b - a) / 0.02) + 2):
                    q = a + tt * (b - a)
                    wx, wy = x0 + c * q[0] - s * q[1], y0 + s * q[0] + c * q[1]
                    if not _point_free(X, Y, ob5, np.array([wx, wy])):
                        hit = True
                        break
                if hit:
                    break
            crossings += hit
    return {"config": cfg["id"], "recall": recall, "false_cells": false_cells,
            "plans": plans, "plans_crossing_obstacle": crossings, "plans_missing": missing,
            "detected_cells": int((det & near_view).sum()), "floor_clusters": int(len(floor.centres)),
            "partition_recall": None}


def main(tag="v0", fparams=None):
    t0 = time.time()
    S = HERE / "splits"
    sets = {"train": json.loads((S / "train.json").read_text())["configs"][:20],
            "appearances-dev": json.loads((S / "appearances.json").read_text())["dev"][:20],
            "distractor-dev": json.loads((S / "distractor.json").read_text())["dev"][:10],
            "arrangements-dev": json.loads((S / "arrangements.json").read_text())["dev"][:10]}
    jobs = [(c, 900 + i, fparams) for cs in sets.values() for i, c in enumerate(cs)]
    with Pool(4) as pool:
        rows = pool.map(run, jobs)
    summ = {}
    for fam in sets:
        rs = [r for r in rows if r["config"].startswith(fam.replace("-dev", "")) and (fam == "train" or "dev" in r["config"])]
        summ[fam] = {"n": len(rs), "recall_median": float(np.median([r["recall"] for r in rs])),
                     "recall_min": float(min(r["recall"] for r in rs)),
                     "false_cells_median": float(np.median([r["false_cells"] for r in rs])),
                     "false_cells_max": int(max(r["false_cells"] for r in rs)),
                     "detected_cells_median": float(np.median([r["detected_cells"] for r in rs])),
                     "plans": int(sum(r["plans"] for r in rs)),
                     "plans_crossing_obstacle": int(sum(r["plans_crossing_obstacle"] for r in rs)),
                     "plans_missing": int(sum(r["plans_missing"] for r in rs))}
    rep = {"tag": tag, "floor_params": fparams, "summary": summ, "rows": rows, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"map_{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(summ, indent=1), rep["wall_s"])


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v0"
    fp = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    main(tag, fp)
