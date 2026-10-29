"""M3: how separable are floor and obstacle pixels under the bot's floor model? (truth only scores)
Per config: the bot's own first look-around fits the floor model; then frames from 3 random free
spots are labelled by MuJoCo segmentation rendering (floor plane vs walls/partition/occluder) and each
pixel's chroma distance to the nearest floor cluster and brightness ratio are recorded. Writes
results/m3/floor_sep.json with per-family distributions and the error rates of candidate thresholds."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configs import _point_free, footprints  # noqa: E402
from interface import Command, Interface  # noqa: E402
from refbot import RefBot  # noqa: E402
from room import World  # noqa: E402

CANDS = [(tol, hi) for tol in (0.02, 0.015, 0.012, 0.01) for hi in (1.6, 1.4, 1.25, 1.15)]


def run(args):
    cfg, seed = args
    rng = np.random.default_rng(seed)
    w = World(cfg)
    it = Interface(w)
    bot = RefBot()
    ctx = bot.ctx(it)
    gen = bot.learn_floor(ctx)
    try:
        cmd = next(gen)
        while True:
            obs = it.tick(cmd)
            bot.sense(obs)
            cmd = gen.send(obs)
    except StopIteration:
        pass
    fm = bot.floor
    m = w.model
    floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obst_g = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("wall_N", "wall_S", "wall_E", "wall_W", "partition", "occluder")}
    seg = mujoco.Renderer(m, 128, 128)
    seg.enable_segmentation_rendering()
    X, Y, ob = footprints(cfg["layout"], pad=0.25)
    rows = {"floor": [], "obst": []}
    for _ in range(3):
        for _ in range(100):
            p = rng.uniform(-1.2, 1.2, 2)
            if _point_free(X, Y, ob, p):
                break
        w.reset(start={"pos": p.round(3).tolist(), "yaw": float(rng.uniform(-np.pi, np.pi))})
        it2 = Interface(w)
        for _ in range(6):
            obs = it2.tick(Command(head=(0.0, 0.15, 0.0, 0.0)))
        fr = obs.frame()
        seg.update_scene(w.duck.data, camera=w.duck.cam_id)
        lab = np.rot90(seg.render()[..., 0], -1)
        ch, br = fm.features(fr)
        d = np.min([np.sqrt(((ch - c) ** 2).sum(-1)) for c in fm.centres], 0)
        r = np.min([np.abs(np.log(br / b)) for b in fm.bright], 0)      # |log brightness ratio|
        up = np.array([[(br[i, j] / fm.bright).max() for j in range(128)] for i in range(128)])
        for key, mask in (("floor", lab == floor_g), ("obst", np.isin(lab, list(obst_g)))):
            if mask.sum():
                idx = np.nonzero(mask.ravel())[0]
                sel = rng.choice(idx, min(400, len(idx)), replace=False)
                rows[key].append(np.stack([d.ravel()[sel], up.ravel()[sel], (br.ravel()[sel] / fm.bright.max())], 1))
    out = {k: (np.concatenate(v).tolist() if v else []) for k, v in rows.items()}
    out["config"] = cfg["id"]
    return out


if __name__ == "__main__":
    S = HERE / "splits"
    cfgs = (json.loads((S / "train.json").read_text())["configs"][:24]
            + json.loads((S / "appearances.json").read_text())["dev"][:16]
            + json.loads((S / "arrangements.json").read_text())["dev"][:8])
    with Pool(4) as pool:
        res = pool.map(run, [(c, 50 + i) for i, c in enumerate(cfgs)])
    summ = {}
    for fam in ("train", "appearances", "arrangements"):
        F = np.array([x for r in res if r["config"].startswith(fam) for x in r["floor"]])
        O = np.array([x for r in res if r["config"].startswith(fam) for x in r["obst"]])
        cand = {}
        for tol, hi in CANDS:
            f_ok = (F[:, 0] < tol) & (F[:, 2] < hi) & (F[:, 2] > 0.35 / 1.0)
            o_ok = (O[:, 0] < tol) & (O[:, 2] < hi) & (O[:, 2] > 0.35)
            cand[f"tol{tol}_hi{hi}"] = {"floor_called_floor": round(float(f_ok.mean()), 3),
                                        "obstacle_called_floor": round(float(o_ok.mean()), 3)}
        summ[fam] = {"floor_px": len(F), "obst_px": len(O),
                     "floor_chroma_p50_p95": np.percentile(F[:, 0], [50, 95]).round(4).tolist(),
                     "obst_chroma_p5_p50": np.percentile(O[:, 0], [5, 50]).round(4).tolist(),
                     "floor_bright_ratio_p50_p99": np.percentile(F[:, 2], [50, 99]).round(3).tolist(),
                     "obst_bright_ratio_p5_p50": np.percentile(O[:, 2], [5, 50]).round(3).tolist(),
                     "candidates": cand}
    (HERE / "results" / "m3" / "floor_sep.json").write_text(json.dumps(summ, indent=1))
    for fam, v in summ.items():
        print(fam, {k: v[k] for k in v if k != "candidates"})
        for k, c in v["candidates"].items():
            print("   ", k, c)
