"""R8 Gate C (representation): record reference-bot episodes with 224 px head-camera frames and
frame-aligned simulator-truth labels.

    ./r8_job.sh rep-rec ../.venv/bin/python r8_rep/rep_record.py --cfg-lo 270 --cfg-hi 300 --plan heldout --max-min 44

Mirrors record.record_episode (bot only, optional noisy-bot bursts with the same RNG stream and rules) and
keeps record.py's conventions:
  * the stored-style 96 px frame of tick t is `record.down(obs.frame())`, drawn by the interface's own
    128 px renderer at the top of tick t, exactly when record.py draws it (cold_renderer=False: the recorder
    draws tick 0, the bot first senses at tick 1), so a clean episode reproduces the stored M4 episode;
  * the 224 px frame comes from a SEPARATE 224x224 renderer of the same head camera, warmed with one
    throw-away draw before tick 0 (the cold-renderer rule: a new renderer's first frame differs by 1-2 grey
    levels), so every stored 224 px frame is a warm frame; it never reaches the bot;
  * a third renderer (segmentation, 224 px, also warmed) counts the lamp / plate pixels in view.

Per episode: episodes/r8_rep/<name>/frames224.npy (T, 224, 224, 3) uint8, uncompressed so training can
memory-map it; episodes/r8_rep/<name>/meta.npz with
  frames96 (T, 96, 96, 3)  stored-style frames (P2-A' / CNN rows of the bake-off)
  cmd_exec, cmd_bot (T, 8) executed / bot command [vx, vy, wz, 4 head, done]
  LABELS ONLY (simulator truth; never an input to any model):
  lamp_on (T,) lamp state; press_count (T,);
  lamp_px, plate_px, pframe_px (T,) visible pixels of the lamp, the button plate and its dark frame at 224 px
  lamp_rc, plate_rc (T, 2) mean (row, col) of those pixels in the upright 224 px frame (-1 if none)
  cam_pos (T, 3), cam_fwd (T, 3) head-camera position and optical axis (world)
  duck_pose (T, 3) trunk x, y, yaw
  plate_bearing, plate_dist, lamp_bearing, lamp_dist (T,): horizontal-plane bearing of the object centre
      from the camera's optical axis (rad, + = left / counter-clockwise) and horizontal distance (m)
  header (JSON): config, task, k, noise, judge result, ticks.
Index: episodes/r8_rep/index.jsonl (one row per episode).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from interface import BOUNDS, Command, Interface  # noqa: E402
from record import cmd_vec, down  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, Judge, make_episode, start_episode  # noqa: E402

OUT = ROOT / "episodes" / "r8_rep"
IMG = 224
NOISE = {"p": 0.0385, "len": [3, 10], "head_frac": 0.6}       # = lewm/record_noisy.py (R6)
# recording plans: (task, k, noisy) per config
PLANS = {
    "train": [("set_lamp", 8, False), ("set_lamp", 9, True), ("go_and_return", 8, True)],
    "heldout": [("set_lamp", 8, False), ("set_lamp", 9, True), ("go_and_return", 8, True),
                ("set_lamp", 10, False), ("set_lamp", 11, True)],
    "train_extra": [("set_lamp", 10, False), ("set_lamp", 11, True)],
    "train_lamp": [("set_lamp", 8, False), ("set_lamp", 9, True)],     # what R8 trained on (go_and_return dropped for time)
}
DEV_SETS = (("instances-dev", 10, 30), ("arrangements-dev", 0, 20), ("appearances-dev", 0, 20), ("distractor-dev", 0, 20))


def _seg_stats(seg, gid):
    m = (seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    n = int(m.sum())
    if n == 0:
        return 0, (-1.0, -1.0)
    r, c = np.nonzero(m)
    return n, (float(r.mean()), float(c.mean()))


def _bearing(cam_pos, fwd, p):
    f = np.array([fwd[0], fwd[1]])
    d = np.array([p[0] - cam_pos[0], p[1] - cam_pos[1]])
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        return 0.0, float(np.linalg.norm(d))
    f = f / nf
    return float(np.arctan2(f[0] * d[1] - f[1] * d[0], f @ d)), float(np.linalg.norm(d))


def record_rep_episode(cfg, task, k, noisy=False, out_dir=OUT, budget_ticks=None, keep=True, split="train"):
    """One bot episode; returns the index row (and writes the files when keep=True)."""
    ep = make_episode(cfg, task, k)
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    judge = Judge(w, ep)
    budget = BUDGET_TICKS[task] if budget_ticks is None else int(budget_ticks)
    fac, bot = controller(task, budget, None)
    gen = fac(iface, ep["instruction"])
    noise = NOISE if noisy else None
    noise_rng = np.random.default_rng([int(cfg["hash"], 16) % 2 ** 32, k, 13])   # = record.py's stream
    burst_left, burst_cmd, n_noisy = 0, None, 0
    m = w.model
    lamp_g = w.lamp_geom
    plate_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "button_plate")
    pframe_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "button_frame")
    cam = w.duck.cam_id
    r224 = mujoco.Renderer(m, IMG, IMG)
    rseg = mujoco.Renderer(m, IMG, IMG)
    rseg.enable_segmentation_rendering()
    w.duck.head_image(r224)                       # warm-up draws (cold-renderer rule), discarded
    w.duck.head_image(rseg)
    F224, F96, CE, CB = [], [], [], []
    L = {k_: [] for k_ in ("lamp_on", "press_count", "lamp_px", "plate_px", "pframe_px", "lamp_rc", "plate_rc",
                           "cam_pos", "cam_fwd", "duck_pose", "plate_bearing", "plate_dist", "lamp_bearing",
                           "lamp_dist")}
    obs = iface.observe()
    t0 = time.time()
    cmd = next(gen)
    while True:
        exec_cmd = cmd
        if noise is not None and not exec_cmd.done:          # record.py's noise rule, same RNG order
            if burst_left == 0 and noise_rng.random() < noise["p"]:
                burst_left = int(noise_rng.integers(noise["len"][0], noise["len"][1] + 1))
                hc = np.asarray(BOUNDS["head"]) * noise.get("head_frac", 0.6)
                burst_cmd = Command(vx=float(noise_rng.uniform(*BOUNDS["vx"])),
                                    wz=float(noise_rng.uniform(*BOUNDS["wz"])),
                                    head=tuple(float(x) for x in noise_rng.uniform(-hc, hc)))
            if burst_left > 0:
                exec_cmd = burst_cmd
                burst_left -= 1
                n_noisy += 1
        F96.append(down(obs.frame()))                    # stored-style frame, interface renderer
        F224.append(w.duck.head_image(r224))
        seg = np.rot90(_render_seg(w, rseg, cam), -1)    # upright, like head_image
        lpx, lrc = _seg_stats(seg, lamp_g)
        ppx, prc = _seg_stats(seg, plate_g)
        fpx, _ = _seg_stats(seg, pframe_g)
        d = w.duck.data
        cpos = d.cam_xpos[cam].copy()
        fwd = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
        pb, pd = _bearing(cpos, fwd, d.geom_xpos[plate_g])
        lb, ld = _bearing(cpos, fwd, d.geom_xpos[lamp_g])
        p = w.duck.pos()
        for key, val in (("lamp_on", w.lamp_on), ("press_count", w.press_count), ("lamp_px", lpx),
                         ("plate_px", ppx), ("pframe_px", fpx), ("lamp_rc", lrc), ("plate_rc", prc),
                         ("cam_pos", cpos), ("cam_fwd", fwd), ("duck_pose", [p[0], p[1], w.duck.yaw()]),
                         ("plate_bearing", pb), ("plate_dist", pd), ("lamp_bearing", lb), ("lamp_dist", ld)):
            L[key].append(val)
        CE.append(cmd_vec(exec_cmd))
        CB.append(cmd_vec(getattr(bot, "proposed", cmd)))
        obs = iface.tick(exec_cmd)
        judge.update(iface)
        if iface.done or iface.tick_count >= budget:
            break
        try:
            cmd = gen.send(obs)
        except StopIteration:
            break
    r224.close()
    rseg.close()
    res = judge.result(iface)
    res = {kk: (v.item() if hasattr(v, "item") else v) for kk, v in res.items()}
    T = len(F224)
    name = f"{split}_{cfg['id']}_{task}_k{k}" + ("_noisy" if noisy else "")
    header = {"format": "microduck-r8rep/1", "config_id": cfg["id"], "task": task, "k": k,
              "instruction": ep["instruction"], "controller": "refbot", "noise": None if noise is None else
              {**noise, "noisy_ticks": n_noisy}, "result": res, "ticks": T, "img": IMG,
              "labels_are_truth": "lamp_on, press_count, *_px, *_rc, cam_*, duck_pose, *_bearing, *_dist: "
                                  "simulator truth, labels only"}
    row = {"name": name, "config": cfg["id"], "cfg": int(cfg["id"].rsplit("-", 1)[1]), "split": split,
           "task": task, "k": k,
           "noisy": noisy, "ticks": T, "success": res.get("success"), "presses": int(L["press_count"][-1]),
           "lamp_view_ticks": int(sum(x >= 16 for x in L["lamp_px"])),
           "plate_view_ticks": int(sum(x >= 16 for x in L["plate_px"])), "wall_s": round(time.time() - t0, 1)}
    if keep:
        d = Path(out_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "frames224.npy", np.stack(F224))
        arrs = {k_: np.asarray(v, np.float32) for k_, v in L.items()}
        arrs["lamp_on"] = arrs["lamp_on"].astype(np.int8)
        arrs["press_count"] = arrs["press_count"].astype(np.int16)
        for k_ in ("lamp_px", "plate_px", "pframe_px"):
            arrs[k_] = arrs[k_].astype(np.int32)
        np.savez_compressed(d / "meta.npz", frames96=np.stack(F96), cmd_exec=np.stack(CE), cmd_bot=np.stack(CB),
                            header=np.array(json.dumps(header)), **arrs)
        row["path"] = str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d)
    else:
        row["_frames224"], row["_frames96"], row["_labels"] = np.stack(F224), np.stack(F96), L
        row["_cmd_exec"] = np.stack(CE)
    return row


def _render_seg(w, renderer, cam):
    renderer.update_scene(w.duck.data, camera=cam)
    return renderer.render()


def _job(args):
    cfg, task, k, noisy = args[:4]
    split = args[4] if len(args) > 4 else "train"
    try:
        return record_rep_episode(cfg, task, k, noisy, split=split)
    except Exception as e:  # keep going; the failure goes to the index
        return {"config": cfg["id"], "task": task, "k": k, "noisy": noisy, "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-lo", type=int, default=0)
    ap.add_argument("--cfg-hi", type=int, default=0)
    ap.add_argument("--plan", choices=sorted(PLANS) + ["dev"], required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--max-min", type=float, default=44.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx = OUT / "index.jsonl"
    done = set()
    if idx.exists():
        done = {(r["config"], r["task"], r["k"]) for r in map(json.loads, open(idx)) if "error" not in r}
    # interleave plans across configs so a time-guard stop leaves a balanced set
    if a.plan == "dev":    # the 80 p2_dev rooms (p2_devdemos.py SETS), set_lamp k=0, clean bot: evaluation only
        from m6_eval import load_set
        jobs = [(c, "set_lamp", 0, False, s) for s, lo, hi in DEV_SETS for c in load_set(s, hi)[lo:hi]
                if (c["id"], "set_lamp", 0) not in done]
    else:
        cfgs = json.loads((ROOT / "splits" / "train.json").read_text())["configs"][a.cfg_lo:a.cfg_hi]
        assert all(c["id"].startswith("train-") for c in cfgs)
        jobs = [(c, t, k, nz) for (t, k, nz) in PLANS[a.plan] for c in cfgs if (c["id"], t, k) not in done]
    t0 = time.time()
    stopped = False
    n = 0
    with Pool(a.procs) as pool, open(idx, "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"{n}/{len(jobs)} episodes, {(time.time() - t0) / 60:.1f} min", flush=True)
            if (time.time() - t0) / 60 > a.max_min:
                stopped = True
                pool.terminate()
                break
    rows = [json.loads(l) for l in open(idx)]
    ok = [r for r in rows if "error" not in r]
    print(json.dumps({"plan": a.plan, "cfg": [a.cfg_lo, a.cfg_hi], "episodes_in_index": len(rows),
                      "errors": len(rows) - len(ok), "ticks_in_index": sum(r["ticks"] for r in ok),
                      "this_run": n, "jobs": len(jobs), "wall_s": round(time.time() - t0),
                      "stopped_by_time_guard": stopped}), flush=True)


if __name__ == "__main__":
    main()
