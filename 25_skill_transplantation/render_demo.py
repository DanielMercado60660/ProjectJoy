"""Reload checkpoints and render the demo grid offscreen (MuJoCo rgb_array) to media/.

Rows: fwd specialist, bwd specialist, naive average, best training-free merge (picked on validation
seeds), DAgger-distilled single network. Columns: the task indicator fed to the network (fwd, bwd).
Each tile shows the running task return. The merged parameters are rebuilt from the specialists
and the lambda recorded in results/merge_s<seed>.json, then saved to checkpoints/demo/.

  ../.venv/bin/python render_demo.py --seed 0 --cond pre
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EP_LEN, TASKS, load_params, make_env, np_mean, policy_input,  # noqa: E402
                    save_params, task_reward)
import merge as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build_policies(seed, cond, ckroot, resdir):
    d = os.path.join(ckroot, f"s{seed}")
    t0, norm = load_params(os.path.join(d, "base.npz" if cond == "pre" else "init.npz"))
    _, norm = load_params(os.path.join(d, "base.npz"))
    A, _ = load_params(os.path.join(d, f"{cond}_fwd.npz"))
    B, _ = load_params(os.path.join(d, f"{cond}_bwd.npz"))
    pols = [("fwd specialist", A), ("bwd specialist", B), ("average a=0.5", M.interpolate(A, B, 0.5))]
    mres = json.load(open(os.path.join(resdir, f"merge_s{seed}.json")))
    L = str(max(int(k) for k in mres[cond]["by_length"]))
    R = mres[cond]["by_length"][L]
    fns = {"task_arith": lambda l: M.task_arithmetic(t0, [A, B], l),
           "ties": lambda l: M.ties(t0, [A, B], l, k=0.2),
           "dare": lambda l: M.dare(t0, [A, B], l, p=0.9, seed=seed)}
    best = max(fns, key=lambda k: max(R[k]["val_scores"]))
    lam = R[best]["lam"]
    pols.append((f"best merge: {best} l={lam}", fns[best](lam)))
    dist = os.path.join(d, f"distill_{cond}_average.npz")
    if os.path.exists(dist):
        pols.append(("DAgger distilled", load_params(dist)[0]))
    os.makedirs(os.path.join(ckroot, "demo"), exist_ok=True)
    save_params(os.path.join(ckroot, "demo", f"s{seed}_{cond}_average.npz"), pols[2][1], norm)
    save_params(os.path.join(ckroot, "demo", f"s{seed}_{cond}_best_merge.npz"), pols[3][1], norm,
                method=best, lam=lam)
    return pols, norm


def run_tile(p, norm, task, seed, every, steps):
    env = make_env(render=True)
    m = env.unwrapped.model
    # the floor plane is drawn only +-40 m by default; enlarge it (and its texture repeat) so the checker
    # stays visible on 300 m runs. Visual only: MuJoCo planes collide as infinite planes anyway.
    scale = 3000.0 / m.geom_size[0][0]
    m.geom_size[0][:2] = 3000.0
    if m.geom_matid[0] >= 0:
        m.mat_texrepeat[m.geom_matid[0]] *= scale
    obs, _ = env.reset(seed=seed)
    tid = np.array([TASKS.index(task)])
    frames, ret = [], 0.0
    for t in range(steps):
        a = np_mean(p, policy_input(norm, obs[None], tid))[0]
        obs, _, _, _, info = env.step(a)
        ret += task_reward(task, info["x_velocity"], a)
        if t % every == 0:
            frames.append((env.render(), ret, info["x_position"]))
    env.close()
    return frames, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="pre")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--steps", type=int, default=EP_LEN)
    ap.add_argument("--out", default=os.path.join(HERE, "media", "demo_grid"))
    ap.add_argument("--eval_seed", type=int, default=10000)
    ap.add_argument("--ckroot", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    pols, norm = build_policies(args.seed, args.cond, args.ckroot, args.results)
    W, H = 320, 240
    tiles, finals = {}, {}
    for name, p in pols:
        for task in TASKS:
            tiles[(name, task)], finals[(name, task)] = run_tile(p, norm, task, args.eval_seed, args.every, args.steps)
            print(f"{name:32s} on {task}: return {finals[(name, task)]:.0f}", flush=True)
    n = len(tiles[(pols[0][0], TASKS[0])])
    head = 28
    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (W * 2 + 150, head + H * len(pols)), (252, 252, 251))
        dr = ImageDraw.Draw(canvas)
        for j, task in enumerate(TASKS):
            dr.text((150 + j * W + 8, 8), f"task indicator = {task}  ({'run +x' if task == 'fwd' else 'run -x'})",
                    fill=(11, 11, 11))
        for r, (name, _) in enumerate(pols):
            y = head + r * H
            dr.text((6, y + H // 2 - 6), name.replace(": ", ":\n"), fill=(11, 11, 11))
            for j, task in enumerate(TASKS):
                img, ret, x = tiles[(name, task)][i]
                canvas.paste(Image.fromarray(img), (150 + j * W, y))
                dr.rectangle([150 + j * W, y, 150 + j * W + 150, y + 30], fill=(252, 252, 251))
                dr.text((150 + j * W + 4, y + 2), f"return {ret:7.0f}\nx = {x:6.1f} m", fill=(11, 11, 11))
        frames.append(np.array(canvas))
    import imageio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimwrite(args.out + ".mp4", frames, fps=25, quality=7, macro_block_size=2)
    Image.fromarray(frames[-1]).save(args.out + "_final.png")
    small = [np.array(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames[::3]]
    imageio.mimwrite(args.out + ".gif", small, duration=0.12, loop=0)
    json.dump({f"{k[0]} | {k[1]}": v for k, v in finals.items()},
              open(args.out + "_returns.json", "w"), indent=1)
    print("wrote", args.out + ".mp4")


if __name__ == "__main__":
    main()
