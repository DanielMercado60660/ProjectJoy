"""Render a GIF: the maze, the bat, its pulses (rays + rings), and the echo profile it hears.

  ../.venv/bin/python demo.py --ckpt checkpoints/bat --agent learned --out media/bat_learned.gif
  ../.venv/bin/python demo.py --agent wf2 --out media/wallfollow_k2.gif
"""
import argparse

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bat_env import N_BINS, N_SECT, RAY_DIR, BatEnv, MazePool, WallFollower, cell_xy, K_RAYS
from evaluate import HELDOUT, LearnedAgent

SECT_NAMES = ["front", "fr-right", "right", "bk-right", "back", "bk-left", "left", "fr-left"]


def trace(agent, env, max_steps=10**9):
    obs = env.reset()
    frames = []
    def snap(o, pulse_prob=None, done=False, success=False):
        frames.append(dict(r=int(env.r[0]), c=int(env.c[0]), hd=int(env.hd[0]), obs=o[0].copy(),
                           heard=bool(o[0, 2 * N_SECT * N_BINS] > 0.5), bump=bool(o[0, 2 * N_SECT * N_BINS + 1] > 0.5),
                           pulses=int(env.pulses[0]), colls=int(env.colls[0]), t=int(env.t[0]),
                           pp=pulse_prob, done=done, success=success))
    snap(obs)
    gr, gc = int(env.gr[0]), int(env.gc[0])
    while True:
        mv, pu = agent.act(obs)
        pp = float(getattr(agent, "last_pulse_prob", [np.nan])[0])
        pre = (int(env.pulses[0] + pu[0]), int(env.colls[0]))
        obs, r, d, info = env.step(mv, pu)
        if d[0]:
            frames.append(dict(frames[-1], r=gr, c=gc, heard=False, done=True, success=bool(info["success"][0]),
                               pulses=int(info["pulses"][0]), colls=int(info["collisions"][0]), t=int(info["t"][0]), pp=pp))
            return frames, (gr, gc)
        snap(obs, pp)
        if len(frames) >= max_steps:
            return frames, (gr, gc)


def render(frames, pool, m, goal, title, out, fps=6):
    h, w = pool.hw[m]
    grid = pool.grids[m]
    imgs = []
    trail = []
    for k, f in enumerate(frames):
        fig = plt.figure(figsize=(9.6, 4.8), dpi=80)
        ax = fig.add_axes([0.02, 0.06, 0.46, 0.86])
        ax.imshow(grid, cmap="Greys", vmin=-0.3, vmax=1.3, extent=(0, grid.shape[1], grid.shape[0], 0))
        trail.append((2 * f["c"] + 1.5, 2 * f["r"] + 1.5))
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#7aa6c2", lw=2, alpha=0.6)
        gy, gx = 2 * goal[0] + 1.5, 2 * goal[1] + 1.5
        ax.plot(gx, gy, "o", ms=11, color="#f2c94c", mec="#8a6d00")
        ax.text(gx, gy + 0.05, "moth", ha="center", va="center", fontsize=5)
        p = cell_xy(f["r"], f["c"])
        if f["heard"]:
            roll = (np.arange(K_RAYS) + 8 * f["hd"]) % K_RAYS
            d = pool.dist[m, f["r"], f["c"], roll]
            for j in range(K_RAYS):
                q = p + d[j] * RAY_DIR[roll[j]]
                ax.plot([p[1], q[1]], [p[0], q[0]], "-", color="#eb5757", lw=0.6, alpha=0.5)
            for rad in (0.8, 1.6, 2.4):
                ax.add_patch(plt.Circle((p[1], p[0]), rad, fill=False, color="#eb5757", lw=1, alpha=0.6))
        hv = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])[f["hd"]]   # (dx, dy) on screen
        tip = np.array([p[1], p[0]]) + 0.55 * np.array([hv[0], -hv[1]])
        perp = np.array([hv[1], hv[0]]) * 0.35
        base = np.array([p[1], p[0]]) - 0.35 * np.array([hv[0], -hv[1]])
        ax.add_patch(plt.Polygon([tip, base + perp, base - perp], color="#2d9cdb" if not f["bump"] else "#eb5757"))
        ax.set_xlim(0, grid.shape[1]); ax.set_ylim(grid.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        status = "REACHED THE MOTH" if f.get("success") else ("TIMED OUT" if f["done"] else "")
        ax.set_title(f"{title}  step {f['t']}  pulses {f['pulses']}  collisions {f['colls']}  {status}", fontsize=8)
        o = f["obs"]
        wall = o[:N_SECT * N_BINS].reshape(N_SECT, N_BINS) if f["heard"] else np.zeros((N_SECT, N_BINS))
        moth = o[N_SECT * N_BINS:2 * N_SECT * N_BINS].reshape(N_SECT, N_BINS) if f["heard"] else np.zeros((N_SECT, N_BINS))
        for i, (arr, name, cmap) in enumerate([(wall, "rock echo", "magma"), (moth, "moth echo", "viridis")]):
            a2 = fig.add_axes([0.58, 0.55 - 0.45 * i, 0.38, 0.36])
            a2.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1.5 if i == 0 else 0.8)
            a2.set_yticks(range(N_SECT)); a2.set_yticklabels(SECT_NAMES, fontsize=6)
            a2.set_xticks([0, 3, 6, 9, 11]); a2.set_xticklabels(["0", "3", "6", "9", "11"], fontsize=6)
            a2.set_title(f"what it hears: {name}" + ("" if f["heard"] else " (silent: no pulse)"), fontsize=8)
            if i == 1:
                a2.set_xlabel("time of flight bin (distance)", fontsize=7)
        if f.get("pp") is not None and not np.isnan(f["pp"]):
            fig.text(0.58, 0.95, f"P(pulse next) = {f['pp']:.2f}", fontsize=8)
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imgs += [imgs[-1]] * fps * 2
    imageio.mimsave(out, imgs, duration=1.0 / fps, loop=0)
    imageio.imwrite(out.replace(".gif", "_last.png"), imgs[-1])
    print(f"wrote {out} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bat")
    ap.add_argument("--agent", default="learned")   # learned | wf1 | wf2 | wf4
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--maze", type=int, default=-1, help="-1: first held-out maze the agent solves")
    ap.add_argument("--out", default="media/bat_learned.gif")
    ap.add_argument("--max_tries", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=10**9, help="truncate the clip (smoke)")
    a = ap.parse_args()
    pool = MazePool([a.size], max(a.max_tries, a.maze + 1), seed=HELDOUT[a.size])
    if a.agent == "learned":
        from model import load
        model, _ = load(a.ckpt)
        make = lambda: LearnedAgent(model, 1, seed=1)
    else:
        make = lambda: WallFollower(1, int(a.agent[2:]))
    tries = [a.maze] if a.maze >= 0 else range(a.max_tries)
    for m in tries:
        env = BatEnv(pool, 1, seed=0, fixed_order=True)
        env.next_maze = m
        frames, goal = trace(make(), env, a.max_steps)
        if frames[-1].get("success") or a.maze >= 0:
            break
    print(f"maze {m}: success={frames[-1].get('success')} steps={frames[-1]['t']}")
    render(frames, pool, m, goal, a.agent, a.out)


if __name__ == "__main__":
    main()
