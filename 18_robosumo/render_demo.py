"""Reload a checkpoint and render bouts offscreen with MuJoCo to GIF/MP4 in media/.

  ../.venv/bin/python render_demo.py --ckpt checkpoints/main.pkl --opponents charger flanker circler counter
"""
import argparse, os, pickle
import numpy as np
import imageio.v2 as imageio
import mujoco
from sumo_env import SumoEnv, make_script, play_bout, FRAME_SKIP, DT
from ppo import NumpyPolicy, load_tuned
from tournament import make
import tournament


def render_bout(env, pa, pb, seed, R=None, fric=None, size=(320, 240), every=2):
    env.rng = np.random.default_rng(seed)
    r = play_bout(env, pa, pb, R=R, fric=fric, record=True)
    renderer = mujoco.Renderer(env.m, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance = 2.9 * env.R
    cam.elevation = -60
    cam.azimuth = 90
    frames = []
    for q in r["frames"][::every]:
        env.d.qpos[:] = q
        mujoco.mj_forward(env.m, env.d)
        renderer.update_scene(env.d, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames, r


def label(frames, text):
    from PIL import Image, ImageDraw
    out = []
    for f in frames:
        im = Image.fromarray(f)
        ImageDraw.Draw(im).text((6, 4), text, fill=(0, 0, 0))
        out.append(np.array(im))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/main.pkl")
    ap.add_argument("--opponents", nargs="+", default=["charger", "flanker", "circler", "counter"])
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--out", default="media/demo")
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--fric", type=float, default=None)
    args = ap.parse_args()
    tournament.init(args.ckpt)
    env = SumoEnv(seed=args.seed)
    all_frames, summary = [], []
    fps = int(round(1 / (DT * FRAME_SKIP) / 2))
    for k, opp in enumerate(args.opponents):
        frames, r = render_bout(env, make("ppo", 0), make(opp, k), args.seed + k, args.R, args.fric)
        res = {1: "PPO (red) wins", -1: f"{opp} wins", 0: "draw"}[r["outcome"]]
        how = " (tip)" if (r["tip_a"] or r["tip_b"]) else (" (ring-out)" if r["outcome"] != 0 else "")
        txt = f"PPO (red) vs {opp} (blue): {res}{how}, {r['steps'] * DT * FRAME_SKIP:.1f}s"
        summary.append(txt)
        print(txt)
        all_frames += label(frames, txt) + label([frames[-1]] * fps, txt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out + ".mp4", all_frames, fps=fps)
    imageio.mimsave(args.out + ".gif", all_frames[::2], duration=2 / fps, loop=0)
    open(args.out + ".txt", "w").write("\n".join(summary) + "\n")
    print("wrote", args.out + ".mp4/.gif")


if __name__ == "__main__":
    main()
