"""Reload checkpoints and render the dance offscreen: each panel shows a controller (solid) over a
translucent ghost of the reference pose, with a beat light and a push marker. Panels side by side.

  ../.venv/bin/python render_demo.py --ckpt checkpoints/ppo_pd.pkl --cond nominal --loops 2 --out media/demo_nominal
"""
import argparse, json, pickle
import numpy as np
import imageio.v2 as imageio
import mujoco
from PIL import Image, ImageDraw
from clip import Clip
from dance_env import DanceEnv, BaselineController, run_clip, DT
from eval import CONDITIONS
from ppo import PolicyController

W, H = 360, 400


def render_log(env, log, title, every=1):
    m = env.m
    d, g = mujoco.MjData(m), mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0, 0.75]
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -8
    opt = mujoco.MjvOption()
    frames = []
    beat = env.clip.beat / env.tempo
    for k in range(0, len(log["t"]), every):
        d.qpos[:] = log["qpos"][k]
        g.qpos[:] = log["ghost"][k]
        mujoco.mj_forward(m, d)
        mujoco.mj_forward(m, g)
        r.update_scene(d, camera=cam)
        n0 = r.scene.ngeom
        mujoco.mjv_addGeoms(m, g, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        for i in range(n0, r.scene.ngeom):
            r.scene.geoms[i].rgba[:] = [0.2, 0.6, 1.0, 0.28]
        img = Image.fromarray(r.render().copy())
        dr = ImageDraw.Draw(img)
        t = log["t"][k]
        bi = int(np.round(t / beat))
        on = abs(t - bi * beat) < 0.08
        col = (255, 60, 60) if on else (90, 90, 90)
        dr.ellipse([W - 40, 10, W - 14, 36], fill=col)
        dr.text((W - 44, 40), f"beat {bi % 4 + 1}", fill=(255, 255, 255))
        dr.multiline_text((8, 8), title, fill=(255, 255, 255), spacing=3)
        dr.text((8, 42), f"t={t:5.2f}s  falls so far: {sum(1 for ft in log['fall_t'] if ft <= t)}", fill=(255, 255, 255))
        dr.text((8, H - 18), "solid = controller, blue ghost = reference", fill=(230, 230, 230))
        f = log["pushing"][k] if len(log["pushing"]) else 0.0
        if abs(f) > 0:
            x0, y0 = W // 2, 80
            dr.line([x0 - 40 * np.sign(f), y0, x0 + 40 * np.sign(f), y0], fill=(255, 220, 0), width=5)
            dr.text((x0 - 30, y0 + 8), f"PUSH {abs(f):.0f} N", fill=(255, 220, 0))
        frames.append(np.array(img))
    r.close()
    return frames


def controllers(ckpts, base_params):
    if base_params.get("k_sw") is not None:   # review addition: anchored baseline (baseline_anchor.py)
        from baseline_anchor import AnchorBaselineController
        out = [("anchored baseline (review)", AnchorBaselineController(base_params))]
    else:
        out = [("PD + feed-forward + balance (tuned baseline)", BaselineController(base_params))]
    for path in ckpts:
        ck = pickle.load(open(path, "rb"))
        out.append((f"PPO residual-{ck['config']['residual']} ({path.split('/')[-1]})",
                    PolicyController(ck, BaselineController(ck["base_params"]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=["checkpoints/ppo_pd.pkl"])
    ap.add_argument("--cond", default="nominal")
    ap.add_argument("--loops", type=float, default=2)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--baseline", default="results/baseline_tuned.json")
    ap.add_argument("--out", default="media/demo_nominal")
    ap.add_argument("--gif_every", type=int, default=2)
    args = ap.parse_args()
    base_params = json.load(open(args.baseline))["params"]
    clip = Clip()
    panels = []
    summary = []
    for title, ctrl in controllers(args.ckpt, base_params):
        env = DanceEnv(clip=clip, seed=args.seed, **CONDITIONS[args.cond])
        log = run_clip(env, ctrl, loops=args.loops, phase=0.0, record=True)
        panels.append(render_log(env, log, f"{title}\n[{args.cond}]"))
        summary.append(dict(controller=title, falls=log["falls"], fall_t=[round(x, 2) for x in log["fall_t"]]))
    n = min(len(p) for p in panels)
    frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)]
    imageio.mimsave(args.out + ".mp4", frames, fps=int(round(1 / DT)), quality=7, macro_block_size=8)
    imageio.mimsave(args.out + ".gif", frames[::args.gif_every], duration=DT * args.gif_every * 1000, loop=0)
    # contact-sheet PNG: one frame per beat of the first loop
    beat_frames = [frames[min(n - 1, int(round(b * clip.beat / DT)))] for b in range(4)]
    imageio.imwrite(args.out + "_beats.png", np.concatenate(beat_frames, axis=0))
    json.dump(summary, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
