"""Reload a checkpoint and render demo clips (offscreen MuJoCo, no window).

Panels: hold still (zero action) | tuned scripted gait | PPO policy (deterministic mean). Same seed,
so same initial egg pose, object properties and target in every panel. The translucent egg to the side shows the target orientation.
Touch sensor zones light up red when they register contact. Text: target rotation, current axis
error, total contact normal force on the egg (budget 10 N).

  ../.venv/bin/python render_demo.py --ckpt checkpoints/run2/latest.npz
"""
import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import json

from baselines import Gait, Zero
from egg_env import FORCE_BUDGET, THR, EggVecEnv
from train import load_policy

W, H = 400, 400


def record(policy, n, seed, tmin, tmax, fric=None, mass=None):
    env = EggVecEnv(n, seed=seed, auto_reset=False, substep_force=True, theta_min=tmin, theta_max=tmax)
    env.fixed_fric, env.fixed_mass = fric, mass
    obs, priv = env.reset()
    if hasattr(policy, "reset"):
        policy.reset(env)
    Q, TS, E, F, D = [], [], [], [], []

    def snap():
        Q.append(np.stack([d.qpos.copy() for d in env.datas]))
        TS.append(np.stack([d.sensordata[env.touch].copy() for d in env.datas]))
        E.append(env.err.copy())
        F.append(env.force_peak.copy())
        D.append(env.dropped.copy())
    snap()
    for t in range(env.T):
        obs, priv, *_ = env.step(policy(obs, priv, env))
        snap()
    return env, dict(q=np.array(Q), ts=np.array(TS), err=np.array(E), force=np.array(F),
                     drop=np.array(D))


class Painter:
    def __init__(self, env):
        self.env = env
        self.m = mujoco.MjModel.from_xml_path(__import__("egg_env").XML)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        # near top-down view: the task is a rotation about world z
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.40, 90.0, -80.0
        self.cam.lookat[:] = [1.06, 0.92, 0.16]
        self.touch_sites = np.array([self.m.sensor_objid[s] for s in env.touch])
        self.m.site_rgba[:, 3] = 0.0

    def frame(self, q, touch, goal_q, label, err, force, delta, dropped):
        e = self.env
        self.d.qpos[:] = q
        self.d.qpos[e.tq:e.tq + 3] = q[e.oq:e.oq + 3] + np.array([0.12, 0.0, 0.0])
        self.d.qpos[e.tq + 3:e.tq + 7] = goal_q
        mujoco.mj_forward(self.m, self.d)
        rgba = self.m.site_rgba
        rgba[self.touch_sites] = [1, 0, 0, 0.0]
        rgba[self.touch_sites[touch > 0], 3] = 0.7
        self.r.update_scene(self.d, self.cam)
        img = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(img)
        ok = err < THR and not dropped
        dr.rectangle([0, 0, W, 58], fill=(0, 0, 0))
        dr.text((6, 4), label, fill=(255, 255, 255))
        dr.text((6, 20), f"target rot {np.degrees(delta):+.0f} deg   axis err {np.degrees(err):5.1f} deg"
                + ("  OK" if ok else ""), fill=(120, 255, 120) if ok else (255, 255, 255))
        dr.text((6, 36), f"contact force {force:5.1f} N (budget {FORCE_BUDGET:.0f})"
                + ("  DROPPED" if dropped else ""),
                fill=(255, 90, 90) if force > FORCE_BUDGET or dropped else (255, 255, 255))
        dr.text((W - 100, H - 16), "ghost = target", fill=(255, 255, 255))
        return np.array(img)


def clip(path, env, recs, labels, i, fps=25):
    p = Painter(env)
    frames = []
    for t in range(recs[0]["q"].shape[0]):
        row = [p.frame(r["q"][t, i], r["ts"][t, i], env.goal_q[i], lab, r["err"][t, i], r["force"][t, i],
                       env.delta[i], r["drop"][t, i]) for r, lab in zip(recs, labels)]
        frames.append(np.concatenate(row, axis=1))
    frames += [frames[-1]] * 20  # hold the last frame
    imageio.mimwrite(path, frames, fps=fps, quality=7, macro_block_size=8)
    Image.fromarray(frames[-1]).save(path.replace(".mp4", "_last.png"))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--theta", type=float, nargs=2, default=[0.15, 0.4])
    ap.add_argument("--tag", default="")
    ap.add_argument("--media", default="media")
    ap.add_argument("--fric", type=float, default=None, help="fixed friction scale (robustness clips)")
    ap.add_argument("--mass", type=float, default=None, help="fixed egg mass scale (robustness clips)")
    ap.add_argument("--baselines", default="checkpoints/baselines.json")
    a = ap.parse_args()
    os.makedirs(a.media, exist_ok=True)
    pol = load_policy(a.ckpt)
    B = json.load(open(a.baselines))["gait"]
    gait = Gait(np.array(B["c"]), np.array(B["A"]), np.array(B["phi"]), B["f"])
    env0, r0 = record(Zero(), a.n, a.seed, *a.theta, a.fric, a.mass)
    envg, rg = record(gait, a.n, a.seed, *a.theta, a.fric, a.mass)
    env1, r1 = record(pol, a.n, a.seed, *a.theta, a.fric, a.mass)
    assert np.allclose(env0.goal_q, env1.goal_q) and np.allclose(envg.goal_q, env1.goal_q)
    fin = r1["err"][-1]
    succ = (fin < THR) & ~r1["drop"][-1]
    print("PPO demo episodes: success", succ.mean(), "| hold-still success", ((r0["err"][-1] < THR) & ~r0["drop"][-1]).mean())
    order = np.argsort(-np.abs(env1.delta))
    good = [i for i in order if succ[i]][:2]
    bad = [i for i in np.argsort(-(fin + 10 * r1["drop"][-1] + 0.1 * r1["force"].max(0))) if not succ[i]][:1]
    labels = ["hold still (zero action)", "scripted gait (CEM-tuned)", "PPO (touch + asym. critic)"]
    print("gait demo success", ((rg["err"][-1] < THR) & ~rg["drop"][-1]).mean())
    for k, i in enumerate(good):
        fr = clip(f"{a.media}/demo{a.tag}_success_{k}.mp4", env1, [r0, rg, r1], labels, i)
        if k == 0:
            imageio.mimwrite(f"{a.media}/demo{a.tag}_success_0.gif", [f[::2, ::2] for f in fr[::2]], duration=80, loop=0)
        print(f"success clip {k}: env {i}, delta {np.degrees(env1.delta[i]):.0f} deg, final err {np.degrees(fin[i]):.1f} deg")
    for i in bad:
        clip(f"{a.media}/demo{a.tag}_failure.mp4", env1, [r0, rg, r1], labels, i)
        print(f"failure clip: env {i}, delta {np.degrees(env1.delta[i]):.0f} deg, final err {np.degrees(fin[i]):.1f} deg, "
              f"dropped {r1['drop'][-1][i]}, peak force {r1['force'][:, i].max():.1f} N")


if __name__ == "__main__":
    main()
