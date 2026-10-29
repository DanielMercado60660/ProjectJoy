"""Render M0 demo media (headless, offscreen MuJoCo):
  media/m0_pad_run.mp4      a gate pad run (target 1.5 m at 135 deg): chase view + head_camera inset
  media/m0_push_walk.mp4    a 2 N, 1 s lateral push while walking at 0.3: fall and get-up
  media/m0_head_camera.png  one head_camera frame (128x128, the size later milestones record)
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from duck import CTRL_DT, BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY, goto_command  # noqa: E402

MEDIA = HERE / "media"
W, H, INSET = 480, 360, 128


class Views:
    def __init__(self, duck):
        self.duck = duck
        self.chase = mujoco.Renderer(duck.model, H, W)
        self.head = mujoco.Renderer(duck.model, INSET, INSET)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.9, -25, 135

    def head_frame(self):
        return self.duck.head_image(self.head)

    def frame(self, text_marker=None):
        self.cam.lookat[:] = self.duck.pos() + np.array([0, 0, -0.03])
        self.chase.update_scene(self.duck.data, camera=self.cam)
        img = self.chase.render().copy()
        hv = self.head_frame()
        img[8:8 + INSET, W - INSET - 8:W - 8] = hv
        img[6:8, W - INSET - 10:W - 6] = 255
        img[8 + INSET:10 + INSET, W - INSET - 10:W - 6] = 255
        img[6:10 + INSET, W - INSET - 10:W - INSET - 8] = 255
        img[6:10 + INSET, W - 8:W - 6] = 255
        if text_marker is not None:  # red bar while a push is applied
            img[H - 12:H - 4, 8:W - 8] = (220, 40, 40) if text_marker else img[H - 12:H - 4, 8:W - 8]
        return img


def pad_run():
    duck = Duck(BodyConfig(**FROZEN_BODY))
    duck.reset()
    v = Views(duck)
    target = 1.5 * np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])
    frames = []
    for k in range(int(40 / CTRL_DT)):
        if np.linalg.norm(duck.pos()[:2] - target) < 0.10:
            break
        duck.set_command(*goto_command(duck, target))
        duck.step()
        if k % 2 == 0:
            frames.append(_with_pad(v, target))
    duck.set_command(0, 0, 0)
    for k in range(50):
        duck.step()
        if k % 2 == 0:
            frames.append(_with_pad(v, target))
    imageio.mimsave(MEDIA / "m0_pad_run.mp4", frames, fps=25, macro_block_size=8)
    imageio.imwrite(MEDIA / "m0_head_camera.png", v.head_frame())
    return len(frames)


def _with_pad(v, target):
    v.cam.lookat[:] = v.duck.pos() + np.array([0, 0, -0.03])
    v.chase.update_scene(v.duck.data, camera=v.cam)
    scn = v.chase.scene
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.1, 0.1, 0.002]),
                            np.array([target[0], target[1], 0.002]), np.eye(3).ravel(),
                            np.array([0.2, 0.8, 0.3, 0.8], np.float32))
        scn.ngeom += 1
    img = v.chase.render().copy()
    img[8:8 + INSET, W - INSET - 8:W - 8] = v.head_frame()
    return img


def push_run():
    duck = Duck(BodyConfig(**FROZEN_BODY))
    duck.reset()
    v = Views(duck)
    v.cam.azimuth = 90
    duck.set_command(0.3, 0, 0)
    frames = []
    for k in range(int(9 / CTRL_DT)):
        pushing = 100 <= k < 150
        duck.push_force = np.array([0.0, 2.0, 0.0]) if pushing else np.zeros(3)
        duck.step()
        if k % 2 == 0:
            frames.append(v.frame(text_marker=pushing))
    imageio.mimsave(MEDIA / "m0_push_walk.mp4", frames, fps=25, macro_block_size=8)
    return len(frames)


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    print("pad frames", pad_run())
    print("push frames", push_run())
