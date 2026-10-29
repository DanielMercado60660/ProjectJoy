"""Offscreen rendering of a recorded episode (list of qpos) to MP4/GIF."""
import numpy as np
import mujoco
import imageio.v2 as imageio
import lava_env as E


def render_frames(env, layout, qpos_list, width=640, height=360, every=2, label=None):
    m, d = env.m, env.d
    env.set_layout(layout)
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.9, 115, -28
    frames = []
    for k in range(0, len(qpos_list), every):
        d.qpos[:] = qpos_list[k]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [d.qpos[0] + 0.25, 0.0, 0.15]
        r.update_scene(d, cam)
        img = r.render().copy()
        if label is not None:
            img = stamp(img, label)
        frames.append(img)
    r.close()
    return frames


def stamp(img, text):
    """Draw a small text label (PIL) in the top-left corner."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 20], fill=(0, 0, 0))
    dr.text((4, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def save_video(frames, path, fps=25):
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
