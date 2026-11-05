"""Camera geometry and image preprocessing for the navigation models (no model code here).

The duck's renders are square (Duck.head_image rotates the sideways camera upright, so the renderer must be square).
The visualnav-transformer training pipeline centre-crops every image to 4:3 (IMAGE_ASPECT_RATIO, w:h) and resizes
it with PIL's default (bicubic) filter to the model size: 85x64 for ViNT, 96x96 for NoMaD (a squash of the 4:3 crop).
"""
import numpy as np
from PIL import Image

RENDER = 160                      # square render side (px)
CROP = (20, 140)                  # rows kept: the central 120 of 160 -> 160 x 120 = 4:3
FOVY = {"normal": 45.0, "wide": 100.0}   # MuJoCo fovy (vertical) in degrees; square render -> hfov = fovy


def crop43(square):
    """(160, 160, 3) upright render -> (120, 160, 3) centre 4:3 crop."""
    assert square.shape[:2] == (RENDER, RENDER), square.shape
    return np.ascontiguousarray(square[CROP[0]:CROP[1]])


def resize(frames, size):
    """uint8 (N, 120, 160, 3) 4:3 crops -> uint8 (N, H, W, 3) at size = (W, H), PIL bicubic (the training resize)."""
    out = np.empty((len(frames), size[1], size[0], 3), np.uint8)
    for i, f in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC))
    return out


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def heading_of_axis(axis):
    """Yaw of a camera optical axis (world 3-vector or (N, 3)) projected on the floor."""
    axis = np.asarray(axis)
    return np.arctan2(axis[..., 1], axis[..., 0])


def to_ground_frame(points_xy, origin_xy, heading):
    """World xy -> the 'camera ground frame' (origin = trunk xy, x = camera heading, y left)."""
    d = np.asarray(points_xy, float) - np.asarray(origin_xy, float)
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)
