"""The reference bot's eyes (bot side): colour segmentation of the head-camera frame and projection of
blobs onto the floor, using only the frame, the bot's body model and its own pose estimate.

Classes (fixed colour categories, a prior shared by every config; shades vary by appearance family):
  pad_red, pad_yellow, pad_green, pad_blue   saturated floor squares
  button                                      cyan floor plate (in a dark frame)
  ball                                        magenta sphere (centre 3.5 cm above the floor)
  mat                                         white floor square (low saturation, bright)
  lamp_on                                     bright, warm, low-saturation patch above the horizon
  lamp_off                                    dark grey compact patch near the horizon, sized like a
                                              10 cm box (gives a distance and a position estimate)
Each detection carries its floor position (mean of the blob's pixels projected to the floor plane at
the object's height), the image area, whether it touches the frame border, and its mean colour.
Thresholds live in PARAMS so they can be tuned (results/m3/tuning.jsonl) on train/dev frames only.
The hue windows are the midpoints between the category hue centres (a prior), not tuned.
"""
import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage

from bodymodel import CAM_FOVY_DEG, FRAME, rot_z, tilt_rotation

PARAMS = {
    "sat_min": 0.45, "val_min": 0.28,
    "hue": {"pad_red": [(330, 360), (0, 22)], "pad_yellow": [(28, 80)], "pad_green": [(85, 160)],
            "button": [(160, 200)], "pad_blue": [(200, 262)], "ball": [(268, 330)]},
    "min_area": {"pad": 6, "button": 6, "ball": 3, "mat": 25, "lamp_on": 3, "lamp_off": 8},
    "mat_sat_max": 0.15, "mat_val_min": 0.62, "mat_max_extent": 0.75, "mat_max_dist": 2.0,
    "lamp_val_min": 0.92, "lamp_sat_max": 0.55,
    # lamp off: a dark, grey, compact patch near the horizon whose size fits a 10 cm box
    "lampoff_val_max": 0.22, "lampoff_sat_max": 0.35, "lampoff_elev": (-0.06, 0.16),
    "lampoff_aspect": (0.5, 2.0), "lampoff_fill_min": 0.6, "lampoff_dist": (0.4, 3.6),
    "max_floor_dist": 4.5,
    "ball_max_area_x": 3.0,          # metres: anything projecting further can't be on this room's floor
}
OBJ_HEIGHT = {"ball": 0.035, "button": 0.005}
FLOOR_CLASSES = ("pad_red", "pad_yellow", "pad_green", "pad_blue", "button", "ball", "mat")


def _camera_rays():
    """Unit ray directions in the camera frame (x right, y up, looking -z) for each pixel of the
    upright head image. head_image rotates the raw render clockwise: upright[i, j] = raw[H-1-j, i]."""
    t = np.tan(np.radians(CAM_FOVY_DEG) / 2)
    n = FRAME
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")   # upright row i, col j
    r, c = n - 1 - j, i                                              # raw row, col
    x = ((c + 0.5) / n * 2 - 1) * t
    y = (1 - (r + 0.5) / n * 2) * t
    d = np.stack([x, y, -np.ones_like(x)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


CAM_RAYS = _camera_rays()
FOCAL_PX = (FRAME / 2) / np.tan(np.radians(CAM_FOVY_DEG) / 2)


class CameraPose:
    """Head-camera pose in the bot's odometry frame from its own pose estimate (x, y, yaw), the gravity
    reading and the head/leg encoders (through the body model)."""

    def __init__(self, body, obs, pose_xy_yaw):
        feet, cp, cm = body.fk(obs.body["joint_pos"])
        R = rot_z(pose_xy_yaw[2]) @ tilt_rotation(obs.body["gravity"])
        q = (R @ feet.T).T
        trunk = np.array([pose_xy_yaw[0], pose_xy_yaw[1], -q[:, 2].min()])
        self.trunk_xy = trunk[:2].copy()
        self.pos = trunk + R @ cp
        self.mat = R @ cm
        self.rays = CAM_RAYS @ self.mat.T          # world-frame ray per pixel

    def floor_points(self, mask, height=0.0):
        """Project the masked pixels onto the plane z=height; returns (N,2) and validity."""
        d = self.rays[mask]
        ok = d[:, 2] < -1e-3
        t = (height - self.pos[2]) / np.where(ok, d[:, 2], -1)
        p = self.pos[:2] + t[:, None] * d[:, :2]
        return p, ok & (t > 0)

    def bearing(self, i, j):
        d = self.rays[int(i), int(j)]
        return float(np.arctan2(d[1], d[0])), float(np.arcsin(np.clip(d[2], -1, 1)))


def hsv(frame):
    return rgb_to_hsv(frame.astype(np.float32) / 255.0)


def masks(frame, P=PARAMS):
    h = hsv(frame)
    H, S, V = h[..., 0] * 360, h[..., 1], h[..., 2]
    out = {}
    sat = (S >= P["sat_min"]) & (V >= P["val_min"])
    for cls, wins in P["hue"].items():
        m = np.zeros_like(sat)
        for lo, hi in wins:
            m |= (H >= lo) & (H < hi)
        out[cls] = m & sat
    out["mat"] = (S <= P["mat_sat_max"]) & (V >= P["mat_val_min"])
    out["lamp_on"] = (V >= P["lamp_val_min"]) & (S <= P["lamp_sat_max"]) & ((H < 75) | (S < 0.08))
    out["lamp_off"] = (V <= P["lampoff_val_max"]) & (S <= P["lampoff_sat_max"])
    return out


def detect(frame, cam: CameraPose, P=PARAMS):
    """Blobs per class with floor positions (floor classes) or bearings (lamp)."""
    dets = []
    up = cam.rays[..., 2]                  # >0: above the horizon
    for cls, m in masks(frame, P).items():
        key = "pad" if cls.startswith("pad") else cls
        if cls == "lamp_on":
            m = m & (up > -0.05)           # on a wall at ~0.3 m: at or above the camera's horizon
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        for k in range(1, n + 1):
            bm = lab == k
            area = int(bm.sum())
            if area < P["min_area"][key]:
                continue
            ii, jj = np.nonzero(bm)
            border = bool(ii.min() == 0 or jj.min() == 0 or ii.max() == FRAME - 1 or jj.max() == FRAME - 1)
            det = {"cls": cls, "area": area, "border": border,
                   "px": [float(ii.mean()), float(jj.mean())],
                   "rgb": frame[bm].mean(0).round(1).tolist()}
            if cls in ("lamp_on", "lamp_off"):
                det["bearing"], det["elev"] = cam.bearing(ii.mean(), jj.mean())
                if cls == "lamp_on" and not border:       # size -> distance for the on-lamp too
                    hpx, wpx = ii.max() - ii.min() + 1, jj.max() - jj.min() + 1
                    det["dist"] = float(0.10 * FOCAL_PX / max(hpx, wpx))
                    det["xy"] = (cam.pos[:2] + det["dist"] * np.array([np.cos(det["bearing"]), np.sin(det["bearing"])])).tolist()
                if cls == "lamp_off":
                    hpx, wpx = ii.max() - ii.min() + 1, jj.max() - jj.min() + 1
                    lo, hi = P["lampoff_elev"]
                    fill = area / float(hpx * wpx)
                    dist = 0.10 * FOCAL_PX / max(hpx, wpx)             # a 10 cm box
                    if not (lo <= det["elev"] <= hi and P["lampoff_aspect"][0] <= hpx / wpx <= P["lampoff_aspect"][1]
                            and fill >= P["lampoff_fill_min"] and P["lampoff_dist"][0] <= dist <= P["lampoff_dist"][1]
                            and not border):
                        continue
                    det["dist"] = float(dist)
                    det["xy"] = (cam.pos[:2] + dist * np.array([np.cos(det["bearing"]), np.sin(det["bearing"])])).tolist()
            else:
                pts, ok = cam.floor_points(bm, OBJ_HEIGHT.get(cls, 0.0))
                if ok.sum() < max(3, 0.5 * area):
                    continue
                pts = pts[ok]
                dist = np.linalg.norm(pts - cam.pos[:2], axis=1)
                if np.median(dist) > P["max_floor_dist"]:
                    continue
                det["xy"] = pts.mean(0).tolist()
                det["extent"] = (pts.max(0) - pts.min(0)).round(3).tolist()
                det["dist"] = float(np.median(dist))
                if cls == "mat" and (max(det["extent"]) > P["mat_max_extent"] or det["dist"] > P["mat_max_dist"]):
                    continue                # long or far white things are walls, not the 40 cm mat
                if cls == "ball":
                    # image area vs a whole 7 cm ball at this distance: a screen patch is far bigger
                    full = np.pi * (0.035 * FOCAL_PX / max(det["dist"], 0.15)) ** 2
                    det["full_area"] = float(full)
                    if area > P["ball_max_area_x"] * full + 20:
                        continue
            dets.append(det)
    return dets
