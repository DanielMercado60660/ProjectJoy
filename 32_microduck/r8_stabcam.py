"""Round 8 Gate B: how steady can the duck's camera view get? Cheapest options first, no building.

Streams, rendered for the same episodes (results/r8/stabcam/protocol.json has the frozen settings):
  head        the head camera as now (128 px, 45 deg, the interface's frame)
  reflex      the head camera with the gaze.py reflex on (a paired physics run: the reflex moves the head)
  trunk       a sim-only camera fixed to the trunk (front-top, tilted down), same intrinsics as the head camera
  oracle      ORACLE (simulator truth): the true head-camera position, orientation = the true head-camera
              orientation through the target filter (roll held at 0, yaw and pitch smoothed)
  eis         electronic stabilisation, sensor-only: a wide render from the head camera, its rotation estimated
              from the gyro + gravity reading + joint encoders through the bot's own camera model (BodyModel),
              warped by the rotation-only homography H = K R K^-1 to the filtered target and cropped to 45 deg
  eis_lag     the same warp with the estimate and target from one 50 Hz sample earlier (20 ms sensor latency)
  eis_lagc    eis_lag with the estimate extrapolated one sample at its last rotation rate (latency compensation)
  eis_reflex  the eis warp on the reflex run (both at once)
  pl_head / pl_eis   phase-locked sampling: the 50 Hz frame at each left-foot touchdown (gait phase from the
              encoders + gravity only, as odometry.py finds the stance foot)

How nothing leaks into the physics: StabWorld is room.World compiled from room.build_spec plus three cameras
(a trunk camera and two pose "slots" on the world body). Cameras have no mass or contacts; qpos stays bit for bit
the same (tests/test_r8_stabcam.py). All extra renders use a separate MjData synced from the live one
(qpos -> mj_kinematics -> mj_camlight), so the duck's own data (whose sensors/xquat belong to the state before
the last substep) is never touched.

Truth use, labelled: the ORACLE stream's pose and every metric (the true camera orientation scores the wobble)
read the simulator. The eis estimate itself reads only gyro, gravity and encoders (what the walking policy reads).
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import room  # noqa: E402
from bodymodel import BodyModel, rot_z, tilt_rotation  # noqa: E402
from duck import DEFAULT_POSE  # noqa: E402

TRAIN = json.loads((HERE / "splits" / "train.json").read_text())["configs"]
OUT = HERE / "results" / "r8" / "stabcam"
DATA = HERE / "data" / "r8_stabcam"

FOVY = 45.0                 # the head camera's (MJCF default) and every output stream's vertical = horizontal FOV
OUT_PX = 128                # the interface frame; metrics at 64 px grey through the 4A pipeline (128 -> 96 -> 64)
WIDE_FOVY = 140.0           # eis source render (dev choice, r8_stabcam_run.py dev + protocol.json: 100/120 deg clamp in turns)
WIDE_PX = 864               # focal 157 px ~ the 128 px / 45 deg output's 154.5: no upsampling at the centre
SUB_DT = 0.02               # 50 Hz policy step
TRUNK_MOUNT = {"pos": [0.05, 0.0, 0.02], "tilt_down_deg": 10.0}   # trunk frame: front-top, 0.14 m above the floor
# head-camera matrix (MJCF, after Duck's 180 deg fix) -> standard upright camera (x right, y up, looking -z):
# the upright head image (head_image's rot90) has right = camera +y and up = camera -x.
P_STD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
FILTER = {"kind": "notch", "stride_hz": 1.86, "ramp_yaw": False, "ramp_pitch": False}  # "ma2", frozen in protocol.json
OUT_CORNER_DEG = float(np.degrees(np.arctan(np.sqrt(2) * np.tan(np.radians(FOVY / 2)))))   # 30.36: output corner ray


def clamp_deg(wide_fovy):
    """Any target whose axis is within this angle of the source axis is 100% inside the wide render."""
    return wide_fovy / 2 - OUT_CORNER_DEG - 0.1


CLAMP_DEG = clamp_deg(WIDE_FOVY)
STANCE_HYST = 0.002         # m: foot-height difference before the stance foot is called switched
CAM = mujoco.mjtObj.mjOBJ_CAMERA


# ------------------------------------------------------------------------------------------------ model / world
def _trunk_quat():
    """Standard camera looking along trunk +x (right = trunk -y, up = trunk +z), pitched nose-down."""
    R0 = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])     # columns: cam x, y, z in trunk coords
    a = np.radians(TRUNK_MOUNT["tilt_down_deg"])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Ry @ R0).reshape(-1))
    return q


def build_model_r8(cfg, offscreen=1024):
    """room.build_spec + a trunk camera and two pose slots (45 deg and wide), a bigger offscreen buffer. Visual only."""
    spec = room.build_spec(cfg)
    spec.visual.global_.offwidth = offscreen
    spec.visual.global_.offheight = offscreen
    spec.body("trunk_base").add_camera(name="r8_trunk", pos=TRUNK_MOUNT["pos"], quat=_trunk_quat(), fovy=FOVY)
    spec.worldbody.add_camera(name="r8_virt", pos=[0, 0, 1], fovy=FOVY)
    spec.worldbody.add_camera(name="r8_wide", pos=[0, 0, 1], fovy=WIDE_FOVY)
    model = spec.compile()
    assert model.nu == 14 and abs(model.opt.timestep - 0.005) < 1e-12
    return model


class StabWorld(room.World):
    """room.World on build_model_r8's model. room.World looks build_model up at call time; it is swapped only
    while this constructor runs (single-threaded workers)."""

    def __init__(self, cfg, body=room.FROZEN_BODY, policy=None, offscreen=1024):
        saved = room.build_model
        room.build_model = lambda c: build_model_r8(c, offscreen)
        try:
            super().__init__(cfg, body, policy)
        finally:
            room.build_model = saved


# ------------------------------------------------------------------------------------------------ geometry
def angles(R):
    """(yaw, pitch, roll) of a standard camera matrix (columns right, up, back): yaw/pitch of the forward axis,
    roll = tilt of the right axis out of the horizontal plane (0 = level horizon)."""
    f = -R[:, 2]
    yaw = np.arctan2(f[1], f[0])
    pitch = np.arcsin(np.clip(f[2], -1, 1))
    roll = np.arctan2(-R[2, 0], R[2, 1])
    return np.array([yaw, pitch, roll])


def level_matrix(yaw, pitch):
    """Standard camera matrix looking at (yaw, pitch) with a level horizon (roll 0)."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def axis_angle_deg(Ra, Rb):
    """Angle between the forward axes of two camera matrices (deg)."""
    return float(np.degrees(np.arccos(np.clip(Ra[:, 2] @ Rb[:, 2], -1, 1))))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------------------------------------ rendering
class Rig:
    """Extra renders of a running StabWorld from a separate MjData (never the duck's own)."""

    def __init__(self, world, sizes=(OUT_PX, WIDE_PX), headlight_diffuse=None):
        m = world.model
        self.world, self.m, self.d = world, m, world.duck.data
        self.headlight = headlight_diffuse      # None = as the room sets it; a number overrides it for these renders only
        self.rd = mujoco.MjData(m)
        self.head = world.duck.cam_id
        self.trunk = mujoco.mj_name2id(m, CAM, "r8_trunk")
        self.virt = mujoco.mj_name2id(m, CAM, "r8_virt")
        self.wide = mujoco.mj_name2id(m, CAM, "r8_wide")
        assert min(self.trunk, self.virt, self.wide) >= 0, "not a StabWorld"
        self.renderers = {s: mujoco.Renderer(m, s, s) for s in sizes}

    def close(self):
        for r in self.renderers.values():
            r.close()

    def sync(self):
        rd = self.rd
        rd.qpos[:] = self.d.qpos
        rd.time = self.d.time
        mujoco.mj_kinematics(self.m, rd)
        mujoco.mj_camlight(self.m, rd)

    def head_pose(self):
        return self.rd.cam_xpos[self.head].copy(), self.rd.cam_xmat[self.head].reshape(3, 3) @ P_STD

    def trunk_pose(self):
        return self.rd.cam_xpos[self.trunk].copy(), self.rd.cam_xmat[self.trunk].reshape(3, 3).copy()

    def _r(self, size):
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(self.m, size, size)
        return self.renderers[size]

    def _draw(self, size, cam):
        """update_scene + render. The headlight (MuJoCo's camera-attached light, which turns with the camera) is
        copied into the scene by update_scene, so an override around that call touches no other renderer."""
        r = self._r(size)
        if self.headlight is None:
            r.update_scene(self.rd, camera=cam)
        else:
            hl = self.m.vis.headlight
            saved = hl.diffuse.copy()
            hl.diffuse[:] = self.headlight
            try:
                r.update_scene(self.rd, camera=cam)
            finally:
                hl.diffuse[:] = saved
        return r.render()

    def render_head(self, size=OUT_PX):
        """The head camera exactly as Duck.head_image renders it (upright)."""
        return np.ascontiguousarray(np.rot90(self._draw(size, self.head), -1))

    def render_trunk(self, size=OUT_PX):
        return self._draw(size, self.trunk).copy()

    def render_pose(self, pos, R, size=OUT_PX, wide=False):
        """A standard camera at pos with orientation R (45 deg, or the wide slot)."""
        slot = self.wide if wide else self.virt
        self.rd.cam_xpos[slot] = pos
        self.rd.cam_xmat[slot] = np.asarray(R).reshape(-1)
        return self._draw(size, slot).copy()


_RAYS = {}


def out_rays(n=OUT_PX, fovy=FOVY):
    """Unit rays of an upright n x n standard camera (row 0 at the top), shape (n, n, 3)."""
    key = (n, fovy)
    if key not in _RAYS:
        t = np.tan(np.radians(fovy) / 2)
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        d = np.stack([((j + 0.5) / n * 2 - 1) * t, (1 - (i + 0.5) / n * 2) * t, -np.ones((n, n))], -1)
        _RAYS[key] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return _RAYS[key]


def warp(src, R_src, R_tgt, n=OUT_PX, fovy=FOVY, src_fovy=WIDE_FOVY):
    """Rotation-only homography H = K_src R_src^T R_tgt K_out^-1, bilinear. src: upright wide render taken (as far as
    the caller knows) at orientation R_src; returns the n x n view at R_tgt (uint8) and its valid-pixel mask."""
    d = out_rays(n, fovy) @ (R_src.T @ R_tgt).T          # output rays in the source camera frame
    z = -d[..., 2]
    ok = z > 1e-6
    t = np.tan(np.radians(src_fovy) / 2)
    N = src.shape[0]
    zz = np.where(ok, z, 1.0)
    col = (d[..., 0] / zz / t + 1) / 2 * N - 0.5
    row = (1 - d[..., 1] / zz / t) / 2 * N - 0.5
    valid = ok & (col >= 0) & (col <= N - 1) & (row >= 0) & (row <= N - 1)
    coords = [np.clip(row, 0, N - 1), np.clip(col, 0, N - 1)]
    out = np.stack([ndimage.map_coordinates(src[..., c].astype(np.float32), coords, order=1) for c in range(3)], -1)
    out[~valid] = 0.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), valid


def grey64(frame):
    """The 4A pipeline (p2_learner.prep_frame without importing MLX): 128 -> record.down's 96 BOX -> 64 BOX,
    then the channel mean, float32 grey levels."""
    f = np.asarray(Image.fromarray(frame).resize((96, 96), Image.Resampling.BOX))
    f = np.asarray(Image.fromarray(f).resize((64, 64), Image.Resampling.BOX))
    return f.astype(np.float32).mean(-1)


def valid64(valid128):
    """A 64 px pixel is valid only if its whole 2 x 2 block at 128 px is."""
    n = valid128.shape[0] // 2
    return valid128.reshape(n, 2, n, 2).all((1, 3))


# ------------------------------------------------------------------------------------------------ sensing
class CamEstimator:
    """Sensor-only head-camera orientation at 50 Hz, the bot's own camera model (perception.CameraPose):
    yaw from integrating the gyro (as odometry.py), tilt from the gravity reading, the head chain from the joint
    encoders through BodyModel.fk. The yaw lives in the odometry frame (start = 0). One fk call per sample also
    gives the feet for the stance detector."""

    def __init__(self, body=None):
        self.body = body or BodyModel()
        self.psi = 0.0

    def update(self, gyro, gravity, joint_pos):
        Rt = tilt_rotation(gravity)
        self.psi += float((rot_z(self.psi) @ Rt @ np.asarray(gyro))[2]) * SUB_DT
        feet, _, cm = self.body.fk(joint_pos)
        R = rot_z(self.psi) @ Rt
        return R @ cm @ P_STD, (Rt @ feet.T).T          # camera (standard), gravity-aligned feet (trunk origin)


class StanceEvents:
    """Gait phase from encoders + gravity only: the stance foot is the lower one in the gravity-aligned trunk frame
    (odometry.py's rule), switched only past a small hysteresis. Event 'L' = the left foot takes over (touchdown)."""

    def __init__(self, hyst=STANCE_HYST):
        self.hyst, self.stance = hyst, None

    def update(self, feet_aligned):
        dz = feet_aligned[1, 2] - feet_aligned[0, 2]      # > 0: left foot lower
        new = self.stance
        if dz > self.hyst:
            new = 0
        elif dz < -self.hyst:
            new = 1
        ev = None
        if self.stance is not None and new != self.stance:
            ev = "L" if new == 0 else "R"
        self.stance = new
        return ev


class TargetFilter:
    """Causal smoothing of (yaw, pitch) of the forward axis; the target view keeps roll at 0 (level horizon).
    kinds:
      ema    first-order low-pass, time constant tau (s)
      ab     alpha-beta (type-2) tracker, natural frequency wn (rad/s), damping 1: no lag on a steady turn
      notch  two cascaded moving averages over one stride period (zeros at the stride frequency and all its
             harmonics), plus, per axis, a ramp term (N-1) * (y_k - y_{k-1}) that cancels the averages' delay
             on a steady turn (yaw); pitch has no sustained ramps, so it is left without (no overshoot)."""

    def __init__(self, params=None):
        self.P = dict(FILTER, **(params or {}))
        self.prev = None          # last unwrapped yaw input
        self.state = None

    def step(self, yaw, pitch):
        if self.prev is None:
            u = np.array([yaw, pitch])
        else:
            u = np.array([self.prev[0] + wrap(yaw - self.prev[0]), pitch])
        self.prev = u
        k = self.P["kind"]
        if k == "ema":
            a = SUB_DT / max(self.P.get("tau", 1.0), SUB_DT)
            self.state = u.copy() if self.state is None else self.state + a * (u - self.state)
            return self.state.copy()
        if k == "ab":
            wn = self.P.get("wn", 2.0)
            k1, k2 = 2 * wn * SUB_DT, (wn * SUB_DT) ** 2
            if self.state is None:
                self.state = {"x": u.copy(), "v": np.zeros(2)}
            s = self.state
            s["x"] = s["x"] + s["v"]
            r = u - s["x"]
            s["x"] = s["x"] + k1 * r
            s["v"] = s["v"] + k2 * r
            return s["x"].copy()
        if k in ("notch", "ma1"):
            N = max(2, int(round(1.0 / (self.P["stride_hz"] * SUB_DT))))
            if self.state is None:
                self.state = {"b1": [u.copy()] * N, "b2": None, "y": None}
            s = self.state
            s["b1"] = s["b1"][1:] + [u]
            m1 = np.mean(s["b1"], 0)
            if k == "ma1":                      # one moving average over a stride: half the delay, wider sidelobes
                return m1
            if s["b2"] is None:
                s["b2"] = [m1] * N
            s["b2"] = s["b2"][1:] + [m1]
            y = np.mean(s["b2"], 0)
            dy = np.zeros(2) if s["y"] is None else y - s["y"]
            s["y"] = y
            ramp = np.array([float(self.P.get("ramp_yaw", True)), float(self.P.get("ramp_pitch", False))])
            return y + ramp * (N - 1) * dy
        raise ValueError(k)


def clamp_target(yaw_t, pitch_t, R_src, max_deg=CLAMP_DEG):
    """Pull a level target (yaw_t, pitch_t) toward the source camera's forward axis until it is within max_deg
    (then every output pixel lies inside the wide render). Returns (yaw, pitch, clamped?)."""
    ys, ps, _ = angles(R_src)
    dy, dp = wrap(yaw_t - ys), pitch_t - ps
    for it in range(4):
        a = axis_angle_deg(level_matrix(ys + dy, ps + dp), R_src)
        if a <= max_deg:
            return ys + dy, ps + dp, it > 0
        s = max_deg / a * 0.98
        dy, dp = dy * s, dp * s
    return ys + dy, ps + dp, True


def detrended(x, k=50):
    """x minus its centred 1 s moving mean (m3_gaze.detrended_std's trend, not reduced to a std)."""
    x = np.unwrap(np.asarray(x, float))
    trend = np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")[:len(x)]
    return x - trend


# ------------------------------------------------------------------------------------------------ the probe
class Probe:
    """Hooks a StabWorld's 50 Hz step. Every step: the true head and trunk camera orientations, the sensor-only
    estimate, the filtered targets, the effective view orientation of each stream, the stance events, and the
    renders the schedule asks for.

    mode "off" (reflex off): streams head, trunk, oracle, eis, eis_lag (+ phase-locked from the 50 Hz head/eis frames)
    mode "on"  (reflex run): streams reflex, eis_reflex, eis_reflex_lag
    render: "50" = head and eis at every walking step (phase-locking needs them) plus the rest at tick ends;
            "10" = everything at tick ends only; "none" = orientations only (filter tuning);
            "none_but_rig" = no metric renders, but renderers for the hooks (video, frame set).
    keep_mats: also keep the raw true/estimated matrices (filter tuning offline)."""

    def __init__(self, world, mode="off", render="50", filt=None, keep_mats=False, headlight_diffuse=None):
        self.w, self.mode, self.render, self.keep_mats = world, mode, render, keep_mats
        self.rig = Rig(world, sizes=() if render == "none" else (OUT_PX, WIDE_PX), headlight_diffuse=headlight_diffuse)
        self.est = CamEstimator()
        self.stance = StanceEvents()
        self.f_or, self.f_eis = TargetFilter(filt), TargetFilter(filt)
        self.prev, self.prev2 = None, None
        names = (("head", "trunk", "oracle", "eis", "eis_lag", "eis_lagc") if mode == "off"
                 else ("reflex", "eis_reflex", "eis_reflex_lag", "eis_reflex_lagc"))
        self.names = names
        self.ang = {n: [] for n in names}
        self.cols = {k: [] for k in ("t", "walk", "tick_end", "offset_deg", "clamped", "vx", "wz", "pos")}
        self.events = []
        self.f10 = {n: [] for n in names}
        self.v10 = {n: [] for n in names if n.startswith("eis")}
        self.f50 = {"head": [], "eis": []} if (mode == "off" and render == "50") else {}
        self.v50 = {"eis": []} if (mode == "off" and render == "50") else {}
        self.s50 = []                             # step index of each 50 Hz frame (rendered only while walking)
        self.mats = {"Rh": [], "Rhat": [], "Rtr": []}
        self.hooks = []                          # extra per-step callbacks f(probe, ctx) (frame set, video)
        self.k = 0
        orig = world.step

        def step():
            r = orig()
            self.on_step()
            return r
        world.step = step

    def close(self):
        self.rig.close()

    def on_step(self):
        w = self.w
        duck = w.duck
        d = duck.data
        rig = self.rig
        rig.sync()
        hp, Rh = rig.head_pose()
        _, Rtr = rig.trunk_pose()
        gyro = d.sensordata[duck.gyro_adr:duck.gyro_adr + 3].astype(np.float64)
        grav = duck.projected_gravity().astype(np.float64)
        jp = (d.qpos[duck.jq] - DEFAULT_POSE).astype(np.float64)
        Rhat, feet = self.est.update(gyro, grav, jp)
        ev = self.stance.update(feet)
        if ev is not None:
            self.events.append((self.k, ev))
        ah = angles(Rh)
        yo, po = self.f_or.step(ah[0], ah[1])
        R_or = level_matrix(yo, po)
        ae = angles(Rhat)
        ye, pe = self.f_eis.step(ae[0], ae[1])
        off = axis_angle_deg(level_matrix(ye, pe), Rhat)
        ye, pe, clamped = clamp_target(ye, pe, Rhat)
        R_tgt = level_matrix(ye, pe)
        if self.prev is None:
            self.prev = (Rhat, R_tgt)
        if self.prev2 is None:
            self.prev2 = self.prev[0]
        R_pred = self.prev[0] @ self.prev2.T @ self.prev[0]     # constant-rate extrapolation of the late estimate
        R_eff = Rh @ Rhat.T @ R_tgt                          # what the warped view actually shows (scored by truth)
        R_lag = Rh @ self.prev[0].T @ self.prev[1]
        R_lagc = Rh @ R_pred.T @ self.prev[1]
        tick_end = (w.step_count % 5 == 0)
        c = duck.command
        walking = bool(c[0] > 0.15)
        cols = self.cols
        cols["t"].append(d.time)
        cols["walk"].append(walking)
        cols["tick_end"].append(tick_end)
        cols["offset_deg"].append(off)
        cols["clamped"].append(clamped)
        cols["vx"].append(float(c[0]))
        cols["wz"].append(float(c[2]))
        cols["pos"].append(hp)
        if self.mode == "off":
            for n, R in (("head", Rh), ("trunk", Rtr), ("oracle", R_or), ("eis", R_eff), ("eis_lag", R_lag),
                         ("eis_lagc", R_lagc)):
                self.ang[n].append(angles(R))
        else:
            for n, R in (("reflex", Rh), ("eis_reflex", R_eff), ("eis_reflex_lag", R_lag), ("eis_reflex_lagc", R_lagc)):
                self.ang[n].append(angles(R))
        if self.keep_mats:
            self.mats["Rh"].append(Rh)
            self.mats["Rhat"].append(Rhat)
            self.mats["Rtr"].append(Rtr)
        ctx = {"hp": hp, "Rh": Rh, "Rtr": Rtr, "R_or": R_or, "Rhat": Rhat, "R_tgt": R_tgt, "prev": self.prev,
               "R_pred": R_pred, "tick_end": tick_end, "walking": walking, "R_eff": R_eff}
        if self.render in ("50", "10"):              # "none_but_rig": renderers for hooks only (video, frame set)
            self._render(ctx)
        for h in self.hooks:
            h(self, ctx)
        self.prev2 = self.prev[0]
        self.prev = (Rhat, R_tgt)
        self.k += 1

    def _render(self, x):
        rig = self.rig
        full = self.render == "50" and self.mode == "off" and x["walking"]
        if not (full or x["tick_end"]):
            return
        head = grey64(rig.render_head())
        wide = rig.render_pose(x["hp"], x["Rh"], WIDE_PX, wide=True)
        e_img, e_val = warp(wide, x["Rhat"], x["R_tgt"])
        eis, eis_v = grey64(e_img), valid64(e_val)
        if full:
            self.s50.append(self.k)
            self.f50["head"].append(head.astype(np.float16))
            self.f50["eis"].append(eis.astype(np.float16))
            self.v50["eis"].append(eis_v)
        if not x["tick_end"]:
            return
        l_img, l_val = warp(wide, x["prev"][0], x["prev"][1])
        lag, lag_v = grey64(l_img), valid64(l_val)
        c_img, c_val = warp(wide, x["R_pred"], x["prev"][1])
        lagc, lagc_v = grey64(c_img), valid64(c_val)
        if self.mode == "off":
            self.f10["head"].append(head.astype(np.float16))
            self.f10["trunk"].append(grey64(rig.render_trunk()).astype(np.float16))
            self.f10["oracle"].append(grey64(rig.render_pose(x["hp"], x["R_or"])).astype(np.float16))
            for n, g, v in (("eis", eis, eis_v), ("eis_lag", lag, lag_v), ("eis_lagc", lagc, lagc_v)):
                self.f10[n].append(g.astype(np.float16))
                self.v10[n].append(v)
        else:
            self.f10["reflex"].append(head.astype(np.float16))
            for n, g, v in (("eis_reflex", eis, eis_v), ("eis_reflex_lag", lag, lag_v), ("eis_reflex_lagc", lagc, lagc_v)):
                self.f10[n].append(g.astype(np.float16))
                self.v10[n].append(v)

    def result(self):
        out = {"ang": {n: np.array(v, np.float64) for n, v in self.ang.items()},
               "events": np.array([(k, 0 if e == "L" else 1) for k, e in self.events], np.int32).reshape(-1, 2)}
        for k, v in self.cols.items():
            out[k] = np.array(v)
        out["f10"] = {n: np.array(v) for n, v in self.f10.items() if len(v)}
        out["v10"] = {n: np.array(v) for n, v in self.v10.items() if len(v)}
        out["f50"] = {n: np.array(v) for n, v in self.f50.items() if len(v)}
        out["v50"] = {n: np.array(v) for n, v in self.v50.items() if len(v)}
        out["s50"] = np.array(self.s50, np.int32)
        if self.keep_mats:
            out["mats"] = {n: np.array(v) for n, v in self.mats.items()}
        return out


# ------------------------------------------------------------------------------------------------ episodes
SCRIPTED = [(0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.3, 0.4), (0.3, -0.4), (0.0, 0.55)]   # = m3_gaze.CONDS
SCRIPT_TICKS, SCRIPT_WALK_TICKS, WINDOW = 130, 100, (150, 500)                        # m3_gaze's recipe


def run_scripted(cfg, vx, wz, mode="off", render="50", filt=None, keep_mats=False, hooks=(), headlight_diffuse=None):
    """m3_gaze.run's segment: walk (vx, wz) for 10 s, then stand 3 s; the reflex (mode 'on') adapts on the bot's
    own camera model exactly as there. Returns the probe's result."""
    from gaze import Gaze
    from interface import Command, Interface
    from odometry import Odometry
    w = StabWorld(cfg)
    pr = Probe(w, mode, render, filt, keep_mats, headlight_diffuse)
    pr.hooks.extend(hooks)
    it = Interface(w)
    bm, od, gz = BodyModel(), Odometry(), Gaze()
    gz.enabled = mode == "on"
    obs = it.observe()
    for k in range(SCRIPT_TICKS):
        walking = k < SCRIPT_WALK_TICKS
        oy, op = gz.offsets(obs.sim_time)
        c = Command(vx=vx if walking else 0.0, wz=wz if walking else 0.0, head=(0.0, op, oy, 0.0))
        obs = it.tick(c)
        od.update(obs)
        _, _, cm = bm.fk(obs.body["joint_pos"])
        gz.update(obs, rot_z(od.psi) @ tilt_rotation(obs.body["gravity"]) @ cm)
    pr.close()
    res = pr.result()
    res["meta"] = {"kind": "scripted", "config": cfg["id"], "vx": vx, "wz": wz, "mode": mode,
                   "headlight_diffuse": headlight_diffuse}
    return res


def run_bot(cfg, k, mode="off", render="50", filt=None, hooks=(), keep_mats=False, offscreen=1024):
    """One reference-bot set_lamp episode (the frozen bot; params {'gaze': True} in mode 'on', as r7_gaze_pregate)."""
    from refbot import controller
    from tasks import BUDGET_TICKS, make_episode, run_episode
    ep = make_episode(cfg, "set_lamp", k)
    w = StabWorld(cfg, offscreen=offscreen)
    pr = Probe(w, mode, render, filt, keep_mats)
    pr.hooks.extend(hooks)
    fac, bot = controller("set_lamp", BUDGET_TICKS["set_lamp"], {"gaze": True} if mode == "on" else None)
    res, _, iface = run_episode(cfg, ep, fac, world=w)
    pr.close()
    out = pr.result()
    out["meta"] = {"kind": "bot", "config": cfg["id"], "task": "set_lamp", "k": int(k), "mode": mode,
                   "success": bool(res["success"]), "ticks": int(iface.tick_count), "falls": int(res.get("falls", 0)),
                   "instruction": ep["instruction"]}
    return out
