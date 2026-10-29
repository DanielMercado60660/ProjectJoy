"""The Microduck room, built with mujoco.MjSpec from a config dict, and the World that runs it.

A config (see configs.py for the generators and splits/ for the frozen sets) holds three parts that
never borrow from each other:
  layout      physical: room size, home mat, pads (colour category + position), partition, button,
              lamp position, occluder, ball, duck start pose, initial lamp state
  appearance  visual only: rgb shades, floor texture, wall colours, lighting (never touches geometry,
              mass, friction or contact)
  distractor  optional flickering screen (visual only, its own RNG)
plus "mujoco_version" (pinned; build refuses a mismatch) and "id".

World = compiled room + the frozen M0 body + the room logic (button -> lamp, flicker). Its full state
(MjData integration state, the duck's controller buffers, lamp flag, press count, flicker RNG, step
counter) can be read and written exactly, which gives bit-exact resets and resumable sessions.
"""
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from duck import BodyConfig, Duck

HERE = Path(__file__).resolve().parent
ROBOT_XML = HERE / "robot" / "robot_allcollisions.xml"
MUJOCO_VERSION = "3.16.0"
FROZEN_BODY = dict(scene="allcollisions", phys_delay=4, jv_lag=1)  # = m0_gate.FROZEN_BODY
PAD_COLOURS = ("red", "green", "blue", "yellow")
FLICKER_PERIOD = 5          # control steps (10 Hz) between distractor colour changes
BUTTON_RELEASE_FRAC = 0.5   # "released" = plate above this fraction of the press depth
BUTTON_REARM_STEPS = 50     # ...continuously for 1 s before the next press counts (one visit = one press;
                            # without it every footstep on the plate toggled the lamp again)


def config_hash(cfg):
    """Stable short hash of a config (ignores its own 'hash' field)."""
    c = {k: v for k, v in cfg.items() if k != "hash"}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def _wall_frame(room, wall, offset, inset):
    """Point on the inner face of a wall ('N','S','E','W'), `offset` along it, and the inward yaw."""
    hx, hy = room["size"][0] / 2, room["size"][1] / 2
    t = room["wall_thickness"] / 2
    if wall == "N":
        return np.array([offset, hy - inset - t]), -np.pi / 2
    if wall == "S":
        return np.array([offset, -hy + inset + t]), np.pi / 2
    if wall == "E":
        return np.array([hx - inset - t, offset]), np.pi
    return np.array([-hx + inset + t, offset]), 0.0


def _quat_z(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def build_spec(cfg):
    """Room + robot as an MjSpec (not yet compiled)."""
    if cfg["mujoco_version"] != mujoco.__version__:
        raise RuntimeError(f"config pinned to MuJoCo {cfg['mujoco_version']}, running {mujoco.__version__}")
    lay, app = cfg["layout"], cfg["appearance"]
    room = lay["room"]
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.option.timestep = 0.005
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    wb = spec.worldbody
    G = mujoco.mjtGeom

    # --- appearance-only assets -------------------------------------------------------------
    fl = app["floor"]
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=getattr(mujoco.mjtBuiltin, f"mjBUILTIN_{fl['builtin'].upper()}"),
                     mark=getattr(mujoco.mjtMark, f"mjMARK_{fl['mark'].upper()}"),
                     rgb1=fl["rgb1"], rgb2=fl["rgb2"], markrgb=fl["markrgb"], random=fl.get("random", 0.0),
                     width=256, height=256)
    mat = spec.add_material(name="floor_mat", texrepeat=[fl["texrepeat"]] * 2, texuniform=True,
                            reflectance=fl["reflectance"])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    sky = app["sky"]
    spec.add_texture(name="sky_tex", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=sky["rgb1"], rgb2=sky["rgb2"],
                     width=128, height=768)
    spec.add_material(name="lamp_mat", emission=0.0)
    li = app["light"]
    wb.add_light(name="room_light", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 3],
                 dir=li["dir"], diffuse=li["diffuse"], ambient=li["ambient"], specular=[0.1] * 3,
                 castshadow=li["castshadow"])
    spec.visual.headlight.ambient = li["headlight_ambient"]
    spec.visual.headlight.diffuse = li["headlight_diffuse"]
    spec.visual.headlight.specular = [0, 0, 0]

    def rgba(rgb):
        return list(rgb) + [1.0]

    # --- floor and walls (physical) -----------------------------------------------------------
    wb.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor_mat")
    hx, hy = room["size"][0] / 2, room["size"][1] / 2
    t, h = room["wall_thickness"] / 2, room["wall_height"] / 2
    for name, pos, size in (("wall_N", [0, hy + t, h], [hx + 2 * t, t, h]),
                            ("wall_S", [0, -hy - t, h], [hx + 2 * t, t, h]),
                            ("wall_E", [hx + t, 0, h], [t, hy, h]),
                            ("wall_W", [-hx - t, 0, h], [t, hy, h])):
        wb.add_geom(name=name, type=G.mjGEOM_BOX, pos=pos, size=size, rgba=rgba(app["wall_rgb"]))

    # --- flat markers: home mat and pads (visual only, 1 mm above the floor) -------------------
    def flat(name, xy, half, rgb):
        wb.add_geom(name=name, type=G.mjGEOM_BOX, pos=[xy[0], xy[1], 0.0005], size=[half, half, 0.0005],
                    rgba=rgba(rgb), contype=0, conaffinity=0, group=1)

    flat("home_mat", lay["home"]["pos"], lay["home"]["size"] / 2, app["home_rgb"])
    for pad in lay["pads"]:
        flat(f"pad_{pad['color']}", pad["pos"], pad["size"] / 2, app["pad_rgb"][pad["color"]])

    # --- partition and occluder (physical boxes) ---------------------------------------------
    if lay["partition"] is not None:
        p = lay["partition"]
        wb.add_geom(name="partition", type=G.mjGEOM_BOX, pos=[*p["pos"], p["height"] / 2],
                    quat=_quat_z(p["yaw"]), size=[p["length"] / 2, p["thickness"] / 2, p["height"] / 2],
                    rgba=rgba(app["partition_rgb"]))
    o = lay["occluder"]
    wb.add_geom(name="occluder", type=G.mjGEOM_BOX, pos=[*o["pos"], o["height"] / 2], quat=_quat_z(o["yaw"]),
                size=[o["length"] / 2, o["thickness"] / 2, o["height"] / 2], rgba=rgba(app["occluder_rgb"]))

    # --- button: a broad plate on a sprung slide joint ---------------------------------------
    b = lay["button"]
    # a dark frame on the floor around the plate so it reads as a button, not a fifth pad
    # (static, visual only: no contact, no mass)
    wb.add_geom(name="button_frame", type=G.mjGEOM_BOX, pos=[*b["pos"], 0.0004],
                size=[b["size"] / 2 + 0.025, b["size"] / 2 + 0.025, 0.0004], rgba=[0.08, 0.08, 0.08, 1],
                contype=0, conaffinity=0, group=1)
    bb = wb.add_body(name="button", pos=[*b["pos"], 0.0])
    bb.add_joint(name="button_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                 range=[-b["travel"], 0.0], stiffness=b["stiffness"], springref=b["preload"],
                 damping=b["damping"])
    plate_half = b["thickness"] / 2
    bb.add_geom(name="button_plate", type=G.mjGEOM_BOX, size=[b["size"] / 2, b["size"] / 2, plate_half],
                pos=[0, 0, b["height"] - plate_half], mass=b["mass"], rgba=rgba(app["button_rgb"]))
    spec.add_exclude(bodyname1="world", bodyname2="button")  # the plate never touches floor/walls

    # --- lamp on a wall (visual only) -------------------------------------------------------
    la = lay["lamp"]
    xy, yaw = _wall_frame(room, la["wall"], la["offset"], inset=0.02)
    wb.add_geom(name="lamp", type=G.mjGEOM_BOX, pos=[*xy, la["height"]], quat=_quat_z(yaw),
                size=[0.02, 0.05, 0.05], rgba=rgba(app["lamp_off_rgb"]), material="lamp_mat",
                contype=0, conaffinity=0, group=1)

    # --- ball (physical, Pollen's floorball) -------------------------------------------------
    ball = wb.add_body(name="ball", pos=[*lay["ball"]["pos"], 0.035])
    ball.add_freejoint(name="ball_free")
    # condim 6 so rolling friction acts at all (Pollen's ball.xml uses the default condim 3, where its
    # 1e-4 rolling friction is ignored and the ball rolls as if on ice); 0.002 = a ball on carpet:
    # launched at 0.45 m/s it stops after ~0.7 m in ~6 s (M2, task 3 needs a ball that comes to rest).
    ball.add_geom(name="ball_geom", type=G.mjGEOM_SPHERE, size=[0.035, 0, 0], mass=0.015,
                  condim=6, friction=[0.5, 0.005, 0.002], rgba=rgba(app["ball_rgb"]))
    # Pollen's ball.xml sets a thin-shell inertia; MjSpec geom mass gives a solid sphere unless overridden.
    ball.mass = 0.015
    ball.inertia = [1.225e-5] * 3
    ball.ipos = [0.0, 0.0, 0.0]   # must be explicit: left unset, the CoM compiled offset by the body
    ball.iquat = [1.0, 0.0, 0.0, 0.0]  # position and gravity spun the ball up (caught by make_splits)
    ball.explicitinertial = True

    # --- distractor screen (visual only) -----------------------------------------------------
    dist = cfg.get("distractor")
    if dist is not None:
        xy, yaw = _wall_frame(room, dist["wall"], dist["offset"], inset=0.006)
        wb.add_geom(name="distractor", type=G.mjGEOM_BOX, pos=[*xy, dist["height"]], quat=_quat_z(yaw),
                    size=[0.005, dist["width"] / 2, dist["tall"] / 2], rgba=[0.5, 0.5, 0.5, 1],
                    contype=0, conaffinity=0, group=1)

    # --- overhead camera for replays (orthographic, whole room) -------------------------------
    wb.add_camera(name="top", pos=[0, 0, 3.0], quat=[1, 0, 0, 0], fovy=max(room["size"]) + 0.3,
                  proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
    return spec


def build_model(cfg):
    model = build_spec(cfg).compile()
    assert model.nu == 14 and abs(model.opt.timestep - 0.005) < 1e-12
    return model


class World:
    """One room, one duck, the room's logic and an exact state."""

    def __init__(self, cfg, body=FROZEN_BODY, policy=None):
        self.cfg = copy.deepcopy(cfg)
        self.model = build_model(self.cfg)
        self.duck = Duck(BodyConfig(**body), policy=policy, seed=int(self.cfg["seed"]), model=self.model)
        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "button_slide")
        self.button_qadr = int(m.jnt_qposadr[jid])
        self.lamp_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "lamp")
        self.lamp_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "lamp_mat")
        self.dist_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "distractor")
        bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qadr, self.ball_vadr = int(m.jnt_qposadr[bj]), int(m.jnt_dofadr[bj])
        self.ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._starts = {}
        self._initial = self._make_initial_state()
        self.reset()

    # ---------------------------------------------------------------------------- state
    def _make_initial_state(self, start=None):
        st = start or self.cfg["layout"]["start"]
        self.duck.reset(x=st["pos"][0], y=st["pos"][1], yaw=st["yaw"])  # mj_resetData + duck pose
        self.lamp_on = bool(self.cfg["layout"]["lamp_initial_on"])
        self.press_count, self.button_down, self.step_count, self.release_steps = 0, False, 0, 0
        self.suppress_presses = 0
        self.flicker_rng = np.random.default_rng([int(self.cfg["seed"]), 7])
        self.flicker_rgb = [0.5, 0.5, 0.5]
        return self.get_state()

    def get_state(self):
        phys, pre = self.duck.physics_state()
        return {
            "config_id": self.cfg["id"], "config_hash": config_hash(self.cfg),
            "mujoco_version": mujoco.__version__, "body": self.duck.cfg.label(), "physics": phys, "physics_pre": pre,
            "duck": self.duck.controller_state(),
            "room": {"lamp_on": self.lamp_on, "press_count": self.press_count,
                     "button_down": self.button_down, "step_count": self.step_count,
                     "release_steps": self.release_steps, "suppress_presses": self.suppress_presses,
                     "flicker_rng": self.flicker_rng.bit_generator.state,
                     "flicker_rgb": list(self.flicker_rgb)},
        }

    def set_state(self, state):
        if state["config_hash"] != config_hash(self.cfg):
            raise ValueError("state belongs to a different config")
        if state["mujoco_version"] != mujoco.__version__:
            raise ValueError("state saved under another MuJoCo version")
        if state["body"] != self.duck.cfg.label():   # e.g. another phys_delay: the buffers would not fit
            raise ValueError(f"state saved for body {state['body']}, this world runs {self.duck.cfg.label()}")
        self.duck.restore_physics(state["physics"], state.get("physics_pre"))
        self.duck.set_controller_state(state["duck"])
        r = state["room"]
        self.lamp_on, self.press_count = bool(r["lamp_on"]), int(r["press_count"])
        self.button_down, self.step_count = bool(r["button_down"]), int(r["step_count"])
        self.release_steps = int(r["release_steps"])
        self.suppress_presses = int(r.get("suppress_presses", 0))
        self.flicker_rng.bit_generator.state = r["flicker_rng"]
        self.flicker_rgb = list(r["flicker_rgb"])
        self._apply_visuals()

    def reset(self, start=None):
        """Back to the config's start, bit for bit. `start` = {"pos": [x, y], "yaw": a} overrides the
        duck's start pose (a task's start condition, e.g. task 3's viewpoint); same lamp/room start."""
        if start is None:
            self.set_state(self._initial)
            return
        key = (tuple(start["pos"]), float(start["yaw"]))
        if key not in self._starts:
            self._starts[key] = self._make_initial_state(start)
        self.set_state(self._starts[key])

    def launch_ball(self, velocity_xy):
        """Give the ball a rolling start (no slip) at the current instant. Written straight into qvel,
        then mj_forward; derived values now belong to the current state, which is exactly what a
        restore with no pre-substep state reproduces, so sessions stay bit-exact (M1 review note)."""
        v = np.array([velocity_xy[0], velocity_xy[1], 0.0])
        w = np.cross([0.0, 0.0, 1.0], v) / 0.035          # rolling: omega = z x v / r
        d = self.duck.data
        d.qvel[self.ball_vadr:self.ball_vadr + 3] = v
        d.qvel[self.ball_vadr + 3:self.ball_vadr + 6] = w
        mujoco.mj_forward(self.model, d)
        self.duck.has_pre_state = False

    # ----------------------------------------------------------------------------- step
    def step(self):
        """One 50 Hz step: the duck's control step, then the button, lamp and distractor logic."""
        obs, action = self.duck.step()
        self.step_count += 1
        b = self.cfg["layout"]["button"]
        depth = -float(self.duck.data.qpos[self.button_qadr])
        if not self.button_down and depth >= b["press_depth"]:
            self.button_down = True
            if self.suppress_presses > 0:       # M4 perturbation: this press doesn't register
                self.suppress_presses -= 1
            else:
                self.press_count += 1
                self.lamp_on = not self.lamp_on
        elif self.button_down:
            self.release_steps = self.release_steps + 1 if depth < BUTTON_RELEASE_FRAC * b["press_depth"] else 0
            if self.release_steps >= BUTTON_REARM_STEPS:
                self.button_down, self.release_steps = False, 0
        if self.dist_geom >= 0 and self.step_count % FLICKER_PERIOD == 0:
            self.flicker_rgb = self.flicker_rng.uniform(0, 1, 3).tolist()
        self._apply_visuals()
        return obs, action

    def _apply_visuals(self):
        app = self.cfg["appearance"]
        rgb = app["lamp_on_rgb"] if self.lamp_on else app["lamp_off_rgb"]
        self.model.geom_rgba[self.lamp_geom, :3] = rgb
        self.model.mat_emission[self.lamp_mat] = app["lamp_on_emission"] if self.lamp_on else 0.0
        if self.dist_geom >= 0:
            self.model.geom_rgba[self.dist_geom, :3] = self.flicker_rgb

    # ---------------------------------------------------------------------------- views
    def button_depth(self):
        return -float(self.duck.data.qpos[self.button_qadr])

    def ball_pos(self):
        return self.duck.data.qpos[self.ball_qadr:self.ball_qadr + 3].copy()
