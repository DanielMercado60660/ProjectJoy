"""The Floor Is Lava: Unitree Go1 (MuJoCo Menagerie) crossing broad stepping stones over a lava floor.

Everything is plain numpy + mujoco. One `LavaEnv` = one robot + one stone layout. The floor plane
is lava: any contact between a robot geom and the floor ends the episode as a failure.

Stones are boxes, each in its own static body, so a new layout only rewrites body_pos / geom_size
(plus the cached bounding volumes) instead of recompiling the XML.

Layout split: training layouts use seeds < HELDOUT_SEED_BASE; held-out layouts use seeds >=
HELDOUT_SEED_BASE. Nothing in training ever calls make_layout with a held-out seed.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(HERE, "..", "vendor", "mujoco_menagerie", "unitree_go1")

HELDOUT_SEED_BASE = 10_000_000
N_STONES = 6                 # stepping stones between start and goal platform
N_SLOTS = N_STONES + 2       # + start platform + goal platform
K_NEXT = 4                   # stones visible in the observation
CTRL_DT = 0.02               # 50 Hz control
SIM_DT = 0.004
N_SUB = int(round(CTRL_DT / SIM_DT))
T_MAX = 20.0                 # seconds per episode
HOME_Q = np.array([0, 0.9, -1.8] * 4, dtype=np.float64)
LEG_NAMES = ["FR", "FL", "RR", "RL"]

# Default layout distribution ("broad stones"): length, width, gap, lateral offset, top height.
LAYOUT_CFG = dict(
    start_len=1.2, goal_len=1.2,
    stone_len=(0.35, 0.55), stone_width=(0.55, 0.80),
    gap=(0.05, 0.20), y_off=0.08, top=(0.08, 0.12), start_top=0.10,
)


def make_layout(seed, cfg=LAYOUT_CFG, flat=False):
    """Return array (N_SLOTS, 5): x_start, x_end, y_center, half_width, top_z for each slot.
    Slot 0 = start platform (robot starts at x=0), slots 1..N_STONES = stones, last = goal.
    flat=True: zero gaps and equal heights (a flat walkway, for gait tuning)."""
    rng = np.random.default_rng(int(seed))
    out = np.zeros((N_SLOTS, 5))
    x = 0.45  # start platform ends 0.45 m in front of the trunk start
    out[0] = [x - cfg["start_len"], x, 0.0, 0.5, cfg["start_top"]]
    for i in range(1, N_STONES + 1):
        g = 0.0 if flat else rng.uniform(*cfg["gap"])
        L = rng.uniform(*cfg["stone_len"])
        W = rng.uniform(*cfg["stone_width"])
        y = 0.0 if flat else rng.uniform(-cfg["y_off"], cfg["y_off"])
        top = cfg["start_top"] if flat else rng.uniform(*cfg["top"])
        out[i] = [x + g, x + g + L, y, W / 2, top]
        x = x + g + L
    g = 0.0 if flat else rng.uniform(*cfg["gap"])
    out[-1] = [x + g, x + g + cfg["goal_len"], 0.0, 0.5, cfg["start_top"]]
    return out


def goal_x(layout):
    """Completion line: trunk 0.35 m past the start of the goal platform (rear feet on it too)."""
    return layout[-1, 0] + 0.35


def build_xml():
    stones = []
    for i in range(N_SLOTS):
        rgba = "0.45 0.42 0.40 1" if 0 < i < N_SLOTS - 1 else "0.35 0.38 0.45 1"
        stones.append(
            f'<body name="stone{i}" pos="{i * 3.0} 5 0">'
            f'<geom name="stone{i}" type="box" size="0.2 0.2 0.05" pos="0 0 0.05" rgba="{rgba}" '
            f'friction="0.8 0.02 0.01" condim="3"/></body>')
    return f"""
<mujoco model="go1 lava">
  <include file="go1.xml"/>
  <option timestep="{SIM_DT}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20" offwidth="960" offheight="540"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.1 0.08" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="lava" builtin="checker" mark="random" random="0.05" rgb1="0.9 0.25 0.02"
      rgb2="0.75 0.12 0.0" markrgb="1 0.8 0.1" width="300" height="300"/>
    <material name="lava" texture="lava" texuniform="true" texrepeat="4 4" emission="0.6"/>
  </asset>
  <worldbody>
    <light pos="2 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="lava"/>
    {''.join(stones)}
    <camera name="side" mode="fixed" pos="2.4 -4.2 1.6" xyaxes="1 0 0 0 0.35 0.94"/>
  </worldbody>
</mujoco>"""


def load_model():
    # go1.xml and its meshes are passed in as in-memory assets (nothing written outside this folder).
    assets = {"go1.xml": open(os.path.join(MENAGERIE, "go1.xml"), "rb").read()}
    for f in os.listdir(os.path.join(MENAGERIE, "assets")):
        assets["assets/" + f] = open(os.path.join(MENAGERIE, "assets", f), "rb").read()
    return mujoco.MjModel.from_xml_string(build_xml(), assets)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class LavaEnv:
    """Single-robot env. Action handling (CPG + residual) lives in the caller via `step_targets`."""

    def __init__(self, model=None):
        self.m = model if model is not None else load_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.stone_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"stone{i}") for i in range(N_SLOTS)]
        self.stone_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"stone{i}") for i in range(N_SLOTS)]
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in LEG_NAMES]
        self.foot_site = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in LEG_NAMES]
        robot_bodies = set()
        for b in range(m.nbody):
            r = b
            while r != 0 and r != self.trunk:
                r = m.body_parentid[r]
            if r == self.trunk:
                robot_bodies.add(b)
        self.robot_geom = np.array([m.geom_bodyid[g] in robot_bodies for g in range(m.ngeom)])
        self.mass = float(sum(m.body_mass[b] for b in robot_bodies))
        self.layout = None

    # ------------------------------------------------------------------ layout
    def set_layout(self, layout):
        m = self.m
        for i in range(N_SLOTS):
            x0, x1, yc, hw, top = layout[i]
            b, g = self.stone_body[i], self.stone_geom[i]
            hx, hz = (x1 - x0) / 2, top / 2
            m.body_pos[b] = [(x0 + x1) / 2, yc, 0.0]
            m.geom_pos[g] = [0, 0, hz]
            m.geom_size[g] = [hx, hw, hz]
            m.geom_rbound[g] = np.sqrt(hx * hx + hw * hw + hz * hz)
            m.geom_aabb[g] = [0, 0, 0, hx, hw, hz]
            m.bvh_aabb[m.body_bvhadr[b]] = [0, 0, hz, hx, hw, hz]  # body-frame box of the single geom
        self.layout = np.array(layout, dtype=np.float64)

    def reset(self, layout, rng=None, init_noise=0.0):
        self.set_layout(layout)
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[2] = 0.27 + layout[0, 4] + 0.005
        if rng is not None and init_noise > 0:
            d.qpos[7:] += rng.uniform(-init_noise, init_noise, 12)
            yaw = rng.uniform(-0.1, 0.1)
            d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qpos[1] += rng.uniform(-0.05, 0.05)
        d.ctrl[:] = HOME_Q
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self.energy = 0.0
        self.x0 = float(d.qpos[0])
        self.lava = False
        self.flip = False
        return self

    # ------------------------------------------------------------------ physics
    def step_targets(self, q_target, push_force=None):
        """Apply joint position targets for one control step (N_SUB physics steps).
        Returns per-step energy (J, sum |tau * qdot| dt) and sets self.lava / self.flip."""
        m, d = self.m, self.d
        d.ctrl[:] = np.clip(q_target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        if push_force is not None:
            d.xfrc_applied[self.trunk, :3] = push_force
        e = 0.0
        for _ in range(N_SUB):
            mujoco.mj_step(m, d)
            e += float(np.abs(d.actuator_force * d.qvel[6:]).sum()) * SIM_DT
            if not self.lava and d.ncon:
                g1 = d.contact.geom1[:d.ncon]
                g2 = d.contact.geom2[:d.ncon]
                hit = ((g1 == self.floor_id) & self.robot_geom[g2]) | ((g2 == self.floor_id) & self.robot_geom[g1])
                if hit.any():
                    self.lava = True
        if push_force is not None:
            d.xfrc_applied[self.trunk, :3] = 0
        self.t += CTRL_DT
        self.energy += e
        R = d.xmat[self.trunk].reshape(3, 3)
        if R[2, 2] < 0.3:  # trunk z-axis tilted more than ~72 degrees: flipped / fallen over
            self.flip = True
        return e

    # ------------------------------------------------------------------ state helpers
    def yaw(self):
        q = self.d.qpos[3:7]
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def stone_index(self, x):
        """Index of the slot whose x-extent contains x (or the next one ahead)."""
        L = self.layout
        idx = np.searchsorted(L[:, 1], x)
        return int(min(idx, N_SLOTS - 1))

    def foot_pos(self):
        return np.array([self.d.site_xpos[s] for s in self.foot_site])

    def foot_contacts(self):
        d = self.d
        c = np.zeros(4)
        if d.ncon:
            g1 = d.contact.geom1[:d.ncon]
            g2 = d.contact.geom2[:d.ncon]
            for i, fg in enumerate(self.foot_geom):
                if (g1 == fg).any() or (g2 == fg).any():
                    c[i] = 1.0
        return c

    def next_stones(self, k=K_NEXT, behind=0.45):
        """Next k slots whose far edge is ahead of (trunk x - behind), in the trunk yaw frame.
        Features per slot: x_start, x_end, y_center (relative), half width, top z - trunk z."""
        d = self.d
        x, y, z = d.qpos[0], d.qpos[1], d.qpos[2]
        yaw = self.yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        L = self.layout
        i0 = int(np.searchsorted(L[:, 1], x - behind))
        feats = np.zeros((k, 5))
        for j in range(k):
            i = min(i0 + j, N_SLOTS - 1)
            x0, x1, yc, hw, top = L[i]
            # rotate (dx, dy) into yaw frame for both ends at stone centre line
            dx0, dx1, dy = x0 - x, x1 - x, yc - y
            feats[j] = [c * dx0 + s * dy, c * dx1 + s * dy, -s * ((dx0 + dx1) / 2) + c * dy, hw, top - z]
        return feats

    def progress(self):
        return float(self.d.qpos[0] - self.x0)
