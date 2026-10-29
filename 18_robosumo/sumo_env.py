"""RoboSumo arena in MuJoCo: two differential-drive wheeled robots in a ring.

Each robot is a box chassis on two velocity-controlled wheels plus two frictionless casters.
Pushing, tipping and traction come from MuJoCo contacts. The ring edge is a rule: a robot whose
chassis centre leaves the ring (xy radius > R) loses. A robot whose chassis tips past ~70 degrees
(up-vector z < 0.35) also loses ("tip"). Both at once, or timeout, is a draw.

Actions are 2 numbers in [-1, 1]: left and right wheel speed commands.
Observations are egocentric (each robot sees the world in its own frame), so one policy can play
either side.
"""
import numpy as np
import mujoco

DT = 0.005          # physics step (s)
FRAME_SKIP = 8      # control at 25 Hz
MAX_STEPS = 250     # 10 s bouts
WHEEL_MAX = 20.0    # rad/s -> 1.0 m/s at wheel radius 0.05
OBS_DIM = 15

ROBOT_XML = """
<body name="{n}" pos="{x} 0 0.06">
  <freejoint name="{n}_free"/>
  <geom name="{n}_chassis" type="box" size="0.14 0.10 0.035" mass="2.0" rgba="{rgba}" friction="0.6 0.005 0.0001"/>
  <geom name="{n}_nose" type="box" size="0.02 0.10 0.02" pos="0.15 0 -0.01" mass="0.1" rgba="0.15 0.15 0.15 1"/>
  <geom name="{n}_castf" type="sphere" size="0.022" pos="0.10 0 -0.035" mass="0.05" condim="1" friction="0 0 0" rgba="0.3 0.3 0.3 1"/>
  <geom name="{n}_castb" type="sphere" size="0.022" pos="-0.10 0 -0.035" mass="0.05" condim="1" friction="0 0 0" rgba="0.3 0.3 0.3 1"/>
  <body name="{n}_wl" pos="0 0.125 -0.01">
    <joint name="{n}_jl" type="hinge" axis="0 1 0" damping="0.001" armature="0.0005"/>
    <geom type="cylinder" size="0.05 0.015" euler="90 0 0" mass="0.2" friction="1.0 0.005 0.0001" rgba="0.1 0.1 0.1 1"/>
  </body>
  <body name="{n}_wr" pos="0 -0.125 -0.01">
    <joint name="{n}_jr" type="hinge" axis="0 1 0" damping="0.001" armature="0.0005"/>
    <geom type="cylinder" size="0.05 0.015" euler="90 0 0" mass="0.2" friction="1.0 0.005 0.0001" rgba="0.1 0.1 0.1 1"/>
  </body>
</body>
"""

XML = """
<mujoco model="robosumo">
  <option timestep="{dt}" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="1024"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.78 0.70" rgb2="0.78 0.74 0.66" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="floor" friction="1.0 0.005 0.0001"/>
    <geom name="ring" type="cylinder" size="1.0 0.001" pos="0 0 0.0005" rgba="0.55 0.35 0.2 1" contype="0" conaffinity="0"/>
    <geom name="ring_in" type="cylinder" size="0.97 0.0015" pos="0 0 0.0006" rgba="0.92 0.88 0.78 1" contype="0" conaffinity="0"/>
    {a}
    {b}
  </worldbody>
  <actuator>
    <velocity joint="a_jl" kv="0.08" ctrlrange="-{wm} {wm}" forcerange="-0.8 0.8"/>
    <velocity joint="a_jr" kv="0.08" ctrlrange="-{wm} {wm}" forcerange="-0.8 0.8"/>
    <velocity joint="b_jl" kv="0.08" ctrlrange="-{wm} {wm}" forcerange="-0.8 0.8"/>
    <velocity joint="b_jr" kv="0.08" ctrlrange="-{wm} {wm}" forcerange="-0.8 0.8"/>
  </actuator>
</mujoco>
"""


def build_model():
    xml = XML.format(dt=DT, wm=WHEEL_MAX,
                     a=ROBOT_XML.format(n="a", x=-0.4, rgba="0.85 0.25 0.2 1"),
                     b=ROBOT_XML.format(n="b", x=0.4, rgba="0.2 0.4 0.85 1"))
    return mujoco.MjModel.from_xml_string(xml)


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_upz(q):
    w, x, y, z = q
    return 1 - 2 * (x * x + y * y)


def yaw_quat(th):
    return np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])


class SumoEnv:
    """One bout. Train-time randomization: ring radius in r_range, floor/wheel friction in f_range."""

    def __init__(self, seed=0, r_range=(0.9, 1.1), f_range=(0.8, 1.2)):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.r_range, self.f_range = r_range, f_range
        self.floor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.ring_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ring")
        self.ringin_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ring_in")
        self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n + "_free")] for n in "ab"]
        self.vadr = [self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n + "_free")] for n in "ab"]
        self.base_friction = self.m.geom_friction.copy()

    def set_conditions(self, R, fric):
        self.R = R
        self.fric = fric
        self.m.geom_size[self.ring_id][0] = R
        self.m.geom_size[self.ringin_id][0] = R - 0.03
        # Scale sliding friction of floor and wheels (MuJoCo uses max of the pair).
        self.m.geom_friction[:, 0] = self.base_friction[:, 0] * fric

    def reset(self, R=None, fric=None):
        rng = self.rng
        R = R if R is not None else rng.uniform(*self.r_range)
        fric = fric if fric is not None else rng.uniform(*self.f_range)
        self.set_conditions(R, fric)
        mujoco.mj_resetData(self.m, self.d)
        # Starts: opposite sides at ~0.4R from centre, random ring angle, headings near facing.
        phi = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(0.3, 0.5) * R
        for i, sgn in enumerate((-1, 1)):
            p = sgn * dist * np.array([np.cos(phi), np.sin(phi)])
            face = np.arctan2(-p[1], -p[0]) + rng.uniform(-0.8, 0.8)
            a = self.qadr[i]
            self.d.qpos[a:a + 3] = [p[0], p[1], 0.06]
            self.d.qpos[a + 3:a + 7] = yaw_quat(face)
        mujoco.mj_forward(self.m, self.d)
        self.t = 0
        return self.obs()

    def state(self, i):
        a, v = self.qadr[i], self.vadr[i]
        q = self.d.qpos[a + 3:a + 7]
        return (self.d.qpos[a:a + 2].copy(), quat_yaw(q), quat_upz(q),
                self.d.qvel[v:v + 2].copy(), self.d.qvel[v + 5])

    def obs_for(self, i):
        p, th, up, v, w = self.state(i)
        po, tho, upo, vo, wo = self.state(1 - i)
        c, s = np.cos(th), np.sin(th)
        rot = lambda x: np.array([c * x[0] + s * x[1], -s * x[0] + c * x[1]])
        R = self.R
        rel = rot(po - p)
        dist = np.linalg.norm(rel) + 1e-6
        ctr = rot(-p)                       # ring centre in own frame
        return np.array([
            *rot(v), w * 0.2,
            *ctr, R - np.linalg.norm(p),     # own edge margin (m)
            *rel, dist,
            *(rot(vo - v)),
            np.cos(tho - th), np.sin(tho - th),
            R - np.linalg.norm(po),          # opponent edge margin
            up,
        ], dtype=np.float32)

    def obs(self):
        return self.obs_for(0), self.obs_for(1)

    def step(self, act_a, act_b):
        """Returns obs pair, outcome (None while running; +1 a wins, -1 b wins, 0 draw), info."""
        ctrl = np.clip(np.concatenate([act_a, act_b]), -1, 1) * WHEEL_MAX
        self.d.ctrl[:] = ctrl
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        (pa, _, upa, _, _), (pb, _, upb, _, _) = self.state(0), self.state(1)
        out_a = np.linalg.norm(pa) > self.R
        out_b = np.linalg.norm(pb) > self.R
        tip_a, tip_b = upa < 0.35, upb < 0.35
        lose_a, lose_b = out_a or tip_a, out_b or tip_b
        info = dict(out_a=out_a, out_b=out_b, tip_a=tip_a, tip_b=tip_b, t=self.t)
        if lose_a and lose_b:
            return self.obs(), 0, info
        if lose_a:
            return self.obs(), -1, info
        if lose_b:
            return self.obs(), 1, info
        if self.t >= MAX_STEPS:
            info["timeout"] = True
            return self.obs(), 0, info
        return self.obs(), None, info


# ---------------------------------------------------------------- scripted opponents
# All scripts read the same egocentric observation the learner sees (matched information).
# Obs indices: 0-1 own vel, 2 yaw rate, 3-4 centre (own frame), 5 own edge margin,
# 6-7 opponent rel pos, 8 dist, 9-10 rel vel, 11-12 rel heading, 13 opp edge margin, 14 up.

def _drive(fwd, turn):
    return np.clip(np.array([fwd - turn, fwd + turn]), -1, 1)


def angle_of(v):
    return np.arctan2(v[1], v[0])


class Charger:
    """Turn toward the opponent and ram at full speed. Params: steering gain, speed, slow-turn angle."""
    name = "charger"

    def __init__(self, k=2.0, speed=1.0, turn_slow=1.0):
        self.k, self.speed, self.turn_slow = k, speed, turn_slow

    def __call__(self, o):
        ang = angle_of(o[6:8])
        fwd = self.speed * (1.0 if abs(ang) < self.turn_slow else 0.2)
        return _drive(fwd, self.k * ang)


class Flanker:
    """Get between the opponent and the ring centre, then push outward along the centre->opponent line.
    Params: standoff behind the opponent (m), gain, push-switch distance."""
    name = "flanker"

    def __init__(self, standoff=0.35, k=2.5, push_dist=0.45, speed=1.0):
        self.standoff, self.k, self.push_dist, self.speed = standoff, k, push_dist, speed

    def __call__(self, o):
        ctr, rel = o[3:5], o[6:8]
        opp_from_ctr = rel - ctr
        n = opp_from_ctr / (np.linalg.norm(opp_from_ctr) + 1e-6)
        target = rel - n * self.standoff          # point on centre side of opponent
        # Are we already roughly behind (centre side of) the opponent?
        me_from_opp = -rel
        behind = np.dot(me_from_opp / (np.linalg.norm(me_from_opp) + 1e-6), -n) > 0.6
        if behind or np.linalg.norm(rel) < self.push_dist:
            target = rel
        ang = angle_of(target)
        fwd = self.speed * (1.0 if abs(ang) < 1.2 else 0.15)
        return _drive(fwd, self.k * ang)


class Circler:
    """Orbit the opponent at a radius, tangentially, and cut in when its side is exposed.
    Params: orbit radius, side-exposure threshold, gain."""
    name = "circler"

    def __init__(self, orbit=0.45, expose=0.6, k=2.5):
        self.orbit, self.expose, self.k = orbit, expose, k

    def __call__(self, o):
        rel, dist = o[6:8], o[8]
        # Opponent's heading in my frame: (cos, sin) = o[11:13]. Its side is exposed when its
        # heading is not pointing at me.
        opp_heading = o[11:13]
        to_me = -rel / (dist + 1e-6)
        facing_me = float(np.dot(opp_heading, to_me))
        if facing_me < self.expose or dist < 0.3:
            target = rel
        else:
            tang = np.array([-rel[1], rel[0]]) / (dist + 1e-6)
            target = rel + tang * self.orbit
        # Never drive out: blend toward centre when near the edge.
        if o[5] < 0.25:
            target = target + o[3:5] * 2
        ang = angle_of(target)
        fwd = 1.0 if abs(ang) < 1.2 else 0.2
        return _drive(fwd, self.k * ang)


class CounterPusher:
    """HELD OUT (never trained against). Hold the centre facing the opponent; when it charges, sidestep
    by turning away, then push it from the side. Params: hold radius, sidestep trigger closing speed, gain."""
    name = "counter"

    def __init__(self, hold=0.25, trigger=0.4, k=3.0, dodge_time=6):
        self.hold, self.trigger, self.k, self.dodge_time = hold, trigger, k, dodge_time
        self.dodge = 0
        self.side = 1.0

    def __call__(self, o):
        ctr, rel, dist = o[3:5], o[6:8], o[8]
        closing = -float(np.dot(o[9:11], rel / (dist + 1e-6)))     # opponent approach speed
        if self.dodge > 0:
            self.dodge -= 1
            return _drive(0.6, self.side * 1.0)
        if dist < 0.55 and closing > self.trigger and self.dodge == 0:
            self.dodge = self.dodge_time
            self.side = 1.0 if rel[1] < 0 else -1.0
            return _drive(0.6, self.side)
        if dist < 0.4 or np.linalg.norm(ctr) < self.hold:
            target = rel                      # push when close or centred
        else:
            target = ctr                      # return to centre
        ang = angle_of(target)
        fwd = 1.0 if abs(ang) < 1.0 else 0.0
        return _drive(fwd, self.k * ang)

    def reset(self):
        self.dodge = 0


class Idle:
    name = "idle"

    def __call__(self, o):
        return np.zeros(2)


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, o):
        return self.rng.uniform(-1, 1, 2)


SCRIPT_CLASSES = {"charger": Charger, "flanker": Flanker, "circler": Circler, "counter": CounterPusher}
TRAIN_SCRIPTS = ["charger", "flanker", "circler"]
HELDOUT_SCRIPTS = ["counter"]


def make_script(name, params=None):
    return SCRIPT_CLASSES[name](**(params or {}))


def play_bout(env, pol_a, pol_b, R=None, fric=None, record=False):
    """Run one bout. pol_* are callables obs->action. Returns dict with outcome and stats."""
    oa, ob = env.reset(R, fric)
    for p in (pol_a, pol_b):
        if hasattr(p, "reset"):
            p.reset()
    frames = []
    contact_steps = 0
    while True:
        (oa, ob), out, info = env.step(pol_a(oa), pol_b(ob))
        if oa[8] < 0.32:
            contact_steps += 1
        if record:
            frames.append(env.d.qpos.copy())
        if out is not None:
            return dict(outcome=out, steps=info["t"], tip_a=bool(info["tip_a"]), tip_b=bool(info["tip_b"]),
                        out_a=bool(info["out_a"]), out_b=bool(info["out_b"]),
                        timeout=bool(info.get("timeout", False)), contact_frac=contact_steps / info["t"],
                        frames=frames)
