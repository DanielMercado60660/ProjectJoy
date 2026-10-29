"""The command interface every mind speaks (frozen in M2; see results/m2/interface_frozen.json).

One tick = 10 Hz. A controller sends a Command; the interface clips it to the bounds below, holds it
for 5 policy steps (50 Hz World.step), and returns an Observation. Nothing else may drive the duck:
skills, the reference bot, learned policies and the oracle all go through Interface.tick().

Command
  twist   vx in [-0.3, 0.4] m/s, vy in [0, 0] (slot kept; lateral steps were measured unusable in M0),
          wz in [-0.6, 0.6] rad/s
  head    neck_pitch, head_pitch, head_yaw, head_roll deltas from the default pose, each within 0.8 x
          its training cap (0.88, 0.88, 1.12, 0.248 rad)
  done    the controller's "I'm finished" bit: the episode ends at the first tick with done=True
          (or when the step budget runs out). A learned policy outputs it like any other field.
The interface does NOT remap commands into what the body can do. The body's quirks (measured in M0
and recorded in the freeze file as notes): forward 0.1-0.15 m/s doesn't start a gait from standing,
in-place turns need |wz| >= 0.5, achieved speed is ~0.44x forward and ~1.4x turning, and it curves
left ~0.22 rad/s when commanded straight. Skills deal with these by closing the loop.

Observation (what a controller may use; ground truth such as the lamp flag, press count or ball
position is NOT in it — task judges read those from the World directly)
  frame()        128x128 RGB from the head camera, rendered on demand at the tick's sim time
  body           gyro (3), projected gravity (3), joint positions minus the default pose (14),
                 joint velocities (14), as the walking policy sees them (without its lags);
                 plus the IMU's 50 Hz samples during the tick: gyro_substeps (5x3) and
                 gravity_substeps (5x3), oldest first (the last row equals gyro / gravity).
                 Added in the M2 review: one gyro sample per 10 Hz tick aliases the gait's yaw
                 wobble, and integrating it drifted 0.2-1.3 rad in 60 s; the 50 Hz samples
                 integrate to ~0.1 rad (results/m2/review_checks.json). Odometry from "commands
                 and IMU" (M3) needs them.
  last_command, tick, sim_time
"""
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from duck import DEFAULT_POSE, HEAD_CAPS

TICK_STEPS = 5                     # 50 Hz policy steps per 10 Hz command tick
TICK_DT = 0.1
FRAME_SIZE = 128
BOUNDS = {
    "vx": (-0.3, 0.4),
    "vy": (0.0, 0.0),
    "wz": (-0.6, 0.6),
    "head": tuple(round(float(c) * 0.8, 4) for c in HEAD_CAPS),   # symmetric, per joint
}
HEAD_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    head: tuple = (0.0, 0.0, 0.0, 0.0)
    done: bool = False

    def clipped(self):
        h = np.clip(np.asarray(self.head, np.float64), -np.asarray(BOUNDS["head"]), BOUNDS["head"])
        return Command(vx=float(np.clip(self.vx, *BOUNDS["vx"])), vy=float(np.clip(self.vy, *BOUNDS["vy"])),
                       wz=float(np.clip(self.wz, *BOUNDS["wz"])), head=tuple(float(x) for x in h),
                       done=bool(self.done))

    def as_vector(self):
        """[vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll, done] — the M5 policy's output."""
        return np.array([self.vx, self.vy, self.wz, *self.head, float(self.done)], np.float32)

    def to_dict(self):
        d = asdict(self)
        d["head"] = list(d["head"])
        return d


@dataclass
class Observation:
    tick: int
    sim_time: float
    body: dict
    last_command: Command
    _iface: object = field(repr=False, default=None)
    _frame: np.ndarray | None = field(repr=False, default=None)

    def frame(self):
        """Head-camera RGB at this tick (rendered once, on first use)."""
        if self._frame is None:
            if self._iface.tick_count != self.tick:
                raise RuntimeError("frame() requested after the world moved on")
            self._frame = self._iface.render_frame()
        return self._frame

    def head_joints(self):
        """Achieved head joint deltas from the default pose (encoders), in HEAD_NAMES order."""
        return np.asarray(self.body["joint_pos"][5:9])


class Interface:
    def __init__(self, world):
        self.world = world
        self.tick_count = 0
        self.last_command = Command()
        self.done = False
        self._renderer = None
        self.log = []          # per tick: the clipped command actually applied
        self._imu = None       # (gyro, gravity) at each policy step of the last tick

    def render_frame(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.world.model, FRAME_SIZE, FRAME_SIZE)
        return self.world.duck.head_image(self._renderer)

    def observe(self):
        d = self.world.duck
        dd = d.data
        gyro, grav = self._imu_sample()
        imu = self._imu if self._imu is not None else [(gyro, grav)] * TICK_STEPS   # tick 0: no history
        body = {
            "gyro": gyro,
            "gravity": grav,
            "joint_pos": (dd.qpos[d.jq] - DEFAULT_POSE).copy(),
            "joint_vel": dd.qvel[d.jv].copy(),
            "gyro_substeps": np.array([g for g, _ in imu]),
            "gravity_substeps": np.array([v for _, v in imu]),
        }
        return Observation(tick=self.tick_count, sim_time=float(dd.time), body=body,
                           last_command=self.last_command, _iface=self)

    def tick(self, cmd: Command):
        """Apply one command for one 10 Hz tick and return the next observation."""
        c = cmd.clipped()
        self.world.duck.set_command(c.vx, c.vy, c.wz, head=c.head)
        imu = []
        for _ in range(TICK_STEPS):
            self.world.step()
            imu.append(self._imu_sample())
        self._imu = imu
        self.tick_count += 1
        self.last_command = c
        self.done = self.done or c.done
        self.log.append(c)
        return self.observe()

    def _imu_sample(self):
        d = self.world.duck
        return (d.data.sensordata[d.gyro_adr:d.gyro_adr + 3].copy(),
                d.projected_gravity().astype(np.float64))

    def frozen_spec(self):
        return frozen_spec()


def frozen_spec():
    return {
        "tick_hz": 10, "policy_steps_per_tick": TICK_STEPS, "frame_size": FRAME_SIZE,
        "command_fields": ["vx", "vy", "wz", *HEAD_NAMES, "done"],
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "head_bounds_names": list(HEAD_NAMES),
        "observation": {"frame": f"{FRAME_SIZE}x{FRAME_SIZE}x3 uint8, head camera, upright",
                        "body": ["gyro(3)", "gravity(3)", "joint_pos minus default(14)", "joint_vel(14)",
                                 "gyro_substeps(5x3), the tick's 50 Hz IMU samples, oldest first",
                                 "gravity_substeps(5x3), likewise"],
                        "also": ["last_command", "tick", "sim_time"],
                        "not_observable": ["lamp flag", "press count", "ball position", "true pose",
                                           "button depth"]},
        "episode_end": "first tick with done=True, or the task's step budget",
        "sessions": "a session boundary falls between skills/ticks; skill-internal state is not saved",
        "body_notes": [
            "forward 0.1-0.15 m/s does not start a gait from standing; 0.2+ does",
            "turning in place needs |wz| >= 0.5 from standing",
            "achieved/commanded: forward ~0.44, backward ~0.52, turn ~1.4 (results/m0/gate.json)",
            "commanded straight, the duck curves left ~+0.22 rad/s (median)",
            "neck_pitch barely responds; head_pitch sits ~0.1-0.17 rad off; head_yaw tracks",
            "turning while standing on the button's edge can re-press it",
        ],
    }
