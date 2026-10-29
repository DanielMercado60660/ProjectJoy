"""The Microduck body: plain CPU MuJoCo + Pollen's velstand walking policy at 50 Hz.

Rebuilt from vendor/microduck_rl/scripts/infer_policy.py (read, not imported) with its
`--no-bam --new-cmd-obs` path: XML position actuators, projected gravity, action_scale
1.0, 5 ms physics step, decimation 4, and the 61-dim observation

    [ gyro(3) | projected gravity(3) | q - DEFAULT_POSE (14) | qdot (14) | last action (14)
      | twist vx, vy, wz (3) | head-pose deltas neck_pitch, head_pitch, head_yaw, head_roll (4)
      | body pose x, y, z, roll, pitch, yaw (6) ]

Options the script has: an action-delay buffer at policy rate (`--delay MIN MAX`), a foot
friction override (`--foot-friction`) and foot contact softness (`--foot-solref`). Its
`--current-limit` needs the motor constant kt from the `bam` package, which is not installed
here; `torque_limit` below caps the actuator force directly instead (XML default 0.96 N*m).
Extra knobs that mirror training rather than the script (all off by default):
`phys_delay` (actuator lag in 5 ms physics steps; training's BAM has delay_min/max_lag 3-6,
read here as physics steps since the vendored code does not state the unit),
`jv_lag` (joint velocity seen one policy step late, as in training) and `kp_scale`.
"""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from policy.velstand_np import DEFAULT_POSE, VelstandPolicy

HERE = Path(__file__).resolve().parent
SCENES = {
    # velstand was trained on the full-collision model (robot_allcollisions.xml).
    "allcollisions": HERE / "robot" / "scene_allcollisions.xml",
    # infer_policy.py's default scene (11 curated collision geoms).
    "groundcontact": HERE / "robot" / "scene.xml",
}
PHYS_DT = 0.005
DECIMATION = 4
CTRL_DT = PHYS_DT * DECIMATION  # 50 Hz
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
HEAD = slice(5, 9)
# Head-pose command caps from infer_policy.py (final training caps per joint).
HEAD_CAPS = np.array([1.1, 1.1, 1.4, 0.31], dtype=np.float32)
# Training's uniform actor-obs noise half-widths (microduck_velocity_env_cfg.py):
# gyro 0.03, projected gravity 0.01, joint pos 0.001, joint vel 0.25.
OBS_NOISE = np.concatenate([np.full(3, 0.03), np.full(3, 0.01), np.full(14, 0.001),
                            np.full(14, 0.25)]).astype(np.float32)


@dataclass
class BodyConfig:
    scene: str = "allcollisions"
    delay_min: int = 0          # policy-rate action delay (infer_policy.py --delay), in 20 ms steps
    delay_max: int = 0
    phys_delay: int = 0         # actuator lag in 5 ms physics steps (training: BAM lag 3-6, unit assumed)
    jv_lag: int = 0             # joint-velocity observation lag in policy steps (training: 1)
    kp_scale: float = 1.0
    torque_limit: float | None = None   # N*m; None keeps the XML forcerange (0.96)
    foot_friction: float | None = None  # infer_policy.py --foot-friction (no-op below the floor's 1.0:
                                        # equal contact priority, MuJoCo takes the larger mu)
    foot_solref: float | None = None    # infer_policy.py --foot-solref
    obs_noise: float = 0.0      # scale on training's uniform obs noise (1.0 = as trained)
    extra: dict = field(default_factory=dict)

    def label(self):
        bits = [self.scene]
        if self.delay_max:
            bits.append(f"delay{self.delay_min}-{self.delay_max}")
        if self.phys_delay:
            bits.append(f"phys{self.phys_delay}")
        if self.jv_lag:
            bits.append(f"jvlag{self.jv_lag}")
        if self.kp_scale != 1.0:
            bits.append(f"kp{self.kp_scale:g}")
        if self.torque_limit is not None:
            bits.append(f"tau{self.torque_limit:g}")
        if self.foot_friction is not None:
            bits.append(f"mu{self.foot_friction:g}")
        if self.foot_solref is not None:
            bits.append(f"solref{self.foot_solref:g}")
        if self.obs_noise:
            bits.append(f"noise{self.obs_noise:g}")
        return "_".join(bits)


def quat_rotate_inverse(q, v):
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Duck:
    def __init__(self, cfg: BodyConfig | None = None, policy: VelstandPolicy | None = None, seed=0,
                 model: mujoco.MjModel | None = None):
        """`model`: a prebuilt model that contains the robot (e.g. a room from room.py);
        otherwise cfg.scene is loaded. Either way the 5 ms physics step is enforced."""
        self.cfg = cfg or BodyConfig()
        self.policy = policy or VelstandPolicy()
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(SCENES[self.cfg.scene]))
        self.model.opt.timestep = PHYS_DT
        self._apply_body_options()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qadr = int(m.jnt_qposadr[jid])
        self.vadr = int(m.jnt_dofadr[jid])
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = int(m.sensor_adr[sid])
        self.jq = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.jv = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        assert m.nu == 14
        self.state_size = mujoco.mj_stateSize(m, STATE_SPEC)
        self.pre_state = np.zeros(self.state_size, np.float64)
        self.has_pre_state = False
        self._fix_head_camera()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _fix_head_camera(self):
        """The MJCF head_camera looks backwards (-x while the duck walks +x). Pollen's own
        simulator (src/mjlab_microduck/sim/camera.py) turns it 180 deg about the camera's right
        axis; the frame then comes out on its side, like the real camera (mounted a quarter turn
        off), so head_image() rotates it upright."""
        m = self.model
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, m.cam_quat[self.cam_id], np.array([0.0, 1.0, 0.0, 0.0]))
        m.cam_quat[self.cam_id] = fixed

    def head_image(self, renderer: mujoco.Renderer):
        """Upright RGB frame from the head camera (renderer must be square for the rotation)."""
        renderer.update_scene(self.data, camera=self.cam_id)
        return np.ascontiguousarray(np.rot90(renderer.render(), -1))

    def _apply_body_options(self):
        m, c = self.model, self.cfg
        if c.kp_scale != 1.0:
            kp = m.actuator_gainprm[:, 0] * c.kp_scale
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
        if c.torque_limit is not None:
            m.actuator_forcerange[:, 0] = -c.torque_limit
            m.actuator_forcerange[:, 1] = c.torque_limit
            m.actuator_forcelimited[:] = 1
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name in ("left_foot_collision", "right_foot_collision"):
                if c.foot_friction is not None:
                    m.geom_friction[g, 0] = c.foot_friction
                if c.foot_solref is not None:
                    m.geom_solref[g] = (c.foot_solref, 1.0)

    # ------------------------------------------------------------------ episode
    def reset(self, x=0.0, y=0.0, yaw=0.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr:self.qadr + 3] = (x, y, 0.125)
        d.qpos[self.qadr + 3:self.qadr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        d.qpos[self.jq] = DEFAULT_POSE
        d.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        self._reset_controller()

    def _reset_controller(self):
        self.last_action = np.zeros(14, np.float32)
        self.command = np.zeros(13, np.float32)
        lag = int(self.rng.integers(self.cfg.delay_min, self.cfg.delay_max + 1)) if self.cfg.delay_max else 0
        self.lag = lag
        self.action_buf = deque([np.zeros(14, np.float32)] * (lag + 1), maxlen=lag + 1)
        self.target_queue = deque([DEFAULT_POSE.copy()] * (self.cfg.phys_delay + 1),
                                  maxlen=self.cfg.phys_delay + 1)
        self.jv_hist = deque([np.zeros(14, np.float32)] * (self.cfg.jv_lag + 1),
                             maxlen=self.cfg.jv_lag + 1)
        self.push_force = np.zeros(3)
        self.steps = 0
        self.has_pre_state = False

    def controller_state(self):
        """Everything outside MjData that the next step depends on (for bit-exact save/restore)."""
        return {
            "last_action": self.last_action.tolist(), "command": self.command.tolist(),
            "lag": self.lag, "action_buf": [a.tolist() for a in self.action_buf],
            "target_queue": [t.tolist() for t in self.target_queue],
            "jv_hist": [j.tolist() for j in self.jv_hist], "push_force": self.push_force.tolist(),
            "steps": self.steps, "rng": self.rng.bit_generator.state,
        }

    def set_controller_state(self, s):
        f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731  (float32 round-trips exactly via JSON floats)
        self.last_action, self.command, self.lag = f32(s["last_action"]), f32(s["command"]), int(s["lag"])
        self.action_buf = deque([f32(a) for a in s["action_buf"]], maxlen=self.lag + 1)
        self.target_queue = deque([f32(t) for t in s["target_queue"]], maxlen=self.cfg.phys_delay + 1)
        self.jv_hist = deque([f32(j) for j in s["jv_hist"]], maxlen=self.cfg.jv_lag + 1)
        self.push_force = np.asarray(s["push_force"], np.float64)
        self.steps = int(s["steps"])
        self.rng.bit_generator.state = s["rng"]

    def set_command(self, vx=0.0, vy=0.0, wz=0.0, head=(0, 0, 0, 0), body=(0, 0, 0, 0, 0, 0)):
        head = np.clip(np.asarray(head, np.float32), -HEAD_CAPS, HEAD_CAPS)
        self.command = np.concatenate([[vx, vy, wz], head, body]).astype(np.float32)

    # ------------------------------------------------------------- observation
    def projected_gravity(self):
        q = self.data.xquat[self.trunk_id].astype(np.float32)
        return quat_rotate_inverse(q, np.array([0, 0, -1], np.float32))

    def observation(self):
        d = self.data
        jv_now = d.qvel[self.jv].astype(np.float32)
        self.jv_hist.append(jv_now)
        jv_obs = self.jv_hist[0]  # oldest = lagged by jv_lag policy steps
        obs = np.concatenate([
            d.sensordata[self.gyro_adr:self.gyro_adr + 3].astype(np.float32),
            self.projected_gravity(),
            d.qpos[self.jq].astype(np.float32) - DEFAULT_POSE,
            jv_obs,
            self.last_action,
            self.command,
        ]).astype(np.float32)
        if self.cfg.obs_noise:
            obs[:34] += self.cfg.obs_noise * self.rng.uniform(-1, 1, 34).astype(np.float32) * OBS_NOISE
        return obs

    # -------------------------------------------------------------------- step
    def step(self):
        """One 50 Hz control step: observe, infer, buffer, 4 physics steps."""
        obs = self.observation()
        action = self.policy(obs).astype(np.float32)
        self.last_action = action.copy()
        self.action_buf.append(action)
        applied = self.action_buf[0]  # lagged by self.lag policy steps (0 = none)
        target = DEFAULT_POSE + applied
        d = self.data
        for i in range(DECIMATION):
            self.target_queue.append(target)
            d.ctrl[:] = self.target_queue[0]
            d.xfrc_applied[self.trunk_id, :3] = self.push_force
            if i == DECIMATION - 1:
                # The next observation reads sensordata/xquat that mj_step computed for the state
                # *before* this last substep (as infer_policy.py does). Keep that state so a saved
                # session can rebuild those derived values exactly (see restore_physics).
                mujoco.mj_getState(self.model, d, self.pre_state, STATE_SPEC)
                self.has_pre_state = True
            mujoco.mj_step(self.model, d)
        self.steps += 1
        return obs, action

    def physics_state(self):
        """(current integration state, state before the last substep or None), both float64 arrays."""
        cur = np.empty(self.state_size, np.float64)
        mujoco.mj_getState(self.model, self.data, cur, STATE_SPEC)
        return cur, (self.pre_state.copy() if self.has_pre_state else None)

    def restore_physics(self, cur, pre):
        """Exact inverse of physics_state(): derived quantities (sensors, xquat, contacts) come from
        `pre` via mj_forward, exactly as mj_step left them; then the integration state is set on top."""
        m, d = self.model, self.data
        if pre is not None:
            mujoco.mj_setState(m, d, np.asarray(pre, np.float64), STATE_SPEC)
            mujoco.mj_forward(m, d)
            mujoco.mj_setState(m, d, np.asarray(cur, np.float64), STATE_SPEC)
            self.pre_state[:] = pre
            self.has_pre_state = True
        else:   # fresh reset: derived quantities belong to the current state
            mujoco.mj_setState(m, d, np.asarray(cur, np.float64), STATE_SPEC)
            mujoco.mj_forward(m, d)
            self.has_pre_state = False

    # ------------------------------------------------------------------- state
    @property
    def time(self):
        return self.data.time

    def pos(self):
        return self.data.qpos[self.qadr:self.qadr + 3].copy()

    def quat(self):
        return self.data.qpos[self.qadr + 3:self.qadr + 7].copy()

    def yaw(self):
        return yaw_of(self.quat())

    def tilt(self):
        """Angle between trunk 'up' and world up, from projected gravity (rad)."""
        g = self.projected_gravity()
        return float(np.arccos(np.clip(-g[2], -1, 1)))

    def planar_velocity(self):
        """(vx, vy) in the robot's yaw frame and the world yaw rate wz."""
        v = self.data.qvel[self.vadr:self.vadr + 3]
        c, s = np.cos(self.yaw()), np.sin(self.yaw())
        vx = c * v[0] + s * v[1]
        vy = -s * v[0] + c * v[1]
        # free-joint angular velocity is in the body frame; rotate to world for yaw rate
        w_body = self.data.qvel[self.vadr + 3:self.vadr + 6]
        q = self.quat()
        w_world = quat_rotate_inverse(np.array([q[0], -q[1], -q[2], -q[3]]), w_body)
        return float(vx), float(vy), float(w_world[2])

    def head_pose_error(self):
        """Joint-space error of the head chain vs DEFAULT_POSE + head command (rad, 4)."""
        q = self.data.qpos[self.jq][HEAD]
        return q - (DEFAULT_POSE[HEAD] + self.command[3:7])
