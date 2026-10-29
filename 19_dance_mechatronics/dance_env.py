"""Tracking environment, the PD(+feed-forward+balance) baseline controller, and the clip metrics.

Control at 50 Hz; MuJoCo runs at 500 Hz (Walker2d-v5 timestep and RK4). Each joint is a PD servo
(general actuator) clamped to the Walker2d torque limit of +-100 Nm. Every controller (baseline and
learned) speaks the same interface: per control step it returns joint targets and a feed-forward
torque; ctrl = target + u_ff / kp, so the servo force is kp*(target-q) - kd*qd + u_ff, clamped.
"""
import numpy as np
import mujoco
from clip import Clip, XML, JOINTS, FOOT_R

CTRL_HZ = 50
NSUB = 10
DT = 1.0 / CTRL_HZ
GROUPS = np.array([0, 1, 2, 0, 1, 2])   # hip, knee, ankle per joint
OBS_DIM = 45


def reward_terms(q, qd, ref, feet, root, root_ref):
    """DeepMimic-style tracking reward, each term in [0, 1]."""
    e_pose = np.sum((q - ref["qpos"][3:]) ** 2)
    e_vel = np.sum((qd - ref["qvel"][3:]) ** 2)
    e_ee = np.sum((feet - ref["foot"]) ** 2)
    e_root = (root[0] - root_ref[0]) ** 2 + (root[1] - root_ref[1]) ** 2
    e_pitch = (root[2] - root_ref[2]) ** 2
    r = (0.5 * np.exp(-2.0 * e_pose) + 0.05 * np.exp(-0.1 * e_vel) + 0.25 * np.exp(-40.0 * e_ee)
         + 0.2 * np.exp(-20.0 * e_root - 5.0 * e_pitch))
    return r


class DanceEnv:
    def __init__(self, clip=None, seed=0, tempo=1.0, friction=1.0, mass=1.0, push=0.0,
                 kp=(300, 300, 150), kd=(30, 30, 15), init_noise=0.02, max_steps=None):
        self.clip = clip or Clip()
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.tempo, self.push, self.init_noise = tempo, push, init_noise
        self.max_steps = max_steps
        self.m.geom_friction[:, 0] *= friction
        self.m.body_mass[1:] *= mass
        self.torso = self.m.body("torso").id
        self.foot_geom = [self.m.geom("foot_geom").id, self.m.geom("foot_left_geom").id]
        self.floor = self.m.geom("floor").id
        self.set_gains(kp, kd)

    # ---------------- setup ----------------
    def set_gains(self, kp, kd):
        self.kp = np.asarray(kp, float)[GROUPS]
        self.kd = np.asarray(kd, float)[GROUPS]
        self.m.actuator_gainprm[:, 0] = self.kp
        self.m.actuator_biasprm[:, 1] = -self.kp
        self.m.actuator_biasprm[:, 2] = -self.kd

    def ref(self, t):
        return self.clip.at(t * self.tempo / self.clip.T)

    def reset(self, phase=None):
        if phase is None:
            phase = self.rng.uniform()
        self.t = phase * self.clip.T / self.tempo
        r = self.ref(self.t)
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[:] = r["qpos"]
        d.qvel[:] = r["qvel"] * self.tempo
        d.qpos[3:] += self.rng.normal(0, self.init_noise, 6)
        d.qvel[3:] += self.rng.normal(0, 5 * self.init_noise, 6)
        mujoco.mj_forward(self.m, d)
        self.steps = 0
        self.push_left = 0
        self.next_push = self._draw_push_time()
        self.prev_feet = self.feet()
        self.prev_contact = self.contacts()
        return self.obs()

    def _draw_push_time(self):
        # next push U(0.3, 1.0) loops later (~1.5 pushes per loop; only when self.push > 0)
        return self.t + self.rng.uniform(0.3, 1.0) * self.clip.T / self.tempo

    # ---------------- state ----------------
    def feet(self):
        return np.array([[self.d.geom_xpos[g][0], self.d.geom_xpos[g][2] - FOOT_R] for g in self.foot_geom])

    def contacts(self):
        """(foot contact flags [2], bad contact: any non-foot geom on the floor)."""
        fc = [False, False]
        bad = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g = c.geom2 if c.geom1 == self.floor else (c.geom1 if c.geom2 == self.floor else -1)
            if g < 0:
                continue
            if g == self.foot_geom[0]:
                fc[0] = True
            elif g == self.foot_geom[1]:
                fc[1] = True
            else:
                bad = True
        return np.array(fc), bad

    def fallen(self):
        q = self.d.qpos
        return bool(q[1] < 0.8 or abs(q[2]) > 1.0 or self.prev_contact[1])

    def com(self):
        return self.d.subtree_com[self.torso][[0, 2]].copy()

    def obs(self):
        d = self.d
        ph = (self.t * self.tempo / self.clip.T) % 1.0
        r_now = self.ref(self.t)
        r_next = self.ref(self.t + DT)
        q, qd = d.qpos[3:], d.qvel[3:]
        a = 2 * np.pi * ph
        o = np.concatenate([
            [np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)],
            q, 0.1 * qd,
            [d.qpos[1] - r_now["qpos"][1], d.qpos[2], d.qpos[2] - r_now["qpos"][2], d.qpos[0] - r_now["qpos"][0]],
            0.3 * d.qvel[:3],
            r_next["qpos"][3:], q - r_now["qpos"][3:],
            0.1 * r_next["qvel"][3:] * self.tempo,
            self.prev_contact[0].astype(float), r_now["contact"].astype(float),
        ])
        return o.astype(np.float32)

    # ---------------- step ----------------
    def step(self, target, u_ff):
        """target: joint targets (6); u_ff: feed-forward torques (6). Returns obs, reward, done, info."""
        d = self.d
        d.ctrl[:] = target + u_ff / self.kp
        if self.push > 0 and self.t >= self.next_push and self.push_left == 0:
            self.push_left = 5  # 0.1 s
            self.push_dir = self.rng.choice([-1.0, 1.0])
            self.next_push = self._draw_push_time()
        if self.push_left > 0:
            d.xfrc_applied[self.torso, 0] = self.push_dir * self.push
            self.push_left -= 1
        else:
            d.xfrc_applied[self.torso, 0] = 0.0
        mujoco.mj_step(self.m, d, NSUB)
        mujoco.mj_forward(self.m, d)
        self.t += DT
        self.steps += 1
        feet = self.feet()
        contact = self.contacts()
        # slip: horizontal motion of a sole that was on the floor at both ends of the step
        both = contact[0] & self.prev_contact[0]
        slip = np.abs(feet[:, 0] - self.prev_feet[:, 0]) * both
        self.prev_feet, self.prev_contact = feet, contact
        r = self.ref(self.t)
        rew = reward_terms(d.qpos[3:], d.qvel[3:] / self.tempo, r, feet, d.qpos[:3], r["qpos"][:3])
        done = self.fallen()
        trunc = self.max_steps is not None and self.steps >= self.max_steps
        info = dict(slip=slip, contact=contact[0], ref_contact=r["contact"], feet=feet, ref=r)
        return self.obs(), (0.0 if done else rew), done, trunc, info


class BaselineController:
    """Joint-space PD tracking of the reference + reference inverse-dynamics feed-forward + a simple
    balance term (stance-ankle torque on COM error, stance-hip torque on torso-pitch error).
    All gains are tuned by tune_baseline.py on training seeds."""

    def __init__(self, params=None):
        p = dict(kp_h=300, kp_k=300, kp_a=150, z_h=0.1, z_k=0.1, z_a=0.1, ff=1.0,
                 k_com=0.0, k_comv=0.0, k_pitch=0.0, k_pitchv=0.0, lead=1.0)
        p.update(params or {})
        self.p = p

    def gains(self):
        p = self.p
        kp = np.array([p["kp_h"], p["kp_k"], p["kp_a"]])
        kd = kp * np.array([p["z_h"], p["z_k"], p["z_a"]])
        return kp, kd

    def __call__(self, env):
        p = self.p
        r = env.ref(env.t + p["lead"] * DT)
        target = r["qpos"][3:].copy()
        ph = ((env.t + DT) * env.tempo / env.clip.T) % 1.0
        n = len(env.clip.t)
        u = p["ff"] * env.clip.tau_ff[int(ph * n) % n].copy()
        d = env.d
        rn = env.ref(env.t)
        com = env.com()
        comv = d.subtree_linvel[env.torso][0] if hasattr(d, "subtree_linvel") else 0.0
        e_c = com[0] - rn["com"][0]
        if p.get("com_rel"):
            # foot-relative variant: COM error measured from the stance sole(s), so slow drift of the whole
            # body along the floor does not show up as a balance error
            st = rn["contact"] if rn["contact"].any() else np.array([True, True])
            feet = env.feet()
            e_c = (com[0] - feet[st, 0].mean()) - (rn["com"][0] - rn["foot"][st, 0].mean())
        e_cv = comv - 0.0                       # reference COM is (near) still by construction
        e_p = d.qpos[2] - rn["qpos"][2]
        e_pv = d.qvel[2] - rn["qvel"][2] * env.tempo
        stance = rn["contact"]
        for s in range(2):
            if stance[s]:
                w = 1.0 / max(1, stance.sum())
                u[3 * s + 2] += w * (p["k_com"] * e_c + p["k_comv"] * e_cv)
                u[3 * s + 0] += w * (p["k_pitch"] * e_p + p["k_pitchv"] * e_pv)
        return target, u


def run_clip(env, ctrl, loops=10, phase=0.0, record=False, recover=True):
    """Run a controller for `loops` clip loops starting at `phase`. A fall is counted, then the body is
    reset to the reference at the current phase (if recover) so long runs give falls-per-loop.
    Returns per-step logs for the metrics."""
    if hasattr(ctrl, "gains"):
        env.set_gains(*ctrl.gains())
    env.reset(phase)
    n_steps = int(round(loops * env.clip.T / env.tempo / DT))
    log = dict(t=[], q=[], q_ref=[], feet=[], feet_ref=[], root=[], root_ref=[], contact=[], ref_contact=[],
               slip=[], rew=[], falls=0, fall_t=[], qpos=[], ghost=[], pushing=[])
    for k in range(n_steps):
        mujoco.mj_subtreeVel(env.m, env.d)
        target, u = ctrl(env)
        _, rew, done, _, info = env.step(target, u)
        r = info["ref"]
        log["t"].append(env.t)
        log["q"].append(env.d.qpos[3:].copy()); log["q_ref"].append(r["qpos"][3:])
        log["feet"].append(info["feet"]); log["feet_ref"].append(r["foot"])
        log["root"].append(env.d.qpos[:3].copy()); log["root_ref"].append(r["qpos"][:3])
        log["contact"].append(info["contact"]); log["ref_contact"].append(r["contact"])
        log["slip"].append(info["slip"]); log["rew"].append(rew)
        if record:
            log["qpos"].append(env.d.qpos.copy()); log["ghost"].append(r["qpos"].copy())
            log["pushing"].append(env.d.xfrc_applied[env.torso, 0])
        if done:
            log["falls"] += 1
            log["fall_t"].append(env.t)
            if not recover:
                break
            ph = (env.t * env.tempo / env.clip.T) % 1.0
            t_keep = env.t
            env.reset(ph)
            env.t = t_keep
            env.next_push = env._draw_push_time()   # reset() drew it relative to the wrapped time
    for k in list(log):
        if isinstance(log[k], list) and k not in ("fall_t",):
            log[k] = np.array(log[k])
    log["loops"] = loops
    return log


def metrics(log, clip, tempo=1.0):
    """Tracking error, balance failures, contact quality, beat alignment from a run_clip log."""
    q, qr = log["q"], log["q_ref"]
    out = {}
    out["joint_rmse_deg"] = float(np.degrees(np.sqrt(np.mean((q - qr) ** 2))))
    out["joint_rmse_deg_per_joint"] = np.degrees(np.sqrt(np.mean((q - qr) ** 2, 0))).round(2).tolist()
    out["foot_pos_err_cm"] = float(100 * np.mean(np.linalg.norm(log["feet"] - log["feet_ref"], axis=2)))
    out["root_z_err_cm"] = float(100 * np.sqrt(np.mean((log["root"][:, 1] - log["root_ref"][:, 1]) ** 2)))
    out["root_x_err_cm"] = float(100 * np.sqrt(np.mean((log["root"][:, 0] - log["root_ref"][:, 0]) ** 2)))
    out["pitch_err_deg"] = float(np.degrees(np.sqrt(np.mean((log["root"][:, 2] - log["root_ref"][:, 2]) ** 2))))
    out["falls"] = int(log["falls"])
    out["falls_per_loop"] = log["falls"] / log["loops"]
    out["mean_reward"] = float(np.mean(log["rew"]))
    # contact quality
    T = clip.T / tempo
    c, cr = log["contact"], log["ref_contact"]
    out["contact_agreement"] = float(np.mean(c == cr))
    out["slip_cm_per_loop"] = float(100 * log["slip"].sum() / log["loops"])
    out["touchdown_err_ms"] = _event_timing(log["t"], c, cr, rising=True)
    out["liftoff_err_ms"] = _event_timing(log["t"], c, cr, rising=False)
    out.update(beat_alignment(log, clip, tempo))
    return out


def _events(t, x, rising):
    x = x.astype(int)
    dx = np.diff(x, axis=0)
    ev = []
    for s in range(x.shape[1]):
        idx = np.where(dx[:, s] == (1 if rising else -1))[0] + 1
        ev.append(t[idx])
    return ev


def _event_timing(t, c, cr, rising):
    """Mean |dt| between each reference contact event and the nearest actual event of the same foot (ms).
    Reference events with no actual event within 0.3 s count as misses."""
    ea, er = _events(t, c, rising), _events(t, cr, rising)
    errs, miss = [], 0
    for s in range(2):
        for te in er[s]:
            if len(ea[s]) == 0:
                miss += 1
                continue
            dtm = np.min(np.abs(ea[s] - te))
            if dtm > 0.3:
                miss += 1
            else:
                errs.append(dtm)
    return dict(mean_ms=float(1000 * np.mean(errs)) if errs else None, misses=miss, n=sum(len(e) for e in er))


def beat_alignment(log, clip, tempo):
    """Kick peaks (max swing-sole height) should land on beats 1 and 3, dip bottoms (min root z) on
    beats 2 and 4. For each beat window, find the actual peak time and compare with the beat time."""
    t = log["t"]
    beat = clip.beat / tempo
    T = 4 * beat
    feet_z = log["feet"][:, :, 1]
    rz = log["root"][:, 1]
    offs = {"kick": [], "dip": []}
    nb = int(np.floor((t[-1] - t[0]) / beat))
    b0 = int(np.ceil(t[0] / beat))
    for b in range(b0, b0 + nb):
        tb = b * beat
        w = (t > tb - 0.5 * beat) & (t < tb + 0.5 * beat)
        if w.sum() < 5 or (t[w][0] > tb - 0.45 * beat) or (t[w][-1] < tb + 0.45 * beat):
            continue
        kind = b % 4
        if kind in (0, 2):
            foot = 0 if kind == 0 else 1
            i = np.argmax(feet_z[w, foot])
            offs["kick"].append(t[w][i] - tb)
        else:
            i = np.argmin(rz[w])
            offs["dip"].append(t[w][i] - tb)
    res = {}
    for k, v in offs.items():
        v = np.array(v)
        res[f"beat_{k}_mean_abs_ms"] = float(1000 * np.mean(np.abs(v))) if len(v) else None
        res[f"beat_{k}_mean_ms"] = float(1000 * np.mean(v)) if len(v) else None
        res[f"beat_{k}_within_50ms"] = float(np.mean(np.abs(v) < 0.05)) if len(v) else None
    return res
