"""CPG trot (the baseline and the base of the residual policy), observations, reward, termination.

`LavaTask.step(action)` with action = zeros(13) is exactly the tuned open-loop CPG baseline; the
learned policy adds a bounded residual on the same CPG: 12 joint-target offsets (x RES_SCALE rad)
plus one gait-frequency modulation. So baseline and policy run the identical episode loop.
"""
import numpy as np
import lava_env as E

L1 = L2 = 0.213          # Go1 thigh and calf lengths
RES_SCALE = 0.25          # rad, residual joint-target range
FREQ_MOD = 0.4            # frequency multiplier range 1 +- FREQ_MOD
STRIDE_MOD = 0.5          # stride-length multiplier range 1 +- STRIDE_MOD
ACT_DIM = 14              # 12 joint residuals + frequency mod + stride mod
SIDE = np.array([-1, 1, -1, 1])   # FR, FL, RR, RL: -1 right, +1 left
FRONT = np.array([1, 1, 0, 0], dtype=bool)

# Hand-picked starting point; tune_cpg.py searches around it and writes cpg_params.json.
DEFAULT_CPG = dict(freq=2.0, stride=0.14, height=0.27, clear=0.07, duty=0.55,
                   xoff_f=0.0, xoff_r=0.0, abd=0.0, dh=0.0,
                   p1=0.5, p2=0.5, p3=0.0, k_yaw=1.0, look=0.6)

# Search ranges for the baseline tuner (also bounds the parameters in general).
CPG_RANGES = dict(freq=(1.2, 3.5), stride=(0.0, 0.30), height=(0.20, 0.31), clear=(0.02, 0.13),
                  duty=(0.40, 0.75), xoff_f=(-0.06, 0.06), xoff_r=(-0.06, 0.06), abd=(-0.15, 0.25),
                  dh=(-0.04, 0.04), p1=(0.0, 1.0), p2=(0.0, 1.0), p3=(0.0, 1.0),
                  k_yaw=(0.0, 3.0), look=(0.2, 1.2))
# Extra parameters of the scripted foothold planner baseline (CPG + planner, see LavaTask(planner=True)).
PLAN_DEFAULT = dict(margin=0.04, dmax=0.10, vgain=1.0)
PLAN_RANGES = dict(margin=(0.0, 0.09), dmax=(0.0, 0.16), vgain=(0.5, 1.5))


def leg_ik(x, z):
    """Planar 2-link IK in the thigh-joint frame (x forward, z up; foot below => z < 0)."""
    r2 = x * x + z * z
    r2 = np.clip(r2, 0.05 ** 2, (L1 + L2 - 1e-3) ** 2)
    cq2 = np.clip(r2 / (2 * L1 * L1) - 1.0, -1.0, 1.0)
    q2 = -np.arccos(cq2)
    alpha = np.arctan2(-x, -z)
    q1 = alpha - q2 / 2.0
    return q1, q2


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LavaTask:
    obs_dim = 3 + 3 + 3 + 1 + 2 + 12 + 12 + ACT_DIM + 2 + 4 + 5 * E.K_NEXT + 12

    def __init__(self, cpg=None, model=None, planner=False):
        self.env = E.LavaEnv(model)
        self.cpg = dict(DEFAULT_CPG if cpg is None else cpg)
        self.planner = planner
        m = self.env.m
        import mujoco
        self.thigh_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n + "_thigh") for n in E.LEG_NAMES]

    def plan_delta(self, legs, stride, s, freq):
        """Scripted foothold planner: at lift-off, predict each foot's touchdown x from the hip position,
        the trunk velocity and the nominal swing; if it lands in a gap (or within `margin` of an edge),
        shift the touchdown by the smallest offset (|offset| <= dmax) that puts it on a stone."""
        p, env = self.cpg, self.env
        L = env.layout
        vx = env.d.qvel[0] * p["vgain"]
        t_sw = s / max(freq, 1e-3)
        for i in legs:
            xoff = p["xoff_f"] if FRONT[i] else p["xoff_r"]
            x_td = env.d.xpos[self.thigh_body[i]][0] + vx * t_sw + stride[i] / 2 + xoff
            lo, hi = L[:, 0] + p["margin"], L[:, 1] - p["margin"]
            if np.any((x_td >= lo) & (x_td <= hi)):
                self.delta[i] = 0.0
                continue
            cands = np.concatenate([lo - x_td, hi - x_td])
            cands = cands[np.argsort(np.abs(cands))]
            self.delta[i] = float(np.clip(cands[0], -p["dmax"], p["dmax"]))

    # ------------------------------------------------------------------ CPG
    def cpg_targets(self, phase, ramp, stride_mul=1.0):
        p = self.cpg
        offs = np.array([0.0, p["p1"], p["p2"], p["p3"]])
        ph = (phase + offs) % 1.0
        s = 1.0 - p["duty"]
        # heading term: aim at the centre line of the stone ahead (same stone info the policy sees)
        env = self.env
        x, y = env.d.qpos[0], env.d.qpos[1]
        i = env.stone_index(x + p["look"] / 2)
        yt = env.layout[i, 2]
        yaw_err = wrap(np.arctan2(yt - y, p["look"]) - env.yaw())
        k = np.clip(p["k_yaw"] * yaw_err, -0.5, 0.5)
        stride = p["stride"] * stride_mul * ramp * (1.0 - SIDE * k)    # right legs (SIDE=-1) stride longer to turn left
        swing = ph < s
        if self.planner:
            lift = np.nonzero(swing & ~self.prev_swing)[0]
            if len(lift):
                self.plan_delta(lift, stride, s, p["freq"])
        self.prev_swing = swing
        dl = self.delta if self.planner else 0.0
        u_sw = np.where(swing, ph / s, 0.0)
        u_st = np.where(swing, 0.0, (ph - s) / (1.0 - s))
        fx = np.where(swing, -stride / 2 + (stride + dl) * (1 - np.cos(np.pi * u_sw)) / 2,
                      stride / 2 + dl - (stride + dl) * u_st)
        fz = np.where(swing, p["clear"] * np.sin(np.pi * u_sw), 0.0)
        H = p["height"] + np.where(FRONT, p["dh"] / 2, -p["dh"] / 2)
        fx = fx + np.where(FRONT, p["xoff_f"], p["xoff_r"])
        q1, q2 = leg_ik(fx, fz - H)
        q = np.empty(12)
        q[0::3] = SIDE * p["abd"]
        q[1::3] = q1
        q[2::3] = q2
        return q, swing

    # ------------------------------------------------------------------ episode
    def reset(self, layout, rng=None, init_noise=0.0):
        self.env.reset(layout, rng, init_noise)
        self.phase = 0.0
        self.delta = np.zeros(4)
        self.prev_swing = np.zeros(4, dtype=bool)
        self.prev_a = np.zeros(ACT_DIM)
        self.best_x = self.env.d.qpos[0]
        self.best_t = 0.0
        self.goal = E.goal_x(layout)
        self.last_x = self.env.d.qpos[0]
        return self.obs()

    def obs(self):
        env, d = self.env, self.env.d
        R = d.xmat[env.trunk].reshape(3, 3)
        grav_b = R.T @ np.array([0, 0, -1.0])
        v_b = R.T @ d.qvel[0:3]
        w_b = d.qvel[3:6]  # free-joint angular velocity is already in the body frame
        yaw = env.yaw()
        i = env.stone_index(d.qpos[0])
        h_rel = d.qpos[2] - env.layout[i, 4]
        feet = env.foot_pos() - d.qpos[0:3]
        c, s = np.cos(yaw), np.sin(yaw)
        feet_y = np.stack([c * feet[:, 0] + s * feet[:, 1], -s * feet[:, 0] + c * feet[:, 1], feet[:, 2]], 1)
        o = np.concatenate([
            grav_b, v_b, w_b * 0.25, [h_rel], [np.sin(yaw), np.cos(yaw)],
            d.qpos[7:] - E.HOME_Q, d.qvel[6:] * 0.1, self.prev_a,
            [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
            env.foot_contacts(), env.next_stones().ravel(), feet_y.ravel(),
        ])
        return o.astype(np.float32)

    def step(self, action, push_force=None):
        env, d = self.env, self.env.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ramp = min(1.0, env.t / 0.5)
        fmul = 1.0 + FREQ_MOD * a[12]
        self.phase = (self.phase + self.cpg["freq"] * fmul * E.CTRL_DT) % 1.0
        q_cpg, _ = self.cpg_targets(self.phase, ramp, 1.0 + STRIDE_MOD * a[13])
        q = q_cpg + RES_SCALE * a[:12]
        e = env.step_targets(q, push_force)
        x = d.qpos[0]
        dx = x - self.last_x
        self.last_x = x
        R = d.xmat[env.trunk].reshape(3, 3)
        i = env.stone_index(x)
        h_rel = d.qpos[2] - env.layout[i, 4]
        # reward: progress dominates; small energy / posture / residual penalties
        r = 10.0 * dx - 0.0005 * e - 0.5 * (R[0, 2] ** 2 + R[1, 2] ** 2) \
            - 0.01 * float(np.sum(a[:12] ** 2)) - 0.005 * float(np.sum((a - self.prev_a) ** 2))
        self.prev_a = a
        if x > self.best_x + 0.05:
            self.best_x, self.best_t = x, env.t
        success = x > self.goal
        collapsed = h_rel < 0.12
        stall = env.t - self.best_t > 3.0
        timeout = env.t >= E.T_MAX - 1e-9
        fail = env.lava or env.flip or collapsed
        if fail:
            r -= 10.0
        if success:
            r += 10.0
        done = bool(fail or success or stall or timeout)
        info = dict(success=bool(success), lava=bool(env.lava), flip=bool(env.flip),
                    collapsed=bool(collapsed and not env.lava and not env.flip),
                    stall=bool(stall and not fail and not success), timeout=bool(timeout and not fail and not success),
                    energy=e, x=float(x))
        return self.obs(), float(r), done, info


def run_episode(task, layout, policy=None, rng=None, init_noise=0.0, push=None, record=None):
    """Run one episode. policy(obs)->action or None for the pure CPG baseline.
    push = (trigger_x, force_vec3, duration_s): apply force on the trunk once the trunk passes trigger_x.
    record: list to append (qpos copy) per control step, for rendering."""
    o = task.reset(layout, rng, init_noise)
    zero = np.zeros(ACT_DIM)
    push_steps, pushed, push_t = 0, False, None
    info = {}
    total_r = 0.0
    n_air, n_steps, max_h = 0, 0, 0.0
    slip_sum, slip_n = 0.0, 0
    prev_feet, prev_c = task.env.foot_pos(), task.env.foot_contacts()
    while True:
        a = zero if policy is None else policy(o)
        pf = None
        if push is not None and not pushed and task.env.d.qpos[0] > push[0]:
            pushed, push_steps, push_t = True, int(round(push[2] / E.CTRL_DT)), task.env.t
        if push_steps > 0:
            pf = push[1]
            push_steps -= 1
        o, r, done, info = task.step(a, pf)
        total_r += r
        n_steps += 1
        c_now = task.env.foot_contacts()
        n_air += int(c_now.sum() == 0)
        feet = task.env.foot_pos()
        both = (c_now > 0) & (prev_c > 0)   # foot in contact at both ends of the control step
        if both.any():
            v = np.linalg.norm((feet - prev_feet)[both, :2], axis=1) / E.CTRL_DT
            slip_sum += float(v.sum()); slip_n += int(both.sum())
        prev_feet, prev_c = feet, c_now
        d_ = task.env.d
        max_h = max(max_h, float(d_.qpos[2] - task.env.layout[task.env.stone_index(d_.qpos[0]), 4]))
        if record is not None:
            record.append(task.env.d.qpos.copy())
        if done:
            break
    env = task.env
    prog = env.progress()
    total = task.goal - env.x0
    out = dict(info)
    out.update(progress=prog, frac=float(np.clip(prog / total, 0, 1)), energy_total=env.energy,
               t=env.t, ret=total_r, pushed=pushed, push_t=push_t,
               airborne_frac=n_air / max(n_steps, 1), max_trunk_height=max_h,
               stance_slip_mps=slip_sum / max(slip_n, 1))
    return out
