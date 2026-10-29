"""Vectorized Shadow-hand egg-rotation env on raw MuJoCo (gymnasium-robotics XML assets, read-only).

Why not the gym env directly: it works once `gymfix` is imported (see test_parity.py, which checks
that this env's 153-d base observation equals the patched gym env's observation on the same state),
but we need per-env friction/mass randomization, a narrow z-rotation target curriculum, contact
force measurement and threaded stepping. Everything the gym env does per step is replicated:
absolute position control (action in [-1,1] mapped onto ctrlrange), 20 substeps of 2 ms (40 ms per
control step), 10 settle steps of zero action at reset, the same touch sensors (92, raw sensordata).

Task (our narrow version of HandManipulateEggRotate):
  * the egg starts lying on its side in the palm (long axis horizontal) with uniform random yaw,
  * the target is the settled egg pose rotated about world z by delta, |delta| ~ U[theta_min,
    theta_max], random sign (theta_min = 0.15 > threshold, so doing nothing does not succeed),
  * error = angle between the egg's long axis and the target's long axis, as lines (the egg is an
    ellipsoid, symmetric about and end-to-end along its long axis, so spin about the axis and a
    180-degree flip are physically indistinguishable),
  * success = error < thr (0.1 rad, the gym rotation threshold) at the final step and not dropped,
  * drop = egg centre below DROP_Z (it has left the palm), episode terminates.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

XML = os.path.join(
    os.path.dirname(__import__("importlib.util").util.find_spec("gymnasium_robotics").origin),
    "envs/assets/hand/manipulate_egg_touch_sensors.xml",
)
N_SUB = 20
T_EP = 100            # control steps per episode (gym default 100 = 4 s)
THR = 0.1             # success threshold on axis error (rad)
DROP_Z = 0.10         # egg centre below this = dropped (resting on palm is ~0.15-0.16)
FORCE_BUDGET = 10.0   # N, total normal contact force on the egg (egg weight 1.48 N); fixed a priori
ACT_DIM = 20
RATE_PEN = 0.001       # action-rate penalty weight (small: noise must not swamp shaping)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def quat_to_zaxis(q):
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def gym_rot_dist(qa, qb):
    """gymnasium-robotics' rotation distance (full quaternion, no symmetry)."""
    qd = quat_mul(qa, qb * np.array([1, -1, -1, -1]))
    return 2 * np.arccos(np.clip(qd[0], -1, 1))


class EggVecEnv:
    """n independent envs stepped by a thread pool (mujoco releases the GIL in mj_step)."""

    def __init__(self, n, seed=0, theta_max=0.4, theta_min=0.15, fric=(0.7, 1.3), mass=(0.7, 1.3),
                 threads=4, auto_reset=True, substep_force=False, T=T_EP):
        self.n, self.T, self.auto_reset, self.substep_force = n, T, auto_reset, substep_force
        self.theta_max, self.theta_min, self.fric, self.mass = theta_max, theta_min, fric, mass
        self.models = [mujoco.MjModel.from_xml_path(XML) for _ in range(n)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        m = self.models[0]
        self.rngs = [np.random.default_rng([seed, i]) for i in range(n)]
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.nthreads = threads
        rj = [j for j in range(m.njnt) if m.joint(j).name.startswith("robot")]
        self.rq = np.array([m.jnt_qposadr[j] for j in rj])
        self.rv = np.array([m.jnt_dofadr[j] for j in rj])
        oj, tj = m.joint("object:joint").id, m.joint("target:joint").id
        self.oq, self.ov = m.jnt_qposadr[oj], m.jnt_dofadr[oj]
        self.tq, self.tv = m.jnt_qposadr[tj], m.jnt_dofadr[tj]
        self.touch = np.array([i for i in range(m.nsensor) if "robot0:TS_" in m.sensor(i).name])
        self.egg_geom = m.geom("object").id
        self.egg_body = m.body("object").id
        self.coll_geoms = np.array([g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]])
        self.base_fric = m.geom_friction.copy()
        self.base_mass = m.body_mass[self.egg_body]
        self.base_inertia = m.body_inertia[self.egg_body].copy()
        cr = m.actuator_ctrlrange
        self.ctrl_c, self.ctrl_r = (cr[:, 1] + cr[:, 0]) / 2, (cr[:, 1] - cr[:, 0]) / 2
        self.ctrl_lo, self.ctrl_hi = cr[:, 0], cr[:, 1]
        # per-env episode state
        self.goal_q = np.zeros((n, 4))
        self.goal_axis = np.zeros((n, 3))
        self.delta = np.zeros(n)
        self.t = np.zeros(n, int)
        self.err = np.zeros(n)
        self.err0 = np.zeros(n)
        self.prev_a = np.zeros((n, ACT_DIM))
        self.fric_k = np.ones(n)
        self.mass_k = np.ones(n)
        self.force = np.zeros(n)        # total normal force on egg at end of last control step
        self.force_peak = np.zeros(n)   # peak over substeps (only if substep_force) else = force
        self.dropped = np.zeros(n, bool)
        self.done = np.zeros(n, bool)
        # force penalty (run1: 0.05/N above 10 N, cap 0.5; run2+: see train.py args)
        self.force_w, self.force_margin, self.force_cap = 0.05, FORCE_BUDGET, 0.5
        self.fixed_fric = None   # set to a float to override randomization (robustness eval)
        self.fixed_mass = None
        self.obs_dim = len(self._obs(0))
        self.priv_dim = len(self._priv(0))

    # ---------- physics helpers ----------
    def _egg_force(self, m, d):
        nc = d.ncon
        if nc == 0:
            return 0.0
        g1, g2 = d.contact.geom1[:nc], d.contact.geom2[:nc]
        idx = np.nonzero((g1 == self.egg_geom) | (g2 == self.egg_geom))[0]
        f6 = np.zeros(6)
        tot = 0.0
        for i in idx:
            mujoco.mj_contactForce(m, d, int(i), f6)
            tot += abs(f6[0])
        return tot

    def _set_props(self, i):
        m, rng = self.models[i], self.rngs[i]
        fk = self.fixed_fric if self.fixed_fric is not None else rng.uniform(*self.fric)
        mk = self.fixed_mass if self.fixed_mass is not None else rng.uniform(*self.mass)
        m.geom_friction[:] = self.base_fric
        m.geom_friction[self.coll_geoms, 0] = self.base_fric[self.coll_geoms, 0] * fk
        m.body_mass[self.egg_body] = self.base_mass * mk
        m.body_inertia[self.egg_body] = self.base_inertia * mk
        self.fric_k[i], self.mass_k[i] = fk, mk

    def reset_one(self, i):
        m, d, rng = self.models[i], self.datas[i], self.rngs[i]
        for _attempt in range(10):
            self._set_props(i)
            mujoco.mj_resetData(m, d)
            yaw = rng.uniform(-np.pi, np.pi)
            q = quat_mul(quat_axis_angle([0, 0, 1], yaw), quat_axis_angle([1, 0, 0], np.pi / 2))
            d.qpos[self.oq:self.oq + 3] += rng.normal(size=3, scale=0.005)
            d.qpos[self.oq + 3:self.oq + 7] = q
            d.ctrl[:] = self.ctrl_c          # zero action, as in gym's reset
            mujoco.mj_step(m, d, nstep=N_SUB * 10)
            if d.qpos[self.oq + 2] > DROP_Z + 0.03:
                break
        qs = d.qpos[self.oq + 3:self.oq + 7].copy()
        lo = self.theta_min
        delta = rng.uniform(lo, self.theta_max) * rng.choice([-1.0, 1.0])
        gq = quat_mul(quat_axis_angle([0, 0, 1], delta), qs)
        self.goal_q[i] = gq / np.linalg.norm(gq)
        self.goal_axis[i] = quat_to_zaxis(self.goal_q[i])
        self.delta[i] = delta
        self.t[i] = 0
        self.prev_a[i] = 0
        self.dropped[i] = False
        self.done[i] = False
        self.err[i] = self._axis_err(i)
        self.err0[i] = self.err[i]
        self.force[i] = self.force_peak[i] = self._egg_force(m, d)

    def _axis_err(self, i):
        a = quat_to_zaxis(self.datas[i].qpos[self.oq + 3:self.oq + 7])
        c = abs(float(a @ self.goal_axis[i])) / (np.linalg.norm(a) + 1e-9)
        return float(np.arccos(np.clip(c, 0, 1)))

    def full_err(self, i):
        return gym_rot_dist(self.datas[i].qpos[self.oq + 3:self.oq + 7], self.goal_q[i])

    # ---------- observations ----------
    def base_obs(self, i):
        """Exactly the gym ContinuousTouchSensors 'observation' (153,)."""
        d = self.datas[i]
        return np.concatenate([d.qpos[self.rq], d.qvel[self.rv], d.qvel[self.ov:self.ov + 6],
                               d.qpos[self.oq:self.oq + 7], d.sensordata[self.touch]])

    def _obs(self, i):
        d = self.datas[i]
        b = self.base_obs(i)
        a = quat_to_zaxis(d.qpos[self.oq + 3:self.oq + 7])
        g = self.goal_axis[i]
        s = 1.0 if a @ g >= 0 else -1.0
        g = g * s  # symmetry-normalized target axis
        touch = np.log1p(b[61:])  # log(1+x) touch, as gym's 'log' option
        return np.concatenate([b[:54], b[54:57] - np.array([1.0, 0.87, 0.16]), b[57:61], a,
                               self.goal_q[i], g, np.cross(a, g), [a @ g, self.err[i]],
                               touch, self.prev_a[i]]).astype(np.float32)

    def _priv(self, i):
        return np.array([self.fric_k[i], self.mass_k[i], self.force[i] / FORCE_BUDGET,
                         self.t[i] / self.T, self.delta[i], self.err0[i]], np.float32)

    # ---------- stepping ----------
    def _step_one(self, i, a):
        m, d = self.models[i], self.datas[i]
        if self.done[i] and not self.auto_reset:
            return 0.0
        d.ctrl[:] = np.clip(self.ctrl_c + a * self.ctrl_r, self.ctrl_lo, self.ctrl_hi)
        if self.substep_force:
            pk = 0.0
            for _ in range(N_SUB):
                mujoco.mj_step(m, d)
                pk = max(pk, self._egg_force(m, d))
            self.force[i] = self._egg_force(m, d)
            self.force_peak[i] = pk
        else:
            mujoco.mj_step(m, d, nstep=N_SUB)
            self.force[i] = self.force_peak[i] = self._egg_force(m, d)
        self.t[i] += 1
        rate = float(np.sum((a - self.prev_a[i]) ** 2))
        self.prev_a[i] = a
        e_prev = self.err[i]
        self.err[i] = self._axis_err(i)
        drop = d.qpos[self.oq + 2] < DROP_Z or not np.isfinite(d.qpos).all()
        self.dropped[i] = drop
        r = 10.0 * (e_prev - self.err[i]) + (0.5 if self.err[i] < THR else 0.0)
        r -= min(self.force_cap, self.force_w * max(0.0, self.force[i] - self.force_margin))  # capped
        r -= RATE_PEN * rate
        if drop:
            r -= 5.0
        self.done[i] = drop or self.t[i] >= self.T
        return r

    def _run(self, fn, idxs):
        if self.pool is None:
            return [fn(i) for i in idxs]
        chunks = [idxs[k::self.nthreads] for k in range(self.nthreads)]
        res = {}

        def work(ch):
            for i in ch:
                res[i] = fn(i)
        list(self.pool.map(work, chunks))
        return [res[i] for i in idxs]

    def reset(self):
        self._run(self.reset_one, list(range(self.n)))
        return self.obs(), self.priv()

    def obs(self):
        return np.stack([self._obs(i) for i in range(self.n)])

    def priv(self):
        return np.stack([self._priv(i) for i in range(self.n)])

    def step(self, actions):
        """Returns obs, priv, reward, done, info. With auto_reset, done envs are reset and the
        returned obs is the first obs of the new episode; info holds the finished-episode stats."""
        actions = np.clip(np.asarray(actions, np.float64), -1, 1)
        was_done = self.done.copy()
        rew = np.array(self._run(lambda i: self._step_one(i, actions[i]), list(range(self.n))))
        done = self.done.copy() & ~(was_done & ~self.auto_reset)
        info = {"err": self.err.copy(), "dropped": self.dropped.copy(), "force": self.force.copy(),
                "force_peak": self.force_peak.copy(), "t": self.t.copy(),
                "success": (self.err < THR) & ~self.dropped, "delta": self.delta.copy()}
        if self.auto_reset and done.any():
            ids = list(np.nonzero(done)[0])
            self._run(self.reset_one, ids)
        return self.obs(), self.priv(), rew.astype(np.float32), done, info

    def render_target(self, i):
        """Place the translucent target egg next to the hand (like gym's _render_callback)."""
        d = self.datas[i]
        d.qpos[self.tq:self.tq + 3] = d.qpos[self.oq:self.oq + 3] + np.array([0.15, 0, 0])
        d.qpos[self.tq + 3:self.tq + 7] = self.goal_q[i]
        d.qvel[self.tv:self.tv + 6] = 0
        mujoco.mj_forward(self.models[i], d)


def run_episodes(env, policy, n_eps, collect_traj=False, budget=FORCE_BUDGET):
    """Run exactly one episode per env slot (env must have auto_reset=False and n == n_eps).
    policy(obs, priv, env) -> actions. Returns per-episode metrics dict."""
    assert not env.auto_reset and env.n == n_eps
    obs, priv = env.reset()
    peak = np.zeros(env.n)
    over = np.zeros(env.n, bool)
    ever = np.zeros(env.n, bool)
    steps_succ = np.zeros(env.n)
    traj = []
    if hasattr(policy, "reset"):
        policy.reset(env)
    for t in range(env.T):
        alive = ~env.done
        a = policy(obs, priv, env)
        obs, priv, r, done, info = env.step(a)
        peak = np.where(alive, np.maximum(peak, info["force_peak"]), peak)
        over |= alive & (info["force_peak"] > budget)
        ever |= alive & info["success"]
        steps_succ += alive & info["success"]
        if collect_traj:
            traj.append(info["err"].copy())
    final_err = env.err.copy()
    full = np.array([env.full_err(i) for i in range(env.n)])
    return {"success": (final_err < THR) & ~env.dropped, "dropped": env.dropped.copy(),
            "final_err": final_err, "err0": env.err0.copy(), "ever_success": ever,
            "frac_steps_success": steps_succ / env.T, "peak_force": peak, "over_budget": over,
            "full_err": full, "delta": env.delta.copy(), "traj": np.array(traj).T if collect_traj else None}
