"""Synthetic reference clip: a planar "kick-dip" groove for the Walker2d body at a fixed tempo.

There is no mocap on disk, so the clip is keyframed. Four beats per loop (100 BPM -> 2.4 s):
  beat 1: right heel-kick (knee flexes ~90 deg, thigh swings back), peak exactly on the beat
  beat 2: dip (both knees bend, pelvis drops), lowest point exactly on the beat
  beat 3: left heel-kick
  beat 4: dip
Walker2d is planar (sagittal), so the "weight shift" is fore-aft over the stance foot, not lateral.

Only the joint keyframes are hand-set. The root (x, z) and the body lean are solved per frame:
  * the stance foot is pinned flat on the floor at a fixed spot (no slip in the reference);
  * the whole-body lean (torso pitch, with the foot joints keeping the soles level) is solved so that the centre of pressure (CoP) from inverse dynamics sits at a
    target point on the stance foot (quasi-static start, then a few fixed-point passes using
    mj_inverse on the finite-differenced trajectory, so the swing-leg momentum is accounted for).
`feasibility()` reports the inverse-dynamics check: CoP inside the foot, normal force > 0, friction
cone, joint torques under the 100 Nm limit.
"""
import numpy as np
import mujoco
from scipy.optimize import brentq

XML = __file__.replace("clip.py", "walker2d_dance.xml")
JOINTS = ["thigh_joint", "leg_joint", "foot_joint", "thigh_left_joint", "leg_left_joint", "foot_left_joint"]
FOOT_R = 0.06          # foot capsule radius
FOOT_LEN = 0.2         # flat part of the sole, from the ankle forward
REF_HZ = 500           # reference sampled at the sim rate

# keyframe parameters (radians); signs follow Walker2d: thigh <= 0 swings back, leg <= 0 flexes knee
# Chosen after the feasibility sweep in RESULTS.md: 100 BPM with 1-beat kicks needed >100 Nm and the CoP
# left the foot; 90 BPM with kicks spread over 1.4 beats keeps the CoP at mid-foot and torques <= ~60 Nm.
DEFAULT = dict(bpm=90.0, thigh0=-0.05, knee0=-0.15, dip_knee=-0.60, kick_knee=-1.2, kick_thigh=-0.15,
               kick_foot=0.3, kick_width=1.4, dip_width=1.0, cop_target=0.10, gain=0.3, passes=10)


def bump(t, center, width, period):
    """Raised-cosine bump of the given width (seconds), periodic."""
    d = (t - center + period / 2) % period - period / 2
    x = np.clip(np.abs(d) / (width / 2), 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def keyframes(t, p):
    """Hand-set channels at times t: torso pitch, both knees, kick envelopes."""
    beat = 60.0 / p["bpm"]
    T = 4 * beat
    kr = bump(t, 0.0, p["kick_width"] * beat, T)      # right kick on beat 1
    kl = bump(t, 2 * beat, p["kick_width"] * beat, T)  # left kick on beat 3
    dip = bump(t, beat, p["dip_width"] * beat, T) + bump(t, 3 * beat, p["dip_width"] * beat, T)
    return dict(kr=kr, kl=kl, dip=dip, T=T, beat=beat)


class Clip:
    def __init__(self, **kw):
        self.p = dict(DEFAULT, **kw)
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.jadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.foot_geom = [self.m.geom("foot_geom").id, self.m.geom("foot_left_geom").id]
        self.build()

    # ---------- kinematics helpers ----------
    def _pose(self, pitch, legs):
        """legs: [(thigh, knee, foot), (thigh, knee, foot)] -> qpos with root at origin."""
        q = np.zeros(self.m.nq)
        q[1] = 1.25
        q[2] = pitch
        q[self.jadr] = np.array(legs).ravel()
        return q

    def _fk(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[1].copy(), [self.d.geom_xpos[g].copy() for g in self.foot_geom]

    def _frame(self, i, pitch, kf, p):
        """Joint angles for frame i given the whole-body lean (torso pitch); the foot joints keep soles level."""
        h = p["thigh0"]
        knee = p["knee0"] + p["dip_knee"] * kf["dip"][i]
        legs = []
        for side, k in ((0, kf["kr"][i]), (1, kf["kl"][i])):
            th = h + p["kick_thigh"] * k
            kn = knee + (p["kick_knee"] - knee) * k
            flat = pitch - th - kn          # foot joint that keeps the sole level
            ft = (1 - k) * flat + k * p["kick_foot"]   # swing foot blends from level to a fixed angle
            legs.append((min(th, 0.0), kn, ft))
        return legs

    def _place(self, q, stance):
        """Translate root so the stance foot is flat on the floor at x(ankle)=0."""
        _, feet = self._fk(q)
        f = feet[stance]
        q = q.copy()
        q[0] -= f[0] - FOOT_LEN / 2
        q[1] -= f[2] - FOOT_R
        return q

    # ---------- build ----------
    def build(self, passes=None):
        p = self.p
        passes = p["passes"] if passes is None else passes
        beat = 60.0 / p["bpm"]
        T = 4 * beat
        n = int(round(T * REF_HZ))
        t = np.arange(n) / REF_HZ
        kf = keyframes(t, p)
        stance = np.where(kf["kr"] > kf["kl"], 1, 0)  # kicking right -> stand on left
        offset = np.zeros(n)                            # CoP->COM correction from inverse dynamics
        self.cop_err_hist = []
        for it in range(passes + 1):
            Q = np.zeros((n, self.m.nq))
            for i in range(n):
                target = p["cop_target"] + offset[i]

                def f(h):
                    q = self._pose(h, self._frame(i, h, kf, p))
                    com, feet = self._fk(q)
                    ankle_x = feet[stance[i]][0] - FOOT_LEN / 2
                    return com[0] - ankle_x - target
                lo, hi = -0.8, 0.8
                h = brentq(f, lo, hi, xtol=1e-12) if f(lo) * f(hi) < 0 else (lo if abs(f(lo)) < abs(f(hi)) else hi)
                Q[i] = self._place(self._pose(h, self._frame(i, h, kf, p)), stance[i])
            self.t, self.T, self.beat, self.qpos, self.stance, self.kf = t, T, beat, Q, stance, kf
            self._derivs()
            self.com_z_mean = float(np.mean([self._fk(q)[0][2] for q in Q[::50]]))
            if it < passes:
                cop = self.inverse()["cop_rel"]
                err = cop - p["cop_target"]
                # cart-table model: cop ~= com - (z_com/g) * com_ddot. Correct the COM by delta with
                # delta - c * delta_ddot = -err (c = z_com/g), solved exactly for a periodic signal in
                # the Fourier domain (it is a low-pass filter, so the iteration stays smooth).
                c = self.com_z_mean / 9.81
                w = 2 * np.pi * np.fft.rfftfreq(n, 1.0 / REF_HZ)
                delta = np.fft.irfft(-np.fft.rfft(err) / (1 + c * w ** 2), n)
                offset += self.p.get("gain", 0.5) * delta
                self.cop_err_hist.append(float(np.abs(err).max()))
        self._contacts()
        self.tau_ff = self.inverse()["tau"]     # reference joint torques (feed-forward for the baseline)

    def _derivs(self):
        dt = 1.0 / REF_HZ
        Q = self.qpos.copy()
        # unwrap root x so differences are periodic (root x returns to start each loop)
        qp, qm = np.roll(Q, -1, 0), np.roll(Q, 1, 0)
        self.qvel = (qp - qm) / (2 * dt)
        self.qacc = (qp - 2 * Q + qm) / dt ** 2

    def _contacts(self):
        """Reference foot positions and contact schedule from FK."""
        n = len(self.t)
        self.foot_pos = np.zeros((n, 2, 2))   # (frame, foot, [x, z]) of the sole centre
        self.com = np.zeros((n, 2))
        for i in range(n):
            com, feet = self._fk(self.qpos[i])
            self.com[i] = com[[0, 2]]
            for s in range(2):
                self.foot_pos[i, s] = feet[s][[0, 2]] - [0, FOOT_R]
        self.contact = self.foot_pos[:, :, 1] < 0.01   # sole within 1 cm of the floor
        self.root_z = self.qpos[:, 1]

    # ---------- inverse dynamics feasibility ----------
    def inverse(self):
        m, d = self.m, self.d
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        n = len(self.t)
        frc = np.zeros((n, m.nv))
        for i in range(n):
            d.qpos[:], d.qvel[:], d.qacc[:] = self.qpos[i], self.qvel[i], self.qacc[i]
            mujoco.mj_inverse(m, d)
            frc[i] = d.qfrc_inverse
        m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        Fx, Fz, My = frc[:, 0], frc[:, 1], frc[:, 2]
        xt, zt = self.qpos[:, 0], self.qpos[:, 1]
        # ground force F at (xp, 0): moment about torso about +y = r_z Fx - r_x Fz, r = (xp-xt, -zt)
        xp = xt - (My + zt * Fx) / Fz
        ankle = np.zeros(n)
        tau = np.zeros((n, 6))
        jacp = np.zeros((3, m.nv))
        foot_body = [m.body("foot").id, m.body("foot_left").id]
        for i in range(n):
            d.qpos[:] = self.qpos[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ankle[i] = d.geom_xpos[self.foot_geom[self.stance[i]]][0] - FOOT_LEN / 2
            # joint torques = inverse-dynamics force minus what the ground force supplies through each
            # loaded foot (J_c^T f); the load is split evenly when both soles are on the floor
            loaded = [self.stance[i]]
            if hasattr(self, "contact") and self.contact[i].all():
                loaded = [0, 1]
            t = frc[i, 3:].copy()
            for s in loaded:
                mujoco.mj_jac(m, d, jacp, None, np.array([xp[i], 0.0, 0.0]), foot_body[s])
                f = np.array([Fx[i], 0.0, Fz[i]]) / len(loaded)
                t -= (jacp.T @ f)[3:]
            tau[i] = t
        return dict(Fx=Fx, Fz=Fz, My=My, cop=xp, cop_rel=xp - ankle, tau=tau)

    def feasibility(self):
        r = self.inverse()
        mu = 1.9
        return dict(
            cop_rel_min=float(r["cop_rel"].min()), cop_rel_max=float(r["cop_rel"].max()),
            cop_inside_foot=bool(((r["cop_rel"] > 0) & (r["cop_rel"] < FOOT_LEN)).all()),
            Fz_min=float(r["Fz"].min()), friction_ratio_max=float(np.abs(r["Fx"] / r["Fz"]).max()),
            friction_ok=bool((np.abs(r["Fx"]) < mu * r["Fz"]).all()),
            tau_max=float(np.abs(r["tau"]).max()), tau_ok=bool(np.abs(r["tau"]).max() < 100),
            swing_clearance_min=float(self.min_swing_clearance()),
        )

    def min_swing_clearance(self):
        """Lowest swing-sole height while the kick envelope is above 0.5 (should be clearly off the floor)."""
        mask = np.maximum(self.kf["kr"], self.kf["kl"]) > 0.5
        sw = 1 - self.stance
        return self.foot_pos[np.arange(len(self.t)), sw, 1][mask].min()

    # ---------- sampling at arbitrary time (tempo scaling via phase) ----------
    def at(self, phase):
        """Reference at phase in [0,1): linear interpolation. Returns dict of arrays."""
        n = len(self.t)
        x = (phase % 1.0) * n
        i0 = int(np.floor(x)) % n
        i1 = (i0 + 1) % n
        w = x - np.floor(x)
        q = (1 - w) * self.qpos[i0] + w * self.qpos[i1]
        qd = (1 - w) * self.qvel[i0] + w * self.qvel[i1]
        fp = (1 - w) * self.foot_pos[i0] + w * self.foot_pos[i1]
        return dict(qpos=q, qvel=qd, foot=fp, contact=self.contact[i0 if w < 0.5 else i1],
                    com=(1 - w) * self.com[i0] + w * self.com[i1])


if __name__ == "__main__":
    import json, sys, time
    t0 = time.time()
    c = Clip()
    print("built in %.1fs" % (time.time() - t0))
    print(json.dumps(c.feasibility(), indent=1))
    print("root x range", c.qpos[:, 0].min(), c.qpos[:, 0].max(), "root z", c.qpos[:, 1].min(), c.qpos[:, 1].max())
    print("joint ranges", c.qpos[:, 3:].min(0).round(2), c.qpos[:, 3:].max(0).round(2))
