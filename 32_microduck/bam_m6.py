"""BAM M6 model of the Dynamixel XL330 servo, in numpy, for the Microduck body (R8 gate A).

Velstand was trained with this actuator (BAM M6 XL330, firmware P-gain 200); M0's body uses
MuJoCo position actuators instead. This module re-creates the BAM torque pipeline on top of
plain CPU MuJoCo, following Pollen's deployment rehearsal (vendor/microduck_rl/scripts/
infer_policy.py, default path = `bam.mujoco.MujocoController`) and, where the two differ,
offering the training-time variant (`bam.mjlab.BamActuator`, MuJoCo Warp) as a switch.

Per 5 ms physics step, before mj_step (MujocoController.update):
  1. supply voltage  V = max(vin - vin_drop_gain * sum|tau_prev|, vin_min)
     (tau_prev = the previous step's computed motor torques, before the force clamp)
  2. firmware P loop  duty = clip((q_target - q) * kp_fw * error_gain, -1, 1)
     [optional firmware current limiter: duty clipped to (kt*dq +- R*I_max) / V first]
  3. DC motor         tau = kt * V*duty / R - kt^2 * dq / R   -> data.ctrl (motor actuator, gear 1,
                      forcerange +- force_vin * kt / R)
  4. friction budget  (M6: Coulomb + Stribeck + directional load-dependent + quadratic) written to
                      model.dof_frictionloss; viscous friction to model.dof_damping. MuJoCo's
                      solver then applies the friction as a (stiffened) frictionloss constraint.
     Inputs: motor-side load = data.qfrc_actuator (previous solve, after the clamp); external
     load = -qfrc_bias + qfrc_constraint - (the dof's own friction-constraint force), all from the
     previous solve; Stribeck factor exp(-|dq / dtheta_s|^alpha) from the current velocity.
Model conversion (bam.mjlab.BamActuator.edit_spec / infer_policy.load_mujoco_with_bam): position
actuators -> motors, joint damping/frictionloss zeroed, stiff friction constraint
(solref -5e4/-2e2, solimp 0.99/0.9999/0.001/0.5/2), armature from the fit. Done here on the
COMPILED model (rooms pass prebuilt models); r8_bam_parity.py checks it equals the MjSpec route.

`semantics`:
  "cpu"   = bam.mujoco.MujocoController + bam.model.Model.compute_frictions exactly. Two quirks
            of that code path are kept for parity: (a) the friction-constraint rows are matched
            by JOINT id (efc_id is a DOF id; on this floating-base robot joint id != dof id, so
            actuators 0-4 subtract nothing and 5-13 subtract another joint's friction);
            (b) the quadratic term is gated by sign(ext) != sign(motor) (a tie gives 0).
  "mjlab" = bam.mjlab.BamActuator._compute_friction_budget (the training code): rows matched by
            DOF id, no sign gate, |motor| == |ext| counts as back-driven. Default.
Delay (training's BamActuatorCfg delay_min_lag 3 / delay_max_lag 6): the position target reaches
the servo `lag` physics steps late. bam's docs give the unit as "simulation steps" (= 5 ms physics
steps here); how mjlab re-samples the lag is not in the bam source, so: one lag per episode drawn
at reset (default) or a fresh draw every physics step (`lag_resample=True`).

Provenance and licence
----------------------
Ported from Rhoban `bam` (Better Actuator Models), https://github.com/Rhoban/bam, commit
62bd8ce12154340be97e06f7f41a0ca8f116d967 (branch mjlab_frictionloss, the pin in
vendor/microduck_rl/uv.lock; whether velstand v8 was trained on exactly this commit cannot be
checked from here). Sources: bam/actuator.py (VoltageControlledActuator), bam/dynamixel/actuator.py
(XL330Actuator), bam/model.py (m6 compute_frictions), bam/mujoco.py (MujocoController.update),
bam/mjlab.py (BamActuator.edit_spec / compute / _compute_friction_budget), and the parameter values
of bam/params/xl330/m6.json (copied below).
  Copyright 2025 Marc Duclusaud & Grégoire Passault. Licensed under the Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0; full text also in robot/LICENSE). This file is a
  derivative work: the code was rewritten (restructured into numpy functions, joint indexing made
  selectable, delay added, compiled-model conversion instead of MjSpec); the parameter values are
  unchanged. Modifications (c) 2026 the ProjectJoy authors, also under Apache-2.0.
  Please cite: M. Duclusaud, G. Passault, V. Padois, O. Ly, "Extended Friction Models for the
  Physics Simulation of Servo Actuators", ICRA 2025.
"""
from collections import deque

import mujoco
import numpy as np

# bam/params/xl330/m6.json @ 62bd8ce (identified on Rhoban's pendulum test bench)
XL330_M6 = {
    "kt": 0.36601349688984386,                       # torque constant [Nm/A] = back-EMF [V s/rad]
    "R": 2.8113923539223227,                         # winding resistance [ohm]
    "armature": 0.0018077432831600838,               # reflected rotor inertia [kg m^2]
    "friction_base": 0.004771183165566,              # Coulomb [Nm]
    "friction_stribeck": 0.004676345799486616,       # extra static friction near zero speed [Nm]
    "load_friction_motor": 0.2667860954283698,       # load-dependent, motor side
    "load_friction_external": 8.515871897059342e-06,  # load-dependent, external side
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,            # Stribeck velocity [rad/s]
    "alpha": 8.683259907618984,                      # Stribeck curvature
    "friction_viscous": 0.005359668274599504,        # viscous [Nm s/rad]
}
# bam/dynamixel/actuator.py XL330Actuator: duty = (q_target - q) * kp * error_gain
XL330_ERROR_GAIN = (4096 / (2 * np.pi)) / (256 * 885)  # counts/rad / (KP divisor * PWM limit)
XL330_MAX_PWM = 1.0
XL330_MAX_CURRENT = 1.75  # A; XL330Actuator's default, which bam.mjlab keeps (infer_policy sets None)
# bam.mjlab.BamActuator._STIFF_SOLREF_FRICTION / _STIFF_SOLIMP_FRICTION (stiff_frictionloss=True)
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)

# Defaults = infer_policy.py's CPU BAM path (vin 7.4 nominal 2S LiPo, sag gain 0.1 V/Nm, floor 6 V,
# kp_fw 200, no current limiter, force range at vin) + training's 3-6 step actuator delay and the
# training friction semantics. Training samples vin ~ U(6.5, 8.2), gain ~ U(0, 0.2) per env and
# sets the force range at 8.2 V.
DEFAULTS = dict(semantics="mjlab", kp_fw=200.0, vin=7.4, vin_drop_gain=0.1, vin_min=6.0,
                max_current=None, force_vin=None, lag_min=3, lag_max=6, lag_resample=False)
_SHORT = dict(semantics="sem", kp_fw="kp", vin="vin", vin_drop_gain="drop", vin_min="vmin",
              max_current="imax", force_vin="fvin", lag_min="lag", lag_max="lagmax", lag_resample="resample")

P = XL330_M6
KT, R = P["kt"], P["R"]
KT2 = KT ** 2


def options(overrides=None):
    """Full option dict: DEFAULTS updated by `overrides` (unknown keys are an error)."""
    o = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown BAM option {k!r}; known: {sorted(DEFAULTS)}")
        o[k] = v
    if o["semantics"] not in ("cpu", "mjlab"):
        raise ValueError("semantics must be 'cpu' or 'mjlab'")
    if not 0 <= o["lag_min"] <= o["lag_max"]:
        raise ValueError("need 0 <= lag_min <= lag_max")
    return o


def label(overrides=None):
    """Short tag of the non-default options (for BodyConfig.label)."""
    o = options(overrides)
    bits = ["bam"]
    for k, v in o.items():
        if v != DEFAULTS[k]:
            bits.append(f"{_SHORT[k]}{v:g}" if isinstance(v, float) else f"{_SHORT[k]}{v}")
    return "-".join(bits)


def force_limit(vin):
    """Motor actuator force range: stall torque at `vin` (bam: vin * kt / R)."""
    return vin * KT / R


# ------------------------------------------------------------------ the equations (pure functions)
def supply_voltage(vin, vin_drop_gain, vin_min, prev_torque):
    """MujocoController.update: battery sag from the previous step's computed motor torques."""
    if vin_drop_gain is None:
        return vin
    load = np.sum(np.abs(prev_torque))
    v = vin - vin_drop_gain * load
    if vin_min is not None:
        v = max(v, vin_min)
    return v


def motor_torque(q_target, q, dq, vin, kp_fw=200.0, max_current=None):
    """VoltageControlledActuator.compute_control + compute_torque (XL330 firmware P loop, DC motor)."""
    duty = (q_target - q) * kp_fw * XL330_ERROR_GAIN
    if max_current is not None:  # firmware current limiter, as a duty-cycle window
        back_emf = KT * dq
        span = R * max_current / vin
        center = back_emf / vin
        duty = np.clip(duty, center - span, center + span)
    duty = np.clip(duty, -XL330_MAX_PWM, XL330_MAX_PWM)
    volts = vin * duty
    torque = KT * volts / R
    torque -= KT2 * dq / R
    return torque


def stribeck(dq):
    """1 when still, 0 when moving fast (exp(-|dq/dtheta_s|^alpha))."""
    return np.exp(-(np.abs(dq / P["dtheta_stribeck"]) ** P["alpha"]))


def friction_budget(motor, external, dq, semantics="mjlab"):
    """Velocity-independent friction budget [Nm] (-> dof_frictionloss). motor = previous
    qfrc_actuator, external = external load torque (see module docstring), dq = joint velocity."""
    s = stribeck(dq)
    if semantics == "cpu":  # bam/model.py Model.compute_frictions, m6 branch, same operation order
        g = np.abs(external * P["load_friction_external"] - motor * P["load_friction_motor"])
        gs = np.abs(external * P["load_friction_external_stribeck"] - motor * P["load_friction_motor_stribeck"])
        fl = P["friction_base"]
        fl += g
        fl += s * P["friction_stribeck"]
        fl += gs * s
        enable = np.sign(external) != np.sign(motor)
        d_motor = np.abs(external) < np.abs(motor)
        d_external = np.abs(external) > np.abs(motor)
        q_motor = P["load_friction_external_quad"] * np.abs(external) ** 2
        q_external = P["load_friction_motor_quad"] * np.abs(motor) ** 2
        fl += s * (d_motor * q_motor + d_external * q_external) * enable
        return fl
    # bam/mjlab.py BamActuator._compute_friction_budget (m6), same operation order
    fl = np.full_like(motor, P["friction_base"])
    fl = fl + s * P["friction_stribeck"]
    g = np.abs(external * P["load_friction_external"] - motor * P["load_friction_motor"])
    fl = fl + g
    gs = np.abs(external * P["load_friction_external_stribeck"] - motor * P["load_friction_motor_stribeck"])
    fl = fl + s * gs
    abs_ext, abs_mot = np.abs(external), np.abs(motor)
    drive = (abs_mot > abs_ext).astype(motor.dtype)
    back = 1.0 - drive
    quad = drive * P["load_friction_external_quad"] * abs_ext ** 2 + back * P["load_friction_motor_quad"] * abs_mot ** 2
    return fl + s * quad


def friction_rows(efc_type, efc_id, efc_force, nefc, slot):
    """Force of each controlled joint's own friction-loss constraint row (previous solve).
    `slot[i]` = actuator index whose matching id is i, else -1 (matching id = dof id for 'mjlab',
    joint id for 'cpu'); bam sums efc_force over matching rows: exactly one row or none here."""
    out = np.zeros(int(slot.max()) + 1)
    if nefc:
        t, i, f = efc_type[:nefc], efc_id[:nefc], efc_force[:nefc]
        rows = np.flatnonzero((t == FRICTION_DOF) & (i < len(slot)))
        if rows.size:
            s = slot[i[rows]]
            keep = s >= 0
            out[s[keep]] = f[rows[keep]]
    return out


def external_torque(qfrc_bias, qfrc_constraint, friction_force):
    """Load on the gearbox: gravity/Coriolis bias + constraint forces, minus the joint's own
    friction-constraint force (so the load-dependent friction does not feed back on itself)."""
    ext = -qfrc_bias + qfrc_constraint
    ext -= friction_force
    return ext


# --------------------------------------------------------------------------- model conversion
def convert_model(model, act_ids, opts):
    """In place, on a compiled MjModel: the selected position actuators become BAM torque motors
    (infer_policy.load_mujoco_with_bam / bam.mjlab.BamActuator.edit_spec). Call mj_setConst
    after creating MjData (BamM6.__init__ does)."""
    act_ids = np.asarray(act_ids)
    lim = force_limit(opts["force_vin"] if opts["force_vin"] is not None else opts["vin"])
    model.actuator_gaintype[act_ids] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_gainprm[act_ids, 0] = 1.0                 # set_to_motor: unit gain
    model.actuator_biastype[act_ids] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_biasprm[act_ids] = 0.0                    # MuJoCo 3.16 set_to_motor clears it
    model.actuator_dyntype[act_ids] = mujoco.mjtDyn.mjDYN_NONE
    model.actuator_ctrlspec[act_ids] = 0                     # input signature: type default (motor)
    model.actuator_forcelimited[act_ids] = 1
    model.actuator_forcerange[act_ids, 0] = -lim
    model.actuator_forcerange[act_ids, 1] = lim
    model.actuator_ctrllimited[act_ids] = 0                  # infer_policy: act.ctrllimited = False
    model.actuator_gear[act_ids] = (1.0, 0, 0, 0, 0, 0)
    dofs = model.jnt_dofadr[model.actuator_trnid[act_ids, 0]]
    model.dof_damping[dofs] = 0.0
    model.dof_dampingpoly[dofs] = 0.0
    model.dof_frictionloss[dofs] = 0.0
    model.dof_solref[dofs] = STIFF_SOLREF_FRICTION
    model.dof_solimp[dofs] = STIFF_SOLIMP_FRICTION
    model.dof_armature[dofs] = P["armature"]                 # MujocoController sets it, then mj_setConst


# ------------------------------------------------------------------------------- the actuator
class BamM6:
    """BAM M6 XL330 on a converted model. Per physics step: `substep(target, rng)` then mj_step."""

    def __init__(self, model, data, act_ids, opts):
        self.model, self.data, self.o = model, data, opts
        self.act_ids = np.asarray(act_ids)
        jnt = model.actuator_trnid[self.act_ids, 0]
        self.qadr = model.jnt_qposadr[jnt]
        self.dadr = model.jnt_dofadr[jnt]
        match = jnt if opts["semantics"] == "cpu" else self.dadr
        self.slot = np.full(max(model.nv, model.njnt), -1)
        self.slot[match] = np.arange(len(self.act_ids))
        self.prev_torque = np.zeros(len(self.act_ids))
        self.lag = 0
        self.history = deque(maxlen=opts["lag_max"] + 1)
        mujoco.mj_setConst(model, data)

    def reset(self, hold, rng):
        """Episode start: friction fields back to their converted (zero) values, clean voltage
        state, delay line filled with `hold`, lag drawn from `rng` (only if lag_max > 0)."""
        self.model.dof_frictionloss[self.dadr] = 0.0
        self.model.dof_damping[self.dadr] = 0.0
        self.prev_torque[:] = 0.0
        lo, hi = self.o["lag_min"], self.o["lag_max"]
        self.lag = int(rng.integers(lo, hi + 1)) if hi else 0
        self.history.clear()
        self.history.extend([np.asarray(hold, np.float32).copy()] * (hi + 1))

    def substep(self, target, rng):
        """Push this physics step's position target through the delay line, then update()."""
        self.history.append(target)
        if self.o["lag_resample"] and self.o["lag_max"]:
            self.lag = int(rng.integers(self.o["lag_min"], self.o["lag_max"] + 1))
        q_target = np.asarray(self.history[-1 - self.lag], np.float64)
        self.update(q_target)
        return q_target

    def update(self, q_target):
        """bam.mujoco.MujocoController.update (or the mjlab friction semantics): writes data.ctrl,
        model.dof_frictionloss and model.dof_damping for the coming mj_step. Returns the torque."""
        m, d, o = self.model, self.data, self.o
        q = d.qpos[self.qadr]
        dq = d.qvel[self.dadr]
        vin = supply_voltage(o["vin"], o["vin_drop_gain"], o["vin_min"], self.prev_torque)
        torque = motor_torque(q_target, q, dq, vin, o["kp_fw"], o["max_current"])
        if o["vin_drop_gain"] is not None:
            self.prev_torque = np.atleast_1d(torque).copy()
        d.ctrl[self.act_ids] = torque
        fr = friction_rows(d.efc_type, d.efc_id, d.efc_force, d.nefc, self.slot)
        ext = external_torque(d.qfrc_bias[self.dadr], d.qfrc_constraint[self.dadr], fr)
        m.dof_frictionloss[self.dadr] = friction_budget(d.qfrc_actuator[self.dadr], ext, dq, o["semantics"])
        m.dof_damping[self.dadr] = P["friction_viscous"]
        return torque

    # ---------------------------------------------------------------- save / restore
    def state(self):
        return {"prev_torque": self.prev_torque.tolist(), "lag": self.lag,
                "history": [np.asarray(h, np.float32).tolist() for h in self.history],
                "frictionloss": self.model.dof_frictionloss[self.dadr].tolist(),
                "damping": self.model.dof_damping[self.dadr].tolist()}

    def set_state(self, s):
        self.prev_torque = np.asarray(s["prev_torque"], np.float64)
        self.lag = int(s["lag"])
        self.history.clear()
        self.history.extend(np.asarray(h, np.float32) for h in s["history"])
        self.model.dof_frictionloss[self.dadr] = s["frictionloss"]
        self.model.dof_damping[self.dadr] = s["damping"]
