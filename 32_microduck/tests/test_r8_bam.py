"""R8 gate A: the BAM M6 actuator port (bam_m6.py) and its BodyConfig option in duck.py.

Defaults bit-identical to the pre-change capture; model conversion; the servo equations by hand;
torque-level parity against bam's own outputs (a subset of the harvested states, no bam needed);
the delay line; save/restore; a short stand-and-walk.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bam_m6  # noqa: E402
import r8_bam_defaults  # noqa: E402
import r8_bam_parity  # noqa: E402
from duck import CTRL_DT, BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY  # noqa: E402
from policy.velstand_np import DEFAULT_POSE, VelstandPolicy  # noqa: E402

PRIMARY = dict(scene="allcollisions", jv_lag=1, actuator="bam_m6")


@pytest.fixture(scope="module")
def policy():
    return VelstandPolicy()


def test_default_bodies_bit_identical_to_pre_change_capture():
    assert r8_bam_defaults.verify()


def test_labels_and_validation(policy):
    assert BodyConfig(**FROZEN_BODY).label() == "allcollisions_phys4_jvlag1"      # unchanged
    assert BodyConfig(**PRIMARY).label() == "allcollisions_bam_jvlag1"
    assert "imax1.75" in BodyConfig(**PRIMARY, bam={"max_current": 1.75}).label()
    for bad in (dict(PRIMARY, phys_delay=4), dict(PRIMARY, kp_scale=1.2), dict(PRIMARY, delay_max=2),
                dict(scene="allcollisions", bam={"vin": 7.0}), dict(PRIMARY, bam={"nope": 1}),
                dict(PRIMARY, actuator="pd")):
        with pytest.raises(ValueError):
            Duck(BodyConfig(**bad), policy=policy)


def test_model_conversion_and_reset_state(policy):
    duck = Duck(BodyConfig(**PRIMARY), policy=policy)
    m, d, dofs = duck.model, duck.data, duck.bam_act.dadr
    assert (m.actuator_gaintype == 0).all() and (m.actuator_biastype == 0).all()     # FIXED / NONE: motors
    assert np.allclose(m.actuator_gainprm[:, 0], 1.0) and not m.actuator_ctrllimited.any()
    assert np.allclose(m.actuator_forcerange[:, 1], 7.4 * bam_m6.KT / bam_m6.R)       # 0.963 N*m
    assert np.allclose(m.dof_armature[dofs], bam_m6.P["armature"])
    assert np.allclose(m.dof_solref[dofs], bam_m6.STIFF_SOLREF_FRICTION)
    assert np.allclose(m.dof_solimp[dofs], bam_m6.STIFF_SOLIMP_FRICTION)
    assert (m.dof_frictionloss[dofs] == 0).all() and (m.dof_damping[dofs] == 0).all()
    assert (d.ctrl == 0).all()                       # ctrl is a torque now, not a position target
    assert 3 <= duck.bam_act.lag <= 6


def test_servo_equations_by_hand():
    kt, R = bam_m6.KT, bam_m6.R
    z = np.zeros(1)
    # small error: the firmware P loop linearises to the XML actuator's kp 0.55 (= bam's to_mujoco)
    tau = bam_m6.motor_torque(np.array([0.1]), z, z, 7.4)
    assert tau[0] == pytest.approx(0.1 * 200 * bam_m6.XL330_ERROR_GAIN * 7.4 * kt / R)
    assert tau[0] / 0.1 == pytest.approx(0.554, abs=1e-3)
    # saturated duty: stall torque vin*kt/R; back-EMF kt^2/R per rad/s
    assert bam_m6.motor_torque(np.array([2.0]), z, z, 7.4)[0] == pytest.approx(7.4 * kt / R)
    assert bam_m6.motor_torque(z, z, np.array([1.0]), 7.4)[0] == pytest.approx(-kt * kt / R)
    # current limiter: stall torque kt * 1.75 A = 0.64 N*m instead of 0.96
    assert bam_m6.motor_torque(np.array([2.0]), z, z, 7.4, max_current=1.75)[0] == pytest.approx(kt * 1.75)
    # battery sag and its floor
    assert bam_m6.supply_voltage(7.4, 0.1, 6.0, np.array([1.0, -2.0])) == pytest.approx(7.1)
    assert bam_m6.supply_voltage(7.4, 0.1, 6.0, np.full(14, 3.0)) == 6.0


def test_friction_budget_by_hand():
    P = bam_m6.P
    mot, ext, still = np.array([0.4]), np.array([-0.2]), np.zeros(1)
    base = (P["friction_base"] + P["friction_stribeck"] + abs(-0.2 * P["load_friction_external"] - 0.4 * P["load_friction_motor"])
            + abs(-0.2 * P["load_friction_external_stribeck"] - 0.4 * P["load_friction_motor_stribeck"]))
    quad = P["load_friction_external_quad"] * 0.2 ** 2          # |motor| > |ext|: drive side
    for sem in ("cpu", "mjlab"):                                 # opposite signs: both include the quad term
        assert bam_m6.friction_budget(mot, ext, still, sem)[0] == pytest.approx(base + quad)
    same_sign = np.array([0.2])
    fl_cpu = bam_m6.friction_budget(mot, same_sign, still, "cpu")[0]
    fl_mjlab = bam_m6.friction_budget(mot, same_sign, still, "mjlab")[0]
    assert fl_mjlab - fl_cpu == pytest.approx(quad)              # the sign gate is the only difference
    fast = np.array([10.0])                                      # Stribeck terms vanish when moving fast
    assert bam_m6.friction_budget(mot, ext, fast, "mjlab")[0] == pytest.approx(
        P["friction_base"] + abs(-0.2 * P["load_friction_external"] - 0.4 * P["load_friction_motor"]), rel=1e-6)


def test_torque_level_parity_with_bam_subset():
    S = dict(np.load(r8_bam_parity.SUBSET))
    assert len(S["k"]) >= 150
    res = r8_bam_parity.check_cpu(S)
    for name in ("torque", "frictionloss", "damping"):
        assert res[name]["max_abs_diff"] <= r8_bam_parity.TOL, (name, res[name])
    full = json.loads((ROOT / "results" / "r8" / "bam" / "parity.json").read_text())
    assert full["states"] >= 1000 and all(v["pass"] for v in full["cpu_semantics_vs_bam_mujoco"].values())


def test_delay_line_physics_steps(policy):
    duck = Duck(BodyConfig(**PRIMARY, bam={"lag_min": 2, "lag_max": 2}), policy=policy)
    act = duck.bam_act
    seen = []
    act.update = lambda q_target, _u=act.update: seen.append(q_target.copy()) or _u(q_target)
    duck.set_command(0.3)
    _, a1 = duck.step()
    _, a2 = duck.step()
    t1, t2 = DEFAULT_POSE + a1, DEFAULT_POSE + a2
    # physics steps 0-1 still see the hold pose, 2-5 the first target, 6-7 the second
    assert np.allclose(seen[1], DEFAULT_POSE) and np.allclose(seen[2], t1) and np.allclose(seen[5], t1)
    assert np.allclose(seen[6], t2) and np.allclose(seen[7], t2)


def test_save_restore_bit_exact(policy):
    body = dict(PRIMARY, bam={"lag_resample": True, "max_current": 1.75})
    a = Duck(BodyConfig(**body), policy=policy, seed=5)
    a.set_command(0.3, 0, 0.4)
    for _ in range(60):
        a.step()
    cur, pre = a.physics_state()
    ctl = json.loads(json.dumps(a.controller_state()))
    for _ in range(40):
        a.step()
    b = Duck(BodyConfig(**body), policy=policy, seed=99)
    b.restore_physics(cur, pre)
    b.set_controller_state(ctl)
    for _ in range(40):
        b.step()
    assert np.array_equal(a.data.qpos, b.data.qpos) and np.array_equal(a.data.qvel, b.data.qvel)


def test_bam_body_stands_and_walks(policy):
    duck = Duck(BodyConfig(**PRIMARY), policy=policy, seed=1)
    tilt = []
    for cmd, secs in (((0, 0, 0), 1.0), ((0.3, 0, 0), 3.0)):
        duck.set_command(*cmd)
        for _ in range(int(secs / CTRL_DT)):
            duck.step()
            tilt.append(duck.tilt())
    assert max(tilt) < np.radians(20) and duck.pos()[0] > 0.15
    assert np.abs(duck.data.ctrl).max() <= duck.model.actuator_forcerange[0, 1] + 1e-12
