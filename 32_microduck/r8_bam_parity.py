"""R8 gate A: parity of bam_m6.py (our numpy port) against Rhoban bam's own code.

Modes (BAMPY = the isolated venv's python that has `bam` installed; never ../.venv):
  convert  [BAMPY -I]  our compiled-model conversion vs infer_policy.load_mujoco_with_bam (MjSpec route
                       + bam.mujoco.MujocoController's armature/mj_setConst): every MjModel array.
                       -> results/r8/bam/parity_convert.json
  record   [../.venv]  closed-loop Duck rollout on the BAM body in 'cpu' semantics, no delay (= Pollen's
                       infer_policy BAM path), 14 s of mixed commands; saves every physics step's position
                       target and the resulting qpos/qvel/ctrl/dof_frictionloss -> data/r8_bam/parity_record.npz
  replay   [BAMPY -I]  feeds those targets, open loop, through infer_policy's model + bam's
                       MujocoController.update() + mj_step; compares step by step (first differing step,
                       max abs diff); harvests the controller's inputs and outputs on >= 1,000 sampled
                       physics steps -> data/r8_bam/parity_states.npz (+ a 200-state subset in
                       results/r8/bam/parity_states_subset.npz for the tests), parity_replay.json
  closedloop [BAMPY -I] infer_policy.py's own BAM control loop (its observation code, onnxruntime, apply_action,
                       bam update + mj_step) vs Duck on the BAM body ('cpu', no delay, jv_lag 0) driven by the same
                       onnxruntime session, closed loop -> results/r8/bam/parity_closedloop.json
  compare  [../.venv]  bam_m6's pure functions on the harvested inputs vs bam's outputs ('cpu' semantics),
                       plus the 'mjlab' friction semantics vs bam.mjlab's own torch functions
                       (_compute_friction_budget, _dof_friction_force, read from the bam source tree given
                       with --bam-src and run with torch: parity use only) -> results/r8/bam/parity.json
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bam_m6  # noqa: E402

OUT = HERE / "results" / "r8" / "bam"
REC = HERE / "data" / "r8_bam" / "parity_record.npz"
STATES = HERE / "data" / "r8_bam" / "parity_states.npz"     # full harvest (gitignored, ~1.6 MB)
SUBSET = OUT / "parity_states_subset.npz"                     # every 7th state, for tests/test_r8_bam.py
SCENE = HERE / "robot" / "scene_allcollisions.xml"
INFER = HERE.parent / "vendor" / "microduck_rl" / "scripts" / "infer_policy.py"
PARITY_BAM = dict(semantics="cpu", lag_min=0, lag_max=0)   # infer_policy's BAM path: no delay
VIN, DROP, VMIN, KP = 7.4, 0.1, 6.0, 200.0                # infer_policy defaults
SCRIPT = [((0, 0, 0), (0, 0, 0, 0), 1.0), ((0.3, 0, 0), (0, 0, 0, 0), 3.0), ((0, 0, 0.6), (0, 0, 0, 0), 2.0),
          ((0.25, 0, -0.5), (0.3, -0.2, 0.5, 0.1), 2.0), ((-0.3, 0, 0), (0, 0, 0, 0), 2.0),
          ((0, 0.3, 0), (0, 0, 0, 0), 2.0), ((0, 0, 0), (-0.4, 0.3, -0.6, 0), 2.0)]
TOL = 1e-9  # stated absolute tolerance [N*m] for the torque-level comparisons


def load_infer_policy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("infer_policy", INFER)
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    return ip


def reference_model(ip):
    """Pollen's CPU BAM path, verbatim: load_bam_model + load_mujoco_with_bam on our scene."""
    bam_model = ip.load_bam_model(KP, VIN, None)
    model, data, ctrl, names = ip.load_mujoco_with_bam(str(SCENE), bam_model, 0.005, DROP, VMIN)
    return bam_model, model, data, ctrl


# ----------------------------------------------------------------------------------- convert
def convert():
    ip = load_infer_policy()
    _, m1, _, _ = reference_model(ip)
    m2 = mujoco.MjModel.from_xml_path(str(SCENE))
    m2.opt.timestep = 0.005
    opts = bam_m6.options(PARITY_BAM)
    bam_m6.convert_model(m2, np.arange(m2.nu), opts)
    d2 = mujoco.MjData(m2)
    mujoco.mj_setConst(m2, d2)
    differ, n = [], 0
    for name in dir(m1):
        if name.startswith("_"):
            continue
        a = getattr(m1, name)
        if not isinstance(a, np.ndarray):
            continue
        b = getattr(m2, name)
        n += 1
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            differ.append({"field": name, "max_abs": float(np.max(np.abs(a.astype(float) - b.astype(float))))
                           if a.shape == b.shape else "shape"})
    for name in ("timestep", "integrator", "cone", "iterations", "ls_iterations", "noslip_iterations"):
        if getattr(m1.opt, name) != getattr(m2.opt, name):
            differ.append({"field": f"opt.{name}"})
    res = {"arrays_compared": n, "differing": differ, "identical": not differ,
           "reference": "infer_policy.load_mujoco_with_bam (MjSpec) + bam.mujoco.MujocoController (armature, mj_setConst)",
           "ours": "bam_m6.convert_model on the compiled MjModel + mj_setConst"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity_convert.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------ record
def record():
    from duck import CTRL_DT, BodyConfig, Duck
    duck = Duck(BodyConfig(scene="allcollisions", jv_lag=1, actuator="bam_m6", bam=PARITY_BAM), seed=0)
    act = duck.bam_act
    targets, post = [], {k: [] for k in ("qpos", "qvel", "ctrl", "frictionloss")}
    upd = act.update

    def logged_update(q_target):   # record what reaches the servo model, then run our port
        targets.append(np.array(q_target, np.float64))
        return upd(q_target)

    act.update = logged_update
    q0, v0 = duck.data.qpos.copy(), duck.data.qvel.copy()
    real_step = mujoco.mj_step

    def logged_step(m, d):
        real_step(m, d)
        post["qpos"].append(d.qpos.copy())
        post["qvel"].append(d.qvel.copy())
        post["ctrl"].append(d.ctrl.copy())
        post["frictionloss"].append(m.dof_frictionloss[act.dadr].copy())

    mujoco.mj_step = logged_step
    try:
        for cmd, head, dur in SCRIPT:
            duck.set_command(*cmd, head=head)
            for _ in range(int(round(dur / CTRL_DT))):
                duck.step()
    finally:
        mujoco.mj_step = real_step
    REC.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REC, targets=np.array(targets), qpos0=q0, qvel0=v0,
                        **{k: np.array(v) for k, v in post.items()})
    print(f"recorded {len(targets)} physics steps -> {REC}; final x {duck.pos()[0]:.3f}, tilt {duck.tilt():.3f}")


# ------------------------------------------------------------------------------------ replay
def replay():
    ip = load_infer_policy()
    _, m, d, ctrl = reference_model(ip)
    rec = np.load(REC)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = rec["qpos0"]
    d.qvel[:] = rec["qvel0"]
    ctrl.reset(d.qpos)
    mujoco.mj_forward(m, d)
    dofs = ctrl.dof_indexes
    n = len(rec["targets"])
    sample = set(range(1, n, 2))      # every other physics step (step 0 too trivial: all-zero context)
    H = {k: [] for k in ("k", "q", "dq", "q_target", "prev_torque", "qfrc_bias", "qfrc_constraint",
                         "qfrc_actuator", "nefc", "efc_type", "efc_id", "efc_force",
                         "torque", "frictionloss", "damping")}
    first_diff, maxdiff = None, {k: 0.0 for k in ("qpos", "qvel", "ctrl", "frictionloss")}
    for k in range(n):
        ctrl.q_target[:] = rec["targets"][k]
        if k in sample:
            H["k"].append(k)
            H["q"].append(d.qpos[ctrl.qpos_indexes].copy())
            H["dq"].append(d.qvel[dofs].copy())
            H["q_target"].append(ctrl.q_target.copy())
            H["prev_torque"].append(ctrl._prev_motor_torque.copy())
            H["qfrc_bias"].append(d.qfrc_bias[dofs].copy())
            H["qfrc_constraint"].append(d.qfrc_constraint[dofs].copy())
            H["qfrc_actuator"].append(d.qfrc_actuator[dofs].copy())
            H["nefc"].append(int(d.nefc))
            H["efc_type"].append(d.efc_type[:d.nefc].copy())
            H["efc_id"].append(d.efc_id[:d.nefc].copy())
            H["efc_force"].append(d.efc_force[:d.nefc].copy())
        ctrl.update()
        if k in sample:
            H["torque"].append(d.ctrl[ctrl.act_indexes].copy())
            H["frictionloss"].append(m.dof_frictionloss[dofs].copy())
            H["damping"].append(m.dof_damping[dofs].copy())
        mujoco.mj_step(m, d)
        here = {"qpos": d.qpos, "qvel": d.qvel, "ctrl": d.ctrl, "frictionloss": m.dof_frictionloss[dofs]}
        for key, v in here.items():
            diff = float(np.max(np.abs(v - rec[key][k])))
            maxdiff[key] = max(maxdiff[key], diff)
            if diff > 0 and first_diff is None:
                first_diff = {"physics_step": k, "field": key, "abs_diff": diff}
    width = max(H["nefc"])
    pad = lambda rows, fill, dt: np.array([np.concatenate([r, np.full(width - len(r), fill)]).astype(dt)  # noqa: E731
                                           for r in rows])
    arrays = {k: np.array(v) for k, v in H.items() if not k.startswith("efc_")}
    arrays.update(efc_type=pad(H["efc_type"], -1, np.int16), efc_id=pad(H["efc_id"], -1, np.int32),
                  efc_force=pad(H["efc_force"], 0.0, np.float64),
                  joint_ids=np.asarray(ctrl.joint_indexes), dof_ids=np.asarray(dofs),
                  friction_dof_type=np.int64(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STATES, **arrays)
    every = {k: (v[::7] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(H["k"]) else v)
             for k, v in arrays.items()}
    np.savez_compressed(SUBSET, **every)
    res = {"physics_steps": n, "sim_s": n * 0.005, "bit_identical": first_diff is None,
           "first_difference": first_diff, "max_abs_diff_over_run": maxdiff,
           "harvested_states": len(H["k"]),
           "reference": "infer_policy.load_mujoco_with_bam + bam.mujoco.MujocoController.update + mj_step, "
                        "fed the recorded per-physics-step targets open loop",
           "ours": "Duck(actuator='bam_m6', bam={'semantics': 'cpu', 'lag_min': 0, 'lag_max': 0}) closed loop"}
    (OUT / "parity_replay.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# -------------------------------------------------------------------------------- closed loop
def closedloop():
    """[BAMPY -I] infer_policy.py's own BAM control loop (PolicyInference: its observation, onnxruntime
    policy, apply_action, bam_ctrl.update + mj_step x4) vs Duck(actuator='bam_m6', 'cpu', no delay,
    no jv lag) driven by the same onnxruntime session, same commands, closed loop on both sides.
    Compares observation, action and qpos at every control step."""
    from duck import CTRL_DT, BodyConfig, Duck
    ip = load_infer_policy()
    _, m, d, ctrl = reference_model(ip)
    onnx = str(HERE.parent / "data" / "microduck_policies" / "velstand.onnx")
    pol = ip.PolicyInference(m, d, walking_onnx_path=onnx, bam_ctrl=ctrl, new_cmd_obs=True,
                             use_projected_gravity=True)
    qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
    d.qpos[qa:qa + 3] = (0.0, 0.0, 0.125)                       # infer_policy.main's start, verbatim
    d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    for i, qi in enumerate(pol.joint_qpos_indices):
        d.qpos[qi] = pol.default_pose[i]
    ctrl.reset(d.qpos)
    pol.set_position_targets(pol.default_pose)
    mujoco.mj_forward(m, d)
    sess = pol.walking_session
    duck = Duck(BodyConfig(scene="allcollisions", actuator="bam_m6", bam=PARITY_BAM),
                policy=lambda o: sess.run(None, {sess.get_inputs()[0].name: o[None]})[0][0], seed=0)
    first, maxd, n = None, {"obs": 0.0, "action": 0.0, "qpos": 0.0}, 0
    for cmd, head, dur in SCRIPT:
        pol.set_vel_cmd(*cmd)
        pol.head_offset[:] = head
        pol._update_command()
        duck.set_command(*cmd, head=head)
        for _ in range(int(round(dur / CTRL_DT))):
            obs_ref = pol.get_observations()
            a_ref = pol.infer()
            pol.apply_action(a_ref)
            for _ in range(4):
                ctrl.update()
                mujoco.mj_step(m, d)
            obs, a = duck.step()
            for key, x, y in (("obs", obs, obs_ref), ("action", a, a_ref), ("qpos", duck.data.qpos, d.qpos)):
                diff = float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
                maxd[key] = max(maxd[key], diff)
                if diff > 0 and first is None:
                    first = {"control_step": n, "field": key, "abs_diff": diff}
            n += 1
    res = {"control_steps": n, "sim_s": n * CTRL_DT, "bit_identical": first is None, "first_difference": first,
           "max_abs_diff": maxd, "final_x": [float(duck.pos()[0]), float(d.qpos[qa])],
           "reference": "infer_policy.PolicyInference (new_cmd_obs, projected gravity, onnxruntime 1.31.0) + "
                        "load_mujoco_with_bam + bam_ctrl.update + mj_step, the script's own loop order",
           "ours": "Duck(actuator='bam_m6', bam={'semantics': 'cpu', 'lag_min': 0, 'lag_max': 0}, jv_lag 0) with the "
                   "same onnxruntime session as its policy"}
    (OUT / "parity_closedloop.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------- compare
def port_outputs(S, semantics):
    """bam_m6's pure functions on harvested inputs -> (torque, frictionloss, damping) per state."""
    ids = S["joint_ids"] if semantics == "cpu" else S["dof_ids"]
    slot = np.full(int(max(S["dof_ids"].max(), S["joint_ids"].max())) + 1, -1)
    slot[ids] = np.arange(len(ids))
    tq, fl = [], []
    for i in range(len(S["k"])):
        vin = bam_m6.supply_voltage(VIN, DROP, VMIN, S["prev_torque"][i])
        tq.append(bam_m6.motor_torque(S["q_target"][i], S["q"][i], S["dq"][i], vin, KP, None))
        nefc = int(S["nefc"][i])
        fr = bam_m6.friction_rows(S["efc_type"][i], S["efc_id"][i], S["efc_force"][i], nefc, slot)
        ext = bam_m6.external_torque(S["qfrc_bias"][i], S["qfrc_constraint"][i], fr)
        fl.append(bam_m6.friction_budget(S["qfrc_actuator"][i], ext, S["dq"][i], semantics))
    return np.array(tq), np.array(fl), np.full_like(np.array(fl), bam_m6.P["friction_viscous"])


def check_cpu(S):
    tq, fl, dmp = port_outputs(S, "cpu")
    out = {}
    for name, ours, ref in (("torque", tq, S["torque"]), ("frictionloss", fl, S["frictionloss"]),
                            ("damping", dmp, S["damping"])):
        d = np.abs(ours - ref)
        out[name] = {"max_abs_diff": float(d.max()), "bit_identical_frac": float(np.mean(d == 0)),
                     "ref_abs_max": float(np.abs(ref).max()), "pass": bool(d.max() <= TOL)}
    return out


def mjlab_reference(S, bam_src):
    """bam.mjlab's own torch code (read from the bam source tree, executed with torch in float64)."""
    import ast
    import types

    import torch
    src = (Path(bam_src) / "bam" / "mjlab.py").read_text()
    tree = ast.parse(src)
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "BamActuator":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name in ("_compute_friction_budget", "_dof_friction_force",
                                                                 "_as_tensor"):
                    funcs[f.name] = ast.get_source_segment(src, f)
    ns = {"torch": torch, "np": np, "_FRICTION_DOF_CONSTRAINT": int(S["friction_dof_type"])}
    code = "from __future__ import annotations\n" + "\n".join(
        "\n".join(line[4:] if line.startswith("    ") else line for line in funcs[n].splitlines()) for n in funcs)
    exec(compile(code, "bam/mjlab.py[BamActuator methods]", "exec"), ns)  # noqa: S102 (parity only)
    P = bam_m6.P
    bm = types.SimpleNamespace(stribeck=True, load_dependent=True, directional=True, quadratic=True,
                               **{k: types.SimpleNamespace(value=v) for k, v in P.items()})
    nv = int(S["dof_ids"].max()) + 1
    dofs = torch.as_tensor(S["dof_ids"])
    ext_l, fl_l = [], []
    for i in range(len(S["k"])):
        nefc = int(S["nefc"][i])
        efc = types.SimpleNamespace(type=torch.as_tensor(S["efc_type"][i][None, :nefc].astype(np.int64)),
                                    id=torch.as_tensor(S["efc_id"][i][None, :nefc].astype(np.int64)),
                                    force=torch.as_tensor(S["efc_force"][i][None, :nefc]))
        me = types.SimpleNamespace(_bam_model=bm, _device="cpu",
                                   _data=types.SimpleNamespace(efc=efc, nefc=torch.tensor([nefc])))
        me._as_tensor = types.MethodType(ns["_as_tensor"], me)
        qfrc_friction = ns["_dof_friction_force"](me, nv)
        bias = torch.zeros(1, nv, dtype=torch.float64)
        cons = torch.zeros(1, nv, dtype=torch.float64)
        bias[0, dofs] = torch.as_tensor(S["qfrc_bias"][i])
        cons[0, dofs] = torch.as_tensor(S["qfrc_constraint"][i])
        ext = -bias[:, dofs] + cons[:, dofs] - qfrc_friction[:, dofs]     # BamActuator.compute, step 3
        vel = torch.as_tensor(S["dq"][i][None])
        s = torch.exp(-torch.pow(torch.abs(vel) / P["dtheta_stribeck"], P["alpha"]))   # step 4
        fl = ns["_compute_friction_budget"](me, torch.as_tensor(S["qfrc_actuator"][i][None]), ext, s)
        ext_l.append(ext[0].numpy())
        fl_l.append(fl[0].numpy())
    return np.array(ext_l), np.array(fl_l)


def check_mjlab(S, bam_src):
    ext_ref, fl_ref = mjlab_reference(S, bam_src)
    slot = np.full(int(S["dof_ids"].max()) + 1, -1)
    slot[S["dof_ids"]] = np.arange(len(S["dof_ids"]))
    ext = np.array([bam_m6.external_torque(S["qfrc_bias"][i], S["qfrc_constraint"][i],
                                           bam_m6.friction_rows(S["efc_type"][i], S["efc_id"][i], S["efc_force"][i],
                                                                int(S["nefc"][i]), slot)) for i in range(len(S["k"]))])
    _, fl, _ = port_outputs(S, "mjlab")
    out = {}
    for name, ours, ref in (("external_torque", ext, ext_ref), ("frictionloss", fl, fl_ref)):
        d = np.abs(ours - ref)
        out[name] = {"max_abs_diff": float(d.max()), "ref_abs_max": float(np.abs(ref).max()),
                     "pass": bool(d.max() <= TOL)}
    # how much the two semantics differ on the same states (information, not a parity check)
    _, fl_cpu, _ = port_outputs(S, "cpu")
    out["cpu_vs_mjlab_frictionloss"] = {"max_abs_diff": float(np.abs(fl_cpu - fl).max()),
                                        "median_abs_diff": float(np.median(np.abs(fl_cpu - fl))),
                                        "median_budget": float(np.median(fl))}
    return out


def compare(bam_src=None):
    S = dict(np.load(STATES))
    res = {"states": int(len(S["k"])), "tolerance_abs_Nm": TOL, "cpu_semantics_vs_bam_mujoco": check_cpu(S)}
    if bam_src:
        res["mjlab_semantics_vs_bam_mjlab_torch"] = check_mjlab(S, bam_src)
        res["mjlab_reference"] = ("bam/mjlab.py BamActuator._dof_friction_force + _compute_friction_budget "
                                  "(commit 62bd8ce), executed with torch float64 on the harvested states")
    res["inputs"] = ("harvested by `replay` from bam.mujoco.MujocoController on Pollen's model, every other "
                     "physics step of a 14 s walk/turn/back/lateral/head script")
    prev = json.loads((OUT / "parity.json").read_text()) if (OUT / "parity.json").exists() else {}
    if not bam_src and "mjlab_semantics_vs_bam_mjlab_torch" in prev:   # keep the one-off torch check
        res["mjlab_semantics_vs_bam_mjlab_torch"] = prev["mjlab_semantics_vs_bam_mjlab_torch"]
        res["mjlab_reference"] = prev.get("mjlab_reference")
    (OUT / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "convert":
        convert()
    elif mode == "record":
        record()
    elif mode == "replay":
        replay()
    elif mode == "closedloop":
        closedloop()
    elif mode == "compare":
        compare(sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--bam-src" else None)
    else:
        sys.exit(__doc__)
