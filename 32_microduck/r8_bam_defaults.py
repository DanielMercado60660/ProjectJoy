"""R8 gate A: proof that the default (XML-actuator) body is untouched by the BAM option.

`capture` was run ONCE, before duck.py gained `actuator="bam_m6"`: it records three default-path
bodies for 300 control steps (5 command segments, head commands included) and saves every
qpos / qvel / ctrl / observation / action to results/r8/bam/default_body_reference.npz.
`verify` re-runs the same rollouts with the current code and requires array_equal on all of
them (tests/test_r8_bam.py calls it).

    ../.venv/bin/python r8_bam_defaults.py capture   # before the change (done 2026-11-05)
    ../.venv/bin/python r8_bam_defaults.py verify    # any time after
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from duck import BodyConfig, Duck  # noqa: E402
from m0_gate import FROZEN_BODY  # noqa: E402

REF = HERE / "results" / "r8" / "bam" / "default_body_reference.npz"
BODIES = {
    "frozen": dict(FROZEN_BODY),
    "plain": {"scene": "allcollisions"},
    # every pre-existing knob at once (exercises the rng draws: policy-rate lag + obs noise)
    "knobs": dict(scene="groundcontact", delay_min=1, delay_max=3, phys_delay=2, jv_lag=1, kp_scale=1.2,
                  torque_limit=0.8, foot_friction=1.3, foot_solref=0.03, obs_noise=1.0),
}
SEQ = [((0.3, 0, 0), (0, 0, 0, 0)), ((0, 0, 0.6), (0.2, 0, 0.3, 0)), ((0.2, 0, -0.5), (0, 0, 0, 0)),
       ((-0.3, 0, 0), (0, -0.3, 0, 0.1)), ((0, 0.3, 0), (0, 0, 0, 0))]
STEPS_PER_SEG = 60


def rollout(body, seed=3):
    duck = Duck(BodyConfig(**body), seed=seed)
    out = {k: [] for k in ("qpos", "qvel", "ctrl", "obs", "action")}
    for cmd, head in SEQ:
        duck.set_command(*cmd, head=head)
        for _ in range(STEPS_PER_SEG):
            obs, act = duck.step()
            out["qpos"].append(duck.data.qpos.copy())
            out["qvel"].append(duck.data.qvel.copy())
            out["ctrl"].append(duck.data.ctrl.copy())
            out["obs"].append(obs.copy())
            out["action"].append(act.copy())
    return {k: np.array(v) for k, v in out.items()}


def capture():
    REF.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, body in BODIES.items():
        for k, v in rollout(body).items():
            arrays[f"{name}/{k}"] = v
    np.savez_compressed(REF, **arrays)
    print(f"wrote {REF} ({len(arrays)} arrays)")


def verify():
    ref = np.load(REF)
    bad = []
    for name, body in BODIES.items():
        got = rollout(body)
        for k, v in got.items():
            if not np.array_equal(ref[f"{name}/{k}"], v):
                bad.append(f"{name}/{k}")
    print("default bodies bit-identical to the pre-change capture" if not bad else f"MISMATCH: {bad}")
    return not bad


if __name__ == "__main__":
    if sys.argv[1:] == ["capture"]:
        if REF.exists():
            sys.exit(f"{REF} exists; it must only be captured once, before the change")
        capture()
    else:
        sys.exit(0 if verify() else 1)
