"""Check the numpy port against onnxruntime on 1,000 random observations.

Two draws of 500 each: (a) around the policy's own normaliser (mean + std * N(0,1)),
which is what the policy sees in use; (b) a wide uniform draw in [-5, 5] to stress the
ELUs far from the training range. Pass: max abs error <= 1e-5 on both.
Writes results/m0/parity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from velstand_np import DEFAULT_ONNX, VelstandPolicy  # noqa: E402

OUT = HERE.parent / "results" / "m0" / "parity.json"


def main(n=1000, seed=20261026):
    pol = VelstandPolicy()
    sess = ort.InferenceSession(str(DEFAULT_ONNX), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    half = n // 2
    obs_a = (pol.mean + pol.std * rng.standard_normal((half, 61))).astype(np.float32)
    obs_b = rng.uniform(-5, 5, size=(n - half, 61)).astype(np.float32)
    report = {"n": n, "seed": seed, "onnxruntime": ort.__version__, "n_params": int(pol.n_params)}
    worst = 0.0
    for name, obs in (("normaliser_draw", obs_a), ("uniform_pm5", obs_b)):
        # onnxruntime one observation at a time, as the 50 Hz loop calls it ([1, 61]).
        ref = np.concatenate([sess.run(None, {"obs": o[None]})[0] for o in obs])
        ours = pol(obs)
        err = np.abs(ref - ours)
        report[name] = {
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "action_abs_max": float(np.abs(ref).max()),
        }
        worst = max(worst, float(err.max()))
    report["max_abs_err"] = worst
    report["pass_1e-5"] = bool(worst <= 1e-5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
