"""NumPy port of Pollen's velstand.onnx walking policy.

The ONNX graph is: (obs - mean) / std -> Gemm 61x512 -> ELU -> Gemm 512x256 -> ELU
-> Gemm 256x128 -> ELU -> Gemm 128x14. We read the initialisers with `onnx` and run
the same maths in float32 numpy. onnxruntime is used only by parity.py to check it.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = HERE.parent.parent / "data" / "microduck_policies" / "velstand.onnx"

# Joint order shared by every Microduck policy (also in the ONNX metadata).
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# STAND2 / HOME pose from scripts/infer_policy.py (full precision; the ONNX
# metadata stores the same values rounded to 3 decimals).
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0))).astype(x.dtype)


class VelstandPolicy:
    """61-dim observation -> 14 joint offsets (motor target = DEFAULT_POSE + action)."""

    def __init__(self, onnx_path=DEFAULT_ONNX, dtype=np.float32):
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        self.meta = {p.key: p.value for p in model.metadata_props}
        self.dtype = dtype
        self.mean = init["obs_normalizer._mean"].reshape(-1).astype(dtype)
        self.std = init["onnx::Div_24"].reshape(-1).astype(dtype)
        self.layers = [
            (init[f"mlp.{i}.weight"].T.astype(dtype).copy(), init[f"mlp.{i}.bias"].astype(dtype))
            for i in (0, 2, 4, 6)
        ]
        assert self.mean.shape == (61,) and self.layers[-1][0].shape == (128, 14)
        names = tuple(self.meta.get("joint_names", "").split(","))
        assert names == JOINT_NAMES, names
        assert float(self.meta.get("action_scale", "1.0")) == 1.0
        self.n_params = sum(w.size + b.size for w, b in self.layers)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=self.dtype)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        assert x.shape[1] == 61, x.shape
        h = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = _elu(h)
        return h[0] if single else h
