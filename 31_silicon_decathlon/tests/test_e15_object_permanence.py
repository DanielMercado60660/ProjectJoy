"""Parity for event 15: the vendored tracker (models/tracker.py) matches the original 15_object_permanence
model, and every dialect (MLX step, NumPy, torch-for-Core-ML) gives the same outputs.

The original model.py is loaded with importlib here only, to verify the copy (verification-only exception).
Comparisons that need fp32 exactness run MLX on the CPU stream: MLX GPU fp32 matmuls with more than one
row carry ~1e-3 relative error on this machine (see RESULTS.md), which is a property of the reference,
not of the copies.
"""
import importlib.util
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import tracker as tr  # noqa: E402


@pytest.fixture(scope="module")
def seq():
    frames, st = tr.recorded("test_long", [0, 1])
    return frames, st


@pytest.fixture(scope="module")
def mlx_model():
    return tr.load_mlx()[0]


def _orig_tracker():
    spec = importlib.util.spec_from_file_location("orig15_model", os.path.join(tr.SRC, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Tracker(K=3, enc_dim=512, hid=256, layers=1)
    m.load_weights(tr.CKPT + ".weights.npz")
    return m


def test_renderer_matches_saved_frames_shape(seq):
    frames, st = seq
    assert frames.shape == (2, 40, 64, 64, 3) and frames.dtype == np.float32
    assert st["pos"].shape == (2, 40, 3, 2)


def test_copy_matches_original_full_sequence(seq, mlx_model):
    frames, _ = seq
    with mx.stream(mx.cpu):
        a = np.array(mlx_model(mx.array(frames)))
        b = np.array(_orig_tracker()(mx.array(frames)))
    assert np.abs(a - b).max() == 0.0


def test_step_matches_full_sequence(seq, mlx_model):
    frames, _ = seq
    with mx.stream(mx.cpu):
        full = np.array(mlx_model(mx.array(frames)))
        h = mx.zeros((2, 256))
        for t in range(frames.shape[1]):
            o, h = mlx_model.step(mx.array(frames[:, t]), h)
            assert np.abs(np.array(o) - full[:, t]).max() < 1e-5


def test_numpy_and_torch_match_mlx_cpu_step(seq, mlx_model):
    torch = pytest.importorskip("torch")
    frames, _ = seq
    npt, ts = tr.NumpyTracker(), tr.torch_step(2)
    h_m, h_n, h_t = mx.zeros((2, 256)), np.zeros((2, 256), np.float32), torch.zeros(2, 256)
    for t in range(0, frames.shape[1]):
        with mx.stream(mx.cpu):
            o_m, h_m = mlx_model.step(mx.array(frames[:, t]), h_m)
            mx.eval(o_m, h_m)
        o_n, h_n = npt.step(frames[:, t], h_n)
        with torch.no_grad():
            o_t, h_t = ts(torch.from_numpy(frames[:, t]), h_t)
        assert np.abs(o_n - np.array(o_m)).max() < 1e-4
        assert np.abs(o_t.numpy().reshape(2, 3, 5) - np.array(o_m)).max() < 1e-4
        assert np.abs(h_t.numpy() - h_n).max() < 1e-4


def test_tracker_still_tracks(seq, mlx_model):
    """Sanity: hidden-frame error of the copy on these two long-occlusion sequences is sub-ball-radius."""
    frames, st = seq
    with mx.stream(mx.cpu):
        out = np.array(mlx_model(mx.array(frames)))
    mu, _, _ = tr.to_pixels(out)
    present = np.broadcast_to(st["present"][:, None, :], st["vis"].shape)
    err = np.linalg.norm(mu - st["pos"], axis=-1)
    assert err[present].mean() < 1.0 and err[present & (st["vis"] == 0)].mean() < 3.5
