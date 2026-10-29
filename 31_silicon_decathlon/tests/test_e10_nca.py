"""Parity for event 10: our MLX / NumPy / torch copies of the NCA vs the sibling's own nca.py.

Verification only: this is the one place the sibling's code is imported.
"""
import importlib.util
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import nca  # noqa: E402

SIB = os.path.join(HERE, "..", "10_self_healing_pixels", "nca.py")


def _sibling():
    spec = importlib.util.spec_from_file_location("sib_nca10", SIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state(n=2, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, nca.SIZE, nca.SIZE, nca.CHANNELS), np.float32)
    c = nca.SIZE // 2
    x[:, c - 12:c + 12, c - 12:c + 12] = rng.uniform(0, 1, (n, 24, 24, nca.CHANNELS))
    return x, nca.fire_masks(1, n, seed=seed + 1)[0]


def test_target_matches_sibling():
    sib = _sibling()
    np.testing.assert_array_equal(nca.load_target(), sib.load_target("lizard"))


def test_mlx_copy_matches_sibling():
    sib = _sibling()
    ref = sib.load_model(nca.CKPT)
    ours = nca.load_mlx()
    x, m = _state()
    with mx.stream(mx.cpu):
        a = np.array(ref(mx.array(x), fire_mask=mx.array(m > 0.5)))
        b = np.array(ours(mx.array(x), mx.array(m)))
    assert np.abs(a - b).max() < 1e-6


def test_numpy_matches_mlx_cpu():
    ours = nca.load_mlx()
    x, m = _state()
    with mx.stream(mx.cpu):
        a = np.array(ours(mx.array(x), mx.array(m)))
    b = nca.NumpyNCA()(x, m)
    assert np.abs(a - b).max() < 1e-5


def test_torch_matches_mlx_cpu():
    import torch
    ours = nca.load_mlx()
    x, m = _state()
    with mx.stream(mx.cpu):
        a = np.array(ours(mx.array(x), mx.array(m)))
    with torch.no_grad():
        b = nca.torch_net()(torch.from_numpy(x), torch.from_numpy(m)).numpy()
    assert np.abs(a - b).max() < 1e-5
