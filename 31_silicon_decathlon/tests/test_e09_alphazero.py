"""e09 parity (added by the reviewer): the vendored copy in models/alphazero.py and its NumPy baseline vs
09_tiny_alphazero's own net.py (loaded with importlib here only, verification-only exception).

  ../.venv/bin/python -m pytest -q tests/test_e09_alphazero.py
"""
import importlib.util
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)   # model paths are relative to the project folder

from models import alphazero as az  # noqa: E402

SIB = os.path.join(HERE, "..", "09_tiny_alphazero")


def _sib():
    if SIB not in sys.path:
        sys.path.insert(0, SIB)   # net.py may import c4 from its own folder
    spec = importlib.util.spec_from_file_location("sib_az_net", os.path.join(SIB, "net.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def test_mlx_and_numpy_match_sibling_net():
    g = _sib()
    sib, _ = g.load_model(az.CKPT)
    ours, _ = az.load_mlx()
    x = az.random_positions(64, seed=3)
    x = x if isinstance(x, np.ndarray) else np.asarray(x)
    with mx.stream(mx.cpu):
        p_ref, v_ref = (np.array(a) for a in sib(mx.array(x)))
        p, v = (np.array(a) for a in ours(mx.array(x)))
    assert np.abs(p - p_ref).max() == 0 and np.abs(v - v_ref).max() == 0
    out = az.NumpyNet()(x)
    pn, vn = (out["policy"], out["value"]) if isinstance(out, dict) else out
    assert np.abs(np.asarray(pn) - p_ref).max() < 1e-4 and np.abs(np.asarray(vn).reshape(-1) - v_ref).max() < 1e-4
