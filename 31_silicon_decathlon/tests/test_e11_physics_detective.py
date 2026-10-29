"""Parity: our copy of the Physics Detective posterior net (numpy + MLX) and data generator against the
sibling project's own sbi.py / physics.py (verification only)."""
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB = os.path.join(HERE, "..", "11_physics_detective")
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import physics_detective as pd  # noqa: E402


def _sibling():
    if SIB not in sys.path:
        sys.path.insert(0, SIB)
    import sbi  # noqa: E402
    return sbi


def test_data_parity():
    sbi = _sibling()
    rng = np.random.default_rng(3)
    d = sbi.sample_designs(rng, 64, pd.MAXN)
    y, c = sbi.observe(sbi.world1_positions(d[..., 0], d[..., 1], 0.3), rng)
    rng2 = np.random.default_rng(3)
    d2 = pd.sample_designs(rng2, 64, pd.MAXN)
    y2, c2 = pd.observe(pd.world1_positions(d2[..., 0], d2[..., 1], 0.3), rng2)
    np.testing.assert_array_equal(d, d2)
    np.testing.assert_array_equal(sbi.featurize(d, y, c), pd.featurize(d2, y2, c2))


def test_net_parity():
    sbi = _sibling()
    ref_model, _ = sbi.load(os.path.join(SIB, "checkpoints", "sbi"))
    f, m, _ = pd.fresh_datasets(200, seed=7)
    with mx.stream(mx.cpu):
        ref = [np.array(a) for a in ref_model(mx.array(f), mx.array(m))]
        ours = [np.array(a) for a in pd.load_mlx()(mx.array(f), mx.array(m))]
    for a, b in zip(ours, ref):
        assert np.abs(a - b).max() < 1e-5
    for a, b in zip(pd.NumpyNet()(f, m), ref):  # exact-erf GELU in float32 numpy
        assert np.abs(a - b).max() < 1e-3
    # grid posterior summary matches the sibling's posterior_on_grid
    with mx.stream(mx.cpu):
        W = sbi.posterior_on_grid(ref_model, f, m, pd.GRID)
    mean, _ = pd.grid_summary(*ref)
    assert np.abs(W @ pd.GRID - mean).max() < 1e-9
