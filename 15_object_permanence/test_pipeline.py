"""Fast unit checks: ../.venv/bin/python -m pytest -q"""
import mlx.core as mx
import numpy as np

import kalman
import metrics
import sim
from model import POS_SCALE, Tracker, gaussian_nll, to_pixels
from train import render_mx


def _tiny(seed=0, **over):
    rng = np.random.default_rng(seed)
    cfg = dict(sim.BASE, **over)
    pos, pres, occ, col, _ = sim.simulate_one(rng, cfg)
    return pos[None], pres[None], occ[None], col[None]


def test_render_parity():
    pos, pres, occ, col = _tiny()
    a = sim.render(pos, pres, occ, col, noise=0)
    b = np.array(render_mx(mx.array(pos), mx.array(pres.astype(np.float32)), mx.array(occ), mx.array(col), noise=0))
    assert np.abs(a - b).max() < 1e-5


def test_balls_in_bounds_and_visible_at_start():
    for s in range(5):
        pos, pres, occ, col = _tiny(s, collisions=True, hidden_turn=0.5)
        p = pos[0][:, pres[0]]
        assert p.min() >= sim.R - 1e-4 and p.max() <= sim.W - sim.R + 1e-4
        vis = sim.occ_visible_frac(p[..., 0], [o for o in occ[0] if o[0] >= 0])
        assert (vis[:2] > 0.999).all()


def test_kalman_tracks_clean_sequence():
    pos, pres, occ, col = _tiny(3)
    frames = sim.render(pos, pres, occ, col, noise=0.0)[0]
    m, c = kalman.track_sequence(frames, col[0], pres[0], q=0.02, r=0.2, bounce=True)
    vis = sim.occ_visible_frac(pos[0][..., 0], [o for o in occ[0] if o[0] >= 0])
    err = np.linalg.norm(m - pos[0], axis=-1)[(vis > 0.999) & pres[0][None]]
    assert err.mean() < 0.5


def test_calibrated_gaussian_has_nominal_coverage():
    rng = np.random.default_rng(0)
    L = np.array([[2.0, 0], [0.7, 0.5]])
    cov = L @ L.T
    gt = rng.normal(size=(20000, 2)) @ L.T
    err, d2, nll = metrics.per_frame(np.zeros_like(gt), np.broadcast_to(cov, (len(gt), 2, 2)), gt)
    assert abs((d2 <= 1).mean() - metrics.EXP1) < 0.02
    assert abs((d2 <= 4).mean() - metrics.EXP2) < 0.02


def test_model_shapes_and_nll_matches_numpy():
    m = Tracker(K=3, enc_dim=32, hid=16)
    out = m(mx.zeros((2, 4, 64, 64, 3)))
    assert out.shape == (2, 4, 3, 5)
    tgt = mx.array(np.random.default_rng(0).normal(size=(2, 4, 3, 2)).astype(np.float32) * 0.1)
    nll_mx = np.array(gaussian_nll(out, tgt))
    mean, cov = to_pixels(np.array(out))
    _, _, nll_np = metrics.per_frame(mean, cov, np.array(tgt) * POS_SCALE + 32)
    # pixel-space NLL = normalized NLL + 2*log(POS_SCALE)
    assert np.allclose(nll_np, nll_mx + 2 * np.log(POS_SCALE), atol=1e-3)
