"""Round 6 tests: ../.venv/bin/python -m pytest -q test_r6.py   (a few seconds; no training, no network)."""
import os
import numpy as np
import mlx.core as mx

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(ROOT, "checkpoints", "followup_val3k", "best.safetensors")


def test_standalone_mel_matches_project_mel():
    import mel, r6_vocoder
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 2) * 0.1).astype(np.float32)
    assert np.array_equal(mel.MEL_FB, r6_vocoder.MEL_FB)
    assert np.array_equal(mel.log_mel(x), r6_vocoder.log_mel(x))


def test_old_checkpoint_loads_and_matches_round3_model():
    from model import load_vocoder
    from r6_vocoder import load, vocode
    m = np.random.default_rng(1).standard_normal((40, 80)).astype(np.float32) - 5
    old, _ = load_vocoder(OLD)
    a = np.array(old(mx.array(m[None]))[0])
    b = vocode(load(OLD), m)
    assert a.shape == b.shape == (39 * 256,)
    assert np.max(np.abs(a - b)) < 1e-5


def test_phase_grid_matches_generator_istft():
    from r6_train import target_stft, phase_loss
    from r6_vocoder import Vocoder
    G = Vocoder(dim=32, n_blocks=1)
    y = mx.array((np.random.default_rng(2).standard_normal((2, 64 * 256)) * 0.1).astype(np.float32))
    S = target_stft(y)
    assert S.shape == (2, 65, 513)
    rec = G.istft(mx.real(S), mx.imag(S))
    assert float(mx.max(mx.abs(rec - y)[:, 1024:-1024])) < 1e-5
    ph = mx.arctan2(mx.imag(S), mx.real(S))
    assert float(phase_loss(ph + 2 * np.pi, y)) < 1e-3  # anti-wrapping: a 2*pi shift costs nothing


def test_discriminators_shapes_and_losses():
    from r6_disc import Discriminators, d_hinge, g_hinge, feat_match
    D = Discriminators("small", 8)
    y = mx.array(np.random.default_rng(3).standard_normal((2, 8192)).astype(np.float32) * 0.1)
    mp, mr = D(y)
    assert len(mp) == 5 and len(mr) == 3
    assert float(feat_match(mp, mp)) == 0.0
    assert np.isfinite(float(d_hinge(mp, mp))) and np.isfinite(float(g_hinge(mr)))
