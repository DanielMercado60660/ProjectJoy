"""Parity for event 29: our flow U-Net / NumPy / torch copies and the classifier vs the sibling's own code.

Verification only: this is the one place the sibling's code is imported.
"""
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import image_gen as IG  # noqa: E402

SIB = os.path.join(HERE, "..", "29_image_gen")


_SIBMODS = {}


def _sib():
    """Import the sibling's train_gen/train_classifier. Its `models.py` clashes with our `models` package,
    so swap ours out of sys.modules while importing, then put it back."""
    if not _SIBMODS:
        ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "models" or k.startswith("models.")}
        sys.path.insert(0, SIB)
        try:
            import train_classifier
            import train_gen
            _SIBMODS.update(tg=train_gen, tc=train_classifier)
        finally:
            sys.path.remove(SIB)
            sys.modules.pop("models", None)
            sys.modules.update(ours)
    return _SIBMODS["tg"], _SIBMODS["tc"]


def _inputs(n=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 32, 32, 3)).astype(np.float32), np.array([0.0, 625.0][:n], np.float32)


def test_unet_matches_sibling():
    tg, _ = _sib()
    _, g, _ = tg.load_generator(IG.CKPT)
    ours = IG.load_mlx()
    x, t = _inputs()
    with mx.stream(mx.cpu):
        a, b = np.array(g(mx.array(x), mx.array(t))), np.array(ours(mx.array(x), mx.array(t)))
    assert np.abs(a - b).max() < 1e-6


def test_sampler_matches_sibling():
    """Our Euler loop (fixed noise in, 8 steps) equals train_gen._sample_batch for the same noise."""
    tg, _ = _sib()
    _, g, cfg = tg.load_generator(IG.CKPT)
    ours = IG.load_mlx()
    key = mx.random.key(3)
    k1, _ = mx.random.split(key)
    with mx.stream(mx.cpu):
        ref = np.array(tg._sample_batch("flow", g, 2, cfg, key, 8, "euler", None, 0.0))
        x = mx.random.normal((2, 32, 32, 3), key=k1)
        for t0, t1 in IG.euler_times(8):
            x = x + (t1 - t0) * ours(x, mx.full((2,), t0 * 1000.0))
        x = np.array(mx.clip(x, -1, 1))
    assert np.abs(ref - x).max() < 1e-5


def test_classifier_matches_sibling():
    _, tc = _sib()
    ref = tc.load_classifier(32)
    ours = IG.load_classifier()
    x = np.random.default_rng(1).integers(0, 256, (4, 32, 32, 3)).astype(np.uint8)
    with mx.stream(mx.cpu):
        logits, _ = ref(mx.array(x).astype(mx.float32) / 127.5 - 1)
        a = np.array(logits)
    with mx.stream(mx.cpu):
        b = np.array(ours(mx.array(x).astype(mx.float32) / 127.5 - 1))
    assert np.abs(a - b).max() < 1e-5


def test_numpy_and_torch_match_mlx_cpu():
    import torch
    ours = IG.load_mlx()
    x, t = _inputs()
    with mx.stream(mx.cpu):
        a = np.array(ours(mx.array(x), mx.array(t)))
    b = IG.NumpyUNet()(x, t)
    with torch.no_grad():
        c = IG.torch_net()(torch.from_numpy(x), torch.from_numpy(t)).numpy()
    assert np.abs(a - b).max() < 1e-4, np.abs(a - b).max()
    assert np.abs(a - c).max() < 1e-4, np.abs(a - c).max()
