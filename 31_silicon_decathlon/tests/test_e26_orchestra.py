"""Parity: vendored Neural Orchestra decoder/classifier (MLX, NumPy, torch) vs the sibling's own model.py.
Verification only: this is the one place that loads sibling code.
  ../.venv/bin/python -m pytest -q tests/test_e26_orchestra.py
"""
import importlib.util
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
from models import orchestra as orc  # noqa: E402

SIB = os.path.join(HERE, "..", "26_neural_orchestra")


def _sib():
    spec = importlib.util.spec_from_file_location("sib_orchestra_model", os.path.join(SIB, "model.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def test_decoder_and_classifier_parity():
    g = _sib()
    sib, _ = g.load_vae(orc.CKPT)
    z, s, style = orc.random_latents(64, seed=1)
    with mx.stream(mx.cpu):
        # the sibling's own decode path, threshold and demo 'sample' mode
        h_ref, v_ref, o_ref, p_ref = g.decode_numpy(sib, z, style, mode="threshold")
        hs_ref, vs_ref, os_ref, _ = g.decode_numpy(sib, z, style, mode="sample", rng=np.random.default_rng(7))
        p, v, o = (np.array(a) for a in orc.load_mlx()(mx.array(z), mx.array(s)))
        clf = g.load_classifier(orc.CLF)
        logits_ref = np.array(clf(mx.array(g.to_input(h_ref, v_ref, o_ref))))
    assert np.abs(p.reshape(p_ref.shape) - p_ref).max() < 1e-5
    h, vv, oo = orc.to_pattern(p, v, o, "threshold", 0)
    assert (h == h_ref).all() and np.abs(vv - v_ref).max() < 1e-5 and np.abs(oo - o_ref).max() < 1e-5
    hs, _, _ = orc.to_pattern(p, v, o, "sample", 7)
    assert (hs == hs_ref).all()
    pn, vn, on = orc.NumpyDecoder()(z, s)
    assert max(np.abs(pn - p).max(), np.abs(vn - v).max(), np.abs(on - o).max()) < 1e-5
    assert np.abs(orc.NumpyClassifier()(h_ref, v_ref, o_ref) - logits_ref).max() < 1e-4


def test_torch_parity():
    import torch
    z, s, _ = orc.random_latents(16, seed=2)
    with mx.stream(mx.cpu):
        ref = [np.array(a) for a in orc.load_mlx()(mx.array(z), mx.array(s))]
    with torch.no_grad():
        out = [a.numpy() for a in orc.torch_decoder()(torch.from_numpy(z), torch.from_numpy(s))]
    assert max(np.abs(a - b).max() for a, b in zip(out, ref)) < 1e-5
