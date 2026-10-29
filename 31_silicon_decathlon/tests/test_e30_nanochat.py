"""Parity: vendored Nanochat GPT (MLX, NumPy, torch) vs the sibling's own gpt.py at fp32.
Verification only: this is the one place that loads sibling code.
  ../.venv/bin/python -m pytest -q tests/test_e30_nanochat.py
"""
import importlib.util
import json
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
from models import nanochat as nc  # noqa: E402

SIB = os.path.join(HERE, "..", "30_nanochat_mlx")


def _sibling():
    spec = importlib.util.spec_from_file_location("sib_gpt", os.path.join(SIB, "gpt.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    cfg = json.load(open(os.path.join(nc.CKPT, "meta.json")))["config"]
    m = g.GPT(g.GPTConfig(**cfg))
    m.load_weights(os.path.join(nc.CKPT, "model.safetensors"))
    m.compute_dtype = mx.float32
    mx.eval(m.parameters())
    return m


def test_rope_matches_mlx():
    x = np.random.default_rng(0).standard_normal((1, 3, 7, 128)).astype(np.float32)
    for off in (0, 255):
        ref = np.array(mx.fast.rope(mx.array(x), 128, traditional=False, base=nc.ROPE_BASE, scale=1.0, offset=off))
        cos, sin = nc.rope_tables(7, 128, off)
        assert np.abs(nc._np_rope(x, cos, sin) - ref).max() < 1e-4


def test_prefill_and_decode_parity():
    sib = _sibling()
    ids = nc.val_window(64, seed=3)[None, :64]
    with mx.stream(mx.cpu):  # CPU fp32 for a tight comparison
        ref = np.array(sib(mx.array(ids)))
        mine = nc.load_mlx()
        out, kv = mine.forward(mx.array(ids))
        assert np.abs(np.array(out) - ref).max() < 1e-4
        # sibling KV-cache decode == my pure-function decode at the last position
        cache = sib.make_cache()
        sib(mx.array(ids[:, :-1]), cache=cache)
        sib_step = np.array(sib(mx.array(ids[:, -1:]), cache=cache))
        _, past = mine.forward(mx.array(ids[:, :-1]))
        step, _ = mine.forward(mx.array(ids[:, -1:]), past)
        assert np.abs(np.array(step) - sib_step).max() < 1e-4
        assert np.abs(sib_step[0, 0] - ref[0, -1]).max() < 1e-3
    npm = nc.NumpyGPT()
    o, past = npm.forward(ids)
    assert np.abs(o - ref).max() < 1e-3
    o1, _ = npm.forward(ids[:, :-1])
    _, past = npm.forward(ids[:, :-1])
    st, _ = npm.forward(ids[:, -1:], past)
    assert np.abs(st[0, 0] - sib_step[0, 0]).max() < 1e-3


def test_torch_prefill_parity():
    import torch
    sib = _sibling()
    ids = nc.val_window(64, seed=4)[None, :64]
    with mx.stream(mx.cpu):
        ref = np.array(sib(mx.array(ids)))
    with torch.no_grad():
        out = nc.torch_prefill(64)(torch.from_numpy(ids)).numpy()
    assert np.abs(out - ref).max() < 1e-3


def test_quantized_is_quantized():
    q = nc.load_quantized(4)
    ids = mx.array(nc.val_window(16, seed=5)[None, :16])
    a, b = np.array(q(ids)), np.array(nc.load_mlx()(ids))
    assert np.abs(a - b).max() > 1e-3
