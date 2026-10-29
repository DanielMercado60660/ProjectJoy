"""Tiny-size checks that the efficiency-pass code paths compute the same thing as the reference."""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt import GPT, config_for_depth, loss_fn  # noqa: E402
from optim import build_optimizer  # noqa: E402
from train import make_update  # noqa: E402


def _model(dtype=mx.float32, **flags):
    m = GPT(config_for_depth(2, 512, 64))
    m.init_weights(0)
    mx.random.seed(3)
    for b in m.h:
        b.attn.c_proj.weight = mx.random.normal(b.attn.c_proj.weight.shape) * 0.02
        b.mlp.c_proj.weight = mx.random.normal(b.mlp.c_proj.weight.shape) * 0.02
    m.lm_head.weight = mx.random.normal(m.lm_head.weight.shape) * 0.05
    m.compute_dtype = dtype
    flags.setdefault("ce_mode", "ref")  # the reference path unless a test asks for another
    m.set_fast_flags(**flags)
    return m


def _batch(seed):
    r = np.random.default_rng(seed)
    x = r.integers(0, 512, (4, 64)).astype(np.int32)
    return x, np.roll(x, -1, axis=1)


def _rel(a, b):
    return float(mx.linalg.norm((a - b).reshape(-1)) / mx.linalg.norm(b.reshape(-1)))


def test_fused_qkv_same_logits_and_cache_path():
    x, _ = _batch(0)
    a = _model()(mx.array(x))
    b = _model(fused_qkv=True)(mx.array(x))
    assert _rel(b, a) < 1e-5
    from engine import generate, generate_naive
    m = _model(fused_qkv=True)
    ids = [int(i) for i in x[0, :10]]
    assert generate_naive(m, ids, 6) == list(generate(m, ids, 6))


def test_kernel_is_the_default_ce_mode():
    assert GPT(config_for_depth(2, 512, 64)).ce_mode == "kernel"


def test_fused_ce_kernel_matches_reference():
    x, y = _batch(1)
    la = loss_fn(_model(), mx.array(x), mx.array(y))
    lb = loss_fn(_model(ce_mode="kernel"), mx.array(x), mx.array(y))
    assert abs(float(la) - float(lb)) < 1e-4
    mask = mx.array(np.tile((np.arange(64) >= 32).astype(np.float32), (4, 1)))
    la = loss_fn(_model(), mx.array(x), mx.array(y), mask)
    lb = loss_fn(_model(ce_mode="kernel"), mx.array(x), mx.array(y), mask)
    assert abs(float(la) - float(lb)) < 1e-4
    from fused_ce import capped_ce, _ref
    z = (mx.random.normal((8, 512)) * 40).astype(mx.bfloat16)
    t = mx.array(np.random.default_rng(2).integers(0, 512, 8).astype(np.int32))
    assert float(mx.abs(capped_ce(z, t) - _ref(z, t)).max()) < 1e-4
    ga = mx.grad(lambda z: capped_ce(z, t).sum())(z).astype(mx.float32)
    gb = mx.grad(lambda z: _ref(z, t).sum())(z).astype(mx.float32)
    assert _rel(ga, gb) < 1e-2


def _deltas(accum_mode="ref", batched=False, addmm=False):
    m = _model()
    opt, _ = build_optimizer(m, weight_decay=0.05, muon_batched=batched, muon_addmm=addmm)
    upd = make_update(m, opt, 2, mode=accum_mode)
    from mlx.utils import tree_flatten
    p0 = {k: mx.array(v) for k, v in tree_flatten(m.parameters())}
    losses = []
    for s in range(2):
        bs = [_batch(10 * s + k) for k in range(2)]
        losses.append(upd(lambda k: bs[k]))
    return {k: v - p0[k] for k, v in tree_flatten(m.parameters())}, losses


def test_fast_accumulation_matches_reference_loop():
    da, la = _deltas("ref")
    db, lb = _deltas("fast")
    assert np.allclose(la, lb, rtol=1e-5)
    # Muon orthogonalises in bf16, so a 1e-7 change in summation order can flip bf16 roundings
    # and move a delta by ~1%; at the real d6 config the two loops agree to ~1e-10
    # (efficiency/parity.json). Gate on the global delta.
    num = sum(float(mx.sum((db[k] - da[k]) ** 2)) for k in da)
    den = sum(float(mx.sum(da[k] ** 2)) for k in da)
    assert (num / den) ** 0.5 < 2e-2


def test_batched_muon_matches_per_matrix_muon():
    da, _ = _deltas(batched=False)
    db, _ = _deltas(batched=True)
    num = sum(float(mx.sum((db[k] - da[k]) ** 2)) for k in da if k.startswith("h."))
    den = sum(float(mx.sum(da[k] ** 2)) for k in da if k.startswith("h."))
    assert (num / den) ** 0.5 < 2e-2
