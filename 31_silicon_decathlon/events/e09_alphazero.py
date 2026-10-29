"""Event 09: Tiny AlphaZero policy/value net (5 residual blocks, 64 channels), one forward pass.

Latency batch 1 = one MCTS leaf evaluation. Throughput batch 256 = a self-play wave.
Task check: does the athlete pick the same best move as MLX fp32 on positions from random play?
"""
import os

import mlx.core as mx
import mlx.utils
import numpy as np

import coreml_util
from models import alphazero as az

EVENT = dict(id="09_alphazero", project="09_tiny_alphazero", model="PVNet 5x64 ResNet (policy + value)",
             checkpoint=az.CKPT, unit="positions", params=None,
             notes="Inputs are positions from random legal play (0-29 plies).")
BATCHES = (1, 256)
SKIPS = {"mlx_gpu_q8": "conv-heavy net; MLX quantization covers Linear layers only",
         "mlx_gpu_q4": "conv-heavy net; MLX quantization covers Linear layers only"}

_cache = {}


def _mlx():
    if "mlx" not in _cache:
        m, _ = az.load_mlx()
        _cache["mlx"] = m
        EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    return _cache["mlx"]


def make_inputs(batch, seed=0):
    return {"x": az.random_positions(batch, seed=seed)}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def fn(inp):
        x = mx.array(inp["x"]).astype(dtype)
        if stream is not None:
            with mx.stream(stream):
                p, v = fwd(x)
                mx.eval(p, v)
        else:
            p, v = fwd(x)
            mx.eval(p, v)
        return {"policy": np.array(p.astype(mx.float32)), "value": np.array(v.astype(mx.float32))}
    return fn


def coreml_packages(batch):
    x = make_inputs(batch)["x"]
    path = coreml_util.export_torch(az.torch_net(), (x,), ["x"], f"e09_alphazero_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = _mlx()
    m16 = az.PVNet()
    m16.load_weights(az.CKPT + ".safetensors")
    m16.eval()
    m16.set_dtype(mx.float16)
    npnet = az.NumpyNet()
    out = {
        "numpy_cpu": lambda inp: dict(zip(("policy", "value"), npnet(inp["x"]))),
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(m16, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
    }
    paths = coreml_packages(batch)
    for name, path in paths.items():
        out[name] = coreml_util.make_athlete(path, name, rename=["policy", "value"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    return dict(policy_max_err=float(np.abs(o["policy"] - ref["policy"]).max()),
                value_max_err=float(np.abs(np.ravel(o["value"]) - np.ravel(ref["value"])).max()))


def task_check(name, batch):
    """Best-move agreement with MLX fp32 on 2000 fresh positions (fixed seed), run at this batch size."""
    if batch != BATCHES[-1]:
        return None
    ath = athletes(batch)
    fn, ref_fn = ath[name], ath["mlx_gpu"]
    agree, n = 0, 0
    for s in range(1, 1 + 2000 // batch):
        inp = make_inputs(batch, seed=100 + s)
        a, r = fn(inp)["policy"].argmax(-1), ref_fn(inp)["policy"].argmax(-1)
        agree += int((a == r).sum()); n += batch
    return dict(best_move_agreement=agree / n, positions=n)


# Tolerances for batches without a task check: fp16 engines drift by ~0.01-0.05 logits here, and a
# best move only flips when two logits are within that of each other. Stated up front, not tuned.
POLICY_TOL, VALUE_TOL = 0.1, 0.05


def qualifies(agreement, task):
    """Same model = picks the same best move as MLX fp32 on >= 99% of positions (throughput batch);
    where no task check ran, policy logits within 0.1 and value within 0.05 of MLX fp32."""
    if task:
        a = task["best_move_agreement"]
        return a >= 0.99, f"best-move agreement {a:.4f} {'>=' if a >= 0.99 else '<'} 0.99"
    p, v = agreement["policy_max_err"], agreement["value_max_err"]
    ok = p <= POLICY_TOL and v <= VALUE_TOL
    return ok, f"policy err {p:.3g} (tol {POLICY_TOL}), value err {v:.3g} (tol {VALUE_TOL})"
