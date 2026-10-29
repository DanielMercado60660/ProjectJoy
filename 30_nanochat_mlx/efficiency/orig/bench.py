"""
Throughput benchmark: tokens/sec, estimated MFU and peak memory for a few model sizes,
with/without mx.compile and bf16 vs fp32 compute. Synthetic tokens, full optimizer step.

MFU denominator: we don't trust a spec-sheet number for the M5 Max GPU, so we measure the
throughput of a large dense matmul in the same dtype and report "MFU vs measured matmul peak".

    ../bin/gpu_run ../.venv/bin/python bench.py --out media/bench.json
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn

from gpt import GPT, config_for_depth, loss_fn
from optim import build_optimizer


def matmul_peak(dtype, n=8192, reps=20):
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    for _ in range(3):
        mx.eval(a @ b)
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
        mx.eval(c)
    dt = time.perf_counter() - t0
    return 2 * n ** 3 * reps / dt


def bench_train(depth, dtype, compile_, B=16, T=1024, steps=10, warmup=3):
    cfg = config_for_depth(depth, 8192, T)
    model = GPT(cfg)
    model.init_weights(0)
    model.compute_dtype = dtype
    opt, _ = build_optimizer(model, weight_decay=0.01)
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def step(x, y):
        loss, g = lg(model, x, y)
        opt.update(model, g)
        return loss

    if compile_:
        step = mx.compile(step, inputs=state, outputs=state)
    x = mx.random.randint(0, cfg.vocab_size, (B, T))
    y = mx.random.randint(0, cfg.vocab_size, (B, T))
    mx.eval(x, y)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    for _ in range(warmup):
        mx.eval(step(x, y), state)
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step(x, y)
        mx.eval(loss, state)
    dt = (time.perf_counter() - t0) / steps
    tps = B * T / dt
    return {"depth": depth, "params_M": model.num_params() / 1e6, "dtype": "bf16" if dtype == mx.bfloat16 else "fp32",
            "compile": compile_, "B": B, "T": T, "step_ms": dt * 1000, "tok_per_s": tps,
            "tflops": model.estimate_flops() * tps / 1e12, "peak_mem_gb": mx.get_peak_memory() / 1e9,
            "loss": float(loss.item())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,6,8,10")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--out", default="media/bench.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    res = {"matmul_peak_tflops": {}}
    for name, dt in [("bf16", mx.bfloat16), ("fp32", mx.float32)]:
        res["matmul_peak_tflops"][name] = matmul_peak(dt) / 1e12
        print(name, "matmul peak", f"{res['matmul_peak_tflops'][name]:.2f} TFLOP/s", flush=True)
    rows = []
    depths = [int(d) for d in a.depths.split(",")]
    for d in depths:
        combos = [(mx.bfloat16, True)] if a.quick else [(mx.bfloat16, True), (mx.bfloat16, False), (mx.float32, True), (mx.float32, False)]
        for dt, c in combos:
            r = bench_train(d, dt, c, B=a.B)
            r["mfu_vs_measured_peak"] = r["tflops"] / res["matmul_peak_tflops"][r["dtype"]]
            rows.append(r)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    # larger device batch for the pilot size (memory check)
    if not a.quick:
        r = bench_train(8, mx.bfloat16, True, B=32)
        r["mfu_vs_measured_peak"] = r["tflops"] / res["matmul_peak_tflops"]["bf16"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    res["rows"] = rows
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
