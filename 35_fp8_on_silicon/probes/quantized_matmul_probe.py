"""Quick probe: quantized matmul (mxfp8/mxfp4/nvfp4/affine) vs bf16 on this Mac, plus gradient support.
Single non-interleaved runs; read as rough. Run: ../../.venv/bin/python quantized_matmul_probe.py"""
import time
import mlx.core as mx


def bench(f, n=20):
    for _ in range(3):
        mx.eval(f())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    mx.synchronize()
    return (time.perf_counter() - t) / n


print("mlx", mx.__version__)
for M, K, N in [(8192, 1024, 4096), (8192, 2048, 8192), (1, 2048, 8192)]:
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    tb = bench(lambda: x @ w.T)
    row = f"M={M} K={K} N={N}: bf16 {tb*1e3:.3f} ms ({2*M*K*N/tb/1e12:.1f} TF/s)"
    for mode in ["mxfp8", "mxfp4", "nvfp4", "affine"]:
        q = mx.quantize(w, mode=mode)
        b = q[2] if len(q) > 2 else None
        tq = bench(lambda: mx.quantized_matmul(x, q[0], q[1], b, transpose=True, mode=mode))
        row += f" | {mode} {tq*1e3:.3f} ms ({tb/tq:.2f}x)"
    print(row)

x = mx.random.normal((64, 1024)).astype(mx.bfloat16)
w = mx.random.normal((256, 1024)).astype(mx.bfloat16)
q = mx.quantize(w, mode="mxfp8")
b = q[2] if len(q) > 2 else None
g = mx.grad(lambda x: mx.quantized_matmul(x, q[0], q[1], b, transpose=True, mode="mxfp8").sum())(x)
mx.eval(g)
print("grad wrt activations: OK", g.shape)
try:
    mx.eval(mx.grad(lambda s: mx.quantized_matmul(x, q[0], s, b, transpose=True, mode="mxfp8").sum())(q[1]))
    print("grad wrt scales: OK")
except Exception as e:
    print("grad wrt weights/scales: ERR", str(e)[:120])
