"""Reviewer probe: does VECLIB_MAXIMUM_THREADS change NumPy (Accelerate) speed here? 2048^2 fp32 matmul."""
import os
import time

import numpy as np

a = np.random.rand(2048, 2048).astype(np.float32)
b = np.random.rand(2048, 2048).astype(np.float32)
a @ b
ts = []
for _ in range(20):
    t = time.perf_counter()
    a @ b
    ts.append(time.perf_counter() - t)
t0, w0 = time.process_time(), time.perf_counter()
for _ in range(20):
    a @ b
cpu = (time.process_time() - t0) / (time.perf_counter() - w0)
print(f"VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS', 'unset')}: best {min(ts) * 1e3:.2f} ms, "
      f"{2 * 2048 ** 3 / min(ts) / 1e12:.2f} TFLOP/s, CPU-s per wall-s {cpu:.2f}")
