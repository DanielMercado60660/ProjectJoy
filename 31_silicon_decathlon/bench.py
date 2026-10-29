"""Timing harness shared by every event in this folder.

Every athlete is a function inputs(dict of numpy) -> outputs(dict of numpy).
Timing is host-to-host: numpy in, numpy out, so every engine pays its own
transfer and sync cost and nothing is left running asynchronously.
"""
import fcntl
import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(HERE, ".bench.lock")


@contextmanager
def exclusive():
    """Only one timed section runs at a time across everyone using this folder."""
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


def busy_processes(min_cpu=20.0):
    """Other processes using real CPU right now (excluding this one)."""
    out = _sh(["ps", "-Ao", "pid,pcpu,comm", "-r"]).splitlines()[1:]
    me = os.getpid()
    busy = []
    for line in out:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, cpu, comm = int(parts[0]), float(parts[1]), parts[2]
        if pid != me and cpu >= min_cpu:
            busy.append(dict(pid=pid, cpu=cpu, comm=os.path.basename(comm)))
    return busy


def env_snapshot():
    batt = _sh(["pmset", "-g", "batt"]).splitlines()
    therm = _sh(["pmset", "-g", "therm"])
    import mlx.core as mx
    try:
        import coremltools as ct
        ctv = ct.__version__
    except Exception:
        ctv = None
    return dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        loadavg=os.getloadavg(),
        power=batt[0] if batt else "",
        battery=batt[1].strip() if len(batt) > 1 else "",
        thermal=therm,
        busy_processes=busy_processes(),
        mlx=mx.__version__,
        coremltools=ctv,
        numpy=np.__version__,
        machine=platform.machine(),
        macos=platform.mac_ver()[0],
    )


def noise_probe():
    """Fixed reference workloads. Compare before/after an event to spot interference."""
    import mlx.core as mx
    a = np.random.default_rng(0).standard_normal((512, 512)).astype(np.float32)
    t = time.perf_counter()
    for _ in range(50):
        a @ a
    cpu_ms = (time.perf_counter() - t) / 50 * 1e3
    g = mx.random.normal((2048, 2048), key=mx.random.key(0))
    mx.eval(g @ g)
    t = time.perf_counter()
    for _ in range(20):
        mx.eval(g @ g)
    gpu_ms = (time.perf_counter() - t) / 20 * 1e3
    return dict(numpy_512_matmul_ms=cpu_ms, mlx_gpu_2048_matmul_ms=gpu_ms)


def _call_times(fn, inputs, budget_s, max_calls):
    """Call fn back to back for about budget_s seconds; return per-call ms."""
    times = []
    end = time.perf_counter() + budget_s
    while len(times) < max_calls:
        t = time.perf_counter()
        fn(inputs)
        t2 = time.perf_counter()
        times.append((t2 - t) * 1e3)
        if t2 > end and len(times) >= 3:
            break
    return times


def race(athletes, inputs, repeats=7, budget_s=0.25, warmup=5, max_calls=2000, seed=0):
    """Time every athlete on the same inputs, interleaved.

    Each round runs every athlete once for about budget_s seconds, in a shuffled order,
    so slow drift (thermal throttling, background load) hits all athletes alike.
    Returns {name: dict(median_ms, p90_ms, min_ms, calls, rounds)}.
    """
    rng = random.Random(seed)
    for name, fn in athletes.items():
        for _ in range(warmup):
            fn(inputs)
    samples = {name: [] for name in athletes}
    round_medians = {name: [] for name in athletes}
    names = list(athletes)
    for _ in range(repeats):
        rng.shuffle(names)
        for name in names:
            ts = _call_times(athletes[name], inputs, budget_s, max_calls)
            samples[name].extend(ts)
            round_medians[name].append(float(np.median(ts)))
    out = {}
    for name, ts in samples.items():
        ts = np.asarray(ts)
        out[name] = dict(
            median_ms=float(np.median(ts)),
            p90_ms=float(np.percentile(ts, 90)),
            min_ms=float(ts.min()),
            calls=int(ts.size),
            rounds=repeats,
            round_median_spread=float(np.max(round_medians[name]) / np.min(round_medians[name])),
        )
    return out


def mlx_peak_memory(fn, inputs):
    """Peak MLX allocation (bytes) during one call. Only meaningful for MLX athletes."""
    import mlx.core as mx
    mx.clear_cache()
    mx.reset_peak_memory()
    fn(inputs)
    return int(mx.get_peak_memory())


def max_abs_err(out, ref):
    return {k: float(np.max(np.abs(np.asarray(out[k], np.float64) - np.asarray(ref[k], np.float64))))
            for k in ref}


def dir_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def mlx_gpu_matmul_rel_err(m=64, k=4096, n=512):
    """Measured precision of an MLX GPU fp32 matmul (M >= 2) vs float64: ~5e-7 = true fp32, ~8e-4 = TF32.
    MLX reads MLX_ENABLE_TF32 once, at the first GPU matmul of the process (default: TF32 on)."""
    import mlx.core as mx
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    ref = a.astype(np.float64) @ w.astype(np.float64)
    with mx.stream(mx.gpu):
        y = mx.array(a) @ mx.array(w)
        mx.eval(y)
    return float(np.abs(np.array(y) - ref).max() / np.abs(ref).max())
