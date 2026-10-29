"""Event 29: Monster Factory flow-matching U-Net (6.47M params, 2.89 GFLOP/image), one network evaluation.

One item = one U-Net forward for one 32x32 image (one Euler step). Batch 1 = one image; batch 64 = the
training batch / sample grid. The README says ~8 Euler steps suffice, so a full sample is 8 items; the
task check records the wall time of the full 8-step sample per athlete (single shot, not raced).
Task check (both batches): 64 (batch 64) or 16 (batch 1) samples from fixed noise, 8 Euler steps, every
step through the athlete host to host; the project's classifier_32 (MLX CPU fp32, the fixed judge) reads
the 6 attributes of each sample; compare labels and pixels to the MLX fp32 athlete's samples.
"""
import time

import mlx.core as mx
import numpy as np

import coreml_util
from models import image_gen as IG

EVENT = dict(id="29_image_gen", project="29_image_gen", model="Flow-matching U-Net 64x[1,2,2], attention at 8x8 (EMA)",
             checkpoint=IG.CKPT, unit="U-Net evaluations (one 32x32 image, one Euler step)", params=None,
             notes="Timed call = one U-Net forward; a full sample = 8 calls (Euler-8). Inputs: Gaussian x, "
                   "t from the Euler-8 grid (fixed seed). Core ML packages are fp16.")
BATCHES = (1, 64)
SKIPS = {"mlx_gpu_q8": "conv-dominated U-Net; MLX quantization covers Linear layers only",
         "mlx_gpu_q4": "conv-dominated U-Net; MLX quantization covers Linear layers only"}
STEPS = 8

_cache = {}


def _model(dtype=mx.float32):
    if ("mlx", dtype) not in _cache:
        m = IG.load_mlx(dtype=dtype)
        _cache[("mlx", dtype)] = m
        if dtype == mx.float32:
            import mlx.utils
            EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    return _cache[("mlx", dtype)]


def make_inputs(batch, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.array([t0 * 1000.0 for t0, _ in IG.euler_times(STEPS)], np.float32)
    return {"x": rng.standard_normal((batch, IG.RES, IG.RES, 3)).astype(np.float32),
            "t": rng.choice(grid, batch).astype(np.float32)}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def fn(inp):
        x, t = mx.array(inp["x"]).astype(dtype), mx.array(inp["t"])
        if stream is not None:
            with mx.stream(stream):
                v = fwd(x, t)
                mx.eval(v)
        else:
            v = fwd(x, t)
            mx.eval(v)
        return {"v": np.array(v.astype(mx.float32))}
    return fn


def coreml_packages(batch):
    inp = make_inputs(batch)
    path = coreml_util.export_torch(IG.torch_net(), (inp["x"], inp["t"]), ["x", "t"], f"e29_image_gen_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = _model()
    npn = IG.NumpyUNet()
    out = {
        "numpy_cpu": lambda inp: {"v": npn(inp["x"], inp["t"])},
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(_model(mx.float16), dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = coreml_util.make_athlete(path, name, rename=["v"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    d = np.abs(o["v"] - ref["v"])
    return dict(v_max_err=float(d.max()), v_mean_err=float(d.mean()))


# ---------- task check ----------
def sample(fn, n, batch, seed=2029):
    """n images from fixed noise, Euler-STEPS, each step through fn at this batch. Returns (x [-1,1], ms per batch)."""
    x = np.random.default_rng(seed).standard_normal((n, IG.RES, IG.RES, 3)).astype(np.float32)
    t_start = time.perf_counter()
    for t0, t1 in IG.euler_times(STEPS):
        for k in range(0, n, batch):
            xb = x[k:k + batch]
            v = fn({"x": xb, "t": np.full((len(xb),), t0 * 1000.0, np.float32)})["v"]
            x[k:k + batch] = xb + (t1 - t0) * v
    ms_per_batch = (time.perf_counter() - t_start) * 1e3 / (n // batch)
    return np.clip(x, -1, 1), ms_per_batch


def task_check(name, batch):
    ath = athletes(batch)
    n = 64 if batch >= 64 else 16
    if "clf" not in _cache:
        _cache["clf"] = IG.load_classifier()
    clf = _cache["clf"]
    if ("ref", batch) not in _cache:
        rx, rms = sample(ath["mlx_gpu"], n, batch)
        _cache[("ref", batch)] = (rx, IG.classify(clf, IG.to_uint8(rx)), rms)
    rx, rl, rms = _cache[("ref", batch)]
    if name == "mlx_gpu":
        x, labels, ms = rx, rl, rms
    else:
        x, ms = sample(ath[name], n, batch)
        labels = IG.classify(clf, IG.to_uint8(x))
    d = np.abs(x - rx)
    return dict(samples=n, steps=STEPS,
                attr_agreement=float((labels == rl).mean()),
                recipe_agreement=float((labels == rl).all(axis=1).mean()),
                per_attr_agreement={a: float((labels[:, j] == rl[:, j]).mean()) for j, a in enumerate(IG.ATTR_NAMES)},
                pixel_max_err=float(d.max()), pixel_mean_err=float(d.mean()),
                sample8_ms_per_call_batch=ms, sample8_note="wall time of one full 8-step sample of this batch, "
                                                           "host to host per step; single shot, not raced")


def qualifies(agreement, task):
    """Same generator if the classifier reads the same attributes off its samples as off MLX fp32's:
    >= 97% of the 6 x N attribute labels agree. Without a task: velocity max err <= 0.05."""
    if task:
        a = task["attr_agreement"]
        return a >= 0.97, (f"attribute labels agree {a:.1%} (recipe {task['recipe_agreement']:.0%}) on "
                           f"{task['samples']} Euler-8 samples; pixel mean err {task['pixel_mean_err']:.3g}")
    e = agreement["v_max_err"]
    return e <= 0.05, f"velocity max err {e:.2g} (tolerance 0.05)"
