"""Event 10: Self-Healing Pixels NCA (lizard_regen, 48 -> 128 -> 16 per-cell MLP), one update step.

One item = one NCA update step of one creature on the 72x72 demo grid (the demo runs one creature;
eval.py batches 32 trials). Inputs are grown creatures (200 steps from the seed, fixed fire masks)
plus a fixed fire mask, so the stochastic update is identical for every athlete.
Task check (both batches): grow -> cut off the left half -> 100 steps of regrowth, every step through the athlete
(host to host), with the same fire masks; RGBA MSE vs the lizard target compared to MLX fp32.
"""
import mlx.core as mx
import numpy as np

import coreml_util
from models import nca

EVENT = dict(id="10_nca", project="10_self_healing_pixels", model="NCA lizard_regen (perceive 3x3 + MLP 48-128-16)",
             checkpoint=nca.CKPT, unit="NCA steps of one 72x72 creature",
             params=48 * 128 + 128 + 128 * 16,
             notes="Fire mask is an input (fixed seed). Inputs are creatures grown 200 steps (MLX CPU, fixed masks). "
                   "Core ML packages are fp16.")
BATCHES = (1, 32)
SKIPS = {"mlx_gpu_q8": "fc1 input dim 48 is not a multiple of any MLX quantization group size (32/64/128); "
                       "the whole net is 8k params",
         "mlx_gpu_q4": "same as q8"}
GROW, REGEN = 200, 100

_cache = {}


def _grown(n):
    """n creatures grown from the seed for GROW steps on the MLX CPU stream with fixed fire masks."""
    if _cache.get("grown_n", 0) < n:
        m = _model()
        masks = nca.fire_masks(GROW, n, seed=12345)
        with mx.stream(mx.cpu):
            x = mx.array(nca.make_seed(nca.SIZE, nca.SIZE, n))
            for t in range(GROW):
                x = m(x, mx.array(masks[t]))
                mx.eval(x)
        _cache["grown"], _cache["grown_n"] = np.array(x), n
    return _cache["grown"][:n]


def _model(dtype=mx.float32):
    if ("mlx", dtype) not in _cache:
        _cache[("mlx", dtype)] = nca.load_mlx(dtype=dtype)
    return _cache[("mlx", dtype)]


def make_inputs(batch, seed=0):
    return {"x": _grown(max(batch, BATCHES[-1]))[:batch].copy(),
            "fire_mask": nca.fire_masks(1, batch, seed=777 + seed)[0]}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def fn(inp):
        x, m = mx.array(inp["x"]).astype(dtype), mx.array(inp["fire_mask"]).astype(dtype)
        if stream is not None:
            with mx.stream(stream):
                y = fwd(x, m)
                mx.eval(y)
        else:
            y = fwd(x, m)
            mx.eval(y)
        return {"x": np.array(y.astype(mx.float32))}
    return fn


def coreml_packages(batch):
    inp = make_inputs(batch)
    path = coreml_util.export_torch(nca.torch_net(), (inp["x"], inp["fire_mask"]), ["x", "fire_mask"],
                                    f"e10_nca_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m, m16 = _model(), _model(mx.float16)
    npn = nca.NumpyNCA()
    out = {
        "numpy_cpu": lambda inp: {"x": npn(inp["x"], inp["fire_mask"])},
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(m16, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = coreml_util.make_athlete(path, name, rename=["x"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    d = np.abs(o["x"] - ref["x"])
    return dict(state_max_err=float(d.max()), rgba_max_err=float(d[..., :4].max()),
                alive_cells_diff=int(np.abs((o["x"][..., 3] > 0.1).sum() - (ref["x"][..., 3] > 0.1).sum())))


def _target():
    if "target" not in _cache:
        _cache["target"] = nca.load_target()
    return _cache["target"]


def _mse(x):
    return ((np.asarray(x, np.float64)[..., :4] - _target()) ** 2).mean(axis=(1, 2, 3))


def regrow(fn, batch):
    """Cut the left half off `batch` grown creatures, run REGEN steps through fn. Returns MSE curve (REGEN+1, batch)."""
    x = _grown(max(batch, BATCHES[-1]))[:batch].copy()
    x[:, nca.half_damage(nca.SIZE, nca.SIZE, 0)] = 0.0
    masks = nca.fire_masks(REGEN, batch, seed=4242)
    curve = [_mse(x)]
    for t in range(REGEN):
        x = fn({"x": x, "fire_mask": masks[t]})["x"]
        curve.append(_mse(x))
    return np.stack(curve)


def task_check(name, batch):
    """Regrowth after a half cut, 100 steps through this athlete at this batch (1 or 32 creatures), vs MLX fp32."""
    ath = athletes(batch)
    if ("ref_curve", batch) not in _cache:
        _cache[("ref_curve", batch)] = regrow(ath["mlx_gpu"], batch)
    ref = _cache[("ref_curve", batch)]
    c = ref if name == "mlx_gpu" else regrow(ath[name], batch)
    pre = _mse(_grown(batch))
    tau = pre + 0.2 * (c[0] - pre)  # eval.py's recovery threshold: 80% of the damage-induced excess removed
    return dict(trials=batch, steps=REGEN, mse_post_damage=float(c[0].mean()), mse_final=float(c[-1].mean()),
                ref_mse_final=float(ref[-1].mean()),
                mse_final_rel_diff=float(abs(c[-1].mean() - ref[-1].mean()) / ref[-1].mean()),
                recovered_frac=float((c[-1] <= tau).mean()), ref_recovered_frac=float((ref[-1] <= tau).mean()),
                max_curve_diff=float(np.abs(c - ref).max()))


def qualifies(agreement, task):
    """Same model if the regrown creature is as good as MLX fp32's (final MSE within 10%, same recovered
    fraction within one trial); without a task result (--no-task), visible RGBA error <= 5e-3 after one step
    (hidden channels are unbounded, so their absolute error is not a fair yardstick)."""
    if task:
        ok = task["mse_final_rel_diff"] <= 0.10 and abs(task["recovered_frac"] - task["ref_recovered_frac"]) <= 1 / task["trials"] + 1e-9
        return ok, f"regrowth MSE {task['mse_final']:.4g} vs ref {task['ref_mse_final']:.4g} " \
                   f"({task['mse_final_rel_diff']:.1%}), recovered {task['recovered_frac']:.2f} vs {task['ref_recovered_frac']:.2f}"
    err = agreement["rgba_max_err"]
    return err <= 5e-3, f"one-step RGBA max err {err:.2g} (tolerance 5e-3)"
