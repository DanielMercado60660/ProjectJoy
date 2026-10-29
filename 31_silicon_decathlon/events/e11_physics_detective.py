"""Event 11: Physics Detective amortized posterior network (Deep Sets, ~150k weights), one forward pass.

One item = one dataset: a set of up to 15 pushes (design + 20 camera frames each), padded to 15 with a
mask, in -> 8-component mixture posterior over the hidden friction mu_k out.
Latency batch 1 = one posterior query (what the detective game / active learner asks after each push).
Throughput batch 1000 = eval_sbi.py's chunk size (run_net bs=1000).
Inputs: fresh simulated datasets, same recipe as eval_sbi.py (mu ~ U(0.05, 0.8), 1-15 pushes), fixed seed.
Task check: on 1000 other fixed datasets, the posterior mean/sd of mu_k (grid summary on host, float64,
identical for every athlete) vs the MLX fp32 reference, and the RMSE of the posterior mean vs the true mu_k.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

import coreml_util
from models import physics_detective as pd

EVENT = dict(id="11_physics_detective", project="11_physics_detective",
             model="Deep-Sets posterior net (enc 42-192-192-128, head 129-192-192-24, 8-comp MoG)",
             checkpoint=pd.CKPT, unit="datasets (posterior queries; each = up to 15 pushes x 20 frames)",
             params=None,
             notes="Inputs: fresh simulated datasets (eval_sbi.py recipe, fixed seed). numpy_cpu uses "
                   "scipy.special.erf for exact GELU. q8/q4: first encoder layer (in 42) and first head "
                   "layer (in 129) stay fp32 (in_dim not a multiple of group size 32). Core ML packages "
                   "are fp16. The MLX GPU fp32 reference runs fp32 matmuls in TF32 by default on this "
                   "machine (MLX 0.32.4; MLX_ENABLE_TF32='' turns it off), so CPU fp32 engines differ "
                   "from it by more than fp32 rounding; see qualifies().")
BATCHES = (1, 1000)
SKIPS = {}
TASK_N, TASK_SEED = 1000, 4242
_cache = {}


def _mlx():
    if "mlx" not in _cache:
        m = pd.load_mlx()
        _cache["mlx"] = m
        EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    return _cache["mlx"]


def make_inputs(batch, seed=0):
    f, m, _ = pd.fresh_datasets(batch, seed=1000 + seed)
    return {"feats": f, "mask": m}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def run(inp):
        out = fwd(mx.array(inp["feats"]).astype(dtype), mx.array(inp["mask"]).astype(dtype))
        mx.eval(*out)
        return out

    def fn(inp):
        if stream is not None:
            with mx.stream(stream):
                out = run(inp)
        else:
            out = run(inp)
        return {k: np.array(v.astype(mx.float32)) for k, v in zip(("logw", "loc", "scale"), out)}
    return fn


def coreml_packages(batch):
    inp = make_inputs(batch)
    path = coreml_util.export_torch(pd.torch_net(), (inp["feats"], inp["mask"]), ["feats", "mask"],
                                    f"e11_physics_detective_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def _quantized(bits):
    m = pd.load_mlx()
    nn.quantize(m, group_size=32, bits=bits,
                class_predicate=lambda p, l: isinstance(l, nn.Linear) and l.weight.shape[1] % 32 == 0)
    return m


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = _mlx()
    m16 = pd.load_mlx()
    m16.set_dtype(mx.float16)
    npnet = pd.NumpyNet()
    out = {
        "numpy_cpu": lambda inp: dict(zip(("logw", "loc", "scale"), npnet(inp["feats"], inp["mask"]))),
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(m16, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
        "mlx_gpu_q8": _wrap_mlx(_quantized(8)),
        "mlx_gpu_q4": _wrap_mlx(_quantized(4)),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = coreml_util.make_athlete(path, name, rename=["logw", "loc", "scale"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    """Raw-output errors (mixture weights in probability space: tiny components' log-weights are
    meaningless) plus the posterior-mean shift in mu units."""
    mo, so = pd.grid_summary(o["logw"], o["loc"], o["scale"])
    mr, sr = pd.grid_summary(ref["logw"], ref["loc"], ref["scale"])
    return dict(weight_max_err=float(np.abs(np.exp(o["logw"]) - np.exp(ref["logw"])).max()),
                loc_max_err=float(np.abs(o["loc"] - ref["loc"]).max()),
                scale_max_err=float(np.abs(o["scale"] - ref["scale"]).max()),
                post_mean_max_err=float(np.abs(mo - mr).max()))


def _task_data():
    if "task" not in _cache:
        _cache["task"] = pd.fresh_datasets(TASK_N, seed=TASK_SEED)
    return _cache["task"]


def _run_all(fn, batch):
    f, m, _ = _task_data()
    outs = []
    for s in range(0, len(f), batch):
        o = fn({"feats": f[s:s + batch], "mask": m[s:s + batch]})
        outs.append(pd.grid_summary(o["logw"], o["loc"], o["scale"]))
    return np.concatenate([a for a, _ in outs]), np.concatenate([b for _, b in outs])


def task_check(name, batch):
    ath = athletes(batch)
    key = ("taskref", batch)
    if key not in _cache:
        _cache[key] = _run_all(ath["mlx_gpu"], batch)
    mr, sr = _cache[key]
    mo, so = _run_all(ath[name], batch)
    mu = _task_data()[2]
    shift = np.abs(mo - mr) / sr
    return dict(datasets=TASK_N,
                median_mean_shift_over_ref_sd=float(np.median(shift)),
                p99_mean_shift_over_ref_sd=float(np.percentile(shift, 99)),
                median_sd_ratio=float(np.median(so / sr)),
                rmse_vs_true_mu=float(np.sqrt(np.mean((mo - mu) ** 2))),
                rmse_ref=float(np.sqrt(np.mean((mr - mu) ** 2))))


def qualifies(agreement, task):
    """Same model = (a) median posterior-mean shift vs the reference <= 0.1 reference posterior sd
    (half the net's own median gap to the exact posterior, 0.20 sd in 11's RESULTS), (b) median
    posterior-sd ratio within 5%, (c) RMSE of the posterior mean vs the true mu_k within 2% of the
    reference's. Set after seeing that the TF32 reference itself differs from CPU fp32 by a median
    0.04 sd (p99 0.34 sd), so a per-dataset max criterion would disqualify the exact fp32 engines.
    Reviewer note: under the true-fp32 reference used by the scored pass NumPy's median shift is 2e-5 sd,
    so that motivation is stale; the external anchor (half the net's 0.20 sd gap) still stands. In the final
    pass mlx_gpu_q8 passes at 0.087 sd and coreml_ane (b=1000) fails at 0.197 sd; neither is on a podium."""
    if task is None:
        e = agreement["post_mean_max_err"]
        return (e <= 0.02, f"posterior mean max err {e:.3g} (no task check)")
    med, ratio = task["median_mean_shift_over_ref_sd"], task["median_sd_ratio"]
    rr = task["rmse_vs_true_mu"] / task["rmse_ref"] - 1
    why = f"median shift {med:.3g} sd, sd ratio {ratio:.3f}, RMSE {task['rmse_vs_true_mu']:.4f} ({rr:+.1%})"
    ok = med <= 0.1 and abs(ratio - 1) <= 0.05 and abs(rr) <= 0.02
    return ok, why
