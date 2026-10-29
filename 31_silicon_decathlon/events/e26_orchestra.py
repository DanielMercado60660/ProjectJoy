"""Event 26: Neural Orchestra drum VAE decoder (beta=1 style-conditional MLP VAE, the demo model).

One item = decoding one 2-bar drum pattern (32 steps x 9 drums) from a latent z (16) + style one-hot (8)
into hit probabilities, velocities and microtiming offsets (288 cells each).
Latency batch 1 = the demo's "New neural groove" button. Throughput batch 2000 = the project's eval
set (250 generated patterns per style x 8 styles).
Task check (at b=2000, fresh fixed-seed latents): patterns built the project's way (threshold p > 0.5,
and the demo's Bernoulli sampling with the same uniforms for every athlete) must match MLX fp32
hit-for-hit, and the project's own style classifier must give the same verdict.
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import mlx.core as mx  # noqa: E402
import mlx.utils  # noqa: E402
import numpy as np  # noqa: E402

import coreml_util  # noqa: E402
from models import orchestra as orc  # noqa: E402

EVENT = dict(id="26_orchestra", project="26_neural_orchestra", model="drum VAE decoder (MLP 24-256-512-864, GELU)",
             checkpoint=orc.CKPT, unit="patterns", params=None,
             notes=("Decoder only (the encoder is not used for generation). z ~ N(0, I), styles cycled. "
                    "Outputs hit probability, velocity, offset per cell. Core ML packages are fp16. "
                    "SURPRISE: the mlx_gpu fp32 reference is itself not fp32-exact at b=2000. MLX 0.32.4 GPU fp32 "
                    "matmul with >= 8 rows has TF32-class error (probe: 512-term sums of N(0,1) entries, max err "
                    "0.07-0.09 vs fp64 for M = 8..2000, 1e-5 at M = 1; mlx_cpu and numpy 1e-4). So numpy/mlx_cpu "
                    "disagree with mlx_gpu on 1.25% of threshold patterns while matching an fp64 decode 100%, and "
                    "mlx_gpu matches fp64 on 98.75%. qualifies() still judges against mlx_gpu as pre-registered; "
                    "task has *_vs_fp64 fields for the exact comparison."))
BATCHES = (1, 2000)
SKIPS = {"mlx_gpu_q8": "quantization reserved for the LLM event (brief); also the first layer's 24 inputs "
                       "don't divide MLX's group size 32/64 (the project's own JS demo does use int8 per-row weights)",
         "mlx_gpu_q4": "same as mlx_gpu_q8"}
QUAL_PATTERN, QUAL_STYLE = 0.99, 0.99   # fixed before any athlete was measured

_cache = {}


def make_inputs(batch, seed=0):
    z, s, _ = orc.random_latents(batch, seed=seed)
    return {"z": z, "style": s}


def _wrap_mlx(m, dtype=mx.float32, stream=None):
    def fn(inp):
        def go():
            outs = m(mx.array(inp["z"]).astype(dtype), mx.array(inp["style"]).astype(dtype))
            mx.eval(outs)
            return outs
        if stream is not None:
            with mx.stream(stream):
                outs = go()
        else:
            outs = go()
        return {k: np.array(a.astype(mx.float32)) for k, a in zip(("p", "vel", "off"), outs)}
    return fn


def coreml_packages(batch):
    inp = make_inputs(batch)
    path = coreml_util.export_torch(orc.torch_decoder(), (inp["z"], inp["style"]), ["z", "style"],
                                    f"e26_orchestra_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = orc.load_mlx()
    EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    m16 = orc.load_mlx(mx.float16)
    npd = orc.NumpyDecoder()
    out = {
        "numpy_cpu": lambda inp: dict(zip(("p", "vel", "off"), npd(inp["z"], inp["style"]))),
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(m16, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
    }
    for name, path in coreml_packages(batch).items():
        cm = coreml_util.make_athlete(path, name, rename=["p", "vel", "off"])
        out[name] = (lambda f: lambda inp: {k: v.reshape(-1, orc.CELLS) for k, v in f(inp).items()})(cm)
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    return {f"{k}_max_err": float(np.abs(np.asarray(o[k], np.float32) - ref[k]).max()) for k in ("p", "vel", "off")}


def task_check(name, batch):
    if batch != BATCHES[-1]:
        return None
    ath = athletes(batch)
    clf = orc.NumpyClassifier()
    inp = make_inputs(batch, seed=101)
    style = np.arange(batch) % orc.N_STYLES

    def patterns(fn):
        o = fn(inp)
        return {mode: orc.to_pattern(o["p"], o["vel"], o["off"], mode, seed=102) for mode in ("threshold", "sample")}
    if "ref_task" not in _cache:
        _cache["ref_task"] = patterns(ath["mlx_gpu"])
        _cache["fp64_task"] = patterns(lambda i: dict(zip(("p", "vel", "off"), orc.decode_fp64(i["z"], i["style"]))))
    ref, got = _cache["ref_task"], patterns(ath[name])
    res = dict(patterns=batch)
    for mode in ("threshold", "sample"):
        h, v, o = got[mode]
        hr = ref[mode][0]
        res[f"{mode}_pattern_match"] = float((h == hr).all(axis=(1, 2)).mean())
        res[f"{mode}_cell_match"] = float((h == hr).mean())
        # informational, not used by qualifies: the same match against a float64 decode
        res[f"{mode}_pattern_match_vs_fp64"] = float((h == _cache["fp64_task"][mode][0]).all(axis=(1, 2)).mean())
    cs = clf(*got["sample"]).argmax(1)
    cr = clf(*ref["sample"]).argmax(1)
    res["style_agreement_with_ref"] = float((cs == cr).mean())
    res["style_acc_vs_condition"] = float((cs == style).mean())
    res["ref_style_acc_vs_condition"] = float((cr == style).mean())
    return res


def qualifies(agreement, task):
    if task:
        ok = task["threshold_pattern_match"] >= QUAL_PATTERN and task["style_agreement_with_ref"] >= QUAL_STYLE
        return ok, (f"threshold patterns identical {task['threshold_pattern_match']:.4f} (need >= {QUAL_PATTERN}), "
                    f"classifier agrees on sampled patterns {task['style_agreement_with_ref']:.4f} (need >= {QUAL_STYLE})")
    ok = agreement["p_max_err"] <= 0.01
    return ok, f"hit-probability max error {agreement['p_max_err']:.2g} (need <= 0.01)"
