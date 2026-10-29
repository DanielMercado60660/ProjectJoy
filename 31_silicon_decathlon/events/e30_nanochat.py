"""Event 30: Nanochat d6 pilot GPT (16.9M params), the one LLM on the shelf.

Two timings, unit = tokens:
  b=1   one KV-cached decode step at position 255 (batch 1), after a 255-token held-out prefix.
        Each athlete prefills that prefix once, in its own precision, and keeps the KV cache; the
        timed call is the single next-token step -> logits (1, V). This is what engine.py's decode
        loop does per token.
  b=256 one 256-token prefill forward (batch 1, lm_head on every position, like engine.py's
        prefill and train.py's bpb eval) -> logits (1, 256, V).
Task checks on held-out ClimbMix (val shard, never trained on): bits-per-byte and greedy next-token
agreement with MLX fp32 over 32 windows of 256 tokens (prefill), and next-token agreement over 128
contexts through the athlete's own decode path (decode).
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")  # cap numpy/Accelerate at 4 threads (shared Mac)

import mlx.core as mx  # noqa: E402
import mlx.utils  # noqa: E402
import numpy as np  # noqa: E402

import coreml_util  # noqa: E402
from models import nanochat as nc  # noqa: E402

T_CTX = 256
EVENT = dict(id="30_nanochat", project="30_nanochat_mlx", model="Nanochat d6 pilot GPT (6x384, 3 heads, vocab 8192)",
             checkpoint=nc.CKPT, unit="tokens", params=None,
             notes=("b=1: one KV-cached decode step at position 255 (prefix cached per athlete, own precision). "
                    "b=256: one 256-token prefill forward with full logits. Inputs are held-out val tokens. "
                    "mlx_gpu_bf16 is extra: the pilot's own training/inference dtype. q8/q4 = nn.quantize "
                    "(group 64, affine) on every Linear incl. lm_head and on the token embedding, fp32 activations. "
                    "Core ML packages are fp16. Core ML decode is stateless with the KV cache as I/O: a 255-token "
                    "prefix package returns K/V (own fp16 precision, kept host-side as fp32 numpy), and every timed "
                    "step passes the full past K/V (2 x 6x3x255x128 fp32 = 4.7 MB) in and gets logits + new K/V "
                    "out; MLX/numpy keep their cache on-device/in-process. bpb at 256-token context is higher than the README's "
                    "1.167 (1024 context, bf16). numpy and mlx_cpu differ from mlx_gpu by 0.033 in logits at b=256 "
                    "(and 0.008 at b=1 through the 255-token prefix): MLX GPU fp32 matmul with >= 8 rows is "
                    "TF32-class (see e26 notes), so the reference, not the CPU engines, carries that error."))
BATCHES = (1, T_CTX)
SKIPS = {}
QUAL_GREEDY, QUAL_BPB = 0.98, 0.01   # thresholds fixed before any athlete was measured

_cache = {}


def _models():
    if "models" not in _cache:
        ms = {"fp32": nc.load_mlx(mx.float32), "fp16": nc.load_mlx(mx.float16), "bf16": nc.load_mlx(mx.bfloat16),
              "q8": nc.load_quantized(8), "q4": nc.load_quantized(4)}
        EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(ms["fp32"].parameters()))
        # weights resident per athlete's model; the harness's memory_bytes is MLX peak *active* memory,
        # which here includes all five MLX models loaded at once (~0.17 GB), so compare these instead
        EVENT["weight_bytes"] = {k: int(sum(v.nbytes for _, v in mlx.utils.tree_flatten(m.parameters())))
                                 for k, m in ms.items()}
        _cache["models"] = ms
        _cache["numpy"] = nc.NumpyGPT()
    return _cache["models"]


def make_inputs(batch, seed=0):
    w = nc.val_window(T_CTX, seed)
    if batch == 1:
        return {"prefix": w[None, :T_CTX - 1], "token": w[None, T_CTX - 1:T_CTX]}
    return {"idx": w[None, :T_CTX]}


def _in_stream(stream, f):
    if stream is None:
        return f()
    with mx.stream(stream):
        return f()


def _mlx_prefill(model, stream=None, compiled=False):
    fwd = mx.compile(model.__call__) if compiled else model

    def fn(inp):
        def go():
            y = fwd(mx.array(inp["idx"]))
            mx.eval(y)
            return y
        return {"logits": np.array(_in_stream(stream, go))}
    return fn


def _mlx_decode(model, stream=None, compiled=False):
    """KV cache of the current prefix is athlete state; the timed call is one token step."""
    state = {}
    L = len(model.h)

    def step(tok, *flat):
        past = [(flat[2 * i], flat[2 * i + 1]) for i in range(L)]
        return model.forward(tok, past)[0]
    stepf = mx.compile(step) if compiled else step

    def fn(inp):
        def go():
            if "prefix" not in state or not np.array_equal(state["prefix"], inp["prefix"]):
                _, kv = model.forward(mx.array(inp["prefix"]))
                flat = [a for kvi in kv for a in kvi]
                mx.eval(flat)
                state.update(prefix=np.array(inp["prefix"]), flat=flat)
            y = stepf(mx.array(inp["token"]), *state["flat"])[:, 0]
            mx.eval(y)
            return y
        return {"logits": np.array(_in_stream(stream, go))}
    return fn


def _numpy_prefill(npm):
    return lambda inp: {"logits": npm.forward(inp["idx"])[0]}


def _numpy_decode(npm):
    state = {}

    def fn(inp):
        if "prefix" not in state or not np.array_equal(state["prefix"], inp["prefix"]):
            state.update(prefix=np.array(inp["prefix"]), past=npm.forward(inp["prefix"])[1])
        return {"logits": npm.forward(inp["token"], state["past"])[0][:, 0]}
    return fn


def _coreml_decode_paths():
    w = make_inputs(1)
    kz = np.zeros((len(_models()["fp32"].h), 3, T_CTX - 1, 128), np.float32)
    pre = coreml_util.export_torch(nc.torch_gpt(T_CTX - 1, return_kv=True), (w["prefix"],), ["idx"],
                                   f"e30_nanochat_prefixkv{T_CTX - 1}_fp16")
    dec = coreml_util.export_torch(nc.torch_gpt(1, T_past=T_CTX - 1, return_kv=True), (w["token"], kz, kz),
                                   ["idx", "past_k", "past_v"], f"e30_nanochat_decode{T_CTX}_fp16")
    return pre, dec


def _coreml_decode(athlete):
    pre_path, dec_path = _coreml_decode_paths()
    pre, dec = coreml_util.load(pre_path, athlete), coreml_util.load(dec_path, athlete)
    pn, dn = coreml_util.output_names(pre_path), coreml_util.output_names(dec_path)  # (logits, k, v)
    state = {}

    def fn(inp):
        if "prefix" not in state or not np.array_equal(state["prefix"], inp["prefix"]):
            o = pre.predict({"idx": inp["prefix"]})
            state.update(prefix=np.array(inp["prefix"]), k=np.asarray(o[pn[1]], np.float32),
                         v=np.asarray(o[pn[2]], np.float32))
        o = dec.predict({"idx": inp["token"], "past_k": state["k"], "past_v": state["v"]})
        return {"logits": np.asarray(o[dn[0]]).reshape(1, -1)}
    return fn


def coreml_packages(batch):
    if batch == 1:
        return {k: _coreml_decode_paths()[1] for k in coreml_util.UNITS}
    idx = make_inputs(batch)["idx"]
    path = coreml_util.export_torch(nc.torch_prefill(T_CTX), (idx,), ["idx"], f"e30_nanochat_prefill{T_CTX}_fp16")
    return {k: path for k in coreml_util.UNITS}


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    ms = _models()
    npm = _cache["numpy"]
    if batch == 1:
        mk, out = _mlx_decode, {"numpy_cpu": _numpy_decode(npm)}
    else:
        mk, out = _mlx_prefill, {"numpy_cpu": _numpy_prefill(npm)}
    out.update({
        "mlx_cpu": mk(ms["fp32"], stream=mx.cpu),
        "mlx_gpu": mk(ms["fp32"]),
        "mlx_gpu_fp16": mk(ms["fp16"]),
        "mlx_gpu_bf16": mk(ms["bf16"]),
        "mlx_gpu_compiled": mk(ms["fp32"], compiled=True),
        "mlx_gpu_q8": mk(ms["q8"]),
        "mlx_gpu_q4": mk(ms["q4"]),
    })
    for name, path in coreml_packages(batch).items():
        out[name] = _coreml_decode(name) if batch == 1 else coreml_util.make_athlete(path, name, rename=["logits"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    a, r = np.asarray(o["logits"], np.float32), np.asarray(ref["logits"], np.float32)
    return dict(logits_max_err=float(np.abs(a - r).max()),
                argmax_agreement=float((a.argmax(-1) == r.argmax(-1)).mean()))


def _logp_of(logits, y):
    lg = np.asarray(logits, np.float64).reshape(-1, logits.shape[-1])
    m = lg.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lg - m).sum(-1))
    return lg[np.arange(len(y)), y] - lse


def task_check(name, batch):
    ath = athletes(batch)
    if batch == 1:
        # next-token agreement through each athlete's own decode path (fresh prefix -> prefill + step)
        seeds = range(2000, 2128)
        key = ("ref_decode",)
        if key not in _cache:
            _cache[key] = [int(ath["mlx_gpu"](make_inputs(1, s))["logits"].argmax()) for s in seeds]
        got = [int(ath[name](make_inputs(1, s))["logits"].argmax()) for s in seeds]
        truth = [int(nc.val_window(T_CTX, s)[T_CTX]) for s in seeds]
        return dict(next_token_agreement=float(np.mean(np.array(got) == np.array(_cache[key]))),
                    top1_acc_vs_truth=float(np.mean(np.array(got) == np.array(truth))), contexts=len(seeds))
    tb = nc.token_bytes()

    def score(fn):
        nats, nbytes, am, ys = 0.0, 0, [], []
        for s in range(1000, 1032):
            w = nc.val_window(T_CTX, s)
            x, y = w[None, :T_CTX], w[1:T_CTX + 1]
            lg = fn({"idx": x})["logits"]
            nb = tb[y]
            nats += float((-_logp_of(lg, y) * (nb > 0)).sum())
            nbytes += int(nb.sum())
            am.append(np.asarray(lg).reshape(T_CTX, -1).argmax(-1)); ys.append(y)
        return nats / (np.log(2) * nbytes), np.concatenate(am), np.concatenate(ys)
    if ("ref_score",) not in _cache:
        _cache[("ref_score",)] = score(ath["mlx_gpu"])
    rb, ram, ys = _cache[("ref_score",)]
    b, am, _ = score(ath[name])
    return dict(bpb=b, bpb_ref=rb, bpb_delta=b - rb, greedy_agreement=float((am == ram).mean()),
                top1_acc_vs_truth=float((am == ys).mean()), tokens=int(ys.size))


def qualifies(agreement, task):
    if task and "greedy_agreement" in task:
        ok = task["greedy_agreement"] >= QUAL_GREEDY and task["bpb_delta"] <= QUAL_BPB
        return ok, (f"greedy agreement {task['greedy_agreement']:.4f} (need >= {QUAL_GREEDY}), "
                    f"bpb delta {task['bpb_delta']:+.4f} (need <= {QUAL_BPB})")
    if task and "next_token_agreement" in task:
        ok = task["next_token_agreement"] >= QUAL_GREEDY
        return ok, f"decode next-token agreement {task['next_token_agreement']:.4f} over {task['contexts']} contexts (need >= {QUAL_GREEDY})"
    ok = agreement["argmax_agreement"] >= QUAL_GREEDY
    return ok, f"argmax agreement {agreement['argmax_agreement']:.4f} (need >= {QUAL_GREEDY})"
