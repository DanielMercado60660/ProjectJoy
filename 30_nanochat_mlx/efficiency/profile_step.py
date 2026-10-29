"""
Where does one optimizer update go? Times the pieces of the real step in isolation (bf16,
compiled, median of repeats, mx.synchronize around each):
  micro      compiled loss+grad of one micro-batch (B x T)       x accum per update
  apply      compiled MuonAdamW update                            x1 per update
  muon/adamw the two optimizer parts separately
  ce_*       softcap + cross-entropy forward+backward on (B*T, V) bf16 raw logits:
             ref (fp32 cap -> fast CE), kernel (fused_ce), bf16cap
  sdpa       causal attention fwd+bwd per layer (B, H, T, hd), times n_layer
  loader     numpy batches for one update

    ../bin/gpu_run ../.venv/bin/python efficiency/profile_step.py --depth 6 --B 32
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from data import TokenStream  # noqa: E402
from fused_ce import capped_ce  # noqa: E402
from gpt import GPT, config_for_depth, loss_fn  # noqa: E402
from optim import build_optimizer  # noqa: E402


def timeit(f, reps=10, warm=3):
    for _ in range(warm):
        mx.eval(f())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(f())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join(HERE, "profile.jsonl"))
    a = ap.parse_args()
    B, T = a.B, a.T
    cfg = config_for_depth(a.depth, 8192, T)
    model = GPT(cfg)
    model.init_weights(0)
    opt, _ = build_optimizer(model, weight_decay=0.01)
    opt.init(model.trainable_parameters())
    state = [model.state, opt.state]
    ts = TokenStream("train", seed=0)
    xn, yn = ts.batch(0, B, T)
    x, y = mx.array(xn), mx.array(yn)
    lg = nn.value_and_grad(model, loss_fn)
    micro = mx.compile(lambda x, y: lg(model, x, y), inputs=state, outputs=state)
    fwd = mx.compile(lambda x, y: loss_fn(model, x, y), inputs=state, outputs=state)
    _, grads = micro(x, y)
    mx.eval(grads)
    res = {"depth": a.depth, "B": B, "T": T, "accum": a.accum, "loadavg": os.getloadavg()}
    res["micro_ms"] = timeit(lambda: micro(x, y))
    res["fwd_ms"] = timeit(lambda: fwd(x, y))

    def _apply(g):
        opt.update(model, g)
    apply = mx.compile(_apply, inputs=state, outputs=state)
    res["apply_ms"] = timeit(lambda: (apply(grads), state)[1])
    # optimizer parts: muon only / adamw only (uncompiled split via MultiOptimizer internals)
    parts = opt._split_dictionary(grads)
    muon = opt.optimizers[0]
    mu_c = mx.compile(lambda g: model.update(muon.apply_gradients(g, model)), inputs=state, outputs=state)
    res["muon_ms"] = timeit(lambda: (mu_c(parts[0]), state)[1])
    # CE path, isolated
    V = cfg.vocab_size
    z = (mx.random.normal((B * T, V)) * 3).astype(mx.bfloat16)
    yy = mx.array(yn.reshape(-1))
    w = mx.random.uniform(shape=(B * T,))
    ref = lambda z: (mx.fast.cross_entropy(15 * mx.tanh(z.astype(mx.float32) / 15), yy) * w).sum()
    ker = lambda z: (capped_ce(z, yy) * w).sum()
    b16 = lambda z: (mx.fast.cross_entropy((15 * mx.tanh(z.astype(mx.float32) / 15)).astype(mx.bfloat16), yy) * w).sum()
    for name, f in [("ce_ref", ref), ("ce_kernel", ker), ("ce_bf16cap", b16)]:
        g = mx.compile(mx.value_and_grad(f))
        res[name + "_ms"] = timeit(lambda: g(z))
    # lm_head GEMMs (fwd + 2 bwd) for scale
    h = mx.random.normal((B * T, cfg.n_embd)).astype(mx.bfloat16)
    W = mx.random.normal((V, cfg.n_embd)).astype(mx.bfloat16)
    res["lm_head_gemms_ms"] = timeit(lambda: [h @ W.T, z @ W, z.T @ h])
    # attention fwd+bwd, one layer
    hd = cfg.n_embd // cfg.n_head
    q, k, v = [mx.random.normal((B, cfg.n_head, T, hd)).astype(mx.bfloat16) for _ in range(3)]
    att = mx.compile(mx.grad(lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal").astype(mx.float32).sum(), argnums=(0, 1, 2)))
    res["sdpa_layer_ms"] = timeit(lambda: att(q, k, v))
    res["sdpa_total_ms"] = res["sdpa_layer_ms"] * cfg.n_layer
    t0 = time.perf_counter()
    for i in range(20):
        for k_ in range(a.accum):
            ts.batch(i * a.accum + k_, B, T)
    res["loader_ms_per_update"] = (time.perf_counter() - t0) / 20 * 1000
    upd = res["micro_ms"] * a.accum + res["apply_ms"] + res["loader_ms_per_update"]
    res["est_update_ms"] = upd
    for kk in ("apply_ms", "muon_ms", "ce_ref_ms", "sdpa_total_ms", "loader_ms_per_update"):
        mult = a.accum if kk in ("ce_ref_ms", "sdpa_total_ms") else 1
        res["frac_" + kk.replace("_ms", "")] = res[kk] * mult / upd
    res["env"] = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
