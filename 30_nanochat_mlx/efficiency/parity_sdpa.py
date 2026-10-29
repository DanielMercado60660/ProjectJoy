"""
Parity for the attention-backward tile override (audit #4). The tile sizes are read from the
process environment (MLX_SDPA_VJP_BQ / MLX_SDPA_VJP_BK), so the reference and the candidate
must run in different processes:

    ../.venv/bin/python efficiency/parity_sdpa.py --save /tmp/ref.npz           # default tiles
    MLX_SDPA_VJP_BQ=512 MLX_SDPA_VJP_BK=512 ../.venv/bin/python efficiency/parity_sdpa.py --compare /tmp/ref.npz

Checks (pilot d6 weights + optimizer state, real batch 4x1024, bf16):
  - forward loss identical, full-model gradient rel error, 2-update parameter-delta rel error
  - isolated SDPA q/k/v gradient rel error (random bf16 inputs, 4x3x1024x128 = d6 shapes)
  - causal leakage through the VJP: loss on positions < 512 only; perturbing every future token
    must leave all parameter gradients unchanged (rel ~ 0)
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import parity as P  # noqa: E402


def sdpa_grads():
    mx.random.seed(5)
    q, k, v = [mx.random.normal((4, 3, 1024, 128)).astype(mx.bfloat16) for _ in range(3)]
    w = mx.random.normal((4, 3, 1024, 128)).astype(mx.bfloat16)
    f = lambda q, k, v: (mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask="causal") * w).astype(mx.float32).sum()
    return mx.grad(f, argnums=(0, 1, 2))(q, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--out", default=os.path.join(HERE, "parity.json"))
    a = ap.parse_args()
    start = "pilot"
    x, y = P.batches(7)[0]
    loss, g = P.loss_grad(P.make_model(start), x, y)
    d, _ = P.run_updates(P.make_model(start), P.make_opt(P.make_model(start), start), set())
    dq, dk, dv = sdpa_grads()
    # leakage: masked loss on t < 512, future tokens perturbed
    mask = np.zeros(x.shape, np.float32)
    mask[:, :512] = 1
    x2 = x.copy()
    x2[:, 512:] = (x2[:, 512:] + 17) % 8192
    y2 = y.copy()
    y2[:, 512:] = (y2[:, 512:] + 17) % 8192
    m = P.make_model(start)
    _, ga = P.loss_grad(m, x, y, mx.array(mask))
    _, gb = P.loss_grad(m, x2, y2, mx.array(mask))
    leak = P.rel(gb, ga)[0]
    env = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
    print(f"env {env}  loss {loss:.6f}  leakage rel {leak:.3g}", flush=True)
    cur = {"loss": np.array(loss), **{"g/" + k: np.array(v) for k, v in g.items()},
           **{"d/" + k: np.array(v) for k, v in d.items()},
           "dq": np.array(dq.astype(mx.float32)), "dk": np.array(dk.astype(mx.float32)),
           "dv": np.array(dv.astype(mx.float32))}
    if a.save:
        np.savez(a.save, **cur)
        return
    ref = dict(np.load(a.compare))
    to = lambda pre, src: {k[len(pre):]: mx.array(v) for k, v in src.items() if k.startswith(pre)}
    row = {"variant": "sdpa", "start": start, "env": env,
           "loss_rel": abs(float(cur["loss"]) - float(ref["loss"])) / abs(float(ref["loss"])),
           "grad_rel": P.rel(to("g/", cur), to("g/", ref))[0],
           "delta_rel": P.rel(to("d/", cur), to("d/", ref))[0],
           "sdpa_qkv_grad_rel": max(math.sqrt(float(((cur[n] - ref[n]) ** 2).sum()) / float((ref[n] ** 2).sum()))
                                    for n in ("dq", "dk", "dv")),
           "leakage_rel": leak}
    row["pass"] = (row["loss_rel"] <= 1e-3 and row["sdpa_qkv_grad_rel"] <= 1e-2 and row["leakage_rel"] < 1e-6
                   and row["delta_rel"] <= 1.5e-2 and row["grad_rel"] <= 1e-2)
    print(json.dumps(row), flush=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(prev + [row], open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
