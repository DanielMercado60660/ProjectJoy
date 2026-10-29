"""Export a checkpoint for the browser: fold BatchNorm into the preceding conv, write
float32 weights as base64 inside JSON, plus a few MLX reference outputs so the JS
forward pass can be checked (web/check_js.js).

  python export_web.py checkpoints/main/gen_080 web/weights.json
"""
import base64
import json
import sys

import numpy as np

import c4
from net import load_model, make_evaluator


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def fold(conv, bn):
    w = np.array(conv.weight)                  # (out, kh, kw, in)
    g, b = np.array(bn.weight), np.array(bn.bias)
    mu, var = np.array(bn.running_mean), np.array(bn.running_var)
    s = g / np.sqrt(var + bn.eps)
    return dict(shape=list(w.shape), data=b64(w * s[:, None, None, None]), bias=b64(b - mu * s))


def lin(l):
    w = np.array(l.weight)
    return dict(shape=list(w.shape), data=b64(w), bias=b64(np.array(l.bias)))


def export(prefix, out_path, n_tests=6):
    model, meta = load_model(prefix)
    spec = dict(stem=fold(model.stem, model.stem_bn),
                blocks=[[fold(b.c1, b.b1), fold(b.c2, b.b2)] for b in model.blocks],
                p_conv=fold(model.p_conv, model.p_bn), p_fc=lin(model.p_fc),
                v_conv=fold(model.v_conv, model.v_bn), v_fc1=lin(model.v_fc1), v_fc2=lin(model.v_fc2),
                meta=dict(source=prefix, gen=meta.get("gen"), config=meta["config"]))
    ev = make_evaluator(model)
    tests = []
    for moves in ([], [3], [3, 3, 2, 4], [0, 1, 0, 1, 0], [3, 2, 3, 2, 4, 4, 1, 5, 5]):
        if len(tests) >= n_tests:
            break
        p, m = c4.from_moves(moves)
        x = c4.features_batch(np.array([p]), np.array([m]))
        lg, v = ev(x)
        tests.append(dict(moves=moves, logits=lg[0].tolist(), value=float(v[0])))
    spec["tests"] = tests
    with open(out_path, "w") as f:
        json.dump(spec, f)
    return spec


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
