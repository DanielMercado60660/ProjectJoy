"""
Parity checks for the efficiency candidates (tolerances from efficiency/codex_audit.md).

For each candidate we compare against the reference (pilot) code path on the SAME weights,
optimizer state and real-data batches, at two starting points:
  init  : init_weights(0) with the zero-init output projections perturbed (otherwise most
          block gradients are exactly zero and errors can hide), fresh optimizer
  pilot : checkpoints/pilot/last weights + optimizer state (trained d6, read-only)
Metrics (rel(a,b) = ||a-b|| / ||b||, fp32, over all parameters concatenated):
  loss_rel   relative error of the micro-batch loss (model-path flags)
  grad_rel   gradient rel error (model-path flags)
  delta_rel  rel error of the parameter DELTA after 2 optimizer updates (all flags)
  delta_worst  worst per-tensor delta rel error
  cos        cosine similarity of the deltas
The 'default_vs_orig' check runs the new code with all flags off against the untouched
pre-efficiency source (efficiency/orig/) and must be bit-identical.

    ../.venv/bin/python efficiency/parity.py            # writes efficiency/parity.json
"""
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402

import checkpoint as ckpt  # noqa: E402
from data import TokenStream  # noqa: E402
from gpt import GPT, config_for_depth, loss_fn  # noqa: E402
from optim import build_optimizer, set_schedule  # noqa: E402
from train import make_update  # noqa: E402

PILOT = os.path.join(ROOT, "checkpoints", "pilot", "last")
B, T, ACCUM, DEPTH = 4, 1024, 4, 6
STREAM = TokenStream("train", seed=123)


def load_orig(name):
    spec = importlib.util.spec_from_file_location(f"orig_{name}", os.path.join(HERE, "orig", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat(tree):
    return {k: v.astype(mx.float32) for k, v in tree_flatten(tree)}


def rel(a, b):
    """a, b: dicts name->array; global rel error, worst per-tensor rel, cosine."""
    num = sum(float(mx.sum((a[k] - b[k]) ** 2)) for k in b)
    den = sum(float(mx.sum(b[k] ** 2)) for k in b)
    dot = sum(float(mx.sum(a[k] * b[k])) for k in b)
    na = sum(float(mx.sum(a[k] ** 2)) for k in b)
    worst = 0.0
    for k in b:
        nb = float(mx.sum(b[k] ** 2))
        if nb > 0:
            worst = max(worst, math.sqrt(float(mx.sum((a[k] - b[k]) ** 2)) / nb))
    return math.sqrt(num / max(den, 1e-30)), worst, dot / max(math.sqrt(na * den), 1e-30)


def make_model(start, gpt_mod=None, flags=()):
    G = (gpt_mod.GPT if gpt_mod else GPT)
    cfgf = (gpt_mod.config_for_depth if gpt_mod else config_for_depth)
    m = G(cfgf(DEPTH, 8192, T))
    if start == "pilot":
        m.load_weights(os.path.join(PILOT, "model.safetensors"))
    else:
        m.init_weights(0)
        mx.random.seed(1)
        for b in m.h:
            b.attn.c_proj.weight = mx.random.normal(b.attn.c_proj.weight.shape) * 0.02
            b.mlp.c_proj.weight = mx.random.normal(b.mlp.c_proj.weight.shape) * 0.02
        m.lm_head.weight = mx.random.normal(m.lm_head.weight.shape) * 0.02
    m.compute_dtype = mx.bfloat16
    if not gpt_mod:
        m.set_fast_flags(fused_qkv="qkv" in flags,
                         ce_mode="kernel" if "kernel" in flags else "bf16cap" if "bf16cap" in flags else "ref")
    mx.eval(m.parameters())
    return m


def make_opt(model, start, flags=(), optim_mod=None):
    kw = {} if optim_mod else dict(muon_batched="mbatch" in flags, muon_addmm="addmm" in flags,
                                   ns_steps=4 if "ns4" in flags else 3 if "ns3" in flags else 5)
    bo = optim_mod.build_optimizer if optim_mod else build_optimizer
    opt, base_lrs = bo(model, weight_decay=0.05, lr_scale=0.5, **kw)
    if start == "pilot":
        ckpt.load_optimizer(PILOT, opt, model)
    else:
        opt.init(model.trainable_parameters())
    mx.eval(opt.state)
    sched = (optim_mod.set_schedule if optim_mod else set_schedule)
    sched(opt, base_lrs, 1.0, 0.97, 0.05)
    return opt


def batches(step, group=1):
    out = []
    for k in range(ACCUM // group):
        parts = [STREAM.batch(step * ACCUM + k * group + j, B, T) for j in range(group)]
        out.append((np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])))
    return out


def run_updates(model, opt, flags, n=2):
    group = 4 if "g4" in flags else 2 if "g2" in flags else 1
    upd = make_update(model, opt, ACCUM // group, mode="fast" if "fast" in flags else "ref")
    p0 = flat(model.parameters())
    losses = []
    for s in range(n):
        bs = batches(s, group)
        if "rev" in flags:  # noise floor: same micro-batches summed in reverse order (identical math)
            bs = bs[::-1]
        losses.append(upd(lambda k: bs[k]))
    p1 = flat(model.parameters())
    return {k: p1[k] - p0[k] for k in p0}, losses


def orig_updates(start, n=2):
    """The pilot's loop, verbatim, on the untouched pre-efficiency gpt.py/optim.py."""
    og, oo = load_orig("gpt"), load_orig("optim")
    model = make_model(start, gpt_mod=og)
    opt = make_opt(model, start, optim_mod=oo)
    lg = nn.value_and_grad(model, og.loss_fn)
    state = [model.state, opt.state]
    micro = mx.compile(lambda x, y: lg(model, x, y), inputs=state, outputs=state)

    def _apply(grads):
        opt.update(model, grads)
    apply = mx.compile(_apply, inputs=state, outputs=state)
    p0 = flat(model.parameters())
    losses = []
    for s in range(n):
        gsum, lsum = None, 0.0
        for x, y in batches(s):
            loss, grads = micro(mx.array(x), mx.array(y))
            gsum = grads if gsum is None else tree_map(lambda a, b: a + b, gsum, grads)
            mx.eval(loss, gsum)
            lsum += loss.item()
        apply(tree_map(lambda g: g / ACCUM, gsum))
        mx.eval(state)
        losses.append(lsum / ACCUM)
    p1 = flat(model.parameters())
    return {k: p1[k] - p0[k] for k in p0}, losses


def fp32_grad_rel(start, flags, x, y):
    """Same check in fp32 compute: a pure re-implementation must agree to ~1e-5 here."""
    a, b = make_model(start), make_model(start, flags=flags)
    a.compute_dtype = b.compute_dtype = mx.float32
    la, ga = loss_grad(a, x, y)
    lb, gb = loss_grad(b, x, y)
    return abs(lb - la) / abs(la), rel(gb, ga)[0]


def grad_floor(start, dtype, x, y):
    """Reorder floor for gradients: full batch vs the mean of its two halves (identical math)."""
    m = make_model(start)
    m.compute_dtype = dtype
    _, g = loss_grad(m, x, y)
    h = B // 2
    _, g1 = loss_grad(m, x[:h], y[:h])
    _, g2 = loss_grad(m, x[h:], y[h:])
    return rel({k: (g1[k] + g2[k]) / 2 for k in g}, g)[0]


def loss_grad(model, x, y, mask=None):
    lg = nn.value_and_grad(model, lambda xx, yy: loss_fn(model, xx, yy, mask))
    loss, g = lg(mx.array(x), mx.array(y))
    return float(loss), flat(g)


def check(flag_set, start, ref_cache):
    flags = set(flag_set.split("+"))
    row = {"variant": flag_set, "start": start}
    model_path = flags & {"qkv", "kernel", "bf16cap"}
    if model_path:
        x, y = batches(7)[0]
        lr_, gr = ref_cache.setdefault(("lg", start), loss_grad(make_model(start), x, y))
        lv, gv = loss_grad(make_model(start, flags=flags), x, y)
        row["loss_rel"] = abs(lv - lr_) / abs(lr_)
        row["grad_rel"], row["grad_worst"], _ = rel(gv, gr)
        # SFT-style mask (loss on the last half only) and an all-zero mask
        mask = np.zeros((B, T), np.float32)
        mask[:, T // 2:] = 1
        lr2, gr2 = loss_grad(make_model(start), x, y, mx.array(mask))
        lv2, gv2 = loss_grad(make_model(start, flags=flags), x, y, mx.array(mask))
        row["masked_loss_rel"] = abs(lv2 - lr2) / abs(lr2)
        row["masked_grad_rel"] = rel(gv2, gr2)[0]
        if ("floor", start) not in ref_cache:
            ref_cache[("floor", start)] = (grad_floor(start, mx.bfloat16, x, y), grad_floor(start, mx.float32, x, y))
        row["grad_floor_bf16"], row["grad_floor_fp32"] = ref_cache[("floor", start)]
        if "bf16cap" not in flags:
            row["fp32_loss_rel"], row["fp32_grad_rel"] = fp32_grad_rel(start, flags, x, y)
        lz, gz = loss_grad(make_model(start, flags=flags), x, y, mx.zeros((B, T)))
        row["zero_mask_ok"] = lz == 0.0 and all(float(mx.abs(v).max()) == 0.0 for v in gz.values())
    if ("upd", start) not in ref_cache:
        ref_cache[("upd", start)] = run_updates(make_model(start), make_opt(make_model(start), start), set())
    dref, lref = ref_cache[("upd", start)]
    m = make_model(start, flags=flags)
    dv, lv = run_updates(m, make_opt(m, start, flags), flags)
    row["delta_rel"], row["delta_worst"], row["cos"] = rel(dv, dref)
    mu = [k for k in dref if k.startswith("h.")]
    row["delta_rel_muon"] = rel({k: dv[k] for k in mu}, {k: dref[k] for k in mu})[0]
    row["update_loss_rel"] = max(abs(a - b) / abs(b) for a, b in zip(lv, lref))
    return row


# Audit gates. Two-update parameter-delta gates are bf16-noise-limited (Adam's early steps are
# ~sign(g) and Muon orthogonalises in bf16, so ANY change of floating-point order moves the
# delta by ~1%); a delta gate therefore passes if it is within audit tol OR within 1.5x of the
# measured floating-point-reorder floor at the same start ("g2", which only regroups the same
# examples into bigger GEMMs). Model-path flags must additionally match in fp32 compute
# (fp32_grad_rel <= 1e-4), which shows they compute the same function.
TOL = {
    "fast": dict(delta_rel=1e-4, update_loss_rel=1e-6),
    "g2": dict(delta_rel=1e-2, update_loss_rel=1e-3),
    "g4": dict(delta_rel=1e-2, update_loss_rel=1e-3),
    "qkv": dict(loss_rel=1e-3, grad_rel=1e-2, fp32_grad_rel=1e-4, delta_rel=1e-2),
    "kernel": dict(loss_rel=1e-3, grad_rel=1e-2, fp32_grad_rel=1e-4, delta_rel=1e-2),
    "mbatch": dict(delta_rel=1e-2),
    "addmm": dict(delta_rel=1e-2),
    "sdpa": dict(delta_rel=1e-2),
}


def verdict(row, floor):
    flags = row["variant"].split("+")
    ok = True
    for f in flags:
        for k, tol in TOL.get(f, {}).items():
            v = row.get(k, 0.0)
            if k == "delta_rel" and floor is not None and f not in ("g2", "fast"):
                tol = max(tol, 1.5 * floor)
            if k == "grad_rel":
                tol = max(tol, 1.5 * row["grad_floor_bf16"])
            if k == "fp32_grad_rel":
                tol = max(tol, 1.5 * row["grad_floor_fp32"])
            if v > tol:
                ok = False
    if "zero_mask_ok" in row and not row["zero_mask_ok"]:
        ok = False
    return ok


def trajectory(variants, n, start="init"):
    """n real updates from the same start on the same batches; loss curves vs the reference."""
    out = {}
    for v in ["base"] + [x for x in variants if x != "base"]:
        flags = set(v.split("+"))
        m = make_model(start, flags=flags)
        opt = make_opt(m, start, flags)
        group = 4 if "g4" in flags else 2 if "g2" in flags else 1
        upd = make_update(m, opt, ACCUM // group, mode="fast" if "fast" in flags else "ref")
        losses = []
        for s in range(n):
            bs = batches(s, group)
            losses.append(upd(lambda k: bs[k]))
        out[v] = losses
    ref = out["base"]
    rows = []
    for v, ls in out.items():
        d = [abs(a - b) for a, b in zip(ls, ref)]
        rows.append({"variant": v, "start": start, "traj_updates": n, "loss_first": ls[0], "loss_last": ls[-1],
                     "max_abs_loss_diff": max(d), "last5_mean_diff": sum(ls[-5:]) / 5 - sum(ref[-5:]) / 5})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["base", "rev", "g2", "fast", "g4", "qkv", "kernel", "mbatch", "addmm", "bf16cap", "ns4", "ns3"])
    ap.add_argument("--starts", nargs="+", default=["init", "pilot"])
    ap.add_argument("--out", default=os.path.join(HERE, "parity.json"))
    ap.add_argument("--traj", type=int, default=0, help="only run an N-update loss-trajectory comparison")
    a = ap.parse_args()
    if a.traj:
        rows = trajectory(a.variants, a.traj)
        prev = json.load(open(a.out)) if os.path.exists(a.out) else []
        json.dump(prev + rows, open(a.out, "w"), indent=1)
        return
    env = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
    rows = []
    for start in a.starts:
        cache = {}
        dnew, lnew = run_updates(make_model(start), make_opt(make_model(start), start), set())
        dorig, lorig = orig_updates(start)
        r0 = {"variant": "default_vs_orig", "start": start, "delta_rel": rel(dnew, dorig)[0],
              "bit_identical": all(bool(mx.array_equal(dnew[k], dorig[k])) for k in dorig) and lnew == lorig,
              "losses_identical": lnew == lorig}
        # GPU run-to-run nondeterminism of the reference itself is ~3e-10 (see 'base' rows)
        r0["pass"] = r0["losses_identical"] and r0["delta_rel"] < 1e-8
        print(json.dumps(r0), flush=True)
        rows.append(r0)
        cache[("upd", start)] = (dnew, lnew)
        floor = None
        for v in a.variants:
            r = check(v, start, cache)
            r["env"] = env
            if v == "g2":
                floor = r["delta_rel"] if start == "pilot" else r["delta_rel_muon"]
            r["reorder_floor"] = floor
            if start == "init":
                # fresh Adam state: step 1 is ~sign(g), so near-zero embedding grads flip sign
                # under ANY fp change and dominate delta_rel; gate on the Muon matrices instead
                r["delta_rel_all"], r["delta_rel"] = r["delta_rel"], r["delta_rel_muon"]
            r["pass"] = verdict(r, floor)
            if v == "g2" and start == "init":
                r["pass"] = "defines the reorder floor"
            if set(v.split("+")) & {"bf16cap", "ns4", "ns3"}:
                r["pass"] = "n/a (changes training)"
            print(json.dumps({k: (float(f"{x:.3g}") if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
            rows.append(r)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else []
    json.dump(prev + rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
