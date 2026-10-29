"""Event 15: Object-permanence tracker (per-frame CNN -> GRU(256) -> Gaussian head per ball), one frame step.

One item = one tracker step for one video stream: a 64x64 RGB frame plus the GRU hidden state in, the
per-ball Gaussian (mean, log sigma, rho) plus the new hidden state out. This is how a live tracker runs
(the original evaluates whole 40-frame sequences at once, but the GRU is causal, so a step is exact).
Latency batch 1 = one camera stream. Throughput batch 50 = 50 streams in lockstep (the original
evaluation batch, `evaluate.run_model(bs=50)`).

Checkpoint: 15_object_permanence/checkpoints/base/final (the README/RESULTS headline model; the hard
model has the identical architecture, so its timings would be the same).

Task check: every athlete tracks recorded held-out `test_long` sequences (long occlusions, where the
hidden state does the work) for all 40 frames, feeding back its *own* hidden state, and is compared
with the MLX fp32 trajectory: tracked mean position (px) and predicted sigma, on all and hidden frames.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

import coreml_util
from models import tracker as tr

EVENT = dict(id="15_object_permanence", project="15_object_permanence",
             model="CoordConv CNN (4 conv) + FC512 + GRU(256) + Gaussian head, 3 ball slots",
             checkpoint=tr.CKPT, unit="tracker steps (one 64x64 frame for one stream, GRU state in/out)",
             params=None,
             notes="Timing inputs: frame 20 of held-out test_long sequences with the fp32 hidden state after "
                   "frames 0-19. q8/q4 quantize the Linear layers (fc 4096x512 + head = 72% of params); "
                   "convs and the GRU matmuls stay fp32. Core ML packages are fp16.")
BATCHES = (1, 50)
SKIPS = {}
TASK_SPLIT, TASK_FIRST, TASK_T = "test_long", 500, 40   # sequences 500.. are not used for timing inputs

_cache = {}


def _mlx():
    if "mlx" not in _cache:
        m, _ = tr.load_mlx()
        _cache["mlx"] = m
        import mlx.utils
        EVENT["params"] = int(sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters())))
    return _cache["mlx"]


def _numpy():
    if "np" not in _cache:
        _cache["np"] = tr.NumpyTracker()
    return _cache["np"]


def make_inputs(batch, seed=0, t=20):
    key = ("inp", batch, seed, t)
    if key not in _cache:
        frames, _ = tr.recorded(TASK_SPLIT, range(seed * 100, seed * 100 + batch))
        h = np.zeros((batch, 256), np.float32)
        npt = _numpy()
        for i in range(t):
            _, h = npt.step(frames[:, i], h)
        _cache[key] = {"frame": np.ascontiguousarray(frames[:, t]), "h": h}
    return _cache[key]


def _wrap_mlx(step, dtype=mx.float32, stream=None):
    def fn(inp):
        f, h = mx.array(inp["frame"]).astype(dtype), mx.array(inp["h"]).astype(dtype)
        if stream is not None:
            with mx.stream(stream):
                o, h2 = step(f, h)
                mx.eval(o, h2)
        else:
            o, h2 = step(f, h)
            mx.eval(o, h2)
        return {"out": np.array(o.astype(mx.float32)), "h": np.array(h2.astype(mx.float32))}
    return fn


def _variant(kind):
    m, _ = tr.load_mlx()
    if kind == "fp16":
        m.set_dtype(mx.float16)
    else:
        nn.quantize(m, group_size=64, bits=int(kind[1:]))
    mx.eval(m.parameters())
    return m


def coreml_packages(batch):
    inp = make_inputs(batch)
    path = coreml_util.export_torch(tr.torch_step(batch), (inp["frame"], inp["h"]), ["frame", "h"],
                                    f"e15_tracker_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def _wrap_coreml(fn, batch):
    def g(inp):
        o = fn(inp)
        return {"out": o["out"].reshape(batch, tr.K, 5), "h": o["h"]}
    return g


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = _mlx()
    npt = _numpy()
    out = {
        "numpy_cpu": lambda inp: dict(zip(("out", "h"), npt.step(inp["frame"], inp["h"]))),
        "mlx_cpu": _wrap_mlx(m.step, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m.step),
        "mlx_gpu_fp16": _wrap_mlx(_variant("fp16").step, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m.step)),
        "mlx_gpu_q8": _wrap_mlx(_variant("q8").step),
        "mlx_gpu_q4": _wrap_mlx(_variant("q4").step),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = _wrap_coreml(coreml_util.make_athlete(path, name, rename=["out", "h"]), batch)
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    mu, s, rho = tr.to_pixels(o["out"])
    mu_r, s_r, rho_r = tr.to_pixels(ref["out"])
    return dict(mean_px_max_err=float(np.linalg.norm(mu - mu_r, axis=-1).max()),
                sigma_rel_max_err=float(np.abs(s / s_r - 1).max()),
                rho_max_err=float(np.abs(rho - rho_r).max()),
                h_max_err=float(np.abs(o["h"] - ref["h"]).max()))


def _trajectory(fn, batch, n_seq):
    """Run n_seq recorded sequences through fn, `batch` streams at a time, carrying fn's own hidden state."""
    frames, st = tr.recorded(TASK_SPLIT, range(TASK_FIRST, TASK_FIRST + n_seq))
    outs = []
    for i in range(0, n_seq, batch):
        h = np.zeros((batch, 256), np.float32)
        seq = []
        for t in range(TASK_T):
            o = fn({"frame": np.ascontiguousarray(frames[i:i + batch, t]), "h": h})
            h = o["h"].astype(np.float32)
            seq.append(o["out"])
        outs.append(np.stack(seq, 1))
    return np.concatenate(outs), st


def task_check(name, batch):
    """Track 50 recorded test_long sequences (40 frames each) with the athlete's own hidden state."""
    n_seq = 50 if batch == 1 else batch * max(1, 50 // batch)
    key = ("ref_traj", batch)
    if key not in _cache:
        _cache[key] = _trajectory(athletes(batch)["mlx_gpu"], batch, n_seq)
    ref, st = _cache[key]
    out, _ = _trajectory(athletes(batch)[name], batch, n_seq)
    mu, s, _ = tr.to_pixels(out)
    mu_r, s_r, _ = tr.to_pixels(ref)
    present = np.broadcast_to(st["present"][:, None, :], st["vis"].shape)
    hidden = present & (st["vis"] == 0)
    dev = np.linalg.norm(mu - mu_r, axis=-1)
    srel = np.abs(s / s_r - 1).max(-1)
    gt_err = np.linalg.norm(mu - st["pos"], axis=-1)
    gt_err_r = np.linalg.norm(mu_r - st["pos"], axis=-1)
    return dict(sequences=int(n_seq), frames=TASK_T, ball_frames=int(present.sum()), hidden_ball_frames=int(hidden.sum()),
                pos_dev_p99_px=float(np.percentile(dev[present], 99)), pos_dev_max_px=float(dev[present].max()),
                hidden_pos_dev_max_px=float(dev[hidden].max()),
                sigma_rel_dev_p99=float(np.percentile(srel[present], 99)),
                hidden_err_px=float(gt_err[hidden].mean()), ref_hidden_err_px=float(gt_err_r[hidden].mean()),
                all_err_px=float(gt_err[present].mean()))


# Tolerances, fixed before any athlete was scored. 0.5 px is the base tracker's own hidden-frame error on
# in-distribution test (0.44 px) and 1/7 of a ball radius; a 10% sigma change is well inside the spread
# between the tracker's 1-sigma coverage (0.42) and the ideal (0.39).
POS_TOL_PX, SIGMA_TOL = 0.5, 0.10
STEP_POS_TOL_PX, STEP_SIGMA_TOL = 0.25, 0.05


def qualifies(agreement, task):
    """Same tracker = over 40-frame recorded sequences with its own hidden state, p99 tracked-position
    deviation from MLX fp32 <= 0.5 px and p99 sigma deviation <= 10%. Without a task check: one step
    within 0.25 px and 5% sigma."""
    if task:
        p, s = task["pos_dev_p99_px"], task["sigma_rel_dev_p99"]
        ok = p <= POS_TOL_PX and s <= SIGMA_TOL
        return ok, f"40-step track: p99 pos dev {p:.3g} px (tol {POS_TOL_PX}), p99 sigma dev {s:.3g} (tol {SIGMA_TOL})"
    p, s = agreement["mean_px_max_err"], agreement["sigma_rel_max_err"]
    ok = p <= STEP_POS_TOL_PX and s <= STEP_SIGMA_TOL
    return ok, f"one step: pos err {p:.3g} px (tol {STEP_POS_TOL_PX}), sigma err {s:.3g} (tol {STEP_SIGMA_TOL})"
