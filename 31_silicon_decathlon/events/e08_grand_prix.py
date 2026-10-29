"""Event 08: Tiny Grand Prix car policy (ES, 13 -> 32 -> 32 -> 2 tanh MLP, 1570 weights), one forward pass.

Latency batch 1 = one car step (the demo / eval drive one policy call per 0.1 s sim step).
Throughput batch 256 = one step of PPO's 256 parallel training envs (train_ppo.py --envs 256).
Inputs are real observations recorded while the numpy policy drives held-out tracks 20-27.
Task check: a vendored copy of the racing sim (models/grand_prix_sim.py); each athlete drives a strict
one-lap time trial (wall touch = DNF) on held-out tracks 0-11 at grip 1.0, at that batch size
(at batch 256 the 12 cars are padded into one 256-row call per step).
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

import coreml_util
from models import grand_prix as gp
from models import grand_prix_sim as sim

EVENT = dict(id="08_grand_prix", project="08_tiny_grand_prix", model="ES car policy MLP 13-32-32-2 (tanh)",
             checkpoint=gp.CKPT, unit="car steps (one observation -> one action)", params=1570,
             notes="Inputs: observations recorded while the numpy policy drives held-out tracks 20-27 "
                   "(fixed). Task check: vendored sim, strict one-lap trial on held-out tracks 0-11, "
                   "grip 1.0. q8/q4 quantize layers 2-3 only (layer 1 has in_dim 13, not a multiple of "
                   "the group size 32), so the first layer stays fp32. Core ML packages are fp16. Under MLX's default TF32 "
                   "GPU matmuls the CPU fp32 engines differed from the MLX GPU reference by ~6e-3 at batch 256; "
                   "with the true-fp32 reference (MLX_ENABLE_TF32=0, used by --final) they agree to ~3e-7.")
BATCHES = (1, 256)
SKIPS = {}
TASK_TRACKS = 12
_cache = {}


def _obs_pool():
    if "pool" not in _cache:
        bank = sim.pack(sim.make_tracks("heldout", 8, start=20))
        r = sim.run_lap_strict(gp.NumpyPolicy(), bank, record_obs=True)
        _cache["pool"] = r["obs"].astype(np.float32)
    return _cache["pool"]


def make_inputs(batch, seed=0):
    pool = _obs_pool()
    idx = np.random.default_rng(seed).choice(len(pool), batch, replace=False)
    return {"obs": np.ascontiguousarray(pool[idx])}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def fn(inp):
        x = mx.array(inp["obs"]).astype(dtype)
        if stream is not None:
            with mx.stream(stream):
                a = fwd(x)
                mx.eval(a)
        else:
            a = fwd(x)
            mx.eval(a)
        return {"action": np.array(a.astype(mx.float32))}
    return fn


def coreml_packages(batch):
    x = make_inputs(batch)["obs"]
    path = coreml_util.export_torch(gp.torch_net(), (x,), ["obs"], f"e08_grand_prix_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def _quantized(bits):
    m = gp.load_mlx()
    nn.quantize(m, group_size=32, bits=bits,
                class_predicate=lambda p, l: isinstance(l, nn.Linear) and l.weight.shape[1] % 32 == 0)
    return m


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = gp.load_mlx()
    m16 = gp.load_mlx()
    m16.set_dtype(mx.float16)
    npnet = gp.NumpyPolicy()
    out = {
        "numpy_cpu": lambda inp: {"action": npnet(inp["obs"])},
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(m16, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
        "mlx_gpu_q8": _wrap_mlx(_quantized(8)),
        "mlx_gpu_q4": _wrap_mlx(_quantized(4)),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = coreml_util.make_athlete(path, name, rename=["action"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    d = np.abs(o["action"].astype(np.float64) - ref["action"])
    return dict(action_max_err=float(d.max()), action_mean_err=float(d.mean()))


def _bank():
    if "bank" not in _cache:
        _cache["bank"] = sim.pack(sim.make_tracks("heldout", TASK_TRACKS))
    return _cache["bank"]


def task_check(name, batch):
    """Each athlete drives the held-out tracks itself (closed loop, its own actions feed back)."""
    fn = athletes(batch)[name]
    bank = _bank()

    def controller(obs):
        obs = obs.astype(np.float32)
        if batch == 1:
            return np.concatenate([fn({"obs": obs[i:i + 1]})["action"] for i in range(len(obs))], 0)
        pad = np.zeros((batch, obs.shape[1]), np.float32)
        pad[:len(obs)] = obs
        return fn({"obs": pad})["action"][:len(obs)]

    r = sim.run_lap_strict(controller, bank)
    f = r["finished"]
    return dict(finished=int(f.sum()), tracks=int(len(f)),
                mean_lap_time_s=float(np.nanmean(r["lap_time"])) if f.any() else None,
                lap_times_s=[None if np.isnan(t) else round(float(t), 1) for t in r["lap_time"]])


def qualifies(agreement, task):
    """Same model = finishes every held-out track (12/12; the fp32 reference does) and the per-call
    action error vs MLX fp32 stays <= 0.05 (actions live in [-1, 1]). Thresholds fixed before timing."""
    err = agreement["action_max_err"]
    if task is not None and task["finished"] < task["tracks"]:
        return False, f"finished {task['finished']}/{task['tracks']} held-out tracks"
    if err > 0.05:
        return False, f"action max err {err:.3g} > 0.05"
    return True, f"finished {task['finished']}/{task['tracks']}, action max err {err:.3g}" if task else \
        f"action max err {err:.3g}"
