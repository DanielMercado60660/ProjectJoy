"""Event 05: Dungeon navigator, PPO + GRU(128) (solver_cur3, ~200k weights), one recurrent step.

One item = one agent step: (obs, hidden state) in -> (action logits, value, next hidden state) out; the
hidden state is an explicit input/output, so every engine pays to move it host <-> device each step.
Latency batch 1 = one agent walking one dungeon (05's demo). Throughput batch 512 = PPO's 512 parallel
training envs (train_solver.py --envs 512).
Inputs: (obs, h) pairs recorded while the numpy policy walks held-out test levels 100-119 of each family.
Task check (greedy actions, fixed levels: test-set indices 0.. of all 5 families, interleaved; 20 levels at
batch 1, 500 at batch 512): (a) closed loop, each athlete walks the levels itself with its own hidden
state: escapes within 200 steps; (b) open loop, along the MLX fp32 reference's own trajectories, the
fraction of steps where the athlete's greedy action (with its own carried hidden state) matches.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

import coreml_util
from models import dungeon as dg

EVENT = dict(id="05_dungeon", project="05_dungeon", model="PPO-GRU navigator (MLP enc 55-128-128, GRU 128, pi/v heads)",
             checkpoint=dg.CKPT, unit="agent steps (one recurrent step, hidden state in/out)", params=None,
             notes="solver_cur3 = PPO-GRU after 3 curriculum rounds (first policy in 05's demo.html). "
                   "Task check uses greedy actions (05's eval samples actions; greedy makes the check "
                   "deterministic). q8/q4: enc1 (in 55) stays fp32. Core ML packages are fp16. Under MLX's default "
                   "TF32 GPU matmuls the CPU fp32 engines differed from the MLX GPU reference by ~0.03 in the "
                   "logits at batch 512; with the true-fp32 reference (MLX_ENABLE_TF32=0, used by --final) they "
                   "agree to ~4e-6.")
BATCHES = (1, 512)
SKIPS = {}
_cache = {}


def _mlx():
    if "mlx" not in _cache:
        m = dg.load_mlx()
        _cache["mlx"] = m
        EVENT["params"] = sum(v.size for _, v in mlx.utils.tree_flatten(m.parameters()))
    return _cache["mlx"]


def _pool():
    if "pool" not in _cache:
        pol = dg.NumpyPolicy()
        env = dg.VecMaze(dg.load_levels(20, start=100))
        h = np.zeros((env.n, dg.HID), np.float32)
        O, H = [], []
        for _ in range(dg.MAX_STEPS):
            live = ~env.done
            obs = env.obs()
            O.append(obs[live]); H.append(h[live])
            logits, _, h = pol(obs, h)
            env.step(logits.argmax(-1))
            if env.done.all():
                break
        _cache["pool"] = (np.concatenate(O).astype(np.float32), np.concatenate(H).astype(np.float32))
    return _cache["pool"]


def make_inputs(batch, seed=0):
    O, H = _pool()
    idx = np.random.default_rng(seed).choice(len(O), batch, replace=len(O) < batch)
    return {"obs": np.ascontiguousarray(O[idx]), "h": np.ascontiguousarray(H[idx])}


def _wrap_mlx(fwd, dtype=mx.float32, stream=None):
    def run(inp):
        out = fwd(mx.array(inp["obs"]).astype(dtype), mx.array(inp["h"]).astype(dtype))
        mx.eval(*out)
        return out

    def fn(inp):
        if stream is not None:
            with mx.stream(stream):
                out = run(inp)
        else:
            out = run(inp)
        return {k: np.array(v.astype(mx.float32)) for k, v in zip(("logits", "value", "h"), out)}
    return fn


def coreml_packages(batch):
    inp = make_inputs(batch)
    path = coreml_util.export_torch(dg.torch_net(), (inp["obs"], inp["h"]), ["obs", "h"],
                                    f"e05_dungeon_b{batch}_fp16")
    return {k: path for k in coreml_util.UNITS}


def _quantized(bits):
    m = dg.load_mlx()
    nn.quantize(m, group_size=32, bits=bits,
                class_predicate=lambda p, l: isinstance(l, nn.Linear) and l.weight.shape[1] % 32 == 0)
    return m


def athletes(batch):
    if ("ath", batch) in _cache:
        return _cache[("ath", batch)]
    m = _mlx()
    m16 = dg.load_mlx()
    m16.set_dtype(mx.float16)
    npnet = dg.NumpyPolicy()
    out = {
        "numpy_cpu": lambda inp: dict(zip(("logits", "value", "h"), npnet(inp["obs"], inp["h"]))),
        "mlx_cpu": _wrap_mlx(m, stream=mx.cpu),
        "mlx_gpu": _wrap_mlx(m),
        "mlx_gpu_fp16": _wrap_mlx(m16, dtype=mx.float16),
        "mlx_gpu_compiled": _wrap_mlx(mx.compile(m)),
        "mlx_gpu_q8": _wrap_mlx(_quantized(8)),
        "mlx_gpu_q4": _wrap_mlx(_quantized(4)),
    }
    for name, path in coreml_packages(batch).items():
        out[name] = coreml_util.make_athlete(path, name, rename=["logits", "value", "h"])
    _cache[("ath", batch)] = out
    return out


def agreement(o, ref):
    return dict(logits_max_err=float(np.abs(o["logits"] - ref["logits"]).max()),
                value_max_err=float(np.abs(np.ravel(o["value"]) - np.ravel(ref["value"])).max()),
                h_max_err=float(np.abs(o["h"] - ref["h"]).max()),
                greedy_action_agreement=float((o["logits"].argmax(-1) == ref["logits"].argmax(-1)).mean()))


def _levels(batch):
    if ("levels", batch) not in _cache:
        _cache[("levels", batch)] = dg.load_levels(4 if batch == 1 else min(100, batch // 5))
    return _cache[("levels", batch)]


def _call(fn, batch, obs, h):
    """One recurrent step for all n agents through an athlete compiled for `batch` rows."""
    n = len(obs)
    if batch == 1:
        outs = [fn({"obs": obs[i:i + 1], "h": h[i:i + 1]}) for i in range(n)]
        return np.concatenate([o["logits"] for o in outs]), np.concatenate([o["h"] for o in outs])
    po = np.zeros((batch, obs.shape[1]), np.float32); po[:n] = obs
    ph = np.zeros((batch, h.shape[1]), np.float32); ph[:n] = h
    o = fn({"obs": po, "h": ph})
    return o["logits"][:n], o["h"][:n]


def _closed_loop(fn, batch, record=False):
    env = dg.VecMaze(_levels(batch))
    h = np.zeros((env.n, dg.HID), np.float32)
    success = np.zeros(env.n, bool)
    log = []
    for _ in range(dg.MAX_STEPS):
        live = ~env.done
        obs = env.obs()
        logits, h2 = _call(fn, batch, obs, h)
        a = logits.argmax(-1)
        if record:
            log.append((obs, live.copy(), a))
        h = np.where(live[:, None], h2, h).astype(np.float32)
        _, succ = env.step(a)
        success |= succ
        if env.done.all():
            break
    return success, env.t.copy(), log


def task_check(name, batch):
    ath = athletes(batch)
    key = ("ref", batch)
    if key not in _cache:
        _cache[key] = _closed_loop(ath["mlx_gpu"], batch, record=True)
    ref_succ, _, ref_log = _cache[key]
    succ, steps, _ = _closed_loop(ath[name], batch)
    # open loop along the reference's trajectories, athlete carries its own hidden state
    n = len(ref_succ)
    h = np.zeros((n, dg.HID), np.float32)
    agree = total = 0
    for obs, live, a_ref in ref_log:
        logits, h2 = _call(ath[name], batch, obs, h)
        agree += int((logits.argmax(-1) == a_ref)[live].sum()); total += int(live.sum())
        h = np.where(live[:, None], h2, h).astype(np.float32)
    return dict(levels=int(n), escaped=int(succ.sum()), ref_escaped=int(ref_succ.sum()),
                mean_steps_escaped=float(steps[succ].mean()) if succ.any() else None,
                open_loop_action_agreement=agree / total, open_loop_steps=total)


def qualifies(agreement, task):
    """Same model = greedy action matches the reference on >= 98% of the reference's own trajectory
    steps (open loop, own hidden state) and it escapes at least as many fixed levels as the reference
    minus max(1, 2%). Thresholds fixed before the first timed run."""
    if task is None:
        a = agreement["greedy_action_agreement"]
        return a >= 0.98, f"greedy action agreement {a:.3f} on the timing inputs (no task check)"
    a, e, r, n = task["open_loop_action_agreement"], task["escaped"], task["ref_escaped"], task["levels"]
    why = f"open-loop action agreement {a:.3f}, escaped {e}/{n} (ref {r})"
    return a >= 0.98 and e >= r - max(1, int(0.02 * n)), why
