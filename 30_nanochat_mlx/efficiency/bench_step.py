"""
Corrected throughput harness: times the REAL pretraining optimizer step.

What is timed per update (exactly what train.py's loop does between its t0/dt):
  set_schedule(...)  ->  train.make_update(...)(get_batch)  with accumulation (default 4
  micro-batches), the real TokenStream loader (random windows from data/tokens), MuonAdamW,
  mx.compile, bf16. tok/s = accum * B * T * n_updates / wall time, with mx.synchronize()
  before and after every timed window.

Several variants can be benchmarked in ONE process; their timed windows are interleaved
(A B C A B C ...) so drift / thermals / background noise hit all variants equally. A variant
is a '+'-joined list of flags:
  base       pilot code path (accum_mode=ref, everything off)
  fast       on-device accumulation, one host sync per update        (audit #2)
  g2, g4     device batch x2/x4 with accumulation /2 /4; each micro-batch is the concatenation
             of the original micro-batches, so the examples per update are unchanged (#3)
  qkv        one GEMM for q/k/v                                      (#5)
  kernel     fused softcap+CE Metal kernel                           (#1)
  bf16cap    capped logits in bf16 before fused CE   [CHANGES TRAINING]  (#9)
  mbatch     stacked (batched) Muon per equal-shape group            (#6)
  addmm      Polar Express via mx.addmm                              (#7)
  ns4, ns3   fewer Polar Express iterations      [CHANGES TRAINING]  (#10)
Process-level knobs (e.g. MLX_SDPA_VJP_BQ/BK attention-backward tiles, #4) are set in the
environment of the whole process; bracket such runs with baseline processes.

    ../bin/gpu_run ../.venv/bin/python efficiency/bench_step.py --depth 6 --B 32 --variants base fast
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from data import TokenStream  # noqa: E402
from gpt import GPT, config_for_depth  # noqa: E402
from optim import build_optimizer, set_schedule  # noqa: E402
from train import make_update  # noqa: E402

FLAGS = {"base", "fast", "g2", "g4", "qkv", "kernel", "bf16cap", "mbatch", "addmm", "ns4", "ns3"}
CHANGES_TRAINING = {"bf16cap", "ns4", "ns3"}


def matmul_peak(n=8192, reps=20, dtype=mx.bfloat16):
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    for _ in range(3):
        mx.eval(a @ b)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(a @ b)
    mx.synchronize()
    return 2 * n ** 3 * reps / (time.perf_counter() - t0) / 1e12


class Variant:
    def __init__(self, spec, depth, B, T, accum, init, total_iters=3000):
        self.spec = spec
        flags = set(spec.split("+"))
        bad = flags - FLAGS
        assert not bad, f"unknown flags {bad}"
        self.flags = flags
        self.group = 4 if "g4" in flags else 2 if "g2" in flags else 1
        assert accum % self.group == 0
        self.B, self.T, self.accum = B, T, accum
        cfg = config_for_depth(depth, 8192, T)
        model = GPT(cfg)
        if init == "pilot":
            from checkpoint import load_meta
            pdir = os.path.join(ROOT, "checkpoints", "pilot", "last")
            assert load_meta(pdir)["config"] == cfg.to_dict(), "pilot checkpoint is d6"
            model.load_weights(os.path.join(pdir, "model.safetensors"))
        else:
            model.init_weights(0)
        model.compute_dtype = mx.bfloat16
        model.set_fast_flags(fused_qkv="qkv" in flags,
                             ce_mode="kernel" if "kernel" in flags else "bf16cap" if "bf16cap" in flags else "ref")
        # same lr/wd arithmetic as train.py at the pilot's total batch
        total_batch = B * T * accum
        lr_scale = (total_batch / 2 ** 19) ** 0.5
        wd = 0.28 * math.sqrt(total_batch / 2 ** 19) * (12 * 91.2e6 / (total_batch * total_iters))
        ns = 4 if "ns4" in flags else 3 if "ns3" in flags else 5
        opt, base_lrs = build_optimizer(model, weight_decay=wd, lr_scale=lr_scale, muon_batched="mbatch" in flags,
                                        muon_addmm="addmm" in flags, ns_steps=ns)
        self.model, self.opt, self.base_lrs, self.wd = model, opt, base_lrs, wd
        self.update = make_update(model, opt, accum // self.group, mode="fast" if "fast" in flags else "ref")
        self.flops_per_token = model.estimate_flops()
        self.train = TokenStream("train", seed=0)
        self.step = 0
        self.windows = []  # (tok/s, loadavg1)
        self.peak_mem = 0.0
        self.loss = None

    def get_batch(self, k):
        s, G, B, T = self.step, self.group, self.B, self.T
        if G == 1:
            return self.train.batch(s * self.accum + k, B, T)
        parts = [self.train.batch(s * self.accum + k * G + j, B, T) for j in range(G)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def one_update(self):
        it = 400 + self.step  # constant-lr phase of the schedule (values only; graph is identical)
        set_schedule(self.opt, self.base_lrs, 1.0, 0.97, self.wd * 0.5 * (1 + math.cos(math.pi * it / 3000)))
        self.loss = self.update(self.get_batch)
        self.step += 1

    def loader_ms(self, n=20):
        t0 = time.perf_counter()
        for i in range(n):
            for k in range(self.accum // self.group):
                self.get_batch(k)
        return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--B", type=int, default=32, help="reference micro-batch rows (before gN grouping)")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--init", choices=["random", "pilot"], default="random")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--per-window", type=int, default=12, help="updates per timed window")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_runs.jsonl"))
    a = ap.parse_args()

    env = {k: v for k, v in os.environ.items() if k.startswith("MLX_")}
    load0 = os.getloadavg()
    peak0 = matmul_peak()
    print(f"[bench] matmul peak {peak0:.1f} TFLOP/s bf16, loadavg {load0}, env {env}", flush=True)
    vs = [Variant(s, a.depth, a.B, a.T, a.accum, a.init) for s in a.variants]
    for v in vs:
        t0 = time.perf_counter()
        for _ in range(a.warmup):
            v.one_update()
        mx.synchronize()
        v.warm_s = time.perf_counter() - t0
        v.loader = v.loader_ms()
        print(f"[bench] warm {v.spec}: {v.warm_s:.1f}s  loss {v.loss:.4f}  loader {v.loader:.1f} ms/update", flush=True)
    for w in range(a.windows):
        order = vs if w % 2 == 0 else vs[::-1]  # alternate order to cancel position effects
        for v in order:
            mx.synchronize()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            for _ in range(a.per_window):
                v.one_update()
            mx.synchronize()
            dt = time.perf_counter() - t0
            tps = a.accum * a.B * a.T * a.per_window / dt
            v.windows.append((tps, os.getloadavg()[0]))
            v.peak_mem = max(v.peak_mem, mx.get_peak_memory() / 1e9)
            print(f"[bench] window {w} {v.spec}: {tps:,.0f} tok/s ({dt / a.per_window * 1000:.0f} ms/update)", flush=True)
    peak1 = matmul_peak()
    load1 = os.getloadavg()
    peak = (peak0 + peak1) / 2
    base_med = None
    for v in vs:
        t = [x for x, _ in v.windows]
        med = statistics.median(t)
        base_med = base_med or med
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": a.tag, "variant": v.spec,
               "changes_training": bool(v.flags & CHANGES_TRAINING), "depth": a.depth,
               "B": a.B * v.group, "accum": a.accum // v.group, "T": a.T, "tokens_per_update": a.accum * a.B * a.T,
               "init": a.init, "env": env, "tok_s_median": med, "tok_s_min": min(t), "tok_s_max": max(t),
               "spread_pct": (max(t) - min(t)) / med * 100, "windows": [round(x) for x in t],
               "rel_to_first": med / base_med, "tflops": v.flops_per_token * med / 1e12,
               "mfu": v.flops_per_token * med / 1e12 / peak, "matmul_peak_tflops": [round(peak0, 2), round(peak1, 2)],
               "peak_mem_gb": v.peak_mem, "loader_ms_per_update": v.loader, "loss_last": v.loss,
               "loadavg_start": load0, "loadavg_end": load1, "loadavg_windows": [round(l, 2) for _, l in v.windows],
               "warmup": a.warmup, "windows_n": a.windows, "per_window": a.per_window}
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
