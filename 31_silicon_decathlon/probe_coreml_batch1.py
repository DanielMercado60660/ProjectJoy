"""Probe for brief finding 1: is Core ML at batch 1 silently running on the CPU whatever you ask for?

For a ladder of conv nets at batch 1 (the real e09 AlphaZero package plus wider random-weight copies of
the same architecture, and one big MLP), this records per compute unit:
  - the compute plan: preferred device per op, and which devices each op *could* run on
  - the plan's estimated cost per device
  - interleaved wall-clock timing (bench.race) and its distribution (p10/p50/p90/p99)
  - process CPU time per call (CPU-seconds burned by this process / wall-seconds): ~1 or more means the
    work ran on CPU cores; well below 1 means it was handed to the GPU or Neural Engine
  - max abs difference between units' outputs (bit-identical outputs = same kernels ran)

Random weights are fine here: this is a placement/timing probe, nothing is scored or trained.

  ../.venv/bin/python probe_coreml_batch1.py            # writes results/probes/coreml_batch1.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import bench  # noqa: E402
import coreml_util  # noqa: E402

OUT = os.path.join(HERE, "results", "probes", "coreml_batch1.json")


def wide_az(channels, blocks=5, seed=0):
    """Same layout as models/alphazero.torch_net (NHWC in, 6x7 board), random weights, wider."""
    import torch
    import torch.nn as tnn
    torch.manual_seed(seed)
    C = channels

    class Res(tnn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = tnn.Conv2d(C, C, 3, padding=1, bias=False); s.b1 = tnn.BatchNorm2d(C)
            s.c2 = tnn.Conv2d(C, C, 3, padding=1, bias=False); s.b2 = tnn.BatchNorm2d(C)

        def forward(s, x):
            return torch.relu(x + s.b2(s.c2(torch.relu(s.b1(s.c1(x))))))

    class Net(tnn.Module):
        def __init__(s):
            super().__init__()
            s.stem = tnn.Conv2d(2, C, 3, padding=1, bias=False); s.bn = tnn.BatchNorm2d(C)
            s.blocks = tnn.ModuleList([Res() for _ in range(blocks)])
            s.head = tnn.Linear(C * 42, 7)

        def forward(s, x):
            x = torch.relu(s.bn(s.stem(x.permute(0, 3, 1, 2))))
            for b in s.blocks:
                x = b(x)
            return s.head(x.permute(0, 2, 3, 1).reshape(-1, C * 42))

    return Net().eval()


def big_mlp(width=4096, depth=4, seed=0):
    import torch
    import torch.nn as tnn
    torch.manual_seed(seed)
    layers = []
    for _ in range(depth):
        layers += [tnn.Linear(width, width), tnn.ReLU()]
    return tnn.Sequential(*layers).eval()


def plan_report(path, athlete):
    """Preferred-device counts, supported-device sets, and estimated cost share per preferred device."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    compiled = ct.utils.compile_model(path)
    plan = MLComputePlan.load_from_path(path=compiled,
                                        compute_units=getattr(ct.ComputeUnit, coreml_util.UNITS[athlete]))
    fn = plan.model_structure.program.functions["main"]
    pref, supported, cost = Counter(), Counter(), Counter()
    heavy = []
    for op in fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        name = lambda d: type(d).__name__.replace("ML", "").replace("ComputeDevice", "")
        p = name(usage.preferred_compute_device)
        sup = "+".join(sorted(name(d) for d in usage.supported_compute_devices))
        pref[p] += 1
        supported[sup] += 1
        c = plan.get_estimated_cost_for_mlprogram_operation(op)
        w = float(c.weight) if c is not None else 0.0
        cost[p] += w
        if op.operator_name in ("ios18.conv", "ios18.linear", "ios17.conv", "ios17.linear", "conv", "linear"):
            heavy.append(dict(op=op.operator_name, preferred=p, supported=sup, cost=round(w, 4)))
    return dict(preferred=dict(pref), supported=dict(supported),
                cost_share={k: round(v, 4) for k, v in cost.items()}, heavy_ops_first3=heavy[:3])


def cpu_ratio(fn, inputs, n):
    """process CPU seconds per wall second over n back-to-back calls."""
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        fn(inputs)
    w, c = time.perf_counter() - w0, time.process_time() - c0
    return dict(wall_ms=w / n * 1e3, cpu_ms=c / n * 1e3, cpu_per_wall=c / w)


def probe(name, module, x, input_name="x", n_cpu=200):
    path = coreml_util.export_torch(module, (x,), [input_name], f"probe_{name}_b{x.shape[0]}_fp16")
    params = sum(p.numel() for p in module.parameters())
    inputs = {input_name: x}
    ath = {u: coreml_util.make_athlete(path, u) for u in coreml_util.UNITS}
    outs = {u: next(iter(f(inputs).values())) for u, f in ath.items()}
    diffs = {f"{a}_vs_{b}": float(np.abs(outs[a].astype(np.float64) - outs[b]).max())
             for a, b in (("coreml_cpu", "coreml_gpu"), ("coreml_cpu", "coreml_ane"), ("coreml_gpu", "coreml_ane"))}
    with bench.exclusive():
        timing = bench.race(ath, inputs, repeats=3, budget_s=0.15, max_calls=3000)
        dist = {}
        for u, f in ath.items():
            ts = np.asarray(bench._call_times(f, inputs, 0.3, 4000))
            dist[u] = {f"p{q}": float(np.percentile(ts, q)) for q in (10, 50, 90, 99)}
            dist[u]["calls"] = int(ts.size)
        cpu = {u: cpu_ratio(f, inputs, max(20, min(n_cpu, int(200 / max(timing[u]["median_ms"], 0.05)))))
               for u, f in ath.items()}
    plans = {u: plan_report(path, u) for u in coreml_util.UNITS}
    row = dict(model=name, params=int(params), input_shape=list(x.shape), package=os.path.relpath(path, HERE),
               timing={u: {k: timing[u][k] for k in ("median_ms", "p90_ms", "min_ms", "calls")} for u in ath},
               distribution=dist, cpu_time=cpu, output_diff=diffs, plan=plans)
    print(f"\n== {name}  params={params:,}  input={tuple(x.shape)}")
    for u in coreml_util.UNITS:
        print(f"  {u:<11} median {timing[u]['median_ms']:8.3f} ms  p90 {timing[u]['p90_ms']:8.3f}  "
              f"cpu/wall {cpu[u]['cpu_per_wall']:.2f}  plan preferred {plans[u]['preferred']}")
    print("  output diffs:", {k: f"{v:.2g}" for k, v in diffs.items()})
    return row


def main():
    from models import alphazero as az
    warnings.simplefilter("ignore")
    rows = []
    x1 = az.random_positions(1, seed=0)
    rows.append(probe("az64_real", az.torch_net(), x1))
    for c in (128, 256, 512):
        rows.append(probe(f"az{c}_random", wide_az(c), x1))
    rows.append(probe("mlp4096x4_random", big_mlp(), np.random.default_rng(0).standard_normal((1, 4096)).astype(np.float32)))
    # batch 256 of the real net, for contrast (the orchestrator saw the units separate here)
    rows.append(probe("az64_real", az.torch_net(), az.random_positions(256, seed=0)))
    res = dict(env=bench.env_snapshot(), rows=rows)
    bench.save_json(res, OUT)
    print("wrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
