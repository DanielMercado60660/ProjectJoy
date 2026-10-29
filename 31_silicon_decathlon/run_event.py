"""Run one event (or all of them): check agreement, race the athletes, write results/<event>.json.

  ../.venv/bin/python run_event.py e09_alphazero            # dev run, marked provisional
  ../.venv/bin/python run_event.py all --final              # the scored pass, idle machine only (true fp32 MLX)
  ../.venv/bin/python run_event.py all --tf32-pass          # then: MLX GPU with TF32 on, vs the saved fp32 refs
  ../.venv/bin/python run_event.py e09_alphazero --no-task --batches 1 --repeats 1 --budget 0.02 \
      --out logs/smoke                                        # fast smoke run, kept out of results/

An event module in events/ defines:
  EVENT    dict(id, project, model, checkpoint, params, unit, notes)
  BATCHES  (latency_batch, throughput_batch), usually (1, N)
  make_inputs(batch, seed) -> dict of numpy arrays
  athletes(batch) -> dict name -> fn(inputs) -> dict of numpy arrays  (reference athlete: 'mlx_gpu')
  SKIPS    dict name -> reason, for athletes this event can't field
  agreement(out, ref) -> dict of floats   (optional; default: max abs error per output)
  task_check(name, batch) -> dict          (optional; a task-level check per athlete)
  coreml_packages(batch) -> dict athlete -> .mlpackage path  (optional; for placement + size)
  qualifies(agreement, task) -> (bool, reason)  (optional; is this athlete still "the same model"?
                                           task is None when there is no task check at that batch
                                           or the run used --no-task. Default: qualifies, reason
                                           "no tolerance defined".)

Each batch entry in the JSON holds agreement, memory_bytes, timing, task, coreml and
qualifies[name] = dict(ok=bool|None, reason=str) (ok None = qualifies() raised; reason says why).
"""
import argparse
import importlib
import os
import sys
import time

# MLX GPU "fp32" matmuls run as TF32 by default (rel err ~8e-4 at M >= 2). MLX reads MLX_ENABLE_TF32 once, at
# the first GPU matmul of the process, so it is set here before anything can run one (verified: setting it
# before the first matmul works even after `import mlx`; setting it after a matmul does nothing).
# --final and --true-fp32 force true fp32 (the reference); --tf32-pass forces TF32 on. Every run *measures*
# the mode (result["mlx_gpu_matmul_rel_err"]) instead of trusting the variable.
if __name__ == "__main__":
    if "--tf32-pass" in sys.argv:
        os.environ["MLX_ENABLE_TF32"] = "1"
    elif "--final" in sys.argv or "--true-fp32" in sys.argv:
        os.environ["MLX_ENABLE_TF32"] = "0"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import bench  # noqa: E402

TF32_THRESHOLD = 1e-5   # measured matmul rel err above this = TF32 (true fp32 ~5e-7, TF32 ~8e-4)
REFS = os.path.join(HERE, "results", "refs")

ATHLETE_ORDER = ["numpy_cpu", "mlx_cpu", "mlx_gpu", "mlx_gpu_fp16", "mlx_gpu_compiled",
                 "mlx_gpu_q8", "mlx_gpu_q4", "coreml_cpu", "coreml_gpu", "coreml_ane"]


def list_events():
    return sorted(f[:-3] for f in os.listdir(os.path.join(HERE, "events"))
                  if f.startswith("e") and f.endswith(".py"))


def wait_for_idle(max_load=4.0, max_wait_s=900):
    t0 = time.time()
    while True:
        load1 = os.getloadavg()[0]
        busy = [p for p in bench.busy_processes(min_cpu=30.0) if "python" in p["comm"].lower()]
        if load1 <= max_load and not busy:
            return dict(waited_s=round(time.time() - t0, 1), load1=load1)
        if time.time() - t0 > max_wait_s:
            return dict(waited_s=round(time.time() - t0, 1), load1=load1, gave_up=True, busy=busy)
        print(f"[run_event] waiting for idle: load {load1:.1f}, busy python {busy}", flush=True)
        time.sleep(15)


def judge(mod, agreement, task):
    """Call the event's qualifies(agreement, task) -> dict(ok, reason). Never raises."""
    q = getattr(mod, "qualifies", None)
    if q is None:
        return dict(ok=True, reason="no tolerance defined")
    try:
        ok, reason = q(agreement, task)
        return dict(ok=bool(ok), reason=str(reason))
    except Exception as e:
        return dict(ok=None, reason=f"qualifies() raised {e!r}"[:200])


def mlx_precision(require_fp32=False, require_tf32=False):
    err = bench.mlx_gpu_matmul_rel_err()
    tf32 = err > TF32_THRESHOLD
    if require_fp32 and tf32:
        raise RuntimeError(f"MLX GPU matmul is TF32 (rel err {err:.2g}) but true fp32 was required; "
                           "launch with MLX_ENABLE_TF32=0 in the environment")
    if require_tf32 and not tf32:
        raise RuntimeError(f"MLX GPU matmul is true fp32 (rel err {err:.2g}) but the TF32 pass needs TF32 on")
    return dict(mlx_tf32=tf32, mlx_gpu_matmul_rel_err=err, mlx_tf32_env=os.environ.get("MLX_ENABLE_TF32"))


def ref_path(event_name, batch):
    return os.path.join(REFS, f"{event_name}_b{batch}.npz")


def run(event_name, final=False, repeats=7, budget_s=0.25, only=None, task=True, batches=None,
        out_dir=None, true_fp32=False):
    mod = importlib.import_module(f"events.{event_name}")
    result = dict(event=mod.EVENT, final=final, batches={}, skips=getattr(mod, "SKIPS", {}),
                  task_checks=bool(task and hasattr(mod, "task_check")),
                  settings=dict(repeats=repeats, budget_s=budget_s, only=only))
    result.update(mlx_precision(require_fp32=final or true_fp32))
    agree_fn = getattr(mod, "agreement", None) or bench.max_abs_err
    with bench.exclusive():
        if final:
            result["idle_wait"] = wait_for_idle()
        result["env_before"] = bench.env_snapshot()
        result["noise_before"] = bench.noise_probe()
        for batch in mod.BATCHES:
            if batches and batch not in batches:
                continue
            inputs = mod.make_inputs(batch, seed=0)
            ath = mod.athletes(batch)
            if only:
                ath = {k: v for k, v in ath.items() if k in only or k == "mlx_gpu"}
            ref = ath["mlx_gpu"](inputs)
            if not result["mlx_tf32"] and out_dir is None:   # true-fp32 reference outputs, for the TF32 pass
                os.makedirs(REFS, exist_ok=True)
                np.savez(ref_path(event_name, batch), **{k: np.asarray(v) for k, v in ref.items()})
            entry = dict(agreement={}, memory_bytes={}, timing={}, task={}, coreml={})
            for name, fn in ath.items():
                entry["agreement"][name] = agree_fn(fn(inputs), ref)
                if name.startswith("mlx"):
                    entry["memory_bytes"][name] = bench.mlx_peak_memory(fn, inputs)
            entry["timing"] = bench.race(ath, inputs, repeats=repeats, budget_s=budget_s)
            if hasattr(mod, "coreml_packages"):
                import coreml_util
                for name, path in mod.coreml_packages(batch).items():
                    entry["coreml"][name] = dict(
                        package=os.path.relpath(path, HERE),
                        size_bytes=bench.dir_size(path),
                        placement=coreml_util.placement(path, name) if name in coreml_util.UNITS else None)
            if task and hasattr(mod, "task_check"):
                for name in ath:
                    entry["task"][name] = mod.task_check(name, batch)
            entry["qualifies"] = {name: judge(mod, entry["agreement"][name], entry["task"].get(name))
                                  for name in ath}
            result["batches"][str(batch)] = entry
        result["noise_after"] = bench.noise_probe()
        result["env_after"] = bench.env_snapshot()
    tag = "" if final else "_provisional"
    path = os.path.join(out_dir or os.path.join(HERE, "results"), f"{event_name}{tag}.json")
    bench.save_json(result, path)
    print_summary(result)
    print(f"wrote {os.path.relpath(path, HERE)}")
    return result


def run_tf32(event_name, repeats=7, budget_s=0.25, batches=None, out_dir=None):
    """TF32 pass (its own process, TF32 on): only mlx_gpu and mlx_gpu_compiled, renamed *_tf32, with agreement
    against the true-fp32 reference outputs the final pass saved in results/refs/. Not interleaved with the
    other athletes (different process). Task checks are not run here: an event's task_check compares an athlete
    with that process's mlx_gpu, which would be TF32 itself; qualifies() gets task=None (tolerance rule)."""
    mod = importlib.import_module(f"events.{event_name}")
    result = dict(event=mod.EVENT, final=True, tf32_pass=True, batches={}, skips={}, task_checks=False,
                  settings=dict(repeats=repeats, budget_s=budget_s))
    result.update(mlx_precision(require_tf32=True))
    agree_fn = getattr(mod, "agreement", None) or bench.max_abs_err
    with bench.exclusive():
        result["env_before"] = bench.env_snapshot()
        result["noise_before"] = bench.noise_probe()
        for batch in mod.BATCHES:
            if batches and batch not in batches:
                continue
            rp = ref_path(event_name, batch)
            if not os.path.exists(rp):
                print(f"[run_event] no true-fp32 reference {os.path.relpath(rp, HERE)}; run --final first. Skipping b={batch}")
                continue
            z = np.load(rp)
            ref = {k: z[k] for k in z.files}
            inputs = mod.make_inputs(batch, seed=0)
            full = mod.athletes(batch)
            ath = {f"{n}_tf32": full[n] for n in ("mlx_gpu", "mlx_gpu_compiled") if n in full}
            entry = dict(agreement={}, memory_bytes={}, timing={}, task={}, coreml={})
            for name, fn in ath.items():
                entry["agreement"][name] = agree_fn(fn(inputs), ref)
                entry["memory_bytes"][name] = bench.mlx_peak_memory(fn, inputs)
            entry["timing"] = bench.race(ath, inputs, repeats=repeats, budget_s=budget_s)
            entry["qualifies"] = {name: judge(mod, entry["agreement"][name], None) for name in ath}
            result["batches"][str(batch)] = entry
        result["noise_after"] = bench.noise_probe()
        result["env_after"] = bench.env_snapshot()
    path = os.path.join(out_dir or os.path.join(HERE, "results"), f"{event_name}_tf32.json")
    bench.save_json(result, path)
    print_summary(result)
    print(f"wrote {os.path.relpath(path, HERE)}")
    return result


def print_summary(result):
    mode = "TF32" if result.get("mlx_tf32") else "true fp32"
    err = result.get("mlx_gpu_matmul_rel_err")
    print(f"== {result['event']['id']}  ({'FINAL' if result['final'] else 'provisional'}"
          f"{', TF32 pass' if result.get('tf32_pass') else ''}; MLX GPU matmul {mode}"
          f"{'' if err is None else f', rel err {err:.1g}'})")
    for batch, e in result["batches"].items():
        names = sorted(e["timing"], key=lambda n: ATHLETE_ORDER.index(n) if n in ATHLETE_ORDER else 99)
        for n in names:
            t = e["timing"][n]
            agree = ", ".join(f"{k}={v:.3g}" for k, v in e["agreement"][n].items())
            q = e.get("qualifies", {}).get(n)
            qs = "" if q is None else ("  [Q]" if q["ok"] else "  [DQ: " + q["reason"] + "]")
            print(f"  b={batch:>5} {n:<17} {t['median_ms']:9.3f} ms (p90 {t['p90_ms']:.3f})  {agree}{qs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event", help="event module name (e.g. e09_alphazero) or 'all'")
    ap.add_argument("--final", action="store_true", help="scored pass: wait for an idle machine")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--budget", type=float, default=0.25, help="seconds per athlete per round")
    ap.add_argument("--only", nargs="*", help="limit to these athletes (mlx_gpu always runs)")
    ap.add_argument("--no-task", action="store_true", help="skip task checks (fast smoke runs)")
    ap.add_argument("--batches", nargs="*", type=int, help="limit to these of the event's BATCHES")
    ap.add_argument("--out", help="write the JSON here instead of results/ (e.g. logs/smoke)")
    ap.add_argument("--true-fp32", action="store_true",
                    help="dev run with MLX GPU in true fp32 (MLX_ENABLE_TF32=0; --final implies it)")
    ap.add_argument("--tf32-pass", action="store_true",
                    help="after --final: time mlx_gpu/mlx_gpu_compiled with TF32 on as *_tf32 athletes, "
                         "agreement vs the saved true-fp32 refs -> results/<event>_tf32.json")
    a = ap.parse_args()
    if a.tf32_pass and (a.final or a.true_fp32):
        ap.error("--tf32-pass runs in its own process; don't combine it with --final/--true-fp32")
    names = list_events() if a.event == "all" else [a.event]
    failed = []
    for n in names:
        try:
            if a.tf32_pass:
                run_tf32(n, repeats=a.repeats, budget_s=a.budget, batches=a.batches, out_dir=a.out)
            else:
                run(n, final=a.final, repeats=a.repeats, budget_s=a.budget, only=a.only, task=not a.no_task,
                    batches=a.batches, out_dir=a.out, true_fp32=a.true_fp32)
        except Exception:   # one broken event must not sink a whole `all` pass
            import traceback
            traceback.print_exc()
            failed.append(n)
            print(f"[run_event] {n} FAILED; continuing", flush=True)
    if failed:
        print(f"[run_event] failed events: {', '.join(failed)}")
        sys.exit(1)
