#!/usr/bin/env python3
"""Round-5 calibration, measured-rate planning, closed-loop evaluation and replay.

No MLX import until an engine or the background load is explicitly constructed.
The command supervisor enforces --max-minutes, including stuck native calls.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from r5_backends import (Backend, Background, DEFAULT_ENGINES, ENGINES, SELECTED,
                         environment, selected_manifest)
from r5_core import Ledger, atomic_json, digest, quantiles, run_episode
from r5_worlds import make_world


def source_hashes():
    paths = sorted(Path(".").glob("r5_*.py"))
    paths += [Path("models/grand_prix.py"), Path("models/grand_prix_sim.py"), Path("models/tracker.py")]
    return {str(p): digest(p) for p in paths}


def checkpoint_id():
    return digest(SELECTED / "manifest.json")


def calibrate(args):
    start = time.perf_counter()
    selected_manifest()
    result = dict(environment_before=environment(), sources=source_hashes(), checkpoint=checkpoint_id(),
                  engines={}, gaps={}, load={}, timings={})
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Launch with MLX_ENABLE_TF32=0 before importing MLX")
    from bench import mlx_gpu_matmul_rel_err
    result["mlx_matmul_relative_error"] = mlx_gpu_matmul_rel_err()
    if result["mlx_matmul_relative_error"] > 1e-5:
        raise ValueError("MLX precision probe is not true-fp32 class")
    # Reference is the exhibit's independent NumPy dialect, same frozen weights.
    for event in ("car", "tracker"):
        world = make_world(event, 850031)
        duration = 1.0 if event == "car" else 40/30
        inputs, expected = [], []
        ref = Backend(event, "numpy_cpu")
        for i in range(40):
            world.advance(i/world.hz)
            inp = world.capture(i)
            inputs.append(inp)
            target = ref.infer(inp)[0]
            expected.append(target)
            world.apply(target)
        result["engines"][event] = {}
        for name in args.engines:
            if time.perf_counter()-start > args.max_minutes*60-5:
                raise TimeoutError("Calibration budget exhausted")
            try:
                backend = Backend(event, name)
                errors = []
                for inp, target in zip(inputs, expected):
                    got, _ = backend.infer(inp)
                    key = "action" if event == "car" else "out"
                    # Tracker compares mean pixels over a recurrent 40-frame rollout.
                    delta = got[key]-target[key]
                    if event == "tracker":
                        delta = np.linalg.norm(delta[..., :2]*32, axis=-1)
                    errors.append(float(np.max(np.abs(delta))))
                tolerance = 0.05 if event == "car" else 0.5
                if max(errors) > tolerance:
                    raise ValueError(f"40-step parity error {max(errors)} > {tolerance}")
                setup = time.perf_counter()
                world = make_world(event, 850032)
                setup_s = time.perf_counter()-setup
                # The same scheduler used in the main run supplies the measured wall rate.
                episode = run_episode(world, backend, "latest", duration)
                m = episode["metrics"]
                result["engines"][event][name] = dict(
                    meta=backend.meta, parity_max_error=max(errors), parity_tolerance=tolerance,
                    world_setup_s=setup_s, wall_s=m["wall_s"], horizon_s=m["horizon_s"],
                    wall_per_horizon=m["wall_s"]/m["horizon_s"],
                    measured_applied_per_second=m["applied"]/m["wall_s"], metrics=m)
                print(f"READY {event} {name} parity={max(errors):.6g} "
                      f"wall={m['wall_s']:.3f}s applied/s={m['applied']/m['wall_s']:.2f}", flush=True)
            except Exception as e:
                result["gaps"][f"{event}/{name}"] = repr(e)
                print(f"GAP {event} {name}: {e}", flush=True)
    bg = None
    try:
        bg = Background()
        before = bg.set_load(True)
        backend = Backend("tracker", "mlx_gpu")
        episode = run_episode(make_world("tracker", 850033), backend, "queue", 0.25, delay_ms=50)
        after = bg.count.value
        bg.set_load(False)
        if after <= before:
            raise RuntimeError("Background completed no matmuls during test")
        result["load"] = dict(ok=True, matmuls=after-before, shape=[1024, 1024],
                              target_duty=0.7, overload_metrics=episode["metrics"])
    except Exception as e:
        result["load"] = dict(ok=False, reason=repr(e))
    finally:
        if bg is not None:
            bg.close()
    result["environment_after"] = environment()
    result["wall_s"] = time.perf_counter()-start
    atomic_json(args.out, result)
    print(f"CALIBRATION {args.out} wall={result['wall_s']:.2f}s", flush=True)
    required = all("numpy_cpu" in result["engines"][ev] and "mlx_gpu" in result["engines"][ev]
                   for ev in ("car", "tracker"))
    if not required or not result["load"]["ok"]:
        raise RuntimeError("Required NumPy/MLX/load path failed; inspect calibration gaps")


def make_plan(pilot, target_seconds, mode, engines, seed_start=910031, split="fresh"):
    """Whole matched blocks only. Missing Core ML engines buy more fresh seeds."""
    if target_seconds <= 0:
        raise ValueError("Target time must be positive")
    events = ["tracker"] if mode == "stress" else ["car", "tracker"]
    available = [e for e in engines if all(e in pilot["engines"][v] for v in events)]
    if not all(e in available for e in ("numpy_cpu", "mlx_gpu")):
        raise ValueError("Need a common NumPy and MLX GPU engine before planning")
    if not pilot["load"].get("ok"):
        raise ValueError("Concurrent load has not passed calibration")
    horizons = {"car": 40.0, "tracker": 40/30}
    delays = [0.0, 50.0] if mode == "stress" else [0.0]
    cells = list(itertools.product(events, available, ("queue", "latest"), ("ambient", "loaded"), delays))
    # Measured scheduler speed plus measured world construction; fixed 0.15 s
    # reserve per episode covers power/load snapshots and load switching.
    block_seconds = sum(horizons[ev]*pilot["engines"][ev][engine]["wall_per_horizon"]
                        + pilot["engines"][ev][engine]["world_setup_s"] + 0.15
                        for ev, engine, *_ in cells)
    rounds = int(target_seconds // block_seconds)
    if rounds < 1:
        raise ValueError(f"Budget cannot fit one balanced block ({block_seconds:.1f}s)")
    jobs = []
    for r in range(rounds):
        # Stable independent order per seed block. No test-result-dependent selection.
        order = np.random.default_rng(503100+r).permutation(len(cells))
        for j in order:
            event, engine, policy, load, delay = cells[j]
            jobs.append(dict(event=event, engine=engine, policy=policy, load=load, delay_ms=delay,
                             seed=seed_start+r, split=split, duration=horizons[event], block=r))
    return dict(mode=mode, engines=available, omitted=[e for e in engines if e not in available],
                rounds=rounds, target_s=target_seconds, measured_block_s=block_seconds,
                predicted_s=rounds*block_seconds, jobs=jobs, sources=pilot["sources"],
                checkpoint=pilot["checkpoint"], pilot_gaps=pilot["gaps"],
                budget_basis="pilot wall/horizon and setup seconds; complete matched seed blocks")


def plan_command(args):
    pilot = json.loads(Path(args.pilot).read_text())
    plan = make_plan(pilot, args.target_minutes*60, args.mode, args.engines, args.seed_start, args.split)
    plan["pilot_file"] = args.pilot
    plan["pilot_sha256"] = digest(args.pilot)
    plan["data_hashes"] = ({"../15_object_permanence/data/test_long.npz":
                           digest("../15_object_permanence/data/test_long.npz")} if args.split == "old" else {})
    atomic_json(args.out, plan)
    print(f"PLAN {args.out}: {plan['rounds']} seed blocks, {len(plan['jobs'])} episodes, "
          f"predicted {plan['predicted_s']/60:.2f} min / target {args.target_minutes:.2f} min; "
          f"omitted={plan['omitted']}")


def run_command(args):
    start = time.perf_counter()
    plan = json.loads(Path(args.plan).read_text())
    selected_manifest()
    if plan["sources"] != source_hashes() or plan["checkpoint"] != checkpoint_id():
        raise ValueError("Code/checkpoint differs from calibration; recalibrate and make a new plan")
    for path, expected in plan.get("data_hashes", {}).items():
        if digest(path) != expected:
            raise ValueError(f"Secondary comparison data changed: {path}")
    # Include schedule itself, model identity and code in the resume contract.
    config = dict(plan=plan, plan_sha256=digest(args.plan), name=args.name)
    state = Ledger(f"checkpoints/r5_{args.name}_state.json", config, args.resume)
    outdir = Path("results/r5") / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    bg, cache = None, {}
    completed_at_start = state.state["step"]
    try:
        bg = Background()
        for index, job in enumerate(plan["jobs"]):
            key = f"episode_{index:05d}"
            receipt = outdir / (key+".json")
            if key in state.state["completed"]:
                if not receipt.exists():
                    raise ValueError(f"Missing committed receipt: {receipt}")
                continue
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous["job"] != job or previous["plan_sha256"] != config["plan_sha256"]:
                    raise ValueError(f"Conflicting orphan receipt: {receipt}")
                state.commit(key)
                continue
            if time.perf_counter()-start + job["duration"]+5 > args.max_minutes*60:
                print("GUARD stopping before next episode; --resume continues the saved schedule", flush=True)
                break
            cache_key = (job["event"], job["engine"])
            if cache_key not in cache:
                cache[cache_key] = Backend(*cache_key)
            backend = cache[cache_key]
            world = make_world(job["event"], job["seed"], job["split"])
            # Warm-up excluded, then run_episode resets hidden state to zero.
            backend.reset()
            backend.infer(world.capture(0))
            count0 = bg.set_load(job["load"] == "loaded")
            before = environment()
            count0 = bg.count.value  # Count only across the timed episode.
            print(f"START {key}", flush=True)
            result = run_episode(world, backend, job["policy"], job["duration"], job["delay_ms"])
            count1 = bg.count.value
            after = environment()
            if job["load"] == "loaded" and count1 <= count0:
                raise RuntimeError("Loaded episode had no concurrent matmuls")
            if job["load"] == "ambient" and count1 != count0:
                raise RuntimeError("Ambient episode overlapped our background worker")
            result.update(job=job, plan_sha256=config["plan_sha256"], backend=backend.meta,
                          env_before=before, env_after=after,
                          background_matmuls=count1-count0, background_pid=bg.process.pid,
                          timing_note="Host-side stages: compute includes native internal dispatch; "
                                      "no hardware kernel-only timing. Ambient can include other projects.")
            atomic_json(receipt, result)
            state.commit(key)
            m = result["metrics"]
            print(f"DONE {key} {job['event']} {job['engine']} {job['policy']} {job['load']} "
                  f"seed={job['seed']} age_p95={m['age_ms']['p95']} "
                  f"coverage={m['action_coverage']:.3f} task={result['task']}", flush=True)
            if args.stop_after and state.state["step"]-completed_at_start >= args.stop_after:
                print("STOP_AFTER checkpoint/resume probe complete", flush=True)
                break
    finally:
        if bg is not None:
            bg.close()
    complete = state.state["step"] == len(plan["jobs"])
    atomic_json(outdir / "run_status.json", dict(complete=complete, done=state.state["step"],
                                                planned=len(plan["jobs"]), plan=args.plan,
                                                wall_this_invocation_s=time.perf_counter()-start))
    print(f"RUN {'COMPLETE' if complete else 'PARTIAL'} {state.state['step']}/{len(plan['jobs'])}", flush=True)


def main():
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    cal.add_argument("--out", default="results/r5/pilot.json")
    pl = sub.add_parser("plan")
    pl.add_argument("--pilot", default="results/r5/pilot.json")
    pl.add_argument("--mode", choices=("nominal", "stress"), default="nominal")
    pl.add_argument("--engines", nargs="+", choices=ENGINES, default=list(DEFAULT_ENGINES))
    pl.add_argument("--target-minutes", type=float, default=23)
    pl.add_argument("--seed-start", type=int, default=910031)
    pl.add_argument("--split", choices=("fresh", "old"), default="fresh")
    pl.add_argument("--out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--stop-after", type=int, default=0)
    for p in (cal, run):
        p.add_argument("--max-minutes", type=float, default=2 if p is cal else 25)
        p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.command == "run" and (not args.name.replace("_", "").replace("-", "").isalnum()
                                  or args.name == "selected"):
        ap.error("--name must be a simple unique run label")
    if args.command != "plan":
        if args.max_minutes <= 0 or args.max_minutes > 45:
            ap.error("--max-minutes must be in (0, 45]")
        if not args.child:
            # Supervise our own process group, never other projects or the slot owner.
            child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--child"], start_new_session=True)
            try:
                code = child.wait(timeout=args.max_minutes*60)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                print("WALL GUARD: stopped own process group; completed receipts survive", flush=True)
                code = 124
            raise SystemExit(code)
    {"calibrate": calibrate, "plan": plan_command, "run": run_command}[args.command](args)


if __name__ == "__main__":
    main()
