"""Wall-clock loop, metrics and atomic episode-boundary resume. NumPy only.

The world belongs to the main thread; one worker owns inference and recurrent
state. Sensor timestamps are actual capture times, never backdated snapshots.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    assert np.isfinite(a).all()
    return dict(n=len(a), **{f"p{p}": float(np.percentile(a, p)) for p in (50, 95, 99)})


@dataclass
class Frame:
    index: int
    scheduled: float
    captured: float
    ready: float
    inputs: dict
    task_active: bool = True


class Pending:
    def __init__(self, policy):
        if policy not in ("queue", "latest"):
            raise ValueError(policy)
        self.policy, self.frames, self.dropped = policy, deque(), 0
        self.peak = 0

    def push(self, frame):
        if self.policy == "latest":
            self.dropped += len(self.frames)
            self.frames.clear()
        self.frames.append(frame)
        self.peak = max(self.peak, len(self.frames))

    def pop(self):
        return self.frames.popleft() if self.frames else None


def run_episode(world, backend, policy, duration, delay_ms=0.0, max_seconds=None):
    """Run for the entire physical horizon, even after a car crashes/finishes.

    Captures continue while the worker is busy. All action applications include
    main-thread pickup delay. At the horizon no pending work is drained or
    credited. An in-flight native call cannot be preempted; watchdog is external.
    """
    period = 1.0 / world.hz
    duration = min(duration, max_seconds) if max_seconds is not None else duration
    pending, rows, capture_rows = Pending(policy), [], []
    backend.reset()
    start = time.perf_counter()
    end = start + duration
    future, submitted = None, None
    next_tick = 0
    missed_sensor = 0
    pool = ThreadPoolExecutor(max_workers=1)

    def infer(frame):
        begun = time.perf_counter()
        out, stages = backend.infer(frame.inputs)
        hold = time.perf_counter()
        if delay_ms:
            time.sleep(delay_ms / 1000)
        finished = time.perf_counter()
        return out, stages, begun, finished, (finished - hold) * 1000

    try:
        while True:
            now = time.perf_counter()
            world.advance(min(now - start, duration))
            if now >= end:
                break
            if future is not None and future.done():
                out, stages, begun, finished, injected = future.result()
                # Poll completion first, then advance old held action to actual application.
                apply_at = time.perf_counter()
                world.advance(min(apply_at - start, duration))
                if apply_at >= end:
                    break
                world.apply(out)
                applied = time.perf_counter()
                age = (applied - submitted.captured) * 1000
                row = dict(frame=submitted.index, captured_s=submitted.captured-start,
                                 applied_s=applied-start, age_ms=age,
                                 task_active=submitted.task_active,
                                 deadline_missed=applied > submitted.scheduled + period,
                                 preprocess_ms=(submitted.ready-submitted.captured)*1000,
                                 queue_ms=(begun-submitted.ready)*1000,
                                 **stages, injected_ms=injected,
                                 pickup_ms=(applied-finished)*1000)
                row["overhead_ms"] = age-sum(row[k] for k in (
                    "preprocess_ms", "queue_ms", "dispatch_ms", "compute_ms",
                    "postprocess_ms", "injected_ms", "pickup_ms"))
                rows.append(row)
                future = submitted = None
            now = time.perf_counter()
            if now >= end:
                break
            if now >= start + next_tick * period:
                # Omitted sensor ticks are losses, not retroactively rendered frames.
                tick = int((now - start) / period)
                missed_sensor += max(0, tick - next_tick)
                scheduled = start + tick * period
                capture_at = now  # The state is advanced to exactly this physical time.
                world.advance(now-start)
                inp = world.capture(tick)
                ready = time.perf_counter()
                active = not (getattr(world, "finished", False) or getattr(world, "crashed", False))
                pending.push(Frame(tick, scheduled, capture_at, ready, inp, active))
                capture_rows.append(dict(frame=tick, jitter_ms=(capture_at-scheduled)*1000))
                next_tick = tick + 1
            if future is None:
                submitted = pending.pop()
                if submitted is not None:
                    future = pool.submit(infer, submitted)
            # Bounded polling adds real, reported pickup latency to every engine.
            remaining = min(end, start + next_tick * period) - time.perf_counter()
            if remaining > 0:
                time.sleep(min(0.0005, remaining))
    finally:
        # Do not reuse recurrent state until the worker has stopped.
        pool.shutdown(wait=True, cancel_futures=True)
    if future is not None:
        future.result()  # A censored call's failure is still a failure, not a valid trial.
    elapsed = time.perf_counter() - start
    scheduled_count = int(np.ceil(duration * world.hz - 1e-9))
    on_time = sum(not r["deadline_missed"] for r in rows)
    metrics = dict(
        age_ms=quantiles([r["age_ms"] for r in rows]),
        active_age_ms=quantiles([r["age_ms"] for r in rows if r["task_active"]]),
        stages_ms={k: quantiles([r[k] for r in rows]) for k in
                   ("preprocess_ms", "queue_ms", "dispatch_ms", "compute_ms",
                    "postprocess_ms", "injected_ms", "pickup_ms", "overhead_ms")},
        sensor_jitter_ms=quantiles([r["jitter_ms"] for r in capture_rows]),
        scheduled=scheduled_count, captured=len(capture_rows), applied=len(rows),
        on_time=on_time, deadline_misses=scheduled_count-on_time,
        deadline_miss_fraction=1-on_time/max(1, scheduled_count),
        action_coverage=len(rows)/max(1, scheduled_count),
        dropped=pending.dropped, pending_at_end=len(pending.frames),
        inflight_at_end=int(future is not None), sensor_ticks_lost=missed_sensor,
        queue_peak=pending.peak, wall_s=elapsed, horizon_s=duration,
    )
    return dict(metrics=metrics, task=world.score(), actions=rows,
                trace=world.trace, geometry=world.geometry())


class Ledger:
    """One replaceable state file; per-episode receipts are small result files.

    Crash between receipt and state publication is recoverable. An interrupted
    episode restarts from its seed, since real elapsed time cannot be resumed.
    """
    def __init__(self, path, config, resume=False):
        self.path, self.config = Path(path), config
        if self.path.exists():
            if not resume:
                raise ValueError(f"{path} exists; use --resume or a new --name")
            self.state = json.loads(self.path.read_text())
            if self.state["config"] != config:
                raise ValueError("Resume config differs from saved config")
        else:
            self.state = dict(config=config, step=0, completed=[], optimizer_state=None,
                              optimizer_reason="Frozen inference benchmark; no optimizer",
                              rng="Per-episode seed and deterministic schedule in config")
            atomic_json(self.path, self.state)

    def commit(self, key):
        if key not in self.state["completed"]:
            self.state["completed"].append(key)
            self.state["step"] = len(self.state["completed"])
            atomic_json(self.path, self.state)
