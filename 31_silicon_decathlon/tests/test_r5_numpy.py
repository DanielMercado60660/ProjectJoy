"""Explicitly selected tests: no old MLX test collection and no MLX imports."""
import ast
import json
from pathlib import Path
import sys
import time
import subprocess
import shutil

import numpy as np
import pytest

from r5_core import Frame, Ledger, Pending, atomic_json, quantiles, run_episode
from r5_run import make_plan
from r5_worlds import CarWorld, TrackerWorld, move_toward
from r5_report import render_report


def test_no_mlx_imported():
    assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)


def test_queue_and_latest_preserve_capture_identity():
    fifo, latest = Pending("queue"), Pending("latest")
    frames = [Frame(i, i*.1, i*.1, i*.1+.01, {}) for i in range(5)]
    for f in frames:
        fifo.push(f)
        latest.push(f)
    assert [fifo.pop().index for _ in frames] == list(range(5))
    assert fifo.dropped == 0
    assert latest.pop() is frames[-1]
    assert latest.dropped == 4 and latest.peak == 1
    assert latest.pop() is None
    with pytest.raises(ValueError):
        Pending("typo")


def test_metrics_empty_and_quantiles():
    assert quantiles([]) == dict(n=0, p50=None, p95=None, p99=None)
    assert quantiles([1, 2, 3])["p50"] == 2
    with pytest.raises(AssertionError):
        quantiles([float("nan")])


class ToyWorld:
    hz = 100.0
    def __init__(self):
        self.t, self.x, self.action = 0.0, 0.0, 1.0
        self.trace = []
    def advance(self, t):
        self.x += max(0, t-self.t)*self.action
        self.t = t
    def capture(self, tick):
        self.trace.append([tick, self.x])
        return {"x": self.x}
    def apply(self, out):
        self.action = out["a"]
    def score(self):
        return {"x": self.x}
    def geometry(self):
        return {}


class ToyBackend:
    def __init__(self, service=0.0):
        self.service = service
        self.seen = []
    def reset(self):
        self.seen = []
    def infer(self, inputs):
        self.seen.append(inputs["x"])
        t = time.perf_counter()
        time.sleep(self.service)
        return {"a": -inputs["x"]}, dict(dispatch_ms=0.0, compute_ms=(time.perf_counter()-t)*1000,
                                         postprocess_ms=0.0)


@pytest.mark.parametrize("policy", ["queue", "latest"])
def test_async_capture_deadlines_and_breakdown(policy):
    backend = ToyBackend(0.025)
    result = run_episode(ToyWorld(), backend, policy, 0.22)
    m = result["metrics"]
    assert m["captured"] > m["applied"] > 0
    assert m["deadline_misses"] == m["scheduled"] - m["on_time"]
    assert m["deadline_miss_fraction"] > 0.8
    assert len(backend.seen) >= m["applied"]
    if policy == "queue":
        assert m["dropped"] == 0 and m["pending_at_end"] > 0
    else:
        assert m["dropped"] > 0 and m["queue_peak"] == 1
    assert m["captured"] == m["applied"]+m["dropped"]+m["pending_at_end"]+m["inflight_at_end"]
    for r in result["actions"]:
        stages = sum(r[k] for k in ("preprocess_ms", "queue_ms", "dispatch_ms", "compute_ms",
                                   "postprocess_ms", "injected_ms", "pickup_ms", "overhead_ms"))
        assert abs(stages-r["age_ms"]) < 1e-6
        assert r["applied_s"] > r["captured_s"]
    # Captures happened at different world states during in-flight inference.
    assert len({x for _, x in result["trace"]}) > 3


def test_no_action_is_penalized_and_guard_clips_horizon():
    result = run_episode(ToyWorld(), ToyBackend(0.08), "latest", 1.0, max_seconds=0.03)
    assert result["metrics"]["applied"] == 0
    assert result["metrics"]["deadline_miss_fraction"] == 1
    assert result["metrics"]["age_ms"]["p99"] is None
    assert result["task"]["x"] == pytest.approx(0.03)


def test_resume_and_config_identity(tmp_path):
    p = tmp_path / "state.json"
    cfg = {"weights": "hash", "seeds": [32, 33], "policy": "latest"}
    l = Ledger(p, cfg)
    l.commit("a")
    l.commit("a")
    assert l.state["step"] == 1
    resumed = Ledger(p, cfg, True)
    resumed.commit("b")
    assert json.loads(p.read_text())["completed"] == ["a", "b"]
    with pytest.raises(ValueError):
        Ledger(p, cfg)
    with pytest.raises(ValueError):
        Ledger(p, dict(cfg, weights="changed"), True)
    assert not p.with_suffix(".json.tmp").exists()


def test_tracker_fresh_seed_and_renderer():
    a, b = TrackerWorld(910031), TrackerWorld(910031)
    c = TrackerWorld(910032)
    assert np.array_equal(a.pos, b.pos)
    assert not np.array_equal(a.pos, c.pos)
    assert a.pos.shape == (40, 3, 2)
    assert np.min(a.pos) >= 3.5 and np.max(a.pos) <= 60.5
    assert a.vis[a.present[None].repeat(40, axis=0)].min() == 0
    x = a.capture(0)["frame"]
    assert x.shape == (1, 64, 64, 3) and x.dtype == np.float32
    assert x.min() >= 0 and x.max() <= 1
    assert np.array_equal(x, b.capture(0)["frame"])
    a.advance(40/30)
    assert a.score()["pursuit_error_px"] > 0  # Never-act cannot get zero error.
    assert a.score()["hidden_tracking_error_px"] is not None


def test_pursuit_speed_bound_and_no_overshoot():
    x = np.zeros((2, 2))
    y = np.array([[30, 40], [0.1, 0]])
    nxt = move_toward(x, y, 10, 0.1)
    assert np.linalg.norm(nxt[0]) == pytest.approx(1)
    assert np.array_equal(nxt[1], y[1])
    assert np.array_equal(move_toward(x, x, 10, 1), x)


def test_car_sensor_matches_existing_geometry_and_stale_hold_moves():
    # No neural network: independent reference ray casting on the same track.
    from models.grand_prix_sim import _cast, RAY_ANGLES
    w = CarWorld(910031)
    tr = w.track
    aa = np.concatenate([tr.left, tr.right])
    bb = np.concatenate([np.roll(tr.left, -1, axis=0), np.roll(tr.right, -1, axis=0)])
    rays = [_cast.py_func(*w.xy, w.yaw+a, aa, bb, len(aa), 60)/60 for a in RAY_ANGLES]
    np.testing.assert_allclose(w.capture(0)["obs"][0, :10], rays, atol=1e-7)
    w.apply({"action": [[0, 1]]})
    start = w.xy.copy()
    w.advance(0.1)
    assert w.v > 0.5 and np.linalg.norm(w.xy-start) > 0.02
    assert 0 < w.progress < 1
    with pytest.raises(ValueError):
        w.apply({"action": [[np.nan, 1]]})


def pilot_fixture():
    return dict(engines={event: {engine: dict(wall_per_horizon=1.01, world_setup_s=0.01)
                                for engine in ("numpy_cpu", "mlx_gpu", "coreml_cpu", "coreml_ane")}
                         for event in ("car", "tracker")}, load={"ok": True}, sources={}, checkpoint="h", gaps={})


def test_budget_scales_seeds_and_every_block_is_matched():
    pilot = pilot_fixture()
    engines = ["numpy_cpu", "mlx_gpu", "coreml_cpu", "coreml_ane"]
    plan = make_plan(pilot, 23*60, "nominal", engines)
    assert plan["rounds"] == 2
    assert plan["predicted_s"] < plan["target_s"]
    assert plan["predicted_s"] > plan["target_s"]*0.9
    for block in range(plan["rounds"]):
        jobs = [j for j in plan["jobs"] if j["block"] == block]
        assert len(jobs) == 32 and len({j["seed"] for j in jobs}) == 1
        assert len({(j["event"], j["engine"], j["policy"], j["load"]) for j in jobs}) == 32
    for ev in pilot["engines"]:
        del pilot["engines"][ev]["coreml_cpu"]
        del pilot["engines"][ev]["coreml_ane"]
    smaller = make_plan(pilot, 23*60, "nominal", engines)
    assert smaller["rounds"] == 4
    assert smaller["omitted"] == ["coreml_cpu", "coreml_ane"]
    stress = make_plan(pilot, 5*60, "stress", engines, seed_start=920031)
    assert {j["delay_ms"] for j in stress["jobs"]} == {0, 50}
    assert {j["event"] for j in stress["jobs"]} == {"tracker"}
    with pytest.raises(ValueError):
        make_plan(pilot, 1, "nominal", engines)


def test_new_files_parse_without_loading_engines():
    for p in Path(".").glob("r5_*.py"):
        ast.parse(p.read_text(), filename=str(p))
    assert "mlx.core" not in sys.modules


def test_censored_worker_error_is_not_hidden():
    class Broken(ToyBackend):
        def infer(self, inputs):
            time.sleep(.03)
            raise ValueError("backend failed after horizon")
    with pytest.raises(ValueError, match="backend failed"):
        run_episode(ToyWorld(), Broken(), "latest", .01)


def test_report_keeps_failures_and_pooled_quantiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("results/r5/fixture")
    root.mkdir(parents=True)
    atomic_json(root / "run_status.json", dict(complete=False, done=2, planned=4))
    for i, ages in enumerate(([1, 2, 3], [100])):
        atomic_json(root / f"episode_{i:05d}.json", dict(
            job=dict(event="car", engine="numpy_cpu", policy="queue", load="ambient", delay_ms=0,
                     seed=i), metrics=dict(scheduled=4, on_time=2-i, applied=len(ages), horizon_s=1),
            task=dict(finished=False, crashed=True, progress_fraction=0.1),
            actions=[dict(age_ms=a) for a in ages], trace=[[0.1, 1, 2, 3, 0.1, 1, 0]],
            geometry=dict(left=[[0, 0], [10, 10]], right=[[1, 1], [9, 9]])))
    assert render_report("fixture", "media/r5_fixture.html") == 2
    page = Path("media/r5_fixture.html").read_text()
    assert "PARTIAL" in page and "unmatched cells" in page
    assert "http://" not in page and "https://" not in page and "__DATA__" not in page
    data = json.loads((root / "summary.json").read_text())["summaries"][0]
    assert data["age_ms"]["p50"] == 2.5  # Pooled actions, not mean episode percentile.
    assert data["task"]["crash_fraction"] == 1
    assert data["action_coverage"] == .5
    assert data["deadline_miss_fraction"] == 5/8
    # Add the second renderer path without pretending the fixture is an evaluation.
    atomic_json(root / "episode_00002.json", dict(
        job=dict(event="tracker", engine="numpy_cpu", policy="latest", load="ambient", delay_ms=0, seed=3),
        metrics=dict(scheduled=1, on_time=1, applied=1, horizon_s=1),
        task=dict(pursuit_error_px=3, tracking_error_px=2, within_5px_fraction=.8),
        actions=[dict(age_ms=2)], geometry=dict(occ=[[20, 30]], present=[True, True, False]),
        trace=[[.1, [[10, 20], [30, 40], [0, 0]], [[11, 20], [31, 40], [0, 0]],
                [[12, 20], [32, 40], [0, 0]]]]))
    render_report("fixture", "media/r5_fixture.html")
    if shutil.which("node"):
        checker = Path(__file__).resolve().parent / "check_r5_demo.js"
        subprocess.run(["node", str(checker), "media/r5_fixture.html"], check=True, capture_output=True)


def test_hard_wall_guard_no_mlx():
    # A deliberately tiny limit expires before the child can reach engine setup.
    cmd = [sys.executable, "r5_run.py", "run", "--plan", "logs/r5/nonexistent_guard_plan.json",
           "--name", "guard_probe", "--max-minutes", "0.00001"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    assert p.returncode == 124 and "WALL GUARD" in p.stdout
