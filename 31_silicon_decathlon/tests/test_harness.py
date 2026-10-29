"""Harness and report tests: timing race, qualifies verdicts, --no-task / --batches / --out, and make_report
(final-over-provisional preference, Core ML fallback detection, medals, demo.html)."""
import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import bench  # noqa: E402
import make_report  # noqa: E402
import run_event  # noqa: E402


def test_race_interleaves_and_reports():
    calls = []
    ath = {"a": lambda i: calls.append("a"), "b": lambda i: calls.append("b")}
    out = bench.race(ath, {}, repeats=3, budget_s=0.001, warmup=1, max_calls=5)
    assert set(out) == {"a", "b"}
    for t in out.values():
        assert t["rounds"] == 3 and t["calls"] >= 9 and t["p90_ms"] >= t["median_ms"] >= t["min_ms"] >= 0


def test_max_abs_err():
    e = bench.max_abs_err({"x": np.array([1.0, 2.0])}, {"x": np.array([1.0, 2.5])})
    assert e == {"x": 0.5}


def test_judge_default_and_errors():
    assert run_event.judge(types.SimpleNamespace(), {}, None) == dict(ok=True, reason="no tolerance defined")
    good = types.SimpleNamespace(qualifies=lambda a, t: (a["e"] < 1, "e small"))
    assert run_event.judge(good, {"e": 0.5}, None) == dict(ok=True, reason="e small")
    assert run_event.judge(good, {"e": 2}, None)["ok"] is False
    bad = types.SimpleNamespace(qualifies=lambda a, t: 1 / 0)
    v = run_event.judge(bad, {}, None)
    assert v["ok"] is None and "ZeroDivisionError" in v["reason"]


def _fake_event(with_qualifies=True):
    m = types.ModuleType("events.zz_fake")
    m.EVENT = dict(id="zz_fake", project="none", model="y = 2x", checkpoint="", unit="rows", params=4, notes="")
    m.BATCHES = (1, 8)
    m.SKIPS = {"coreml_ane": "test"}
    m.make_inputs = lambda batch, seed=0: {"x": np.random.default_rng(seed).standard_normal((batch, 4)).astype(np.float32)}
    m.athletes = lambda batch: {"numpy_cpu": lambda i: {"y": i["x"] * 2},
                                "mlx_gpu": lambda i: {"y": i["x"] * 2},
                                "wrong": lambda i: {"y": i["x"] * 3}}
    m.task_check = lambda name, batch: {"ran": True}
    if with_qualifies:
        m.qualifies = lambda a, t: (a["y"] < 1e-6, f"err {a['y']:.2g}, task {'yes' if t else 'no'}")
    return m


@pytest.mark.parametrize("task", [True, False])
def test_run_writes_qualifies(tmp_path, task):
    sys.modules["events.zz_fake"] = _fake_event()
    try:
        r = run_event.run("zz_fake", repeats=1, budget_s=0.001, task=task, batches=[8], out_dir=str(tmp_path))
    finally:
        del sys.modules["events.zz_fake"]
    assert list(r["batches"]) == ["8"]
    e = r["batches"]["8"]
    assert e["qualifies"]["numpy_cpu"]["ok"] is True and e["qualifies"]["wrong"]["ok"] is False
    assert (e["task"] != {}) == task and r["task_checks"] == task
    assert ("task yes" in e["qualifies"]["numpy_cpu"]["reason"]) == task
    saved = json.load(open(tmp_path / "zz_fake_provisional.json"))
    assert saved["batches"]["8"]["qualifies"]["wrong"]["ok"] is False


def test_run_default_verdict_without_qualifies(tmp_path):
    sys.modules["events.zz_fake"] = _fake_event(with_qualifies=False)
    try:
        r = run_event.run("zz_fake", repeats=1, budget_s=0.001, task=False, batches=[1], out_dir=str(tmp_path))
    finally:
        del sys.modules["events.zz_fake"]
    assert r["batches"]["1"]["qualifies"]["wrong"] == dict(ok=True, reason="no tolerance defined")


def test_cli_has_flags():
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_event.py"), "--help"],
                         capture_output=True, text=True, cwd=HERE).stdout
    for flag in ("--no-task", "--batches", "--out", "--final", "--only"):
        assert flag in out


def test_e09_qualifies():
    from events import e09_alphazero as e
    assert e.qualifies({}, {"best_move_agreement": 0.995})[0]
    assert not e.qualifies({}, {"best_move_agreement": 0.98})[0]
    assert e.qualifies({"policy_max_err": 0.01, "value_max_err": 0.01}, None)[0]
    assert not e.qualifies({"policy_max_err": 0.5, "value_max_err": 0.01}, None)[0]


# ---------------------------------------------------------------- make_report
def _result(final, ane_ms=1.0, ane_place=None, gpu_ok=True):
    timing = lambda ms: dict(median_ms=ms, p90_ms=ms * 1.1, min_ms=ms * 0.9, calls=10, rounds=1)
    entry = dict(
        agreement={n: {"err": 0.0} for n in ("numpy_cpu", "mlx_gpu", "coreml_cpu", "coreml_ane", "coreml_gpu")},
        memory_bytes={"mlx_gpu": 1000}, task={},
        timing={"numpy_cpu": timing(4.0), "mlx_gpu": timing(2.0), "coreml_cpu": timing(0.5),
                "coreml_ane": timing(ane_ms), "coreml_gpu": timing(0.1)},
        coreml={"coreml_cpu": dict(placement={"CPU": 10}), "coreml_ane": dict(placement=ane_place or {"NeuralEngine": 10}),
                "coreml_gpu": dict(placement={"GPU": 10})},
        qualifies={"numpy_cpu": dict(ok=True, reason="r"), "mlx_gpu": dict(ok=True, reason="r"),
                   "coreml_cpu": dict(ok=True, reason="r"), "coreml_ane": dict(ok=True, reason="r"),
                   "coreml_gpu": dict(ok=gpu_ok, reason="r")})
    return dict(event=dict(id="zz", params=100, unit="rows", model="m"), final=final, batches={"1": entry}, skips={})


def test_report_prefers_final_and_ranks(tmp_path):
    json.dump(_result(False, ane_ms=0.2), open(tmp_path / "zz_provisional.json", "w"))
    json.dump(_result(True, ane_ms=0.3, ane_place={"CPU": 10}, gpu_ok=False), open(tmp_path / "zz.json", "w"))
    json.dump({"rows": []}, open(tmp_path / "not_an_event.json", "w"))
    res = make_report.load_results(str(tmp_path))
    assert list(res) == ["zz"] and res["zz"]["kind"] == "final"
    ev = make_report.build_rows(res)
    rows = {r["athlete"]: r for r in ev[0]["batches"][0]["rows"]}
    assert rows["coreml_ane"]["fell_back"] and rows["coreml_ane"]["medal"] is None    # ran on CPU: no medal
    assert rows["coreml_gpu"]["medal"] is None                                        # DQ: no medal
    assert rows["coreml_cpu"]["medal"] == "gold" and rows["mlx_gpu"]["medal"] == "silver"
    assert rows["numpy_cpu"]["medal"] == "bronze" and rows["mlx_gpu"]["x_numpy"] == 2.0
    medals = make_report.medal_table(ev)
    assert medals[0]["athlete"] == "coreml_cpu"
    pts = make_report.crossover_points(ev)
    # the fallen-back coreml_ane duplicates coreml_cpu and is left out of the crossover
    assert pts[0]["best_cpu"] == "coreml_cpu" and pts[0]["best_acc"] == "mlx_gpu"


def test_report_outputs_with_no_and_some_results(tmp_path):
    for evs in ([], make_report.build_rows({"zz": dict(kind="provisional", path="x", data=_result(False))})):
        medals = make_report.medal_table(evs)
        make_report.write_summary(evs, medals, make_report.crossover_points(evs), str(tmp_path / "s.md"))
        make_report.leaderboard_png(evs, str(tmp_path / "l.png"))
        make_report.crossover_png(make_report.crossover_points(evs), None, str(tmp_path / "c.png"))
        make_report.demo_html(evs, medals, str(tmp_path / "d.html"))
        html = open(tmp_path / "d.html").read()
        assert "/*__DATA__*/" not in html and "http://" not in html and "https://" not in html
        if shutil.which("node"):
            r = subprocess.run(["node", os.path.join(HERE, "tests", "check_demo.js"), str(tmp_path / "d.html")],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("env,tf32", [("0", False), ("1", True)])
def test_mlx_tf32_switch_is_measured(env, tf32):
    """MLX_ENABLE_TF32 decides GPU fp32 matmul precision; the harness measures it rather than trusting it."""
    code = "import bench; print(bench.mlx_gpu_matmul_rel_err())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE,
                         env={**os.environ, "MLX_ENABLE_TF32": env}).stdout.strip().splitlines()[-1]
    assert (float(out) > run_event.TF32_THRESHOLD) == tf32
