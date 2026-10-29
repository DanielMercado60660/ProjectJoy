"""Run this file explicitly: existing test_pipeline.py contains MLX training tests."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from env import VecRooms, N_TYPES, N_COLORS, CELL_FEAT, VIEW
from r5_core import (Frontier, SeenMap, ReplayGraph, signed_progress, transfer_metrics,
                     TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS, OLD_SEEDS, TASKS)
from evaluate_r5 import evaluate, task_success, new_env


def test_no_mlx_imported():
    # A separate interpreter also keeps the original full pytest command valid:
    # test_pipeline.py legitimately imports MLX before this file runs on Claude's Mac.
    code = ("import sys; import r5_core, classical_r5, evaluate_r5, render_r5, report_r5; "
            "import planner_r5_t2, progress_r5_t2, run_r5_t2, diagnose_r5_t2; "
            "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1])


def test_split_disjoint():
    groups = [set(x) for x in (TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS, OLD_SEEDS)]
    for i, g in enumerate(groups):
        for h in groups[i + 1:]:
            assert g.isdisjoint(h)


def test_signed_progress_noise_zero_and_terminal_mask():
    old = np.ones(5); new = np.array([1, .5, 1.5, 0, 4])
    np.testing.assert_allclose(signed_progress(old, new, [1, 1, 1, 0, 1]), [0, .5, -.5, 0, -1])
    rng = np.random.default_rng(61)
    # Opposite independent noise errors have exactly cancelling signed reward.
    e = rng.random(100)
    assert signed_progress(np.r_[e, 1-e], np.r_[1-e, e], np.ones(200)).mean() == pytest.approx(0)


@pytest.mark.parametrize("tv", [False, True])
def test_frontier_odometry_and_plans_execute(tv):
    env = VecRooms(1, max_steps=513, noisy_tv=tv, seed=18)
    f = Frontier(seed=3); f.observe(env.obs[0]); obs = [env.obs[0].copy()]; actions = []
    for _ in range(512):
        action = f.act(); nxt, _, _ = env.step([action]); f.observe(nxt[0], action)
        assert f.map.pos == tuple(env.pos[0] - env.start)
        obs.append(nxt[0].copy()); actions.append(action)
    # Structural sanity for the meaningful baseline, not a claimed headline result.
    assert env.rooms[0].sum() >= 4
    assert env.inter[0].sum() >= 3
    graph = ReplayGraph(obs, actions); found = 0
    for task in TASKS:
        plan, expanded = graph.plan(task, 256, 4096)
        assert expanded <= 4096
        if plan is None:
            continue
        found += 1
        fresh = VecRooms(1, max_steps=257, noisy_tv=tv, seed=999)
        for action in plan:
            fresh.step([action])
        assert task_success(fresh, [task])[0], (task, plan)
    assert found >= 2


def test_noise_not_memory_identity_and_hidden_history_preserved():
    env = VecRooms(1, max_steps=10, noisy_tv=True)
    obs = env.obs[0].copy(); other = obs.copy()
    cells = other[:-5].reshape(VIEW, VIEW, CELL_FEAT)
    cells[..., N_TYPES + N_COLORS:] = .123  # arbitrary TV bits, same geometry
    a, b = SeenMap(), SeenMap(); a.observe(obs); b.observe(other)
    assert a.signature() == b.signature()
    b.cells[(99, 99)] = (8, 6)  # remembered distant switch state
    assert a.signature() != b.signature()


def test_empty_replay_failure_cost_and_coverage():
    rows = [dict(success=False, steps=0, planned=False), dict(success=True, steps=10, planned=True)]
    m = transfer_metrics(rows, 128)
    assert m == dict(success_rate=.5, capped_steps=69., success_steps=10., plan_coverage=.5, trials=2)
    assert transfer_metrics([rows[0]], 128)["success_steps"] is None


def test_noise_stream_invariant_to_batching():
    e = new_env([72500, 72501], 9, True); solo = new_env([72501], 9, True)
    for _ in range(5):
        e.step([0, 1]); solo.step([1])
        np.testing.assert_array_equal(e.obs[1], solo.obs[0])


def test_fixed_budget_eval_reproducible_and_not_reset():
    args = dict(method="random", seeds=[72500, 72501], tv=True, explore_steps=24, task_steps=12, record=True)
    a, rec = evaluate(**args); b, _ = evaluate(**args)
    a.pop("elapsed_seconds"); b.pop("elapsed_seconds")
    assert a == b
    assert rec["actions"].shape == (24,)
    assert rec["observations"].shape[0] == 25
    assert a["transfer"]["trials"] == 14
    assert all(1 <= e["rooms"] <= 9 and 0 <= e["tv_steps"] <= 24 for e in a["episodes"])
    assert a["transfer"]["capped_steps"] <= 12


def test_planner_expansion_and_action_caps():
    e = VecRooms(1, max_steps=9); observations = [e.obs[0].copy()]
    for _ in range(8):
        e.step([3]); observations.append(e.obs[0].copy())
    graph = ReplayGraph(observations, [3]*8)
    route, count = graph.plan(("reach", 8), budget=2, max_expansions=1)
    assert route is None and count == 1


@pytest.mark.parametrize("method", ["frontier", "random"])
def test_classical_resume_matches_uninterrupted(tmp_path, method):
    from classical_r5 import train_classical
    import pickle
    common = dict(method=method, seed=510, tv=True, n_envs=2, episode_steps=16, max_minutes=.1)
    full = tmp_path / "full.pkl"; part = tmp_path / "part.pkl"
    train_classical(**common, steps=96, checkpoint=full, log_dir=tmp_path / "full")
    train_classical(**common, steps=32, checkpoint=part, log_dir=tmp_path / "part")
    out = train_classical(**common, steps=96, checkpoint=part, log_dir=tmp_path / "part")
    assert out["status"] == "complete" and out["steps"] == 96
    a, b = (pickle.loads(p.read_bytes()) for p in (full, part))
    assert a["rng"] == b["rng"] and a["totals"] == b["totals"]
    for key in VecRooms._STATE:
        np.testing.assert_array_equal(a["env"][key], b["env"][key])
    assert a["frontier"].map.signature() == b["frontier"].map.signature()
    assert a["frontier"].act() == b["frontier"].act()


def test_receipt_demo_and_report_refuses_incomplete(tmp_path):
    from evaluate_r5 import save_eval
    from render_r5 import render
    from report_r5 import report
    from PIL import Image
    result, recording = evaluate("frontier", [72500], tv=True, explore_steps=16, task_steps=8, record=True)
    receipt = tmp_path / "trial.json"; save_eval(receipt, result, recording)
    gif = tmp_path / "demo.gif"; render(receipt, gif)
    with Image.open(gif) as img:
        assert img.n_frames > 1 and img.size == (340, 292)
    with pytest.raises(FileExistsError):
        save_eval(receipt, result)
    with pytest.raises(FileNotFoundError):
        report(tmp_path, [510], tmp_path / "summary.md")
    assert not (tmp_path / "summary.md").exists()


def test_wall_guard_prevents_new_work(tmp_path):
    import time
    from classical_r5 import train_classical
    out = train_classical("random", 510, False, 10000, tmp_path / "guard.pkl",
                          tmp_path, max_minutes=.001, n_envs=2, episode_steps=16)
    assert out["status"] == "partial_wall_guard" and out["steps"] == 0
    with pytest.raises(TimeoutError):
        evaluate("random", [72500], deadline=time.monotonic() - 1)
