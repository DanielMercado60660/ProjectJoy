"""Pure NumPy/stdlib tests: no MLX import, device probing, or neural execution."""
import itertools
import json
import sys
from argparse import Namespace

import numpy as np
import pytest
import sim
import r5_core as c
from r5_planning import choose, score_endpoints, random_search, cem_search
from experiment_r5 import prepare, fit, summarize, evaluate, task_predictions


def scenes(n=48, seed=71):
    return sim.random_init(n, np.random.default_rng(seed))


def test_no_mlx_imported():
    assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)


def test_parameterized_integrator_matches_true_simulator():
    x = scenes()
    actual, flags = sim.simulate(x, 30)
    assert flags[..., 1].any() and flags[..., 0].any()
    np.testing.assert_allclose(c.physics_rollout(x, 30), actual, atol=3e-13, rtol=0)


def test_walls_backend_and_numpy_baseline_match_existing():
    x = scenes()
    expected = sim.walls_only(x, 4)
    s = x
    for t in range(1, 5):
        s = c.walls_step(s)
        np.testing.assert_allclose(s, expected[:, t], atol=1e-14, rtol=0)
    np.testing.assert_allclose(c.physics_rollout(x, 4, walls=True), expected, atol=1e-14, rtol=0)


def test_walls_arbitrary_ensemble_leading_dimensions():
    x = np.stack([scenes(7), scenes(7, 72)])
    assert c.walls_step(x).shape == (2, 7, 12)
    np.testing.assert_allclose(c.walls_step(x)[1], c.walls_step(x[1]))


def test_collision_fit_can_identify_exact_law():
    x = scenes(100)
    truth, _ = sim.simulate(x, 8)
    losses = []
    params = c.parameter_candidates(n=20)
    for p in params:
        losses.append(np.square(c.physics_rollout(x, 8, p) - truth).mean())
    selected = params[int(np.argmin(losses))]
    assert selected == dict(restitution=1.0, separation=1.0)
    assert min(losses) < 1e-20


def test_pair_features_shapes_and_symmetric_gate():
    x = scenes(20)
    features, gate = c.pair_features(x, c.walls_step(x))
    assert features.shape == (20, 3, 3, 12)
    assert gate.shape == (20, 3, 3)
    np.testing.assert_array_equal(gate, gate.swapaxes(1, 2))
    assert not np.diagonal(gate, axis1=1, axis2=2).any()


def test_correction_conserves_sum_and_is_permutation_equivariant():
    x = scenes(20)
    weight = np.random.default_rng(81).normal(size=(12, 4))

    def reference(s):
        base = c.walls_step(s)
        features, gate = c.pair_features(s, base)
        edges = np.tanh(features @ weight)
        residual = c.antisymmetric_correction(edges, gate)
        np.testing.assert_allclose(residual.sum(axis=1), 0, atol=1e-14)
        return base + residual.reshape(-1, 12)

    prediction = reference(x).reshape(-1, 3, 4)
    for perm in itertools.permutations(range(3)):
        perm = list(perm)
        reordered = x.reshape(-1, 3, 4)[:, perm].reshape(-1, 12)
        np.testing.assert_allclose(reference(reordered).reshape(-1, 3, 4), prediction[:, perm], atol=1e-14)


def test_far_apart_balls_receive_no_learned_correction():
    x = np.array([[.15, .2, 0, 0, .5, .6, 0, 0, .85, .2, 0, 0]])
    _, gate = c.pair_features(x, c.walls_step(x))
    assert not gate.any()
    assert not c.antisymmetric_correction(np.ones((1, 3, 3, 4)), gate).any()


def test_head_on_contact_gate_is_open():
    x = np.array([[.4, .5, 1, 0, .54, .5, -1, 0, .8, .8, 0, 0]])
    _, gate = c.pair_features(x, c.walls_step(x))
    assert gate[0, 0, 1] and not gate[0, 0, 2]


def test_sample_windows_respect_bootstrap_scene_pool():
    x, flags = sim.simulate(scenes(20), 10)
    a = c.sample_windows(x, flags, 40, 3, np.random.default_rng(2), np.array([3, 3]))
    assert a.shape == (40, 4, 12)
    assert all(any(np.array_equal(row, x[3, t:t + 4]) for t in range(8)) for row in a)


def test_sample_windows_reject_too_long_horizon():
    with pytest.raises(ValueError):
        c.sample_windows(np.zeros((2, 2, 12)), np.zeros((2, 1, 2), bool), 3, 2, np.random.default_rng())


def test_forecast_splits_include_wall_and_ball_counts():
    truth = np.zeros((3, 3, 12))
    pred = truth.copy()
    pred[0, 1:, 0::4] = 1
    flags = np.zeros((3, 2, 2), bool)
    flags[0, 0, 0] = True
    flags[1, 1, 1] = True
    m = c.forecast_metrics(pred, truth, flags)
    assert m["1"]["all"]["mean"] == pytest.approx(1 / 3)
    assert m["1"]["collision"]["n"] == 1
    assert m["2"]["collision"]["n"] == 2
    assert m["2"]["ball_collision"]["n"] == 1
    assert m["2"]["no_collision"]["median"] == 0


def test_nonfinite_forecasts_are_not_silently_dropped():
    assert c.stats([0, np.inf])["mean"] is None
    assert c.stats([0, np.inf])["nonfinite"] == 1
    assert c.stats([]) == dict(n=0, mean=None, median=None, nonfinite=0)


def test_fooled_definition_and_legacy_severe_rate_differ():
    rows = [dict(dream=.01, real=.08), dict(dream=.02, real=.14),
            dict(dream=.01, real=.03), dict(dream=.2, real=.01)]
    m = c.planning_metrics(rows)
    assert m["fooled_rate"] == .5
    assert m["severe_fooled_rate"] == .25
    assert m["fooled_given_dream_success"] == pytest.approx(2 / 3)
    assert m["real_success"] == .5 and m["coverage"] == 1


def test_undefined_conditional_fooled_is_null():
    assert c.planning_metrics([dict(dream=None, real=.2)])["fooled_given_dream_success"] is None


def test_exact_paired_test_known_case_and_holm():
    test = c.paired_test([True] * 6, [False] * 6)
    assert test["p_exact"] == .03125
    assert test["difference_ci95"] == [1, 1]
    tests = c.holm_adjust({"one": test, "two": c.paired_test([True], [True])})
    assert tests["one"]["p_holm"] == .0625
    assert tests["two"]["p_holm"] == 1


def test_tied_ranks_and_uncertainty_metrics():
    np.testing.assert_equal(c.average_ranks([3, 1, 1, 2]), [3, .5, .5, 2])
    m = c.uncertainty_metrics([0, 0, 1, 2], [.01, .01, .1, .2])
    assert m["spearman"] == pytest.approx(1)
    assert m["auc_error_ge_radius"] == 1
    assert c.uncertainty_metrics([0, 0], [.01, .1])["spearman"] is None
    assert c.uncertainty_metrics([0, 0], [.01, .1])["auc_error_ge_radius"] == .5


def test_random_candidates_nested_deterministic_and_task_specific():
    np.testing.assert_array_equal(c.candidates(5, 64), c.candidates(5, 4096)[:64])
    assert not np.array_equal(c.candidates(4, 64), c.candidates(5, 64))
    assert np.max(np.abs(c.candidates(1, 4096))) <= 1.5


def test_risk_penalty_changes_choice_and_all_invalid_still_commits():
    endpoints = np.zeros((2, 2, 12))
    endpoints[:, 0, 0] = [-.5, .5]  # mean is a perfect hit; high disagreement
    endpoints[:, 1, 0] = [.1, .1]
    v = np.array([[0, 0], [1, 1]])
    assert choose(v, endpoints, np.zeros(2))["velocity"] == [0, 0]
    assert choose(v, endpoints, np.zeros(2), 1)["velocity"] == [1, 1]
    endpoints[:] = np.nan
    with np.errstate(invalid="ignore"):
        row = choose(v, endpoints, np.zeros(2))
    assert row["all_invalid"] and row["invalid_candidates"] == 2
    assert row["velocity"] == [0, 0] and row["dream"] is None
    # Invalid other-ball coordinates must invalidate an apparent ball-0 hit too.
    endpoints[:] = 0
    endpoints[:, :, 4] = np.nan
    row = choose(v, endpoints, np.zeros(2))
    assert row["all_invalid"] and row["dream"] is None


def test_cem_budget_common_first_population_and_incumbent():
    calls = []

    def predict(v):
        calls.append(v.copy())
        result = np.zeros((1, len(v), 12))
        result[0, :, :2] = v
        return result

    result = cem_search(predict, np.zeros(2), 2, [1, 2, 4], 16)
    assert len(calls) == 4 and sum(len(v) for v in calls) == 64
    np.testing.assert_equal(calls[0], c.candidates(2, 16))
    assert result["4"]["score"] <= result["2"]["score"] <= result["1"]["score"]
    rand = random_search(predict, np.zeros(2), 2, [16])
    assert rand["16"]["velocity"] == result["1"]["velocity"]


def test_json_is_atomic_and_forbids_nonfinite(tmp_path):
    path = tmp_path / "x.json"
    c.write_json(path, dict(a=1))
    with pytest.raises(ValueError):
        c.write_json(path, dict(a=np.nan))
    assert c.read_json(path) == dict(a=1)


def test_wall_guard_bounds_and_expiry():
    with pytest.raises(ValueError):
        c.Budget(46)
    b = c.Budget(1)
    b.start -= 61
    with pytest.raises(c.BudgetExpired):
        b.check()


def test_numpy_prepare_fit_end_to_end(tmp_path):
    data = tmp_path / "data"
    prepare(Namespace(out=str(data), scenes=64, val_scenes=24, frames=12), c.Budget(1))
    dest = tmp_path / "fit.json"
    fit(Namespace(data=str(data), out=str(dest), horizon=3, samples=64, candidates=20, refine=1), c.Budget(1))
    result = c.read_json(dest)
    assert result["complete"]
    assert result["selected"]["parameters"] == dict(restitution=1., separation=1.)
    assert result["selected"]["val_loss"] < 1e-8
    assert not (data / "test.npz").exists()
    assert c.SEEDS["test"] not in (0, 1, c.SEEDS["train"], c.SEEDS["val"], c.SEEDS["tune"])


def test_summary_requires_complete_pairing(tmp_path):
    (tmp_path / "tasks").mkdir()
    c.write_json(tmp_path / "tasks/00000.json", dict(task_id=0, rows={"a/random/64": {}}))
    c.write_json(tmp_path / "tasks/00001.json", dict(task_id=1, rows={"b/random/64": {}}))
    with pytest.raises(ValueError, match="paired"):
        summarize(tmp_path)


class FakePredictor:
    """Classical stand-in to test evaluation plumbing without loading a network."""
    model = Namespace(members=3)

    def rollout(self, init, horizon, final=False, budget=None):
        p = c.physics_rollout(init, horizon, final=final, budget=budget)
        return np.stack([p, p, p])


def tiny_eval_args(tmp_path):
    return Namespace(selection=str(tmp_path / "selection.json"), out=str(tmp_path / "eval"),
                     tasks=2, forecast_scenes=4, horizon=3, shots=[4, 8], cem_iters=[1, 2],
                     population=4, matched_cost=True, seed=51901, action_seed=51902,
                     label="numpy_plumbing_only", chunk=8, legacy_data="")


def test_task_matrix_counts_and_cost_matching(tmp_path):
    a = tiny_eval_args(tmp_path)
    rows = task_predictions(scenes(1)[0], np.array([.4, .5]), 0, FakePredictor(),
                            dict(restitution=1, separation=1), a, .5, c.Budget(1))
    assert len(rows) == 20
    for row in rows.values():
        assert row["task_id"] == 0 and row["real"] >= 0
    for count in a.shots:
        mean = rows[f"hybrid_mean/random/{count}"]
        fitted = rows[f"fitted/random_member_matched/{count}"]
        assert fitted["member_frames"] == mean["member_frames"]
        assert fitted["candidates"] == 3 * count
        assert mean["dream"] == pytest.approx(mean["real"], abs=1e-14)


def test_numpy_eval_resume_and_legacy_plumbing(tmp_path, monkeypatch):
    import experiment_r5
    source = tmp_path / "checkpoint"
    source.mkdir()
    weights = source / "model.safetensors"
    weights.write_text("Not weights: this is a pure NumPy evaluator plumbing test.\n")
    a = tiny_eval_args(tmp_path)
    c.write_json(a.selection, dict(selected=dict(checkpoint=str(source)),
                                   weights_sha256=c.digest(weights), parameters=dict(restitution=1, separation=1), beta=.5))
    monkeypatch.setattr(experiment_r5, "model_loader", lambda *_: (FakePredictor(), {}, source))
    evaluate(a, c.Budget(1))
    result = c.read_json(tmp_path / "eval/summary.json")
    assert result["status"]["complete"]
    assert len(result["metrics"]) == 20
    task = tmp_path / "eval/tasks/00000.json"
    before = task.read_bytes(), task.stat().st_mtime_ns
    evaluate(a, c.Budget(1))
    assert (task.read_bytes(), task.stat().st_mtime_ns) == before
    legacy = tmp_path / "legacy.npz"
    truth, flags = sim.simulate(scenes(4), 3)
    np.savez(legacy, states=truth, flags=flags)
    a.legacy_data, a.out = str(legacy), str(tmp_path / "old")
    evaluate(a, c.Budget(1))
    np.testing.assert_equal(c.read_json(tmp_path / "old/tasks/00000.json")["scene"], truth[0, 0])
    a.horizon += 1
    with pytest.raises(ValueError, match="specification"):
        evaluate(a, c.Budget(1))


def test_checkpoint_hash_mismatch_rejected_before_mlx_import(tmp_path):
    a = tiny_eval_args(tmp_path)
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "model.safetensors").write_text("placeholder")
    c.write_json(a.selection, dict(selected=dict(checkpoint=str(source)), weights_sha256="wrong"))
    with pytest.raises(ValueError, match="changed"):
        evaluate(a, c.Budget(1))
