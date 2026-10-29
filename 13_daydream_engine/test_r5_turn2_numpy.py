"""Turn-2 tests: NumPy only. Neural/VJP/resume execution belongs to GPU smoke."""
from argparse import Namespace
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import sim
import r5_core as c
from r5_soft_contact import BOUNDS, validate_parameters, soft_rollout, proposals, fit_soft
from experiment_r5 import prepare, evaluate, task_predictions, tune
from test_r5_numpy import FakePredictor, tiny_eval_args

PARAMS = dict(stiffness=1800., damping_ratio=.15, contact_radius=.06)


def test_soft_no_contact_matches_known_walls():
    x = np.array([[.15, .8, 0, 0, .5, .8, 0, 0, .85, .8, 0, 0]])
    np.testing.assert_allclose(soft_rollout(x, 10, PARAMS), sim.walls_only(x, 10), atol=1e-14)


def test_soft_contact_conserves_pair_momentum_and_is_not_exact_law():
    x = np.array([[.4, .5, 1, 0, .51, .5, -1, 0, .8, .8, 0, 0]])
    soft = soft_rollout(x, 1, PARAMS)[:, -1]
    hard = c.physics_rollout(x, 1)[:, -1]
    assert np.max(np.abs(soft - hard)) > .01
    assert soft[0, 2] < 1 and soft[0, 6] > -1
    dv = soft.reshape(1, 3, 4)[..., 2:].sum(axis=1) - x.reshape(1, 3, 4)[..., 2:].sum(axis=1)
    np.testing.assert_allclose(dv, [[0, -3 * sim.G * sim.FRAME_DT]], atol=1e-13)


def test_soft_simultaneous_contacts_are_permutation_equivariant():
    x = np.array([[.4, .5, 1, 0, .5, .5, 0, 0, .6, .5, -1, 0]])
    truth = soft_rollout(x, 3, PARAMS).reshape(1, 4, 3, 4)
    for perm in itertools.permutations(range(3)):
        p = list(perm)
        swapped = x.reshape(1, 3, 4)[:, p].reshape(1, 12)
        np.testing.assert_allclose(soft_rollout(swapped, 3, PARAMS).reshape(1, 4, 3, 4), truth[:, :, p], atol=1e-13)


def test_soft_parameter_search_excludes_infinite_stiffness_and_exact_projection():
    ps = proposals(192)
    assert len(ps) == 192 and len({tuple(p.values()) for p in ps}) == 192
    assert ps[:48] == proposals(48)
    for p in ps:
        validate_parameters(p)
    for bad in (dict(PARAMS, stiffness=np.inf), dict(PARAMS, stiffness=100000),
                dict(PARAMS, contact_radius=.1), dict(PARAMS, separation=1)):
        with pytest.raises(ValueError):
            validate_parameters(bad)


def test_geometry_finite_at_self_edges_zero_velocity_and_coincident_balls():
    x = np.tile([.5, .5, 0, 0], (2, 3)).astype(np.float32)
    features, gate = c.geometry_pair_features(x, c.walls_step(x))
    assert features.shape == (2, 3, 3, 24)
    assert np.isfinite(features).all()
    assert not np.diagonal(gate, axis1=-2, axis2=-1).any()


def test_geometry_model_reference_is_permutation_equivariant():
    x = sim.random_init(24, np.random.default_rng(78))
    w = np.random.default_rng(73).normal(size=(24, 4))
    def predict(st):
        base = c.walls_step(st)
        f, g = c.geometry_pair_features(st, base)
        return base.reshape(-1, 3, 4) + c.antisymmetric_correction(np.tanh(f @ w), g)
    p = [1, 2, 0]
    expected = predict(x)[:, p]
    actual = predict(x.reshape(-1, 3, 4)[:, p].reshape(-1, 12))
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_event_windows_align_with_state_targets():
    states = np.zeros((3, 9, 12))
    flags = np.zeros((3, 8, 2), bool)
    for scene in range(3):
        states[scene, :, 0] = scene * 100 + np.arange(9)
        flags[scene, ::scene + 1, 1] = True
    win, event = c.sample_windows(states, flags, 50, 3, np.random.default_rng(3), return_flags=True)
    for w, f in zip(win, event):
        scene, start = divmod(int(w[0, 0]), 100)
        np.testing.assert_equal(f, flags[scene, start:start + 3, 1])


def test_lr_schedule_is_deterministic_across_resume_and_stays_bounded():
    full = [c.learning_rate_at(i, .001, 10, .05) for i in range(21)]
    resumed = [c.learning_rate_at(i, .001, 10, .05) for i in range(7, 21)]
    assert full[7:] == resumed
    assert full[0] == .001 and full[-1] == .00005
    assert np.all(np.diff(full) <= 0)
    assert c.learning_rate_at(1000000, .001) == .001


def test_fresh_seed_guard_rejects_all_previous_test_streams():
    c.require_fresh_turn2(c.TURN2_SEEDS["scene"], c.TURN2_SEEDS["actions"])
    for seed, actions, legacy in ((51901, 62002, ""), (62001, 51902, ""),
                                  (52001, 62002, ""), (62001, 62002, "data/test.npz"),
                                  (51001, 62002, ""), (53003, 62002, ""),
                                  (62001, 62001, "")):
        with pytest.raises(ValueError):
            c.require_fresh_turn2(seed, actions, legacy)


def test_soft_fit_train_val_only_end_to_end(tmp_path):
    data = tmp_path / "data"
    prepare(Namespace(out=str(data), scenes=64, val_scenes=24, frames=12), c.Budget(1))
    args = Namespace(data=str(data), out=str(tmp_path / "soft.json"), proposals=48, refine=2,
                     shortlist=4, samples=64, val_scenes=16, fit_horizon=3, horizon=8)
    fit_soft(args, c.Budget(1))
    result = c.read_json(args.out)
    assert result["complete"] and len(result["trials"]) == 50
    assert result["family"] == "finite_spring_damper_v1" and not result["contains_exact_hard_law"]
    assert not result["oracle"]
    assert result["selected"]["train_loss"] > 1e-9
    assert result["selected"]["val_mean_error"] == min(r["val_mean_error"] for r in result["trials"] if "val_mean_error" in r)
    assert not (data / "test.npz").exists()


def test_turn2_task_matrix_contains_oracle_soft_and_frozen_reference(tmp_path):
    a = tiny_eval_args(tmp_path)
    rows = task_predictions(sim.random_init(1, np.random.default_rng(37))[0], np.array([.4, .5]), 0,
                            FakePredictor(), dict(restitution=1, separation=1), a, .5, c.Budget(1),
                            PARAMS, FakePredictor(), 1.)
    assert len(rows) == 34
    assert not any(k.startswith("fitted/") for k in rows)
    assert set(k.split('/')[0] for k in rows) == {"walls", "oracle_exact", "soft_contact", "hybrid_mean", "hybrid_risk", "reference_mean", "reference_risk"}
    assert rows["soft_contact/random_member_matched/8"]["member_frames"] == rows["hybrid_mean/random/8"]["member_frames"]


def test_turn2_eval_plumbing_and_rejects_old_seed_before_loading_mlx(tmp_path, monkeypatch):
    import experiment_r5
    a = tiny_eval_args(tmp_path)
    a.turn2 = True
    source = tmp_path / "checkpoint"
    source.mkdir()
    weights = source / "model.safetensors"
    weights.write_text("NumPy stand-in, not neural weights")
    c.write_json(a.selection, dict(selected=dict(checkpoint=str(source)), weights_sha256=c.digest(weights),
                                   parameters=dict(restitution=1, separation=1), beta=.5,
                                   soft_contact=dict(parameters=PARAMS),
                                   reference=dict(checkpoint=str(source), weights_sha256=c.digest(weights), beta=1)))
    with pytest.raises(ValueError, match="new scene"):
        evaluate(a, c.Budget(1))
    # Plumbing uses smoke-only streams, never the reserved headline seeds.
    a.seed, a.action_seed = 61041, 61042
    monkeypatch.setattr(experiment_r5, "model_loader", lambda *_: (FakePredictor(), {}, source))
    evaluate(a, c.Budget(1))
    summary = c.read_json(Path(a.out) / "summary.json")
    assert summary["status"]["complete"] and len(summary["metrics"]) == 34
    assert c.read_json(Path(a.out) / "manifest.json")["turn"] == 2
    assert "soft_contact" in c.read_json(Path(a.out) / "forecast.json")["metrics"]
    assert "reference_mean/random/8" in summary["selected_uncertainty"]
    assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)


def test_validation_selection_freezes_soft_fit_and_reference_without_test_data(tmp_path, monkeypatch):
    import experiment_r5
    data = tmp_path / "data"
    prepare(Namespace(out=str(data), scenes=32, val_scenes=12, frames=5), c.Budget(1))
    cfg = dict(train_sha256=c.digest(data / "train.npz"), val_sha256=c.digest(data / "val.npz"))
    exact = tmp_path / "exact.json"
    c.write_json(exact, dict(complete=True, **cfg, selected=dict(parameters=dict(restitution=1., separation=1.))))
    soft = tmp_path / "soft.json"
    c.write_json(soft, dict(complete=True, **cfg, family="finite_spring_damper_v1", selected=dict(parameters=PARAMS)))
    source = tmp_path / "weights"
    source.mkdir()
    (source / "model.safetensors").write_text("NumPy predictor stub only")
    old = tmp_path / "old_selection.json"
    c.write_json(old, dict(selected=dict(checkpoint=str(source)), weights_sha256=c.digest(source / "model.safetensors"), beta=1))
    monkeypatch.setattr(experiment_r5, "model_loader", lambda *_: (FakePredictor(), dict(config=cfg, step=3), source))
    out = tmp_path / "selection.json"
    a = Namespace(out=str(out), data=str(data), tasks=2, forecast_val_scenes=4, horizon=3,
                  fit=str(exact), soft_fit=str(soft), checkpoints=[str(source)], chunk=8,
                  shots=4, tune_seed=53003, betas=[0., 1.], reference_selection=str(old))
    tune(a, c.Budget(1))
    result = c.read_json(out)
    assert result["complete"] and result["forecast_val_scenes"] == 4
    assert result["soft_contact"]["parameters"] == PARAMS
    assert result["soft_contact"]["fit_sha256"] == c.digest(soft)
    assert result["reference"]["beta"] == 1 and result["beta"] == 0
    assert not (data / "test.npz").exists()
