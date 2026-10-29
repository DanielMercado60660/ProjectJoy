"""Classical risk control and paired/resumable evaluation; no MLX imports."""
from argparse import Namespace
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import sim
import r5_classical_risk as cr
from r5_core import Budget, BudgetExpired, candidates, digest, launch_states, physics_rollout, read_json, write_json
from r5_planning import choose
from r5_soft_contact import soft_rollout


class FakeHybrid:
    model = Namespace(members=5)

    def rollout(self, initial, horizon, final=False, budget=None):
        result = physics_rollout(initial, horizon, final=final, budget=budget)
        return np.stack([result] * 5)


@pytest.fixture
def frozen(tmp_path):
    checkpoint = tmp_path / "final_generation"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test-only dummy; never load with MLX")
    trials = [dict(parameters=dict(stiffness=800. + i * 100, damping_ratio=.1, contact_radius=.06),
                   val_mean_error=.02 + i * .001) for i in range(6)]
    fit = dict(complete=True, family="finite_spring_damper_v1", selected=trials[0], trials=trials,
               train_sha256="train", val_sha256="val", horizon=2)
    fitpath = tmp_path / "soft_fit.json"
    write_json(fitpath, fit)
    scenes = sim.random_init(2, np.random.default_rng(53003))
    targets = np.random.default_rng(53004).uniform(.15, .85, (2, 2))
    beta_rows = {str(b): [] for b in [0., .25, .5, 1., 2.]}
    for t, (scene, target) in enumerate(zip(scenes, targets)):
        v = candidates(t, 8, 53005)
        end = FakeHybrid().rollout(launch_states(scene, v), 2, final=True)
        for b, rows in beta_rows.items():
            r = choose(v, end, target, float(b))
            r.update(task_id=t, real=r["dream"])
            rows.append(r)
    selection = dict(complete=True, selected=dict(checkpoint=str(checkpoint), members=5,
        config=dict(train_sha256="train", val_sha256="val")), weights_sha256=digest(checkpoint / "model.safetensors"),
        beta=.5, beta_rows=beta_rows, validation_seed=53003,
        soft_contact=dict(parameters=trials[0]["parameters"], fit=str(fitpath), fit_sha256=digest(fitpath)))
    path = tmp_path / "selection.json"
    write_json(path, selection)
    return path


def tuning_args(frozen, tmp_path):
    return Namespace(selection=str(frozen), out=str(tmp_path / "classical_selection.json"), shots=8)


def eval_args(frozen, tmp_path):
    return Namespace(selection=str(frozen), classical_selection=str(tmp_path / "classical_selection.json"),
        out=str(tmp_path / "headline"), tasks=3, shots=[4, 8], seed=75041, action_seed=75042,
        target_seed=75141, chunk=8)


def test_top_five_only_validated_unique_stable_ranking(frozen):
    _, fit = cr.frozen_inputs(frozen)
    duplicate = copy.deepcopy(fit["trials"][0])
    fit["trials"] = [dict(parameters=duplicate["parameters"], train_loss=0),
                     dict(duplicate, val_mean_error=None), dict(duplicate, val_mean_error=float("nan")),
                     duplicate] + list(reversed(fit["trials"]))
    selected = cr.top_validated(fit)
    assert len(selected) == 5
    assert selected[0]["trial_index"] == 3
    assert [m["val_mean_error"] for m in selected] == sorted({r["val_mean_error"] for r in selected})
    assert len({tuple(m["parameters"].values()) for m in selected}) == 5
    fit["trials"] = [duplicate] * 10
    with pytest.raises(ValueError, match="distinct validated"):
        cr.top_validated(fit)


def test_soft_ensemble_is_exact_stack_of_individual_models(frozen):
    _, fit = cr.frozen_inputs(frozen)
    members = cr.top_validated(fit)
    contact = np.array([[.4, .5, 1, 0, .51, .5, -1, 0, .8, .8, 0, 0]])
    ensemble = cr.soft_ensemble(contact, 2, members, Budget(.2))
    assert ensemble.shape == (5, 1, 12) and np.isfinite(ensemble).all()
    assert np.ptp(ensemble[..., :2], axis=0).max() > 0
    for i, member in enumerate(members):
        np.testing.assert_array_equal(ensemble[i], soft_rollout(contact, 2, member["parameters"], final=True))


def test_beta_tiebreak_matches_hybrid_rule():
    metrics = {"0.0": dict(real_success=.5, real_distance=dict(mean=.2)),
               "0.5": dict(real_success=.5, real_distance=dict(mean=.1)),
               "1.0": dict(real_success=.5, real_distance=dict(mean=.1))}
    assert cr.select_beta(metrics) == .5
    metrics["0.0"]["real_success"] = .6
    assert cr.select_beta(metrics) == 0.


def test_tuning_replays_frozen_validation_and_preserves_inputs(frozen, tmp_path):
    before = digest(frozen)
    a = tuning_args(frozen, tmp_path)
    cr.tune(a, Budget(.2))
    selected = read_json(a.out)
    assert selected["complete"] and selected["frozen_validation_replay_matches"]
    assert selected["protocol"] == dict(tasks=2, shots=8, horizon=2, scene_seed=53003,
        target_seed=53004, action_seed=53005, betas=[0., .25, .5, 1., 2.])
    assert selected["beta"] == cr.select_beta(selected["beta_metrics"])
    assert digest(frozen) == before
    assert not any(k.startswith("mlx") for k in sys.modules)
    with pytest.raises(FileExistsError):
        cr.tune(a, Budget(.2))


@pytest.mark.parametrize("field", ["real", "velocity", "task_id"])
def test_validation_replay_rejects_changed_protocol(frozen, tmp_path, field):
    s = read_json(frozen)
    s["beta_rows"]["0.0"][0][field] = {"real": 100., "velocity": [100., 100.], "task_id": 99}[field]
    write_json(frozen, s)
    a = tuning_args(frozen, tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        cr.tune(a, Budget(.2))
    assert not Path(a.out).exists()


def test_fresh_guard_rejects_every_consumed_stream_in_every_role():
    cr.require_fresh(**dict(seed=73001, action_seed=73002, target_seed=73101))
    for old in (51901, 51902, 52001, 52101, 62001, 62002, 62101, 62201, 51001, 51002, 53003, 53004, 53005):
        for role in range(3):
            seeds = [75041, 75042, 75141]
            seeds[role] = old
            with pytest.raises(ValueError, match="fresh"):
                cr.require_fresh(*seeds)
    with pytest.raises(ValueError):
        cr.require_fresh(75041, 75041, 75141)


def test_task_choices_scores_and_compute_controls(frozen, tmp_path):
    a = tuning_args(frozen, tmp_path)
    cr.tune(a, Budget(.2))
    classical, frozen_info = read_json(a.out), read_json(frozen)
    scene = sim.random_init(1, np.random.default_rng(75333))[0]
    target = np.array([.5, .5])
    record = cr.task_choices(scene, target, 0, FakeHybrid(), frozen_info, classical, [4, 8], 75334, Budget(.2))
    assert len(record["rows"]) == 12
    v = candidates(0, 8, 75334)
    endpoints = cr.soft_ensemble(launch_states(scene, v), 2, classical["members"], Budget(.2))
    for n in [4, 8]:
        for method, beta in [("classical_mean", 0.), ("classical_risk", classical["beta"])]:
            expected = choose(v[:n], endpoints[:, :n], target, beta)
            row = record["rows"][f"{method}/random/{n}"]
            assert all(row[k] == value for k, value in expected.items())
        for method in ("hybrid_mean", "hybrid_risk", "classical_mean", "classical_risk", "soft_contact_5x"):
            assert record["rows"][f"{method}/random/{n}"]["member_frames"] == 5 * n * 2
        assert record["rows"][f"soft_contact_5x/random/{n}"]["candidates"] == n * 5
        assert record["rows"][f"soft_contact/random/{n}"]["member_frames"] == n * 2


def test_partial_resume_and_paired_summary(frozen, tmp_path, monkeypatch):
    cr.tune(tuning_args(frozen, tmp_path), Budget(.2))
    monkeypatch.setattr(cr, "model_loader", lambda *args: (FakeHybrid(), None, None))
    original = cr.task_choices
    def stop_after_one(scene, target, task_id, *args):
        if task_id == 1:
            raise BudgetExpired()
        return original(scene, target, task_id, *args)
    monkeypatch.setattr(cr, "task_choices", stop_after_one)
    a = eval_args(frozen, tmp_path)
    cr.evaluate(a, Budget(.2))
    out = Path(a.out)
    assert read_json(out / "status.json")["completed_tasks"] == 1
    first = digest(out / "tasks/00000.json")
    monkeypatch.setattr(cr, "task_choices", original)
    cr.evaluate(a, Budget(.2))
    assert digest(out / "tasks/00000.json") == first
    summary = read_json(out / "summary.json")
    assert summary["status"]["complete"] and summary["task_ids"] == [0, 1, 2]
    assert len(summary["metrics"]) == 12 and summary["holm_family_size"] == 30
    assert all(t["p_exact"] <= t["p_holm"] <= 1 for t in summary["paired_tests"].values())
    assert all(m["coverage"] == 1 and m["n"] == 3 for m in summary["metrics"].values())
    assert len(summary["selected_uncertainty"]) == 8
    # Fresh uninterrupted run produces exactly the same paired decisions and metrics.
    reference = Namespace(**vars(a)); reference.out = str(tmp_path / "uninterrupted")
    cr.evaluate(reference, Budget(.2))
    assert read_json(Path(reference.out) / "summary.json")["metrics"] == summary["metrics"]
    for t in range(3):
        assert read_json(out / f"tasks/{t:05d}.json")["rows"] == read_json(Path(reference.out) / f"tasks/{t:05d}.json")["rows"]
    def no_load(*args):
        raise AssertionError("complete resume should not reload MLX")
    monkeypatch.setattr(cr, "model_loader", no_load)
    cr.evaluate(a, Budget(.2))
    assert read_json(out / "summary.json")["paired_tests"] == summary["paired_tests"]
    a.action_seed += 10
    with pytest.raises(ValueError, match="resume inputs"):
        cr.evaluate(a, Budget(.2))


def test_frozen_input_integrity_guard(frozen):
    selection = read_json(frozen)
    Path(selection["selected"]["checkpoint"], "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        cr.frozen_inputs(frozen)


def test_existing_four_planner_arms_unchanged(frozen, tmp_path):
    from experiment_r5 import task_predictions
    cr.tune(tuning_args(frozen, tmp_path), Budget(.2))
    classical = read_json(tmp_path / "classical_selection.json")
    selection = read_json(frozen)
    scene = np.array([.4, .5, 1, 0, .51, .5, -1, 0, .8, .8, 0, 0])
    target = np.array([.6, .4])
    a = Namespace(horizon=2, shots=[4, 8], action_seed=75444, matched_cost=True, cem_iters=[1], population=4)
    old = task_predictions(scene, target, 0, FakeHybrid(), dict(restitution=1., separation=1.), a,
        selection["beta"], Budget(.2), soft_parameters=selection["soft_contact"]["parameters"])
    new = cr.task_choices(scene, target, 0, FakeHybrid(), selection, classical, a.shots, a.action_seed, Budget(.2))["rows"]
    for method in cr.METHODS[:4]:
        for n in a.shots:
            oldkey = f"soft_contact/random_member_matched/{n}" if method == "soft_contact_5x" else f"{method}/random/{n}"
            for key in ("velocity", "dream", "disagreement", "score", "real", "real_position", "member_frames", "candidates"):
                assert new[f"{method}/random/{n}"][key] == old[oldkey][key]
