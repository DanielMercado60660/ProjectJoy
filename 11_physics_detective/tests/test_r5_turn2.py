"""CPU-only scientific and persistence checks for the single gain-model fix."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics import MU_RATIO, NT, loglik_frames, positions
from r5_core import PlannerConfig, TerminalTask, evaluate_world, true_observation, true_stop
from r5_gain import GainBelief, GainConfig, GainTask, followup_world
from r5_report_turn2 import render
from run_r5 import atomic_json, read_json
from run_r5_turn2 import (select_candidates, summarize_followup, validation_candidates)


TINY = PlannerConfig(401, 8, 4, 25)
GAIN = GainConfig(301, 31)
BASELINE = {"lsq": {"protocol": "adaptive_10_65", "offset": 0.0},
            "tobit": {"protocol": "adaptive_10_90", "offset": 0.0}}
SELECTED = {"gain_bayes": {"owner": "eig", "half_width": .15},
            "gain_mle": {"owner": "lsq", "half_width": .25}}


def test_joint_likelihood_is_exact_existing_simulator_likelihood():
    post = GainBelief(GAIN, .15)
    action = np.array([8., .3])
    y, cens = true_observation(action, .37, 1.08, np.zeros(NT))
    post.update(action, y, cens)
    ll = loglik_frames(y, cens, positions(*action, post.mu[None, :], MU_RATIO * post.mu[None, :], k=post.gain[:, None]))
    assert post.logp.shape == (31, 301)
    assert np.allclose(post.logp, ll - ll.max())
    assert post.weights.sum() == pytest.approx(1)


def test_gain_task_integrates_conditional_success_with_correct_marginal_mass():
    post = GainBelief(GainConfig(401, 11), .25)
    post.logp = np.random.default_rng(79).normal(size=post.logp.shape)
    actions = np.array([[3., .3], [7., .4], [10., .3], [15., .2]])
    task = GainTask(.55, post, actions)
    weights, expected = post.weights, np.zeros(len(actions))
    for j, gain in enumerate(post.gain):
        effective_actions = actions.copy()
        effective_actions[:, 0] *= gain
        expected += TerminalTask(.55, effective_actions).probabilities(post.mu, weights[j] / weights[j].sum()) * weights[j].sum()
    assert np.allclose(task.probabilities(post), expected, atol=1e-13)
    assert np.all((task.probabilities(post) >= 0) & (task.probabilities(post) <= 1))


def test_joint_fit_recovers_two_known_parameters():
    post = GainBelief(GainConfig(1201, 81), .15)
    mu, gain = .37, 1.08
    for action in (np.array([5., .6]), np.array([10., .2]), np.array([12., .3]), np.array([8., .5])):
        post.update(action, *true_observation(action, mu, gain, np.zeros(NT)))
    stats = post.stats()
    assert abs(stats["posterior_mean"] - mu) < .02
    assert abs(stats["gain_posterior_mean"] - gain) < .03
    assert stats["ci90"][0] < mu < stats["ci90"][1]
    assert stats["gain_ci90"][0] < gain < stats["gain_ci90"][1]
    mm, mg = post.mle()
    assert abs(mm - mu) < .015 and abs(mg - gain) < .025
    for key in ("mu_conditional_underresolved_mass", "gain_conditional_underresolved_mass", "joint_max_cell_mass"):
        assert 0 <= stats[key] <= 1


def test_followup_preserves_identical_observations_cost_and_original_records():
    original = evaluate_world(73002, 0, TINY, BASELINE)
    corrected = followup_world(original, SELECTED, GAIN, TINY.grid_size)
    assert corrected[:len(original)] == original and len(corrected) == 62
    lookup = {(r["family"], r["method"], r["budget"]): r for r in corrected}
    for row in corrected[len(original):]:
        owner = lookup[row["family"], row["probe_owner"], row["budget"]]
        assert row["history"] == owner["history"]
        assert row["attempts"] == row["budget"] + 1
        assert row["stop"] == pytest.approx(float(true_stop(np.array(row["action"]), row["mu_true"], row["gain_true"])))
        if row["method"] == "same_probes_unit_bayes":
            for key in ("action", "posterior_mean", "posterior_sd", "predicted_success", "ci90", "success"):
                assert row[key] == owner[key]


def test_equal_validation_candidate_budget_and_explicit_tie_rule():
    original = evaluate_world(73002, 0, TINY, BASELINE)
    candidates = validation_candidates(original, GAIN, None)
    assert len(candidates) == 8
    assert all(c["trials"] == 6 for c in candidates)
    for method in ("gain_bayes", "gain_mle"):
        choices = [c for c in candidates if c["method"] == method]
        assert len(choices) == 4
        assert {(c["owner"], c["half_width"]) for c in choices} == {("eig", .15), ("eig", .25), ("lsq", .15), ("lsq", .25)}
    selected = select_candidates(candidates)
    assert set(selected) == {"gain_bayes", "gain_mle"}


def test_complete_summary_and_offline_failure_report(tmp_path):
    rows = followup_world(evaluate_world(73002, 0, TINY, BASELINE), SELECTED, GAIN, TINY.grid_size)
    state = dict(status="complete", rows=rows, config={"worlds": 1}, elapsed_seconds=1.)
    summary = summarize_followup(state)
    assert len(summary["groups"]) == 62
    assert all(g["n"] == 1 for g in summary["groups"])
    assert len(summary["gain_fix_paired_differences"]) == 48
    checkpoint = tmp_path / "checkpoint.json"
    atomic_json(checkpoint, state)
    render(checkpoint, str(tmp_path / "demo.html"), str(tmp_path / "summary.json"), 1)
    html = (tmp_path / "demo.html").read_text()
    assert "TINY PIPELINE CHECK" in html and "https://" not in html and "fetch(" not in html
    svgs = re.findall(r"<svg.*?</svg>", html, re.S)
    assert len(svgs) in (7, 10)
    for svg in svgs:
        ET.fromstring(svg)
    state["status"] = "partial"
    with pytest.raises(ValueError, match="Partial"):
        summarize_followup(state)


@pytest.fixture
def original_tuning(tmp_path):
    output = tmp_path / "original_tuning.json"
    subprocess.run([sys.executable, str(ROOT / "run_r5.py"), "tune", "--worlds", "1", "--seed", "73001",
                    "--grid-size", "101", "--checkpoint", str(tmp_path / "original_checkpoint.json"),
                    "--out", str(output), "--max-minutes", "1"], check=True, capture_output=True)
    return output


def tiny_command(original_tuning):
    return [sys.executable, str(ROOT / "run_r5_turn2.py"), "--baseline-tuning", str(original_tuning),
            "--mu-size", "101", "--gain-size", "11", "--base-grid-size", "101",
            "--base-particles", "4", "--base-fantasies", "2", "--base-stride", "50"]


def test_turn2_wall_guard_resume_and_old_seed_rejection(tmp_path, original_tuning):
    base = tiny_command(original_tuning)
    command = base + ["tune", "--worlds", "1", "--seed", "73003", "--checkpoint", str(tmp_path / "guard.json"),
                      "--out", str(tmp_path / "tuned.json")]
    result = subprocess.run(command + ["--max-minutes", "0.000000001"], capture_output=True, text=True)
    assert result.returncode == 75 and "WALL GUARD" in result.stdout
    assert read_json(tmp_path / "guard.json")["next_world"] == 0
    subprocess.run(command + ["--resume", "--max-minutes", "1"], capture_output=True, check=True)
    assert read_json(tmp_path / "tuned.json")["status"] == "complete"
    evaluation = base + ["evaluate", "--worlds", "1", "--seed", "73004", "--mu-size", "201",
                         "--gain-size", "21", "--tuning", str(tmp_path / "tuned.json"), "--max-minutes", "1",
                         "--checkpoint", str(tmp_path / "refined.json"), "--out", str(tmp_path / "refined_summary.json")]
    rejected = subprocess.run(evaluation, capture_output=True, text=True)
    assert rejected.returncode != 0 and "--resolution-check" in rejected.stderr
    subprocess.run(evaluation + ["--resolution-check"], capture_output=True, check=True)
    assert read_json(tmp_path / "refined.json")["config"]["resolution_check"] is True
    for seed in (6202611, 6202612, 6202613):
        result = subprocess.run(base + ["tune", "--worlds", "1", "--seed", str(seed), "--max-minutes", "1",
                                       "--checkpoint", str(tmp_path / "forbidden.json"), "--out", str(tmp_path / "forbidden_out.json")], capture_output=True, text=True)
        assert result.returncode != 0 and "FRESH seeds" in result.stderr
        assert not (tmp_path / "forbidden.json").exists()


def test_turn2_actual_kill_resume_matches_uninterrupted(tmp_path, original_tuning):
    base = tiny_command(original_tuning)
    tuning = tmp_path / "tuning.json"
    subprocess.run(base + ["tune", "--worlds", "1", "--seed", "73003", "--checkpoint", str(tmp_path / "tune_state.json"),
                           "--out", str(tuning), "--max-minutes", "1"], check=True, capture_output=True)
    common = base + ["evaluate", "--worlds", "2", "--seed", "73004", "--tuning", str(tuning), "--max-minutes", "1"]
    checkpoint = tmp_path / "killed.json"
    command = common + ["--checkpoint", str(checkpoint), "--out", str(tmp_path / "resumed_summary.json")]
    with (tmp_path / "killed.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if checkpoint.exists() and read_json(checkpoint)["next_world"] >= 1:
                    child.kill()
                    child.wait(timeout=5)
                    break
                if child.poll() is not None:
                    pytest.fail("Child completed before the actual kill check")
                time.sleep(.005)
            else:
                pytest.fail("No completed world checkpoint within 20 seconds")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    assert read_json(checkpoint)["status"] == "partial"
    subprocess.run(command + ["--resume"], check=True, capture_output=True)
    reference = tmp_path / "reference.json"
    subprocess.run(common + ["--checkpoint", str(reference), "--out", str(tmp_path / "reference_summary.json")], check=True, capture_output=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("selection_seconds", "correction_seconds")} for r in rows]
    assert strip(read_json(checkpoint)["rows"]) == strip(read_json(reference)["rows"])
    assert read_json(checkpoint)["status"] == "complete"
