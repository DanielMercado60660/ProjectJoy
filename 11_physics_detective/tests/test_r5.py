"""R5 CPU checks: numpy/scipy/pytest only. Never import MLX or old SBI tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics import DESIGNS, G, MU_HI, MU_LO, MU_RATIO, NT, positions, world1_positions
from r5_core import (AIM_OFFSETS, FAMILIES, FINAL_ACTIONS, METHODS, PROTOCOLS,
                     Belief, Guard, PlannerConfig, TerminalTask, TimeLimit,
                     baseline_probe, cdf_at, design_scores, evaluate_world, make_world,
                     quantiles, rng_for, success_bounds, true_observation, true_stop, tune_world)
from run_r5 import (add_tuning, atomic_json, choose_tuning, load_state, paired_delta,
                    read_json, summarize, wilson)


TINY = PlannerConfig(401, 8, 4, 25)
TUNING = {"lsq": {"protocol": "adaptive_12_65", "offset": 0.0},
          "tobit": {"protocol": "adaptive_12_65", "offset": 0.0}}


def test_true_physics_gain_limit_threshold_and_stop():
    for gain in (.88, 1.0, 1.12):
        mu = .37
        force = MU_RATIO * mu * G / gain
        assert np.all(positions(force * .999, .3, mu, MU_RATIO * mu, k=gain) == 0)
        action = np.array([force * 1.1, .4])
        trace = positions(*action, mu, MU_RATIO * mu, k=gain, times=np.linspace(0, 20, 201))
        assert np.all(np.diff(trace) >= -1e-12)
        assert trace[-1] == pytest.approx(true_stop(action, mu, gain))
    assert np.allclose(positions(8, .3, .3, .36, k=1), world1_positions(8, .3, .3))


def test_success_interval_inversion_matches_forward_physics():
    rng = np.random.default_rng(19)
    mus = rng.uniform(MU_LO, MU_HI, 100)
    for target in (.25, .55, .85):
        lo, hi = success_bounds(FINAL_ACTIONS, target)
        predicted = (mus[:, None] > lo) & (mus[:, None] < hi)
        actual = abs(true_stop(FINAL_ACTIONS[None, :, :], mus[:, None]) - target) < .025
        assert np.array_equal(predicted, actual)


def test_uniform_midcell_intervals_and_batched_cdf():
    post = Belief(401)
    q = np.array([.05, .25, .5, .75, .95])
    assert np.allclose(quantiles(post.grid, post.weights, q), MU_LO + q * (MU_HI - MU_LO))
    x = np.array([-.5, .05, .2, .425, .8, 2])
    expected = np.clip((x - MU_LO) / (MU_HI - MU_LO), 0, 1)
    assert np.allclose(cdf_at(x, post.grid, np.stack([post.weights] * 3)), expected[None])
    task = TerminalTask(.5)
    analytic = np.maximum(0, np.minimum(task.upper, MU_HI) - np.maximum(task.lower, MU_LO)) / (MU_HI - MU_LO)
    assert np.allclose(task.probabilities(post.grid, post.weights), analytic)


def test_censored_fit_and_posterior_find_known_mu():
    mu, post = .31, Belief(6001)
    for action in (np.array([12, .4]), np.array([8, .3]), np.array([5, .5])):
        y, cens = true_observation(action, mu, 1, np.zeros(NT))
        post.update(action, y, cens)
    assert abs(post.estimate("tobit") - mu) < .004
    assert abs(post.estimate("lsq") - mu) < .004
    assert abs(post.estimate("eig") - mu) < .004
    stats = post.stats()
    assert stats["ci90"][0] < mu < stats["ci90"][1]
    assert stats["posterior_sd"] < .01
    assert post.weights.sum() == pytest.approx(1)


def test_all_censored_lsq_flat_fit_is_centre_tobit_uses_censoring():
    post = Belief(401)
    post.update(np.array([2., .1]), np.full(NT, np.nan), np.ones(NT, dtype=bool))
    assert post.estimate("lsq") == pytest.approx((MU_LO + MU_HI) / 2)
    assert post.estimate("tobit") < .15


def test_both_planners_same_draws_shapes_and_bounded_utilities():
    post, task = Belief(401), TerminalTask(.5)
    a = design_scores(post, task, np.random.default_rng(21), TINY)
    b = design_scores(post, task, np.random.default_rng(21), TINY)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    designs, info, voi = a
    assert len(designs) == len(DESIGNS[::25])
    assert np.all(np.isfinite(info)) and np.all(np.isfinite(voi))
    assert info.min() >= -1e-12 and info.max() <= np.log(TINY.particles) + 1e-12
    assert voi.min() >= -1 and voi.max() <= 1
    assert info.max() > info[0]  # a 1 N push is a poor generic identification probe


def test_value_of_information_formula_against_brute_bayes_risk():
    # Reproduce one nontrivial candidate independently, including the particle
    # model's censoring likelihood, and check both expected utility definitions.
    from physics import SIGMA, VIEW, loglik_frames
    post, task = Belief(401), TerminalTask(.65)
    cfg = PlannerConfig(401, 8, 4, 40)
    designs, info, voi = design_scores(post, task, np.random.default_rng(100), cfg)
    rng = np.random.default_rng(100)
    particles = quantiles(post.grid, post.weights, (np.arange(cfg.particles) + .5) / cfg.particles)
    sources = (((np.arange(cfg.fantasies) + rng.random(cfg.fantasies)) / cfg.fantasies) * cfg.particles).astype(int)
    eps = rng.standard_normal((cfg.fantasies, NT))
    d = designs[1]
    x = world1_positions(*d, particles)
    values, gains = [], []
    for source, e in zip(sources, eps):
        y = x[source] + SIGMA * e
        ll = loglik_frames(y, y > VIEW, x)
        w = np.exp(ll - ll.max())
        w /= w.sum()
        values.append(task.probabilities(particles, w).max())
        gains.append(np.sum(w * np.log(np.maximum(w * cfg.particles, 1e-300))))
    assert info[1] == pytest.approx(np.mean(gains))
    assert voi[1] == pytest.approx(np.mean(values) - task.probabilities(particles, np.ones(8) / 8).max())


def test_protocols_are_in_shared_menu_and_tuning_has_all_candidates():
    for protocol in PROTOCOLS:
        for mu in (.05, .4, .8):
            for step in range(4):
                action = baseline_probe(protocol, mu, step, target=.55)
                assert np.any(np.all(np.isclose(DESIGNS, action), axis=1))
    rows = tune_world(51001, 0, 401)
    assert len(rows) == 2 * len(PROTOCOLS) * len(AIM_OFFSETS)
    assert all(r["trials"] == 3 and 0 <= r["successes"] <= 3 for r in rows)
    totals = add_tuning([], rows)
    assert set(choose_tuning(totals)) == {"lsq", "tobit"}


def without_timing(rows):
    return [{k: v for k, v in r.items() if k != "selection_seconds"} for r in rows]


def test_complete_world_pairing_attempts_and_no_probe_oracles():
    rows = evaluate_world(51002, 0, TINY, TUNING)
    assert len(rows) == 38
    for r in rows:
        assert r["attempts"] == r["budget"] + 1
        assert len(r["history"]) == r["budget"]
        assert r["success"] == (abs(r["stop"] - r["target"]) <= .025)
        assert np.any(np.all(FINAL_ACTIONS == r["action"], axis=1))
        if r["family"] == "matched" and r["method"] in ("mu_oracle", "world_oracle"):
            assert r["success"]
    assert len({r["mu_true"] for r in rows}) == len({r["target"] for r in rows}) == 1
    for method in METHODS:
        first = [r["history"][0]["action"] for r in rows if r["method"] == method and r["budget"] == 1]
        assert first[0] == first[1]  # family labels/gain are unavailable to the policy
    mu1, target1, gain1, noise1 = make_world(51002, 0)
    mu2, target2, gain2, noise2 = make_world(51002, 0)
    assert (mu1, target1, gain1) == (mu2, target2, gain2)
    assert np.array_equal(noise1, noise2)


def test_wall_guard_and_checkpoint_reject_config_drift(tmp_path):
    guard = Guard(.001)
    guard.deadline = 0
    with pytest.raises(TimeLimit):
        guard.check()
    cfg = {"seed": 7}
    state = load_state(tmp_path / "state.json", cfg, False)
    atomic_json(tmp_path / "state.json", state)
    with pytest.raises(ValueError, match="Refusing"):
        load_state(tmp_path / "state.json", cfg, False)
    with pytest.raises(ValueError, match="mismatch"):
        load_state(tmp_path / "state.json", {"seed": 8}, True)
    assert load_state(tmp_path / "state.json", cfg, True) == state


def test_summary_not_pseudoreplicated_and_partial_rejected():
    rows = evaluate_world(51002, 0, TINY, TUNING)
    state = dict(status="partial", rows=rows, config={"worlds": 1}, elapsed_seconds=1.)
    with pytest.raises(ValueError, match="partial"):
        summarize(state)
    state["status"] = "complete"
    report = summarize(state)
    assert len(report["groups"]) == 38
    assert all(g["n"] == 1 for g in report["groups"])
    assert all(x["paired_se"] is None for x in report["paired_success_differences"])
    for g in report["groups"]:
        assert g["mean_attempts"] == g["budget"] + 1
        assert 0 <= g["success_wilson95"][0] <= g["success_wilson95"][1] <= 1
    state["rows"] = rows[:-1]
    with pytest.raises(ValueError, match="matrix"):
        summarize(state)


def test_paired_statistics_and_wilson():
    assert wilson(5, 10) == pytest.approx([.2365930905, .7634069095])
    result = paired_delta([1, 1, 0, 0], [0, 1, 0, 1])
    assert result["difference"] == 0
    assert result["paired_se"] == pytest.approx(np.sqrt(2 / 3) / 2)
    assert rng_for(1, 2, 3).random() != rng_for(1, 2, 4).random()


def test_legacy_split_reuses_only_original_friction():
    for i in (0, 1, 5):
        legacy = make_world(51002, i, legacy_mu=True)
        fresh = make_world(51002, i)
        assert legacy[0] == np.random.default_rng([2026, i]).uniform(MU_LO, MU_HI)
        assert legacy[1:3] == fresh[1:3]
        assert np.array_equal(legacy[3], fresh[3])


def test_cli_guard_partial_tuning_resume(tmp_path):
    ckpt, output = tmp_path / "guard.json", tmp_path / "tuning.json"
    command = [sys.executable, str(ROOT / "run_r5.py"), "tune", "--worlds", "1", "--seed", "51001",
               "--grid-size", "101", "--checkpoint", str(ckpt), "--out", str(output)]
    result = subprocess.run(command + ["--max-minutes", "0.000000001"], capture_output=True, text=True)
    assert result.returncode == 75 and "WALL GUARD" in result.stdout
    assert read_json(ckpt)["next_world"] == 0 and not output.exists()
    subprocess.run(command + ["--max-minutes", "1", "--resume"], check=True, capture_output=True)
    assert read_json(ckpt)["status"] == "complete" and read_json(output)["status"] == "complete"


def test_saved_checkpoint_demo_is_offline_and_valid_svg(tmp_path):
    import re
    import xml.etree.ElementTree as ET
    from r5_report import render
    state = dict(status="complete", rows=evaluate_world(51002, 0, TINY, TUNING),
                 config={"worlds": 1}, elapsed_seconds=1.)
    checkpoint = tmp_path / "eval.json"
    atomic_json(checkpoint, state)
    output = tmp_path / "demo.html"
    render(checkpoint, output, tmp_path / "summary.json", 1)
    html = output.read_text()
    assert "TINY PIPELINE CHECK" in html and "Saved consequence" in html
    assert "https://" not in html and "http://" not in html and "fetch(" not in html
    figures = re.findall(r"<svg.*?</svg>", html, re.S)
    assert len(figures) == 8
    for figure in figures:
        ET.fromstring(figure)


def test_process_kill_resume_matches_uninterrupted(tmp_path):
    """Actually SIGKILL a child after an atomic save, then reload in a fresh process."""
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
    base = [sys.executable, str(ROOT / "run_r5.py")]
    tuning = tmp_path / "tuning.json"
    subprocess.run(base + ["tune", "--worlds", "1", "--seed", "51001", "--grid-size", "101",
                          "--checkpoint", str(tmp_path / "tune_ckpt.json"), "--out", str(tuning),
                          "--max-minutes", "1"], check=True, env=env, capture_output=True)
    common = ["evaluate", "--worlds", "2", "--seed", "51002", "--grid-size", "101", "--particles", "4",
              "--fantasies", "2", "--design-stride", "50", "--tuning", str(tuning), "--max-minutes", "1"]
    ckpt = tmp_path / "interrupted.json"
    command = base + common + ["--checkpoint", str(ckpt), "--out", str(tmp_path / "resumed_summary.json")]
    with (tmp_path / "killed.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if ckpt.exists() and read_json(ckpt)["next_world"] >= 1:
                    proc.kill()
                    proc.wait(timeout=5)
                    break
                if proc.poll() is not None:
                    pytest.fail("Child finished before checkpoint kill could be tested")
                time.sleep(.005)
            else:
                pytest.fail("Child did not checkpoint in 20 seconds")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
    assert read_json(ckpt)["status"] == "partial"
    subprocess.run(command + ["--resume"], check=True, env=env, capture_output=True)
    whole = tmp_path / "whole.json"
    subprocess.run(base + common + ["--checkpoint", str(whole), "--out", str(tmp_path / "whole_summary.json")],
                   check=True, env=env, capture_output=True)
    resumed, reference = read_json(ckpt), read_json(whole)
    assert resumed["status"] == "complete" and resumed["next_world"] == 2
    assert without_timing(resumed["rows"]) == without_timing(reference["rows"])
