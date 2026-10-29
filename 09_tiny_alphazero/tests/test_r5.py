"""Only NumPy/numba/SciPy; deliberately does not import r5_mlx or net."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c4
from mcts import BatchMCTS, uniform_evaluator
from r5_core import (Search, allocation, canonical_keys, probabilities, choose, features,
                     learner_features, mistakes, calibration, report_moves, group_masks)
from r5_data import candidate, arrays
from r5_baselines import fit_baselines, scores, macro_nll
from r5_run import decision, games, secondary


def test_no_mlx_import():
    assert not any(x == "mlx" or x.startswith("mlx.") for x in sys.modules)


def test_mirror_keys_and_candidate_legality():
    rng = np.random.default_rng(501)
    for g in range(3):
        row = candidate(rng, g)
        p, m = c4.from_moves(row["moves"])
        pp, mm = c4.from_moves([6-c for c in row["moves"]])
        keys = canonical_keys(np.array([p, pp]), np.array([m, mm]))
        assert keys[0] == keys[1]
        assert (p, m) == (row["pos"], row["mask"])
        assert c4.status(p, m) == 0


def test_exact_allocation_ties_and_fractional_position():
    x = allocation(np.array([0., 1., 1., -.1]), 2, 9, 23)
    np.testing.assert_array_equal(x, [3, 9, 9, 2])
    assert allocation(np.ones(4), 2, 9, 8).sum() == 8
    assert allocation(np.ones(4), 2, 9, 36).sum() == 36
    with pytest.raises(AssertionError):
        allocation(np.ones(4), 2, 9, 37)


def test_zero_entropy_norm_and_illegal_mask():
    legal = np.array([[True, False, False, False, False, False, False]])
    p = probabilities(np.zeros((1, 7)), legal)
    assert p[0, 0] == 1 and p.sum() == 1
    assert choose(np.zeros((1, 7)), p, legal)[0] == 0
    for y, p in (([0, 0], [0, 0]), ([1, 1], [1, 1]), ([0, 1], [.5, .5])):
        r = calibration(np.array(y), np.array(p))
        assert np.isfinite([r["nll"], r["brier"], r["ece"]]).all()


def test_mistakes_count_rescues_and_introduced_separately():
    v = np.array([[1, 0, -1, -999, -999, -999, -999],
                  [0, -1, -999, -999, -999, -999, -999],
                  [-1, -1, -999, -999, -999, -999, -999]])
    r = report_moves(v, np.array([0, 1, 0]), np.array([1, 0, 1]), 12, .2)
    assert r["rescued"] == r["introduced"] == 1 and r["net_avoided"] == 0
    assert r["nonlost_n"] == 2 and r["nonlost_error"] == .5
    with pytest.raises(AssertionError):
        mistakes(v, np.array([3, 0, 0]))
    v[0, 0] = 999
    with pytest.raises(AssertionError):
        mistakes(v, np.array([0, 0, 0]))


def boards():
    seqs = [[0, 0, 1, 1, 2, 4], [3, 2, 3, 2], [0, 1, 2, 3]]
    pm = np.array([c4.from_moves(x) for x in seqs])
    return pm[:, 0], pm[:, 1]


def test_continuation_equals_existing_mcts_including_terminal_visits():
    p, m = boards()
    old = BatchMCTS(3, 12).search(p, m, uniform_evaluator)
    new = Search(p, m, uniform_evaluator, capacity=12)
    new.advance(np.full(3, 4)); new.advance(np.full(3, 12))
    for x, y in zip(old, new.stats()):
        np.testing.assert_allclose(x, y, equal_nan=True)
    np.testing.assert_equal(new.stats()[0].sum(1), [12, 12, 12])


def test_reorder_restores_original_board_ids_and_partial_budgets():
    p, m = boards()
    s = Search(p, m, uniform_evaluator, capacity=14)
    s.advance(np.full(3, 2))
    score = np.array([.2, .1, .9]); target = allocation(score, 2, 14, 24)
    s.sort(score); s.advance(target[s.ids])
    np.testing.assert_array_equal(s.count[np.argsort(s.ids)], target)
    for i in range(3):
        ref = BatchMCTS(1, int(target[i])).search(p[i:i+1], m[i:i+1], uniform_evaluator)
        for x, y in zip(s.stats(), ref):
            np.testing.assert_allclose(x[i], y[0], equal_nan=True)
    assert np.all(np.array([c4.can_play(m[i], int(x)) for i, x in enumerate(s.moves())]))


def test_deadline_and_feature_shape_zero_safe():
    p, m = boards(); s = Search(p, m, uniform_evaluator, capacity=14)
    s.advance(np.full(3, 8), deadline=time.monotonic()-1)
    assert s.count.sum() == 0
    x, moves, policy = features(p, m, s, ab_depths=())
    assert x.shape == (3, 117) and np.isfinite(x).all()
    assert np.all(learner_features(x)[:, 7:11] == 0)


def test_decision_includes_overhead_and_exact_simulation_total():
    from types import SimpleNamespace
    a = SimpleNamespace(low=2, fixed=6, high=14)
    p, m = boards()
    for name in ("fixed", "entropy", "uncertainty", "disagreement", "learned"):
        r = decision(p, m, name, "sim", uniform_evaluator, lambda x: x[:, 0], {}, a)
        assert r["simulations"] == 18 and r["seconds"] > 0
    r = decision(p, m, "entropy", "time", uniform_evaluator, None, {}, a, seconds=1e-9)
    assert r["simulations"] == 6 and r["time_ratio"] > 1  # mandatory overhead is not hidden


def test_tuning_uses_full_validation_and_hard_draws():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 117)).astype(np.float32)
    x[:, 5:11] = rng.integers(0, 2, (40, 6))
    y = (x[:, 0] > 0).astype(np.float32)
    values = np.tile([1, 0, -1, -999, -999, -999, -999], (40, 1))
    values[::3, 0] = 0
    group = np.arange(40) % 3
    fit = fit_baselines(x, y, x, y, values, group, time.monotonic()+20)
    assert len(fit["trials"]) == 582
    for name in ("rule", "logistic"):
        p = scores(name, x, fit, None)
        assert p.shape == (40,) and np.isfinite(p).all()
        assert np.isclose(macro_nll(y, p, values, group), fit[name]["score"])
    masks = group_masks(values, group, y)
    assert masks["all"].sum() == 40 and masks["drawn"].sum() == 14


def test_colour_paired_games_and_cost_accounting():
    from types import SimpleNamespace
    a = SimpleNamespace(low=2, fixed=6, high=14)
    result = games(a, uniform_evaluator, None, {}, "entropy", np.random.default_rng(77),
                   time.monotonic()+10, pairs=2)
    assert result["completed_pairs"] == 2 and result["unfinished_games"] == 0
    assert result["wins"]+result["draws"]+result["losses"] == 4
    for row in result["costs"].values():
        assert row["simulations"] == row["decisions"]*6
    for i, trace in enumerate(result["moves"]):
        moves = result["starts"][i//2]+trace
        p, m = c4.from_moves(moves)
        assert c4.status(p, m) != 0


def test_secondary_full_smoke_groups_and_no_mlx():
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    a = SimpleNamespace(low=2, fixed=4, high=12, batch=3, smoke=True)
    baselines = dict(rule=dict(depth=0, signal=0, weight=0., scale=1., bias=-1.),
                     logistic=dict(weights=[0.]*118, mean=[0.]*117, std=[1.]*117))
    Path("results/r5").mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir="results/r5", prefix="astra_test_") as directory:
        out = Path(directory)
        secondary(a, uniform_evaluator, lambda x: np.full(len(x), .5), baselines, out,
                  time.monotonic()+10)
        report = json.loads((out/"secondary.json").read_text())
        assert report["complete"] and len(report["summary"]) == 14
        assert all(r["n"] == 3 and r["simulations"] == 12 for r in report["summary"].values())
    assert not any(x == "mlx" or x.startswith("mlx.") for x in sys.modules)


def test_followup_counterfactual_allocation_and_puct():
    from types import SimpleNamespace
    from r5_followup import counterfactual_moves
    p, m = boards()
    a = SimpleNamespace(low=8, fixed=32, high=64, allocation_floor=24, c_puct=3.)
    risk = np.array([.2, .1, .9])
    r = decision(p, m, "learned", "sim", uniform_evaluator, lambda _: risk, {}, a)
    cache = np.empty((3, 65), int)
    for i in range(3):
        s = Search(p[i:i+1], m[i:i+1], uniform_evaluator, capacity=64, c_puct=3.)
        for budget in range(65):
            s.advance(np.array([budget])); cache[i, budget] = s.moves()[0]
    expected = counterfactual_moves(risk, 24, 64, cache)
    np.testing.assert_equal(r["move"], expected)
    np.testing.assert_equal(r["per_position_sims"], [24, 24, 48])
    assert r["simulations"] == 96


def test_followup_all_validation_grid():
    from r5_followup import select, error_objective, counterfactual_moves, PUCTS
    rng = np.random.default_rng(34)
    x = rng.normal(size=(12, 117)).astype(np.float32)
    moves = np.tile(np.arange(129) % 2, (12, 1))
    values = np.tile([1, -1, -1, -1, -1, -1, -1], (12, 1))
    values[::3, 0] = 0
    data = dict(values=values, group=np.arange(12) % 3)
    cache = {cp:dict(x=x, moves=moves) for cp in PUCTS}
    b = dict(logistic=dict(weights=[0.]*118, mean=[0.]*117, std=[1.]*117))
    s = select(data, cache, lambda z: np.full(len(z), .5), b)
    assert s["validation_rows"] == 12 and len(s["trials"]) == 639 and len(s["choices"]) == 7
    assert len([r for r in s["trials"] if r["method"] == "fixed"]) == 3
    assert len([r for r in s["trials"] if r["method"] == "rule"]) == 576
    for name, row in s["choices"].items():
        assert row["objective"] == min(r["objective"] for r in s["trials"] if r["method"] == name)


def test_fresh_builder_seed_and_exclusion_resume(monkeypatch):
    from tempfile import TemporaryDirectory
    import r5_data
    monkeypatch.setattr(r5_data, "exclusions", lambda: (set(), dict(replays=[])))
    seeds = dict(train=1, val=23, test=37)
    with TemporaryDirectory(dir="results/r5", prefix="astra_builder_") as d:
        b = r5_data.Builder(d, seeds=seeds, extra_keys=[123], extra_manifest=[dict(test=True)])
        b.add("val", 3, time.monotonic()+10); b.save()
        r = r5_data.Builder(d, seeds=seeds, extra_keys=[123], extra_manifest=[dict(test=True)])
        assert b.rows == r.rows and 123 in r.seen and len(r.rows["val"]) == 3
        assert b.rng["val"].random() == r.rng["val"].random()
        with pytest.raises(AssertionError):
            r5_data.Builder(d, seeds=dict(train=1,val=24,test=37), extra_manifest=[dict(test=True)])


def test_demo_reports_written_file(monkeypatch, capsys):
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from r5_run import demo
    base = str(Path("results/r5").resolve())
    with TemporaryDirectory(dir=base, prefix="astra_demo_") as d:
        monkeypatch.chdir(d)
        root = Path("results/r5/unit_demo"); root.mkdir(parents=True)
        Path("media").mkdir()
        np.savez(root/"test.npz", values=np.ones((1, 7), int))
        (root/"evaluation.json").write_text(json.dumps(dict(summary={"sim/fixed":dict(mistakes=0,net_avoided=0,seconds=.01,simulations=32)},
                       records=[], config=dict(fixed=32), test_sha256="fixture")))
        demo(SimpleNamespace(run="unit_demo",max_minutes=1,low=8,fixed=32,high=128,holdout=3,batch=64,
                             demo_out="media/r5_unit.html"))
        record = json.loads(capsys.readouterr().out)
        assert record["event"] == "demo_written" and record["bytes"] > 0
        html = Path(record["path"]).read_text()
        assert "</html>" in html and "throughput" in html and "http" not in html


def test_game_summary_respects_smoke_budget():
    from r5_analyze import game_summary
    row = dict(method="learned", paired_batch=0, starts=[[]], results=[1,-1],
               completed_pairs=1, unfinished_games=0, wins=1, draws=0, losses=1, score=.5,
               costs={k:dict(simulations=8,decisions=2,seconds=.01) for k in ("learned","fixed")})
    assert game_summary([row], mean_simulations=4)["learned"]["score"] == .5
