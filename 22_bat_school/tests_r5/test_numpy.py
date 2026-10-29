"""Run only this directory on Astra: no MLX, directly or transitively."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from env_r5 import (make_pool, CaveEnv, assert_disjoint, OBS_DIM, ECHO, HEARD,
                    REMAIN, INITIAL, WAIT, FORWARD, BACK, LEFT, RIGHT, BUDGETS,
                    MAX_STEPS, DR, DC, splat, GROUPS)
from baseline_r5 import AcousticPlanner, candidate
from eval_r5 import suite, run_budget, summarize, Behavior, selection_score
from io_r5 import save_checkpoint, load_checkpoint, Deadline
from math_r5 import gae, validation_due


@pytest.fixture(scope="module")
def pool():
    return make_pool("val", 2)


def test_no_mlx_import():
    assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)


def test_official_splits_and_legacy_layouts():
    train, val, test, legacy = (make_pool("train", 256), make_pool("val", 32),
                               make_pool("test", 128), make_pool("legacy", 200))
    assert_disjoint(train, val, test, legacy)
    assert set(train.group) == set(val.group) == set(test.group) == set(range(4))
    assert len(set(train.hashes)) == train.n
    from bat_env import MazePool
    assert legacy.hashes[:200] == MazePool([5], 200, 5005, rays=False).hashes
    assert legacy.hashes[-200:] == MazePool([6], 200, 6006, rays=False).hashes


def test_turn2_full_split_is_new_and_balanced():
    fresh = make_pool("test_turn2", 128)
    assert_disjoint(fresh, make_pool("train", 256), make_pool("val", 32), make_pool("test", 128), make_pool("legacy", 200))
    assert fresh.seed == 225400 and len(set(fresh.hashes)) == 512
    assert np.array_equal(np.bincount(fresh.group), [128]*4)
    assert all(int(h[:8], 16) % 4 == 3 for h in fresh.hashes)


def test_matched_validation_is_load_independent():
    updates = range(1, 5601)
    for elapsed in (0., 10., 9999.):
        assert [u for u in updates if validation_due(u, elapsed, 120., 1000)] == [1, 1000, 2000, 3000, 4000, 5000]
    assert validation_due(17, 120., 120.)
    assert not validation_due(1000, 119., 120.)
    # A resumed interval depends on total updates, not invocation count.
    assert not validation_due(1001, 0., 0., 1000)
    assert validation_due(2000, 0., 0., 1000)
    with pytest.raises(ValueError):
        validation_due(1, 0., 0., -1)


def test_reciprocal_shutters_and_motion(pool):
    e = CaveEnv(pool, pool.n, fixed=True)
    changed = False
    for i, size in enumerate(pool.sizes):
        for r in range(size):
            for c in range(size):
                for d in range(4):
                    if pool.open[i, r, c, d]:
                        rr, cc, dd = r+DR[d], c+DC[d], (d+2) % 4
                        for key in ("period", "phase", "closed"):
                            a = getattr(pool, key)
                            assert a[i, r, c, d] == a[i, rr, cc, dd]
                        times = np.arange(30)
                        values = e.shut(i, r, c, d, times)
                        if pool.closed[i, r, c, d] > 0:
                            assert values.any() and (~values).any()
                            changed = True
    assert changed


def test_no_free_pulse_and_exact_budget(pool):
    e = CaveEnv(pool, pool.n, fixed=True, budget=2, auto_reset=False)
    o = e.reset()
    assert o.shape == (pool.n, OBS_DIM)
    assert not o[:, :ECHO+1].any()
    assert np.all(o[:, REMAIN] == 2/64)
    for t in range(4):
        o, _, _, info = e.step(np.full(pool.n, WAIT), np.ones(pool.n, bool))
        assert np.all(info["pulses"] == min(t+1, 2))
        assert np.all(o[:, HEARD] == (t < 2))
        assert np.all(o[:, REMAIN] == max(1-t, 0)/64)
    assert not o[:, :ECHO].any()
    assert np.all(o[:, INITIAL] == 2/64)


def test_collision_checks_arrival_shutter(pool):
    p = copy.deepcopy(pool)
    e = CaveEnv(p, 1, budget=4, auto_reset=False)
    e.reset()
    m = int(np.flatnonzero(p.group == 1)[0])
    r, c, d = np.argwhere(p.open[m])[0]
    e.m[0], e.r[0], e.c[0], e.hd[0], e.t[0] = m, r, c, d, 0
    p.period[m, r, c, d], p.phase[m, r, c, d], p.closed[m, r, c, d] = 4, 3, 2
    # open at t=0, closed at t=1: a stale echo cannot promise safe passage.
    assert not e.shut(m, r, c, d, 0)
    _, _, _, info = e.step([FORWARD], [False])
    assert info["bump"][0] and (e.r[0], e.c[0]) == (r, c)
    e.t[0] = 2
    _, _, _, info = e.step([FORWARD], [False])
    assert info["moved"][0] and (e.r[0], e.c[0]) == (r+DR[d], c+DC[d])


def test_pulse_schedules_do_not_change_future_noise(pool):
    a = CaveEnv(pool, pool.n, fixed=True, seed=123, budget=64, auto_reset=False)
    b = CaveEnv(pool, pool.n, fixed=True, seed=123, budget=64, auto_reset=False)
    a.reset(); b.reset()
    for t in range(5):
        a.step(np.full(pool.n, WAIT), np.full(pool.n, t % 2 == 0))
        b.step(np.full(pool.n, WAIT), np.zeros(pool.n, bool))
    oa, *_ = a.step(np.full(pool.n, WAIT), np.ones(pool.n, bool))
    ob, *_ = b.step(np.full(pool.n, WAIT), np.ones(pool.n, bool))
    np.testing.assert_array_equal(oa[:, :ECHO], ob[:, :ECHO])


def test_checkpoint_exact_environment_continuation(pool, tmp_path):
    e = CaveEnv(pool, pool.n, seed=7, budget=8)
    e.reset()
    actions = np.arange(pool.n) % 5
    for _ in range(10):
        e.step(actions, np.ones(pool.n, bool))
    rng = np.random.default_rng(72)
    payload = dict(environment=e.state_dict(), rng=rng.bit_generator.state,
                   optimizer={"step": np.array(3), "m": [np.zeros(2), np.ones(2)]}, config={"seed": 7})
    path = tmp_path/"state.npz"
    save_checkpoint(path, payload)
    saved = load_checkpoint(path)
    clone = CaveEnv(pool, pool.n, seed=99, budget=8)
    clone.load_state_dict(saved["environment"])
    for _ in range(8):
        x = e.step(actions, np.ones(pool.n, bool))
        y = clone.step(actions, np.ones(pool.n, bool))
        for i in range(3):
            np.testing.assert_array_equal(x[i], y[i])
        for key in x[3]:
            np.testing.assert_array_equal(x[3][key], y[3][key])
    assert saved["optimizer"]["step"].item() == 3
    save_checkpoint(path, payload)
    assert [f.name for f in tmp_path.iterdir()] == ["state.npz"]


def test_wait_timeout_is_failure_and_frozen(pool):
    e = CaveEnv(pool, pool.n, fixed=True, budget=8, auto_reset=False)
    e.reset()
    for _ in range(MAX_STEPS+2):
        _, reward, _, _ = e.step(np.full(pool.n, WAIT), np.zeros(pool.n, bool))
    assert e.finished.all() and not e.success.any()
    assert np.all(e.t == MAX_STEPS) and np.all(reward == 0)


def test_gae_terminal_and_zero_variance():
    r = np.array([[1., 1.], [5., 5.]], np.float32)
    v = np.zeros_like(r)
    d = np.array([[1., 0.], [1., 1.]])
    a, ret = gae(r, v, d, np.array([999., 999.]), gamma=1., lam=1.)
    np.testing.assert_array_equal(a, [[1., 6.], [5., 5.]])
    np.testing.assert_array_equal(a, ret)
    zero = (v-v.mean())/np.sqrt(max(float(v.var()), 1e-8))
    assert np.isfinite(zero).all()


def test_splat_conserves_amplitude():
    d = np.array([[.5, 1., 3.7, 11.5]])
    amp = np.array([[1., .5, .3, .2]])
    np.testing.assert_allclose(splat(d, amp).sum(-1), amp)


@pytest.mark.parametrize("mode", ["event", "uncertainty", "every"])
def test_baseline_full_grid_and_reproducibility(pool, mode):
    factory = lambda n: AcousticPlanner(n, mode)
    a = suite(pool, factory)
    assert len(a["rows"]) == len(GROUPS)*len(BUDGETS)
    for row in a["rows"]:
        assert row["pulses"] <= row["budget"]
        assert np.isfinite(row["completion"])
    e, _ = run_budget(pool, factory, 8)
    assert e == [r for r in a["episodes"] if r["budget"] == 8]
    with pytest.raises(ValueError):
        summarize([r for r in e if r["group"] != "hard"])


def test_candidates_change_controller_and_schedule():
    configs = [candidate(np.random.default_rng(i), i) for i in range(12)]
    for key in configs[0]:
        assert len({c[key] for c in configs}) > 1, key


def test_behavior_counts_and_no_spurious_repeat(pool):
    e = CaveEnv(pool, pool.n, fixed=True, budget=8, auto_reset=False)
    e.reset(); b = Behavior(pool.n)
    for t in range(2):
        move = np.full(pool.n, WAIT)
        obs, _, _, info = e.step(move, np.ones(pool.n, bool))
        b.update(e, move, obs, info, np.ones(pool.n, bool))
    assert np.all(b.counts["wait_steps"] == 2)
    assert np.all(b.counts["double_listens"] == 1)
    assert np.all(b.counts["repeat_pairs"] == 1)
    assert np.all(b.counts["translate_listens"] == 0)
    assert np.all(b.counts["translate_pairs"] == 0)


def test_static_reference_and_headless_render(pool, tmp_path):
    from demo_r5 import render
    episodes, frames = run_budget(pool, lambda n: AcousticPlanner(n, "every"), 64, trace=True)
    static = [e for e in episodes if e["group"] == "static"]
    assert all(e["success"] for e in static)
    ep = static[0]
    path = tmp_path/"baseline.gif"
    render(pool, ep["cave"], frames, ep, path, Deadline(.1))
    assert path.read_bytes().startswith(b"GIF89a")


def test_moth_is_occluded_and_silent_state_does_not_reveal_map(pool):
    p = copy.deepcopy(pool)
    e = CaveEnv(p, 1, budget=8, auto_reset=False)
    e.reset()
    e.m[:] = 0; e.r[:] = 0; e.c[:] = 0; e.hd[:] = 1
    # A wall immediately east blocks an east-facing moth. Compare exact same
    # noise realization with a goal perpendicular to that ray.
    p.open[0, 0, 0, 1] = False
    p.wall_distance[0, 0, 0, 1] = .5
    p.goal[0] = [0, 2]
    state = copy.deepcopy(e.rng.bit_generator.state)
    blocked = e.echo()[0, 1, 0].copy()
    p.goal[0] = [2, 0]
    e.rng.bit_generator.state = state
    away = e.echo()[0, 1, 0]
    np.testing.assert_array_equal(blocked, away)
    state = copy.deepcopy(e.rng.bit_generator.state)
    a = e.observe(np.zeros(1, bool))
    e.r[:] = 1; e.c[:] = 1; e.hd[:] = 3
    e.rng.bit_generator.state = state
    b = e.observe(np.zeros(1, bool))
    np.testing.assert_array_equal(a, b)


def test_deadline():
    with pytest.raises(ValueError):
        Deadline(0)
    d = Deadline(1)
    d.end = d.start-1
    with pytest.raises(TimeoutError):
        d.check()


def test_never_commit_cannot_win_selection():
    waiting = [dict(n=32, completion=0., collisions=0., steps=320., pulses=0.) for _ in range(16)]
    risky = [dict(n=32, completion=0., collisions=320., steps=320., pulses=64.) for _ in range(16)]
    risky[0]["completion"] = 1/32
    assert selection_score(risky) > selection_score(waiting)
    efficient = copy.deepcopy(risky)
    efficient[0]["collisions"] = 0.
    assert selection_score(efficient) > selection_score(risky)


def test_turn2_gate_rejects_unequal_selection_and_partial_work(tmp_path, monkeypatch):
    """A wall-limited or extra-selected arm must not pass as matched work."""
    import json
    from check_turn2_r5 import check_run
    from io_r5 import write_json
    monkeypatch.chdir(tmp_path)
    work = Path('logs/r5/control')
    work.mkdir(parents=True)
    config = dict(seed=61, entropy=.015)
    rows = [dict(group=g, budget=b, n=32) for g in GROUPS for b in BUDGETS]
    validation = [dict(update=u, score=i/8, rows=rows)
                  for i, u in enumerate([1, 1000, 2000, 3000, 4000, 5000, 5600])]
    records = [dict(update=u, loss=0., entropy=.4, grad_norm=.1, reward_mean=-.01,
                    action_fractions=[1., 0., 0., 0., 0.]) for u in range(1, 5601)]
    (work/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    def write_validation(v):
        (work/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in v))
    write_validation(validation)
    saved = dict(update=5600, env_steps=22937600, optimizer={'step': np.array(67200)}, config=config)
    save_checkpoint(work/'latest.npz', saved)
    selected = work/'selected.npz'
    save_checkpoint(selected, dict(saved, validation=validation[-1], best_score=validation[-1]['score']))
    summary = dict(update=5600, target_reached=True, selected=str(selected), total_seconds=600,
                   best_score=validation[-1]['score'])
    write_json(work/'summary.json', summary)
    assert check_run('control', 61, .015) == config
    write_validation(validation[:2]+[dict(update=1500, score=0., rows=rows)]+validation[2:])
    with pytest.raises(AssertionError):
        check_run('control', 61, .015)
    write_validation(validation)
    summary['target_reached'] = False
    write_json(work/'summary.json', summary)
    with pytest.raises(AssertionError):
        check_run('control', 61, .015)
