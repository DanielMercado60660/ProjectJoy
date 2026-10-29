"""Round-5 simulator / estimator / metric / recurrence tests. NO MLX imports."""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from r5_baseline import AdaptiveRay, DEFAULT
from r5_common import GROUPS, atomic_npz
from r5_env import SurfaceRace, finish_flags, make_bank, suite
from r5_eval import ADAPT_CAP, Events, run, summarize, validate
from r5_policy import HIDDEN, Legacy, Policy, gae, init_params, load_selected, save_selected, sequence, step
from train_r5 import chunks, rollout
from racing.sim import DT, G, MAX_STEER, V_SCALE, WHEELBASE


@pytest.fixture
def tmp_path():
    # Keep even ephemeral test writes inside this project.
    import tempfile
    root = Path(__file__).resolve().parents[1] / 'results/r5/pytest_tmp'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as path:
        yield Path(path)


@pytest.fixture(scope='module')
def bank():
    return make_bank('val', 2)


def test_sensor_causal_paint_and_surface_state(bank):
    a, b = SurfaceRace(bank, 2), SurfaceRace(bank, 2)
    oa = a.reset(np.random.default_rng(8), ids=np.array([0, 1]), modes=np.zeros(2, int))
    ob = b.reset(np.random.default_rng(8), ids=np.array([0, 1]), modes=np.zeros(2, int))
    b.paint[:] = 100
    b.low[:] = .3
    assert np.array_equal(oa, ob)  # invisible grip and paint
    assert np.array_equal(a.observe(), b.observe())
    before = a.base.yaw.copy()
    o, _, _ = a.step(np.tile([.1, 1.], (2, 1)))
    np.testing.assert_allclose(o[:, 13], (a.base.yaw - before) / DT)
    st = a.state()
    b.restore(st)
    for _ in range(5):
        ra = a.step(np.tile([-.1, .2], (2, 1)))
        rb = b.step(np.tile([-.1, .2], (2, 1)))
        for x, y in zip(ra, rb):
            assert np.array_equal(x, y)


def test_spatial_patch_and_bidirectional_switch(bank):
    e = SurfaceRace(bank, 2)
    e.reset(np.random.default_rng(1), ids=np.array([0, 1]), modes=np.array([0, 1]))
    e.start[:] = .3
    e.width[:] = .25
    e.up[:] = 0
    assert np.allclose(e.grip_now(), e.high)
    e.base.idx[0] = int(.4 * bank.n_pts[0])
    e.base.prog[1] = .4 * bank.length[1]
    assert np.allclose(e.grip_now(), e.low)
    e.base.idx[0] = int(.7 * bank.n_pts[0])
    e.up[1] = 1
    assert np.allclose(e.grip_now(), e.high)
    e.base.prog[1] = 0
    assert e.grip_now()[1] == e.low[1]


def test_estimator_zero_and_no_evidence_not_equality():
    c = AdaptiveRay(dict(DEFAULT, rise=0.))
    o = np.zeros((2, 14), np.float32)
    assert np.isfinite(c(o)).all()
    o[:, 10] = 10 / V_SCALE
    c(o)
    c(o)  # coast at constant speed: no saturation evidence
    assert (c.mu >= DEFAULT['prior'] - 1e-6).all()
    assert (c.measurements == 0).all()


def test_estimator_recovers_saturated_grip_without_truth_input():
    c = AdaptiveRay(dict(DEFAULT, prior=1., rise=0.))
    o = np.zeros((1, 14), np.float32)
    o[:, 10] = 20 / V_SCALE
    c(o)
    o[:, 11] = .8
    o[:, 13] = .45 * G / 20
    c(o)
    assert abs(c.mu[0] - .45) < .025
    assert c.measurements[0] == 1
    assert abs(20 * np.tan(.8 * MAX_STEER) / WHEELBASE) > o[0, 13]


def test_full_splits_include_hard_and_fresh_seeds(bank):
    ids, modes, groups, seed = suite(bank, 'val')
    assert set(groups) == set(GROUPS)
    assert len(ids) == 12
    assert all((groups == g).sum() == 2 for g in GROUPS)
    assert set(t.seed for t in bank.tracks).isdisjoint(t.seed for t in make_bank('test', 2).tracks)
    assert set(t.seed for t in bank.tracks).isdisjoint(t.seed for t in make_bank('train', 2).tracks)
    v = validate(AdaptiveRay(), bank, max_steps=8)
    assert set(v['groups']) == set(GROUPS)
    assert all(g['n'] == 2 for g in v['groups'].values())


def test_finish_step_crash_never_finishes():
    assert finish_flags(np.array([101, 101, 99]), np.array([100]*3), np.array([False, True, False])).tolist() == [True, False, False]


def test_adaptation_crash_unreached_and_sustained_envelope():
    e = Events(np.array([0, 1, 2]))
    for t in range(5):
        e.update(t, np.array([t == 0, False, False]), np.array([-1, 0, 0]),
                 np.array([True, True, True]), np.array([False, False, False]))
    assert e.status[0, 0] == 'recovered'
    assert e.latency[0, 0] == pytest.approx(.1)
    e.update(5, np.array([True, False, False]), np.ones(3), np.ones(3, bool), np.array([True, False, False]))
    e.finish()
    assert e.status[0, 1] == 'terminated'
    assert e.status[1, 0] == 'unreached'
    assert e.latency[0, 1] == ADAPT_CAP and e.latency[1, 0] == ADAPT_CAP
    assert e.planned.sum() == 3
    assert not e.rows(1)[0]['exposed']


def test_dead_slow_policy_pays_timeout_and_unreached(bank):
    ids, modes, _, seed = suite(bank, 'val')
    r = run(lambda o: np.zeros((len(o), 2)), bank, ids, modes, seed, 10)
    s = summarize(r)
    assert s['completion'] == 0 and s['crash_rate'] == 0 and s['timeout_rate'] == 1
    assert s['mean_dnf_penalized_lap_s'] == 1
    assert s['adaptation_capped_s'] == ADAPT_CAP and s['events_exposed'] == 0
    assert s['event_recovery_coverage'] == 0


def test_common_scenarios_identical_candidates(bank):
    ids = np.tile([0, 1], 3)
    modes = np.tile([0, 1], 3)
    r = run(AdaptiveRay(), bank, ids, modes, 99, 100, repeat_scenarios=2)
    for k in ('utility', 'finished', 'crashed', 'lap_penalty_s', 'progress'):
        assert np.array_equal(r[k][:2], r[k][2:4])
        assert np.array_equal(r[k][:2], r[k][4:])


def test_policy_warmstart_parity_reset_and_save(tmp_path):
    p = init_params()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 5, 14)).astype(np.float32)
    x[..., :10] = np.clip(x[..., :10], 0, 1)
    h = np.zeros((5, HIDDEN), np.float32)
    mean, value, end = sequence(p, x, h, np.zeros((3, 5)), True)
    old = Legacy()
    np.testing.assert_allclose(mean, np.stack([old(o) for o in x]), atol=1e-6)
    assert np.isfinite(sequence(p, np.zeros_like(x), h, np.ones((3, 5)), True)[0]).all()
    # Make the memory output nonzero so the reset assertion is meaningful.
    p['out'][:] = rng.normal(size=p['out'].shape).astype(np.float32) * .1
    m, _, _ = sequence(p, x, h, np.ones((3, 5)), True)
    f, _, _ = sequence(p, x, h, np.zeros((3, 5)), False)
    np.testing.assert_allclose(m, f, atol=1e-6)
    q = tmp_path / 'selected.npz'
    save_selected(q, p, dict(arm='rnn'))
    restored, meta = load_selected(q)
    np.testing.assert_array_equal(restored(x[0]), Policy(p)(x[0]))


def test_chunk_order_and_terminal_gae():
    a = np.arange(8 * 3 * 2).reshape(8, 3, 2)
    b = chunks(a, 4)
    for chunk in range(2):
        for env in range(3):
            np.testing.assert_array_equal(b[:, chunk * 3 + env], a[chunk * 4:(chunk + 1) * 4, env])
    reward = np.array([[1.], [10.]], np.float32)
    adv, ret = gae(reward, np.zeros_like(reward), np.array([100.]), np.array([[1.], [1.]]))
    np.testing.assert_array_equal(adv, reward)


def test_rollout_exact_resume_including_hidden_rng_and_sim(bank, tmp_path):
    p = init_params()
    p['out'][:] = .01
    env = SurfaceRace(bank, 4)
    rng = np.random.default_rng(52)
    env.reset(rng)
    h = np.zeros((4, HIDDEN), np.float32)
    _, h, _ = rollout(p, env, rng, h, 8, 10, True)
    state, rngstate = env.state(), json.dumps(rng.bit_generator.state)
    path = tmp_path / 'state.npz'
    atomic_npz(path, **state)
    expected, he, _ = rollout(p, env, rng, h.copy(), 8, 10, True)
    other = SurfaceRace(bank, 4)
    with np.load(path) as z:
        other.restore(dict(z))
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = json.loads(rngstate)
    actual, ha, _ = rollout(p, other, rng2, h.copy(), 8, 10, True)
    for k in expected:
        np.testing.assert_array_equal(expected[k], actual[k])
    np.testing.assert_array_equal(he, ha)


def test_classical_tuner_resume_matches_uninterrupted():
    """Tiny actual CEM generation checks optimizer/RNG/selection persistence."""
    import subprocess
    import uuid
    tag = 'r5_smoke_numpy_' + uuid.uuid4().hex[:10]
    args = [sys.executable, 'tune_r5.py', '--population', '4', '--episodes', '6',
            '--n-train', '1', '--n-val', '1', '--max-steps', '12', '--eval-every', '1', '--max-minutes', '.3']
    for name, generations, resume in [(tag + '_whole', 2, False), (tag, 1, False), (tag, 2, True)]:
        command = args + ['--tag', name, '--max-generations', str(generations)]
        if resume:
            command += ['--resume', f'checkpoints/{name}_baseline_state.npz']
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    with np.load(f'checkpoints/{tag}_baseline_state.npz') as a, np.load(f'checkpoints/{tag}_whole_baseline_state.npz') as b:
        assert int(a['generation']) == 2
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_numpy_export_eval_and_headless_demo(tmp_path):
    import subprocess
    import uuid
    import re
    import shutil
    from r5_common import write_json
    tag = 'r5_smoke_numpy_' + uuid.uuid4().hex[:10]
    p = init_params()
    for arm in ('rnn', 'ff'):
        save_selected(f'checkpoints/{tag}_{arm}_selected.npz', p, dict(arm=arm, test_only_untrained=True))
    write_json(f'checkpoints/{tag}_baseline_selected.json', dict(params=DEFAULT, test_only_untrained=True))
    for script in ('r5_eval.py', 'demo_r5.py'):
        args = [sys.executable, script, '--tag', tag, '--max-steps', '10', '--max-minutes', '.3']
        if script == 'r5_eval.py':
            args += ['--n', '1']
        subprocess.run(args, check=True, capture_output=True, text=True, timeout=20)
    result = json.loads(Path(f'results/r5/{tag}_test.json').read_text())
    assert result['complete'] and len(result['methods']) == 6
    assert all(set(m['groups']) == set(GROUPS) for m in result['methods'].values())
    assert len(result['paired']) == 4
    html = Path(f'media/{tag}_demo.html').read_text()
    assert 'http://' not in html and 'https://' not in html
    assert 'test_only_untrained' not in html  # demo shows checkpoint paths, no training claim
    if shutil.which('node'):
        script = tmp_path / 'demo.js'
        script.write_text(re.search(r'<script>(.*?)</script>', html, re.S).group(1))
        subprocess.run(['node', '--check', str(script)], check=True, capture_output=True, timeout=5)


def test_demo_subbatch_has_exact_headline_surfaces(bank):
    ids, modes, groups, seed = suite(bank, 'test')
    full = run(AdaptiveRay(), bank, ids, modes, seed, 120, record=True)
    chosen = np.array([0, 3, 4, 7, 8, 11])
    sub = run(AdaptiveRay(), bank, ids[chosen], modes[chosen], seed, 120, record=True)
    np.testing.assert_array_equal(full['trajectory'][:, chosen], sub['trajectory'])


def test_legacy_split_retains_original_two_to_one_geometry_counts():
    bank = make_bank('legacy', 1)
    assert [t.seed for t in bank.tracks] == [1_000_000, 1_000_001, 2_000_000]
    ids, _, labels, _ = suite(bank, 'legacy')
    assert len(ids) == 9
    assert (labels == 'normal_patch').sum() == 2
    assert (labels == 'hard_patch').sum() == 1


def test_all_round5_tracks_at_full_run_sizes():
    """Exercise exactly the documented sizes, including all hard and legacy tracks.

    Every generated geometry is checked against the untouched original generator
    at its recorded effective seed. This also covers all 150 legacy geometries.
    """
    from racing.track import SPLITS, make_track
    from r5_env import TRACK_RETRY_BASE, TRACK_RETRY_LIMIT
    seen = set()
    report = {}
    for split, n in [('train', 64), ('val', 16), ('test', 64), ('legacy', 50)]:
        bank = make_bank(split, n)
        expected = 3 * n if split == 'legacy' else 2 * n
        assert len(bank.tracks) == len(bank.r5_track_manifest) == expected
        seeds = {t.seed for t in bank.tracks}
        assert len(seeds) == expected and not seen.intersection(seeds)
        seen.update(seeds)
        for track, record in zip(bank.tracks, bank.r5_track_manifest):
            requested, retry = record['requested_seed'], record['retry']
            assert 0 <= retry <= TRACK_RETRY_LIMIT
            assert track.seed == record['effective_seed']
            assert track.seed == (requested if retry == 0 else TRACK_RETRY_BASE + 256 * requested + retry)
            if split == 'legacy':
                assert retry == 0
            original = make_track(track.seed - SPLITS[record['geometry']], record['geometry'])
            for field in ('center', 'tangent', 'normal', 'curvature', 'half_width', 'length'):
                np.testing.assert_array_equal(getattr(track, field), getattr(original, field))
                assert np.isfinite(getattr(track, field)).all()
        report[split] = dict(tracks=expected, replacements=[r for r in bank.r5_track_manifest if r['retry']])
    assert len(seen) == 438
    print('\nFULL_R5_TRACK_GENERATION ' + json.dumps(report, sort_keys=True))


def test_rejected_seed_retry_is_deterministic_and_keeps_old_failure():
    from racing.track import make_track
    from r5_env import make_r5_track, TRACK_RETRY_BASE
    # The original API must still behave exactly as it did before this fix.
    with pytest.raises(RuntimeError, match='could not generate a valid track for seed 30049'):
        make_track(30049, 'hard')
    a, ra = make_r5_track(30049, 'hard')
    b, rb = make_r5_track(30049, 'hard')
    assert ra == rb
    assert ra['requested_seed'] == 2_030_049
    assert 1 <= ra['retry'] <= 32
    assert a.seed == TRACK_RETRY_BASE + 256 * 2_030_049 + ra['retry']
    np.testing.assert_array_equal(a.center, b.center)


def test_track_retry_does_not_hide_other_errors_and_is_bounded(monkeypatch):
    import r5_env
    def unrelated(seed, geometry):
        raise RuntimeError('unrelated generator defect')
    monkeypatch.setattr(r5_env, 'make_track', unrelated)
    with pytest.raises(RuntimeError, match='unrelated generator defect'):
        r5_env.make_r5_track(30049, 'hard')
    attempts = []
    def exhausted(seed, geometry):
        attempts.append(seed)
        raise RuntimeError(f'could not generate a valid track for seed {seed}')
    monkeypatch.setattr(r5_env, 'make_track', exhausted)
    with pytest.raises(RuntimeError, match='exhausted 32 deterministic replacements'):
        r5_env.make_r5_track(30049, 'hard')
    assert len(attempts) == 33 and len(set(attempts)) == 33
