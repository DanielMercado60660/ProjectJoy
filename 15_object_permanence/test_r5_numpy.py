"""Explicit NumPy/SciPy-only suite. Never collects the legacy MLX tests."""
import sys
import subprocess
import numpy as np
from scipy.special import logsumexp
import sim
from r5_data import frames, cues, look_flags, random_look, view_occ
from r5_classical import (physical_step, association_loglik, systematic_resample,
                           kalman_track, particle_track, initial_velocity_proposal)
from r5_metrics import (nll, moments, calibration_d2, uncertainty_look,
                        score_sequence, peek_value, validate, wall_scores, aggregate)


def scene(seed=91):
    p, present, occ, colors, turned = sim.simulate_one(np.random.default_rng(seed), dict(sim.BASE, **sim.HARD))
    vis = sim.occ_visible_frac(p[..., 0], occ[occ[:, 0] >= 0])
    return dict(pos=p[None], present=present[None], occ=occ[None], colors=colors[None],
                turned=turned[None], vis=vis[None], noise_seed=np.int64(seed))


def test_suite_does_not_import_mlx():
    # Isolate the import audit so full pytest may also collect legacy MLX tests.
    subprocess.run([sys.executable, '-c',
        "import sys, r5_data, r5_classical, r5_metrics; "
        "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)"], check=True)


def test_peek_changes_one_frame_one_wall_with_common_noise():
    d = scene()
    look = (12, 0)
    base, reveal = frames(d, 0), frames(d, 0, look)
    assert not np.array_equal(base[12], reveal[12])
    assert np.array_equal(base[:12], reveal[:12]) and np.array_equal(base[13:], reveal[13:])
    lo, hi = d['occ'][0, 0]
    outside = (np.arange(64)+1 <= lo) | (np.arange(64) >= hi)
    assert np.array_equal(base[12, :, outside], reveal[12, :, outside])
    assert look_flags(40, look).sum() == 1
    assert np.array_equal(d['occ'][0, 1:], view_occ(d['occ'][0], 0)[1:])


def test_cue_is_reproducible_and_future_independent():
    d = scene()
    cue = cues(d, [0])
    d['pos'][:, 1:] = -999
    assert np.array_equal(cue, cues(d, [0]))
    assert np.max(np.abs(cue-d['pos'][:, 0])) < 2
    assert random_look(d['occ'][0], 4) == random_look(d['occ'][0], 4)


def test_mixture_nll_matches_manual_and_is_permutation_invariant():
    w = np.array([.3, .7])
    mean = np.array([[-4., 0.], [4., 0.]])
    cov = np.array([np.eye(2), np.eye(2)*2])
    target = np.array([3., 1.])
    expected = -logsumexp(np.log(w)-np.log(2*np.pi)-.5*np.log(np.linalg.det(cov))
                          -.5*np.einsum('ki,kij,kj->k', target-mean, np.linalg.inv(cov), target-mean))
    assert np.allclose(nll((w, mean, cov), target), expected)
    assert np.allclose(nll((w[::-1], mean[::-1], cov[::-1]), target), expected)
    m, c = moments((w, mean, cov))
    assert c[0, 0] > 10 and np.allclose(m, [1.6, 0])


def test_rosenblatt_calibrates_correlated_multimodal_density():
    rng = np.random.default_rng(5)
    count = 16000
    w = np.array([.25, .75])
    m = np.array([[-4., 2.], [4., -2.]])
    c = np.array([[[2., .8], [.8, 1.]], [[1., -.6], [-.6, 2.]]])
    k = rng.choice(2, count, p=w)
    target = np.array([rng.multivariate_normal(m[j], c[j]) for j in k])
    dist = (np.broadcast_to(w, (count, 2)), np.broadcast_to(m, (count, 2, 2)),
            np.broadcast_to(c, (count, 2, 2, 2)))
    d2 = calibration_d2(dist, target)
    assert abs((d2 <= 1).mean()-(1-np.exp(-.5))) < .02
    assert abs((d2 <= 4).mean()-(1-np.exp(-2))) < .02


def test_physics_bounce_collision_and_hidden_turn_order():
    rng = np.random.default_rng(1)
    p = np.array([[[4., 20.], [57., 20.]]])
    v = np.array([[[-1., 0.], [1., 0.]]])
    p2, v2, _ = physical_step(p, v, np.zeros((1, 2), bool), np.array([[-1., -1.]]), rng)
    assert np.allclose(p2[0, 0], [4., 20.]) and np.allclose(v2[0, 0], [1., 0.])
    p = np.array([[[20., 20.], [28., 20.]]])
    v = np.array([[[1., 0.], [-1., 0.]]])
    p2, v2, _ = physical_step(p, v, np.zeros((1, 2), bool), np.array([[-1., -1.]]), rng)
    assert np.allclose(v2, -v) and np.allclose(p2, p+v)
    p = np.array([[[23., 20.]]])
    v = np.array([[[1., 0.]]])
    p2, v2, hidden = physical_step(p, v, np.zeros((1, 1), bool), np.array([[20., 35.]]), rng, turn_probability=1.)
    assert hidden.all() and np.allclose(p2, p+v)
    assert not np.allclose(v2, v)  # turn changes NEXT position, not current observation


def test_association_preserves_detection_order_invariance():
    pos = np.array([[[10., 10.], [40., 40.]], [[12., 10.], [38., 40.]]])
    dets = [(np.array([10., 10.]), 38), (np.array([40., 40.]), 38)]
    a = association_loglik(pos, dets, np.array([[-1., -1.]]), 1., .1)
    b = association_loglik(pos, dets[::-1], np.array([[-1., -1.]]), 1., .1)
    assert np.allclose(a, b) and a[0] > a[1]
    ix = systematic_resample(np.array([0., 1., 0.]), np.random.default_rng(1))
    assert np.array_equal(ix, [1, 1, 1])


def test_guided_initial_proposal_has_correct_importance_weights():
    rng = np.random.default_rng(123)
    pos = np.tile([[[10., 10.]]], (50000, 1, 1))
    v, log_ratio = initial_velocity_proposal(pos, [(np.array([11., 9.]), 38)], rng, .4)
    assert (abs(v) >= .6-1e-9).all() and (abs(v) <= 2.2+1e-9).all()
    # Importance-weighted moments recover the symmetric uniform prior, even
    # though most proposal draws move towards the detection.
    weights = np.exp(log_ratio)
    assert abs(weights.mean()-1) < .06
    assert np.max(abs((weights[:, None]*v[:, 0]).mean(0))) < .08


def test_cue_binding_handles_nearly_equal_x_and_reversed_slots():
    p = np.zeros((1, 8, 3, 2), np.float32)
    p[0, :, 0] = [10.05, 48]
    p[0, :, 1] = [10., 12]
    present = np.array([True, True, False])
    occ = np.array([[25., 38.], [-1., -1.]], np.float32)
    f = sim.render(p, present[None], occ[None], np.tile(sim.SAME, (1, 3, 1)), noise=0)[0]
    dist, _ = kalman_track(f, p[0, 0], present, occ, q=.03, r=.2)
    m, c = moments(dist)
    assert np.max(np.linalg.norm(m[:, :2]-p[0, :, :2], axis=-1)) < .25
    reversed_dist, _ = kalman_track(f, p[0, 0, [1, 0, 2]], present, occ, q=.03, r=.2)
    assert np.allclose(moments(reversed_dist)[0][:, :2], m[:, [1, 0]])


def test_classical_finite_causal_and_paired_peek_pipeline():
    d = scene()
    f = frames(d, 0)
    cue, pres, occ = cues(d, [0])[0], d['present'][0], d['occ'][0]
    a, diag = particle_track(f, cue, pres, occ, seed=8, particles=24)
    validate(a)
    short, _ = particle_track(f[:12], cue, pres, occ, seed=8, particles=24)
    assert all(np.array_equal(x[:12], y) for x, y in zip(a, short))
    look = uncertainty_look(a, pres, occ, 0.)
    assert look[0] == 4
    after, _ = particle_track(frames(d, 0, look), cue, pres, occ, look, seed=8, particles=24)
    assert all(np.array_equal(x[:look[0]], y[:look[0]]) for x, y in zip(a, after))
    score = score_sequence(after, d['pos'][0], d['vis'][0], pres)
    score['peek'] = peek_value(a, after, d['pos'][0], pres, look)
    assert score['peek']['cost_views'] == 1 and np.isfinite(score['all']['nll'])
    assert aggregate([score])['all']['n'] == 40*pres.sum()
    assert 0 < diag['min_ess_fraction'] <= 1


def test_uncertainty_policy_is_causal_and_forces_one_look():
    w = np.ones((40, 3, 1))
    m = np.ones((40, 3, 1, 2))*30
    c = np.tile(np.eye(2), (40, 3, 1, 1, 1))
    c[10:] *= 100
    pres, occ = np.ones(3, bool), np.array([[20., 40.], [-1., -1.]])
    look = uncertainty_look((w, m, c), pres, occ, 10.)
    assert look[0] == 11
    c[12:] *= 10000
    assert uncertainty_look((w, m, c), pres, occ, 10.) == look
    assert uncertainty_look((w, m, c), pres, occ, np.inf)[0] == 31
    assert np.isneginf(wall_scores((w, m, c), pres, occ)[:, 1]).all()


def test_zero_peek_effect_scores_exactly_zero():
    d = scene()
    dist, _ = kalman_track(frames(d, 0), cues(d, [0])[0], d['present'][0], d['occ'][0])
    value = peek_value(dist, dist, d['pos'][0], d['present'][0], (12, 0))
    for key in ('spread_reduction_px2', 'later_error_gain_px', 'later_nll_gain', 'final_swap_gain'):
        assert value[key] == 0
