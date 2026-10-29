"""Fast checks of the physics, inference, SBI network and the non-identifiable world."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics import (DESIGNS, G, MU_RATIO, NT, TIMES, loglik_frames, observe, positions,  # noqa: E402
                     rest_distance, world1_positions)
from inference import GridPosterior, eig, lsq_fit, run_experiment  # noqa: E402


def test_static_push_does_not_move():
    mu = 0.5
    F = 0.99 * MU_RATIO * mu * G
    assert np.all(world1_positions(F, 1.0, mu) == 0)
    assert world1_positions(1.01 * MU_RATIO * mu * G, 1.0, mu)[-1] > 0


def test_trajectory_physics():
    x = positions(8.0, 0.3, 0.3, 0.36, times=np.linspace(0.01, 10, 2000))
    assert np.all(np.diff(x) >= -1e-12)                       # never slides backwards
    assert np.isclose(x[-1], rest_distance(8.0, 0.3, 0.3, 0.36))
    # textbook check: a1 = F/m - mu g, stop distance = x1 + v1^2/(2 mu g)
    a1 = 8.0 - 0.3 * G
    assert np.isclose(x[-1], 0.5 * a1 * 0.09 + (a1 * 0.3) ** 2 / (2 * 0.3 * G))


def test_censored_likelihood_is_finite():
    rng = np.random.default_rng(0)
    y, c = observe(world1_positions(15, 1.0, 0.1), rng)
    assert c.any()
    ll = loglik_frames(y, c, world1_positions(15, 1.0, np.linspace(0.05, 0.8, 50)[:, None]))
    assert np.all(np.isfinite(ll))


def test_posterior_and_lsq_find_truth():
    rng = np.random.default_rng(1)
    post, data = GridPosterior(), []
    for F, T in [(8, 0.3), (6, 0.5), (9, 0.3)]:
        y, c = run_experiment(0.33, F, T, rng)
        post.update(F, T, y, c); data.append((F, T, y, c))
    lo, hi = post.interval(0.99)
    assert lo < 0.33 < hi and post.sd() < 0.02
    mu_ls, se = lsq_fit(data)
    assert abs(mu_ls - 0.33) < 0.03 and np.isfinite(se)


def test_eig_prefers_informative_pushes():
    post = GridPosterior()
    s = eig(post.particles(32), DESIGNS, M=1, rng=np.random.default_rng(0))
    weak = s[(DESIGNS[:, 0] == 1.0)].max()      # 1 N never moves the block for mu >= 0.085
    assert s.max() > 1.0 and weak < 0.3 and np.all(s > -0.05)   # nested-MC estimate can dip slightly below 0


def test_world3_only_depends_on_ratio():
    from deeper import sim3
    a = sim3(9.0, 0.4, np.log(1.0), np.log(1.5))
    b = sim3(9.0, 0.4, np.log(2.0), np.log(3.0))
    assert np.allclose(a, b)


def test_sbi_network_trains_saves_and_loads(tmp_path):
    from sbi import CONFIG, load, posterior_on_grid, simulate_batch, train
    from inference import GRID
    cfg = dict(CONFIG, steps=30, batch=64, hidden=32, embed=16, comps=2)
    train(cfg, str(tmp_path))
    model, cfg2 = load(str(tmp_path))
    f, m, mu = simulate_batch(np.random.default_rng(3), 4)
    W = posterior_on_grid(model, f, m, GRID)
    assert W.shape == (4, len(GRID)) and np.allclose(W.sum(1), 1) and np.all(np.isfinite(W))
