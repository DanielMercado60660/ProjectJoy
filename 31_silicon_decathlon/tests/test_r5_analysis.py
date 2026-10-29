"""NumPy-only checks for post-hoc predictive and paired analysis."""
import json
import sys

import numpy as np
import pytest

from analyze_r5 import corr, demean, paired, relationship, stats


def test_ols_holds_out_entire_seed_and_predicts_known_relation():
    x = np.tile(np.arange(4), 3).astype(float)
    seeds = np.repeat([1, 2, 3], 4)
    r = relationship(x, 2*x+5, seeds)
    assert r["independent_seeds"] == 3
    assert r["within_seed_slope_per_ms"] == pytest.approx(2)
    assert r["leave_one_seed_out"]["r2"] == pytest.approx(1)
    assert r["leave_one_seed_out"]["rmse"] < 1e-12
    # An excellent within-seed association need not generalize scene difficulty.
    y = 2*x+100*seeds
    r = relationship(x, y, seeds)
    assert r["within_seed_r2"] == pytest.approx(1)
    assert r["leave_one_seed_out"]["r2"] < 0


def test_constant_success_is_not_a_correlation():
    assert corr([1, 2, 3], [1, 1, 1]) is None
    assert np.allclose(demean([1, 3, 100, 104], [1, 1, 2, 2]), [-1, 1, -2, 2])
    assert stats([1, 2, 3, 100])["p50"] == 2.5


def test_pairs_match_all_conditions_and_serialize_numpy_outcomes():
    es = []
    for seed in (1, 2):
        for engine in ("a", "b"):
            for policy in ("queue", "latest"):
                es.append(dict(job=dict(seed=seed, engine=engine, policy=policy),
                               file=f"{seed}/{engine}/{policy}",
                               value=10*seed+(0 if policy == "queue" else -seed)))
    r = paired(es, "policy", "queue", "latest", lambda e: np.mean([e["value"]]))
    assert r["negative_pairs"] == 4
    assert r["seed_mean_deltas"] == {"1": -1, "2": -2}
    assert r["deltas"]["mean"] == -1.5
    json.dumps(r, allow_nan=False)
    with pytest.raises(AssertionError):
        paired(es[:-1], "policy", "queue", "latest", lambda e: e["value"])
    assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)
