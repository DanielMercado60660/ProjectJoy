"""Turn-2 data/identity/schedule/baseline checks. No MLX import or execution."""
import itertools
import sys
import subprocess
import numpy as np
from r5_data import FRESH_SEED, SMOKE_SEED
from r5_turn2_utils import (TURN2_FRESH_SEED, permute_queries, query_features,
                            learning_rate, baseline_grids)


def test_no_mlx_and_disjoint_headline_seed_ranges():
    subprocess.run([sys.executable, '-c',
        "import sys, r5_turn2_utils; "
        "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)"], check=True)
    # No turn-2 headline scenes are generated or inspected by this test.
    ranges = [set(range(seed, seed+1000)) for seed in (FRESH_SEED, SMOKE_SEED, TURN2_FRESH_SEED)]
    assert all(not a.intersection(b) for a, b in itertools.combinations(ranges, 2))


def test_query_label_permutation_keeps_identity_and_absent_masks():
    pos = np.arange(2*5*3*2).reshape(2, 5, 3, 2).astype(np.float32)
    pres = np.array([[True, True, False], [True, True, True]])
    cue = pos[:, 0]+.25
    original = pos.copy()
    order = np.array([[2, 0, 1], [1, 2, 0]])
    p, pr, c = permute_queries(pos, pres, cue, order)
    assert np.array_equal(pos, original)  # renderer input was not mutated
    for b in range(2):
        for j in range(3):
            assert np.array_equal(p[b, :, j], original[b, :, order[b, j]])
            assert pr[b, j] == pres[b, order[b, j]]
            assert np.array_equal(c[b, j], cue[b, order[b, j]])
    inverse = np.argsort(order, axis=1)
    restored = permute_queries(p, pr, c, inverse)
    assert all(np.array_equal(a, b) for a, b in zip(restored, (pos, pres, cue)))


def test_query_features_are_equivariant_and_do_not_sort_near_ties():
    cue = np.array([[[10.01, 48.], [10., 12.], [45., 30.]]])
    present = np.array([[1., 1., 0.]])
    occ = np.array([[[24., 37.], [-1., -1.]]])
    ref = query_features(cue, present, occ)
    assert ref.shape == (1, 3, 8)
    assert ref[0, 0, 1] > ref[0, 1, 1]  # y cue resolves the x near-tie
    assert np.all(ref[0, 2, :2] == 0)
    for order in itertools.permutations(range(3)):
        order = list(order)
        candidate = query_features(cue[:, order], present[:, order], occ)
        assert np.array_equal(candidate, ref[:, order])
    shared = np.broadcast_to(cue[:, :1], cue.shape)
    assert np.array_equal(query_features(shared, np.ones_like(present), occ)[:, 0],
                          query_features(shared, np.ones_like(present), occ)[:, 1])


def test_lr_schedule_is_fixed_across_segment_boundaries_and_decays():
    total, peak = 120000, 3e-4
    assert learning_rate(0, total, peak) == peak*.05
    assert np.isclose(learning_rate(1000, total, peak), peak)
    assert np.isclose(learning_rate(40000, total, peak), peak)
    assert np.isclose(learning_rate(total, total, peak), peak*.05)
    rates = np.array([learning_rate(s, total, peak) for s in range(40000, total+1, 100)])
    assert (np.diff(rates) <= 1e-15).all()
    for resume in (40000, 80000):
        assert learning_rate(resume, total, peak) <= learning_rate(resume-1, total, peak)+1e-15
    assert all(np.isfinite(learning_rate(s, 4)) for s in range(5))


def test_expanded_filter_grid_brackets_both_previous_boundary_winners():
    k1, p1 = baseline_grids(512, False)
    k2, p2 = baseline_grids(512, True)
    assert k1 == k2 and len(p2) == 400 and len(k2) == 24
    def configs(grid):
        return {tuple(sorted(c.items())) for c in grid}
    assert configs(p1).issubset(configs(p2))
    for key, previous in (('q', .5), ('miss', .35)):
        vals = [row[key] for row in p2]
        assert min(vals) < previous < max(vals)
    assert all(c['particles'] == 512 for c in p2)
