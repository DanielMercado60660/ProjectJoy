"""Struct model variant (round-3 follow-up): reduces to the original model when the wing is unbreakable,
never creates energy, and folds long thin wings on hard throws but not the textbook glider."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim  # noqa: E402
import sim_struct as ss  # noqa: E402
from evaluate import BASELINE  # noqa: E402


def test_unbreakable_wing_matches_original_model():
    G = np.vstack([BASELINE, np.random.default_rng(1).uniform(0, 1, (15, 8))])
    c, g = sim.training_conditions()
    a = sim.evaluate_batch(G, c, g, 0.005, 60.0, 0)
    b = ss.evaluate_batch(G, c, g, 0.005, 60.0, 0, 1e12)
    assert np.array_equal(a, b[:, :, :sim.N_STATS])


def test_struct_energy_never_increases_still_air():
    G = np.random.default_rng(0).uniform(0, 1, (32, 8))
    c, g = sim.still_air()
    s = ss.evaluate_batch(G, c, g, 0.005, 60.0, 0, ss.SIGMA)
    assert (s[:, 0, 3] <= 1e-9).all()


def test_long_thin_wing_folds_on_hard_throw():
    long_thin = sim.phys_units_to_genes([0.50, 0.035, 0.05, 0.12, 0.018, 0.0, 3.0, 2.0])
    hard = np.array([[10.0, 9.0, 0.0, 0.08, 0.0, 1.0, 1.0, 0.0]])
    s = ss.evaluate_batch(np.vstack([long_thin, BASELINE]), hard, np.zeros((1, sim.N_GUST, 5)),
                          0.005, 60.0, 0, ss.SIGMA)
    assert s[0, 0, 2] == -2 and s[1, 0, 2] == 1
