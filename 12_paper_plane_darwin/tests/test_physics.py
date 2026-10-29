"""Physics sanity tests: no free energy in still air, step-size convergence, stall/stability basics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim  # noqa: E402
from evaluate import BASELINE  # noqa: E402


def test_energy_never_increases_still_air_random_designs():
    rng = np.random.default_rng(0)
    G = rng.uniform(0, 1, (64, 8))
    c, g = sim.still_air()
    s = sim.evaluate_batch(G, c, g, 0.005, 60.0, 0)
    assert (s[:, 0, 3] <= 1e-9).all()      # max per-step relative energy gain
    assert (s[:, 0, 4] < 0).all()          # net energy lost


def test_dt_halving_baseline():
    c, g = sim.training_conditions()
    a = sim.evaluate_batch(BASELINE[None], c, g, 0.005, 60.0, 0)[0, :, 0]
    b = sim.evaluate_batch(BASELINE[None], c, g, 0.0025, 60.0, 0)[0, :, 0]
    assert np.all(np.abs(a - b) <= 0.005 * np.abs(b) + 0.05)


def test_baseline_is_a_stable_glider():
    eig, st, conv = sim.glide_eigen(BASELINE)
    assert conv and eig.real.max() < 0
    assert -st[2] / st[3] > 4.0            # sensible glide ratio


def test_unstable_cg_crashes_short():
    u = [0.30, 0.06, 0.25, 0.18, 0.0, 0.0, 2.0, -1.0]   # no nose ballast -> CG behind neutral point
    g = sim.phys_units_to_genes(u)
    assert sim.describe(g)["static_margin"] < 0
    c, gu = sim.still_air()
    s, _ = sim.fly(g, c[0], gu[0])
    sb, _ = sim.fly(BASELINE, c[0], gu[0])
    assert s[0] < 0.5 * sb[0]
