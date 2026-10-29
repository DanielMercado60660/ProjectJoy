"""Parity: our copies of the Grand Prix policy (numpy + MLX) and of the racing sim against the
sibling project's own code (verification only; event/model modules never import sibling code)."""
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB = os.path.join(HERE, "..", "08_tiny_grand_prix")
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import grand_prix as gp  # noqa: E402
from models import grand_prix_sim as sim  # noqa: E402


def _sibling():
    if SIB not in sys.path:
        sys.path.insert(0, SIB)
    from racing import evaluate, policy, track  # noqa: E402
    return evaluate, policy, track


def test_policy_parity():
    _, policy, _ = _sibling()
    theta, _ = policy.load_policy(os.path.join(SIB, "checkpoints", "es_best.npz"))
    obs = np.random.default_rng(0).uniform(-1, 1, (512, 13)).astype(np.float32)
    ref = policy.forward(theta, obs)
    assert np.abs(gp.NumpyPolicy()(obs) - ref).max() < 1e-6
    m = gp.load_mlx()
    with mx.stream(mx.cpu):
        assert np.abs(np.array(m(mx.array(obs))) - ref).max() < 1e-5
    for i in range(16):  # MLX GPU at batch 1
        assert np.abs(np.array(m(mx.array(obs[i:i + 1])))[0] - ref[i]).max() < 1e-5


def test_sim_parity():
    evaluate, policy, track = _sibling()
    theta, _ = policy.load_policy(os.path.join(SIB, "checkpoints", "es_best.npz"))
    ctrl = policy.MLPController(theta)
    ref = evaluate.run_lap(ctrl, track.pack(track.make_tracks("heldout", 6)), mode="strict")
    ours = sim.run_lap_strict(gp.NumpyPolicy(), sim.pack(sim.make_tracks("heldout", 6)))
    assert (ref["finished"] == ours["finished"]).all()
    np.testing.assert_array_equal(ref["lap_time"], ours["lap_time"])
