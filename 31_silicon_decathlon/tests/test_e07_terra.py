"""Parity for event 07: our HNet / NumPy / torch copies, cost and A* vs the sibling's own terra package.

Verification only: this is the one place the sibling's code is imported.
"""
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from models import terra as T  # noqa: E402

SIB = os.path.join(HERE, "..", "07_terra")


def _sib():
    if SIB not in sys.path:
        sys.path.append(SIB)
    from terra import cost, learn, planners, terrain
    return cost, learn, planners, terrain


def test_mlx_copy_matches_sibling():
    _, learn, _, _ = _sib()
    ref = learn.HNet(base=T.config()["base"])
    ref.load_weights(T.CKPT + ".safetensors")
    ref.eval()
    ours = T.load_mlx()
    x = T.sample_inputs([0, 5])
    with mx.stream(mx.cpu):
        a, b = np.array(ref(mx.array(x))), np.array(ours(mx.array(x)))
    assert np.abs(a - b).max() < 1e-6


def test_features_match_sibling():
    """Our goal channels equal the sibling's; the map reconstruction gives back the generator's kind."""
    cost, learn, _, terrain = _sib()
    v = T.val()
    gr, gc = v["goal"][3]
    np.testing.assert_allclose(T.goal_features(96, 96, gr, gc), learn.goal_features(96, 96, gr, gc), atol=1e-7)
    seed = 500000 + int(v["map_index"][3])   # val maps are seeds 500000..
    gen = terrain.generate(seed, 96)
    elev, kind = T.map_of(3)
    np.testing.assert_array_equal(kind, gen["kind"])
    np.testing.assert_allclose(elev, gen["elev"], atol=1e-3)  # stored as float16


def test_numpy_and_torch_match_mlx_cpu():
    import torch
    ours = T.load_mlx()
    x = T.sample_inputs([1, 2])
    with mx.stream(mx.cpu):
        a = np.array(ours(mx.array(x)))
    b = T.NumpyHNet()(x)
    with torch.no_grad():
        c = T.torch_net()(torch.from_numpy(x)).numpy()
    assert np.abs(a - b).max() < 1e-4
    assert np.abs(a - c).max() < 1e-4


def test_cost_and_astar_match_sibling():
    cost, _, planners, _ = _sib()
    elev, kind = T.map_of(7)
    ec = T.edge_costs(elev, kind)
    np.testing.assert_array_equal(ec, cost.edge_costs(elev, kind))
    G = cost.Graph(elev, kind)
    adj = T.adjacency(elev, kind)
    v = T.val()
    gr, gc = v["goal"][7]
    g = int(gr * 96 + gc)
    s = int(np.flatnonzero(kind.ravel() != T.WATER)[100])
    h = T.octile_field(96, 96, gr, gc).ravel().tolist()
    r = planners.astar(G, s, g, h)
    c, e = T.astar(adj, s, g, h)
    assert abs(c - r.cost) < 1e-9 and e == r.expanded
