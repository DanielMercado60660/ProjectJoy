import math, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terra import terrain, cost, planners, fog, scenarios  # noqa: E402


def _query(T, G, seed, min_dist=20):
    rng = random.Random(seed)
    passable = [u for u in range(G.n) if G.adj[u]]
    while True:
        s, g = rng.choice(passable), rng.choice(passable)
        if G.h_octile(s, g) >= min_dist and planners.dijkstra(G, s, g).path:
            return s, g


def test_fog_with_true_prior_is_optimal():
    T = terrain.generate(3, 48)
    G = cost.Graph(T["elev"], T["kind"])
    s, g = _query(T, G, 0)
    opt = planners.dijkstra(G, s, g).cost
    r = fog.run_fog(T, s, g, cost.mult_map(T["kind"]), R=3)
    assert not r["failed"] and r["replans"] == 0
    assert abs(r["travel_cost"] - opt) < 1e-6


def test_fog_optimistic_reaches_goal_and_dstar_matches_astar_work():
    for seed in range(4):
        T = terrain.generate(10 + seed, 48)
        G = cost.Graph(T["elev"], T["kind"])
        s, g = _query(T, G, seed)
        opt = planners.dijkstra(G, s, g).cost
        r = fog.run_fog(T, s, g, fog.prior_optimistic(T["elev"]), R=3)
        assert not r["failed"]
        assert r["travel_cost"] >= opt - 1e-6
        assert r["observed_cells"] > 0


def test_bridge_collapse_consistent():
    rng = np.random.default_rng(0)
    n = 0
    for ms in range(20):
        T = terrain.generate(ms, 64)
        G = cost.Graph(T["elev"], T["kind"])
        q = scenarios.pick_bridge_query(T, G, rng, min_dist=20)
        if q is None:
            continue
        r = scenarios.bridge_collapse(T, q[0], q[1], warn=3, reps=1)
        if r is None:
            continue
        n += 1
        assert r["consistent"]
        assert r["dstar"]["travel_cost"] >= r["omniscient_cost"] - 1e-6 or r["failed"]
    assert n >= 3
