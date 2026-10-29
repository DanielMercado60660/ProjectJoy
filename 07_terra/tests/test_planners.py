import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terra import terrain, cost, planners  # noqa: E402
from terra.terrain import WATER, BRIDGE, GRASS, ROCK, FOREST  # noqa: E402

SIZE = 40


def rand_passable(G, rng):
    while True:
        u = rng.randrange(G.n)
        if G.adj[u]:
            return u


@pytest.mark.parametrize("seed", range(6))
def test_terrain_basics(seed):
    T = terrain.generate(seed, SIZE)
    assert T["elev"].shape == (SIZE, SIZE)
    assert (T["kind"] == WATER).any()
    for br in T["bridges"]:
        for r, c in br:
            assert T["kind"][r, c] == BRIDGE
    T2 = terrain.generate(seed, SIZE)
    assert np.array_equal(T["kind"], T2["kind"]) and np.array_equal(T["elev"], T2["elev"])


def test_rivers_flow_downhill():
    elev = terrain.fbm(np.random.default_rng(0), 32, 32)
    filled = terrain.priority_flood(elev)
    assert (filled >= elev - 1e-12).all()
    recv, acc = terrain.flow_accumulation(filled)
    h, w = elev.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            br, bc = recv[r, c]
            assert br >= 0, "interior cell of filled DEM must have a downhill receiver"
            assert filled[br, bc] < filled[r, c]


@pytest.mark.parametrize("seed", range(4))
def test_cost_symmetric_and_walls(seed):
    T = terrain.generate(seed, SIZE)
    G = cost.Graph(T["elev"], T["kind"])
    kind = T["kind"].ravel()
    w = G.w
    for u in range(G.n):
        for v, c in G.adj[u]:
            assert kind[u] != WATER and kind[v] != WATER
            back = [cc for vv, cc in G.adj[v] if vv == u]
            assert len(back) == 1 and back[0] == c
            ur, uc = divmod(u, w); vr, vc = divmod(v, w)
            d = math.hypot(ur - vr, uc - vc)
            assert c >= d * cost.MIN_UNIT_COST - 1e-12
            if ur != vr and uc != vc:  # no corner cutting
                assert kind[ur * w + vc] != WATER and kind[vr * w + uc] != WATER


@pytest.mark.parametrize("seed", range(8))
def test_astar_matches_dijkstra_and_admissible(seed):
    T = terrain.generate(seed, SIZE)
    G = cost.Graph(T["elev"], T["kind"])
    rng = random.Random(seed)
    for _ in range(5):
        s, g = rand_passable(G, rng), rand_passable(G, rng)
        gr, gc = divmod(g, G.w)
        h = cost.octile_field(G.h, G.w, gr, gc).ravel()
        d = planners.dijkstra(G, s, g)
        a = planners.astar(G, s, g, h)
        assert (d.cost == a.cost) or abs(d.cost - a.cost) < 1e-9
        if d.path:
            assert abs(cost.path_cost(G, a.path) - a.cost) < 1e-9
            assert a.expanded <= d.expanded
        # admissibility: h <= true cost-to-go everywhere
        dist = np.array(planners.dijkstra_all(G, g))
        fin = np.isfinite(dist)
        assert (h[fin] <= dist[fin] + 1e-9).all()


def test_scipy_oracle_agrees():
    """Independent check of dijkstra_all against scipy's csgraph on the vectorized costs."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    T = terrain.generate(11, SIZE)
    G = cost.Graph(T["elev"], T["kind"])
    rows, cols, vals = [], [], []
    for u in range(G.n):
        for v, c in G.adj[u]:
            rows.append(u); cols.append(v); vals.append(c)
    M = csr_matrix((vals, (rows, cols)), shape=(G.n, G.n))
    src = rand_passable(G, random.Random(1))
    ref = sp_dijkstra(M, indices=src)
    mine = np.array(planners.dijkstra_all(G, src))
    assert np.allclose(np.where(np.isfinite(ref), ref, -1), np.where(np.isfinite(mine), mine, -1))


@pytest.mark.parametrize("seed", range(10))
def test_dstar_lite_matches_dijkstra_under_changes(seed):
    """Agent walks along the D* Lite path; random cells flip (walls, forests, bridges vanish,
    new bridges appear) between moves. After every replan, D* Lite's cost-to-goal must equal
    Dijkstra from scratch on the modified map, and its extracted path must achieve that cost."""
    T = terrain.generate(100 + seed, SIZE)
    G = cost.Graph(T["elev"], T["kind"])
    rng = random.Random(seed)
    s, goal = rand_passable(G, rng), rand_passable(G, rng)
    D = planners.DStarLite(G, s, goal)
    D.compute()
    ref = planners.dijkstra(G, s, goal)
    assert abs(D.cost_to_goal - ref.cost) < 1e-9 or D.cost_to_goal == ref.cost
    pos = s
    for step in range(25):
        # move up to 3 steps along current path
        p = D.path()
        if p and len(p) > 1:
            pos = p[min(len(p) - 1, rng.randint(1, 3))]
            D.move_to(pos)
        if pos == goal:
            break
        # random terrain edits (never on the agent or goal)
        cells = []
        for _ in range(rng.randint(1, 6)):
            r, c = rng.randrange(G.h), rng.randrange(G.w)
            if r * G.w + c not in (pos, goal):
                cells.append((r, c))
        if not cells:
            continue
        newkind = rng.choice([WATER, WATER, GRASS, ROCK, FOREST, BRIDGE])
        changed = G.set_cells(cells, kind_value=newkind)
        if rng.random() < 0.3:  # also change elevation sometimes
            r, c = cells[0]
            changed |= G.set_cells([(r, c)], elev_value=rng.random())
        D.update_vertices(changed)
        D.compute()
        ref = planners.dijkstra(G, pos, goal)
        got = D.cost_to_goal
        if math.isinf(ref.cost):
            assert math.isinf(got)
            assert D.path() == []
        else:
            assert abs(got - ref.cost) < 1e-6, (step, got, ref.cost)
            path = D.path()
            assert path[0] == pos and path[-1] == goal
            assert abs(cost.path_cost(G, path) - ref.cost) < 1e-6


def test_local_rebuild_matches_full_rebuild():
    T = terrain.generate(5, SIZE)
    G = cost.Graph(T["elev"], T["kind"])
    rng = random.Random(0)
    for _ in range(20):
        cells = [(rng.randrange(SIZE), rng.randrange(SIZE)) for _ in range(3)]
        G.set_cells(cells, kind_value=rng.choice([WATER, GRASS, ROCK, BRIDGE]))
    H = cost.Graph(G.elev, G.kind)
    for u in range(G.n):
        assert sorted(G.adj[u]) == sorted(H.adj[u])
