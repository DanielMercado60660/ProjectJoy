"""Stress test: D* Lite vs Dijkstra-from-scratch on 300 maps (48x48), up to 40 replans each,
with terrain edits concentrated near the current path. Prints the number of mismatches.

  ../.venv/bin/python stress_dstar.py        # ~30 s; RESULTS.md: 4610 checks, 0 mismatches
"""
import math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from terra import terrain, cost, planners
from terra.terrain import WATER, GRASS, ROCK, FOREST, BRIDGE


def rand_passable(G, rng):
    while True:
        u = rng.randrange(G.n)
        if G.adj[u]:
            return u


bad = checks = unreachable = 0
for seed in range(300):
    T = terrain.generate(1000 + seed, 48)
    G = cost.Graph(T["elev"], T["kind"])
    rng = random.Random(seed)
    s, goal = rand_passable(G, rng), rand_passable(G, rng)
    D = planners.DStarLite(G, s, goal)
    D.compute()
    pos = s
    for step in range(40):
        p = D.path()
        if p and len(p) > 1:
            pos = p[min(len(p) - 1, rng.randint(1, 3))]
            D.move_to(pos)
        if pos == goal:
            break
        if p and rng.random() < 0.7:  # edits near the path, so they matter
            q = p[rng.randrange(len(p))]
            r0, c0 = divmod(q, G.w)
            cells = [(min(G.h - 1, max(0, r0 + rng.randint(-2, 2))), min(G.w - 1, max(0, c0 + rng.randint(-2, 2))))
                     for _ in range(rng.randint(1, 5))]
        else:
            cells = [(rng.randrange(G.h), rng.randrange(G.w)) for _ in range(3)]
        cells = [x for x in cells if x[0] * G.w + x[1] not in (pos, goal)]
        if not cells:
            continue
        ch = G.set_cells(cells, kind_value=rng.choice([WATER, WATER, GRASS, ROCK, FOREST, BRIDGE]))
        D.update_vertices(ch)
        D.compute()
        ref = planners.dijkstra(G, pos, goal)
        got = D.cost_to_goal
        checks += 1
        if math.isinf(ref.cost):
            unreachable += 1
            ok = math.isinf(got)
        else:
            ok = abs(got - ref.cost) < 1e-6 and abs(cost.path_cost(G, D.path()) - ref.cost) < 1e-6
        if not ok:
            bad += 1
            print("MISMATCH", seed, step, got, ref.cost)
print(f"checks {checks}  mismatches {bad}  (goal unreachable in {unreachable})")
