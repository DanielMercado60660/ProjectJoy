"""Scenarios. Bridge collapse: the agent follows its optimal route; when it is `warn` steps
before the first bridge on that route, the bridge disappears (turns to water). D* Lite repairs
its search tree incrementally; A* (and Dijkstra) replan from scratch from the agent's position.
"""
import copy
import gc as _gc
import math
import time
import numpy as np

from . import cost, planners, terrain
from .terrain import WATER


def pick_bridge_query(T, G, rng, min_dist=35, tries=400):
    """Random start/goal whose optimal path uses a bridge. Returns (s, g, ref) or None."""
    kind = T["kind"].ravel()
    passable = np.flatnonzero(kind != WATER)
    bridge_cells = {r * G.w + c: i for i, b in enumerate(T["bridges"]) for r, c in b}
    for _ in range(tries):
        s, g = int(rng.choice(passable)), int(rng.choice(passable))
        if G.h_octile(s, g) < min_dist:
            continue
        ref = planners.dijkstra(G, s, g)
        if not ref.path:
            continue
        on = [k for k, u in enumerate(ref.path) if u in bridge_cells]
        if on and on[0] > 8:
            return s, g, ref, bridge_cells[ref.path[on[0]]], on[0]
    return None


def _best_of(fn, reps):
    _gc.disable()
    best, res = math.inf, None
    for _ in range(reps):
        res = fn()
        best = min(best, res.ms)
    _gc.enable()
    res.ms = best
    return res


def bridge_collapse(T, s, g, warn=6, record=False, reps=3):
    """Run the scenario for D* Lite and A*-from-scratch on identical maps. Both agents walk
    the same route (D* Lite's initial optimal path) so they replan from the same cell.
    Returns a dict, or None if the route crosses no bridge."""
    out = {}
    # --- D* Lite ---------------------------------------------------------------------
    G = cost.Graph(T["elev"], T["kind"])
    D = planners.DStarLite(G, s, g)
    e0, ms0 = D.compute()
    path0 = D.path()
    bridge_cells = {r * G.w + c: i for i, b in enumerate(T["bridges"]) for r, c in b}
    on = [k for k, u in enumerate(path0) if u in bridge_cells]
    if not on:
        return None
    first_bridge_step, bridge_id = on[0], bridge_cells[path0[on[0]]]
    collapse_at = max(0, first_bridge_step - warn)
    pos = s
    traveled = 0.0
    traj = [pos]
    for k in range(1, collapse_at + 1):
        nxt = path0[k]
        traveled += dict(G.adj[pos])[nxt]
        pos = nxt
        traj.append(pos)
    D.move_to(pos)
    changed = G.set_cells(T["bridges"][bridge_id], kind_value=WATER)
    # replan latency = update_vertices + compute; best of `reps` runs on copies, GC off
    best = math.inf
    _gc.disable()
    for _ in range(reps):
        Dc = copy.copy(D)
        Dc.g, Dc.rhs, Dc.inU, Dc.U = list(D.g), list(D.rhs), list(D.inU), list(D.U)
        t0 = time.perf_counter()
        Dc.update_vertices(changed)
        e1, _ = Dc.compute()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    _gc.enable()
    D, ms1 = Dc, best
    path1 = D.path()
    remaining = D.cost_to_goal
    out["dstar"] = dict(init_expanded=e0, init_ms=ms0, replan_expanded=e1, replan_ms=ms1,
                        travel_cost=traveled + remaining)
    # --- A* and Dijkstra from scratch on the modified map ----------------------------
    gr, gc = divmod(g, G.w)
    hfield = cost.octile_field(G.h, G.w, gr, gc).ravel().tolist()
    G0 = cost.Graph(T["elev"], T["kind"])
    a0 = _best_of(lambda: planners.astar(G0, s, g, hfield), reps)
    a1 = _best_of(lambda: planners.astar(G, pos, g, hfield), reps)
    d0 = _best_of(lambda: planners.dijkstra(G0, s, g), reps)
    d1 = _best_of(lambda: planners.dijkstra(G, pos, g), reps)
    out["astar"] = dict(init_expanded=a0.expanded, init_ms=a0.ms, replan_expanded=a1.expanded,
                        replan_ms=a1.ms, travel_cost=traveled + a1.cost)
    out["dijkstra"] = dict(init_expanded=d0.expanded, init_ms=d0.ms, replan_expanded=d1.expanded,
                           replan_ms=d1.ms, travel_cost=traveled + d1.cost)
    # omniscient: knew the bridge was gone from the start
    omni = planners.dijkstra(G, s, g)
    out["omniscient_cost"] = omni.cost
    out["planned_cost"] = a0.cost
    out["failed"] = math.isinf(remaining)
    out["bridge"] = bridge_id
    out["consistent"] = (math.isinf(remaining) and math.isinf(a1.cost)) or abs(remaining - a1.cost) < 1e-6
    if record:
        out["traj"] = traj
        out["path0"] = path0
        out["path1"] = path1
        out["omni_path"] = omni.path
        out["collapse_pos"] = pos
    return out


def run_bridge_benchmark(seeds, size=96, warn=6, per_map=2, seed=0, reps=3):
    rng = np.random.default_rng(seed)
    rows = []
    for ms in seeds:
        T = terrain.generate(ms, size)
        if not T["bridges"]:
            continue
        G = cost.Graph(T["elev"], T["kind"])
        for _ in range(per_map):
            q = pick_bridge_query(T, G, rng)
            if q is None:
                continue
            s, g = q[0], q[1]
            r = bridge_collapse(T, s, g, warn, reps=reps)
            if r is None:
                continue
            r.update(map_seed=int(ms), s=s, g=g)
            rows.append(r)
    return rows
