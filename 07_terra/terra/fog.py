"""Partial information ("fog"). The agent knows the elevation map (a topo map) but not the
terrain kind: water, bridges, forest and rock are revealed only within sensing radius R of
where it has been. Unknown cells get a *belief* multiplier from a prior:

  optimistic : every unknown cell is grass (free-space assumption, the classic D* Lite setup)
  learned    : MLX U-Net that sees only the elevation map and predicts P(water) and the
               expected land multiplier; unknown cost = (1-p)*m_land + p*WATER_SOFT
  hydrology  : the generator's own river/lake model run on the known elevation (a hand-built
               domain prior that happens to be *exactly* how the maps were made, so it is an
               upper reference, not a fair competitor)

Predicted water is "soft" (expensive, WATER_SOFT) instead of impassable, so a prior can never
make the goal look unreachable. The real world always uses the true cost definition: travel
cost is measured with cost.Graph on the true map. The agent replans with D* Lite on the belief
map; for compute comparison, A* re-plans from scratch on the same belief after each change.
"""
import math
import time
import numpy as np

from . import cost, planners, terrain

WATER_SOFT = 12.0


def prior_optimistic(elev):
    return np.ones(elev.shape)


def prior_hydrology(elev):
    river, lake, _, _ = terrain.hydrology(elev)
    return np.where(river | lake, WATER_SOFT, 1.0)


def prior_from_probs(p_water, m_land):
    return (1 - p_water) * np.maximum(m_land, 1.0) + p_water * WATER_SOFT


def disk_offsets(R):
    return [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1) if dr * dr + dc * dc <= R * R]


def run_fog(T, s, g, prior, R=5, compare_astar=True, max_steps=None, record=False):
    """Walk from s to g under fog. Returns metrics dict."""
    elev, kind = T["elev"], T["kind"]
    h, w = kind.shape
    true_mult = cost.mult_map(kind)
    Gt = cost.Graph(elev, kind)
    belief = np.asarray(prior, np.float64).copy()
    known = np.zeros((h, w), bool)
    Gb = cost.Graph(elev, kind, belief)  # kind only matters through mult here
    offs = disk_offsets(R)
    hfield = cost.octile_field(h, w, *divmod(g, w)).ravel().tolist()

    def sense(u):
        r0, c0 = divmod(u, w)
        cells, vals = [], []
        for dr, dc in offs:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < h and 0 <= c < w and not known[r, c]:
                known[r, c] = True
                if Gb.mult[r, c] != true_mult[r, c]:
                    cells.append((r, c)); vals.append(true_mult[r, c])
        newly = len(offs)  # (not used)
        if cells:
            return Gb.set_cells(cells, mult_values=vals)
        return set()

    sense(s)
    D = planners.DStarLite(Gb, s, g)
    e, ms = D.compute()
    stats = dict(replans=0, dstar_expanded=e, dstar_ms=ms, astar_expanded=0, astar_ms=0.0)
    if compare_astar:
        a = planners.astar(Gb, s, g, hfield)
        stats["astar_expanded"] += a.expanded; stats["astar_ms"] += a.ms
    pos, travel, steps = s, 0.0, 0
    traj = [s]
    max_steps = max_steps or 6 * h * w
    failed = False
    while pos != g:
        p = D.path(max_len=h * w)
        if len(p) < 2 or steps >= max_steps:
            failed = True
            break
        nxt = p[1]
        c = dict(Gt.adj[pos]).get(nxt)
        if c is None:  # cannot happen: the next edge is always inside the sensed disk
            failed = True
            break
        travel += c
        pos = nxt; steps += 1; traj.append(pos)
        D.move_to(pos)
        changed = sense(pos)
        if changed:
            stats["replans"] += 1
            t0 = time.perf_counter()
            D.update_vertices(changed)
            e, _ = D.compute()
            stats["dstar_ms"] += (time.perf_counter() - t0) * 1e3
            stats["dstar_expanded"] += e
            if compare_astar:
                a = planners.astar(Gb, pos, g, hfield)
                stats["astar_expanded"] += a.expanded; stats["astar_ms"] += a.ms
    out = dict(travel_cost=travel if not failed else math.inf, steps=steps, failed=failed,
               observed_cells=int(known.sum()), **stats)
    if record:
        out["traj"] = traj
        out["known"] = known
    return out
