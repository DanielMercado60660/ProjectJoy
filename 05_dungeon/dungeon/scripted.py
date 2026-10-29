"""Scripted solvers.

Partial-observation baselines (same V x V window as the learned agent, same step limit):
  random_walk  : uniform random legal move (no memory)
  wall_follow  : left-hand rule (keeps its own heading)
  tremaux      : depth-first exploration with a perfect memory of visited cells
                 (a "random walk with memory": unvisited neighbour first, else backtrack)
All three switch to a shortest path through *seen* floor once the goal has been seen.

Full-map reference:
  oracle       : BFS shortest path (the optimal path length).
"""
import numpy as np
from numba import njit

from .env import R, MAX_STEPS
from .mazes import bfs_dist

DR = np.array([-1, 0, 1, 0], np.int64)
DC = np.array([0, 1, 0, -1], np.int64)
ALGOS = ("random_walk", "wall_follow", "tremaux")


@njit(cache=True)
def _reveal(known, walls, r, c):
    H, W = walls.shape
    for i in range(r - R, r + R + 1):
        for j in range(c - R, c + R + 1):
            if 0 <= i < H and 0 <= j < W:
                known[i, j] = 1 + walls[i, j]  # 1 floor, 2 wall


@njit(cache=True)
def _step_towards(known, r, c, gr, gc):
    """First move of a shortest path through known floor (known==1); -1 if none."""
    H, W = known.shape
    kw = np.ones((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            if known[i, j] == 1:
                kw[i, j] = 0
    d = bfs_dist(kw, gr, gc)
    if d[r, c] < 0:
        return -1
    for k in range(4):
        nr, nc = r + DR[k], c + DC[k]
        if 0 <= nr < H and 0 <= nc < W and d[nr, nc] == d[r, c] - 1 and kw[nr, nc] == 0:
            return k
    return -1


@njit(cache=True)
def run_scripted(algo, walls, sr, sc, gr, gc, T, seed, traj):
    """algo: 0 random_walk, 1 wall_follow, 2 tremaux. Writes positions into traj (T+1, 2).
    Returns (success, steps)."""
    np.random.seed(seed)
    H, W = walls.shape
    known = np.zeros((H, W), np.uint8)
    visited = np.zeros((H, W), np.uint8)
    stack_r = np.empty(H * W * 4, np.int64)
    stack_c = np.empty(H * W * 4, np.int64)
    sp = 0
    r, c = sr, sc
    heading = np.random.randint(4)
    goal_seen = False
    traj[0, 0], traj[0, 1] = r, c
    visited[r, c] = 1
    for t in range(1, T + 1):
        _reveal(known, walls, r, c)
        if abs(gr - r) <= R and abs(gc - c) <= R:
            goal_seen = True
        a = -1
        if goal_seen:
            a = _step_towards(known, r, c, gr, gc)
        if a < 0:
            if algo == 0:
                legal = np.empty(4, np.int64)
                nl = 0
                for k in range(4):
                    if walls[r + DR[k], c + DC[k]] == 0:
                        legal[nl] = k
                        nl += 1
                a = legal[np.random.randint(nl)] if nl > 0 else np.random.randint(4)
            elif algo == 1:
                for turn in (3, 0, 1, 2):  # left, straight, right, back
                    k = (heading + turn) % 4
                    if walls[r + DR[k], c + DC[k]] == 0:
                        a = k
                        break
                if a < 0:
                    a = heading
            else:
                order = np.random.permutation(4)
                for q in range(4):
                    k = order[q]
                    nr, nc = r + DR[k], c + DC[k]
                    if walls[nr, nc] == 0 and visited[nr, nc] == 0:
                        a = k
                        stack_r[sp], stack_c[sp] = r, c
                        sp += 1
                        break
                if a < 0:
                    if sp > 0:
                        sp -= 1
                        pr, pc = stack_r[sp], stack_c[sp]
                        for k in range(4):
                            if r + DR[k] == pr and c + DC[k] == pc:
                                a = k
                    else:
                        a = np.random.randint(4)
        nr, nc = r + DR[a], c + DC[a]
        if walls[nr, nc] == 0:
            r, c = nr, nc
            heading = a
        visited[r, c] = 1
        traj[t, 0], traj[t, 1] = r, c
        if r == gr and c == gc:
            return True, t
    return False, T


def run_batch(algo, levels, T=MAX_STEPS, seed=0, keep_traj=False):
    k = ALGOS.index(algo) if isinstance(algo, str) else algo
    n = len(levels["walls"])
    succ = np.zeros(n, bool)
    steps = np.zeros(n, np.int64)
    trajs = []
    for i in range(n):
        tr = np.zeros((T + 1, 2), np.int64)
        s, g = levels["start"][i], levels["goal"][i]
        succ[i], steps[i] = run_scripted(k, levels["walls"][i], s[0], s[1], g[0], g[1], T, seed * 100003 + i, tr)
        if keep_traj:
            trajs.append(tr[:steps[i] + 1].copy())
    out = dict(success=succ, steps=steps)
    if keep_traj:
        out["traj"] = trajs
    return out


def oracle_lengths(levels):
    return np.array([bfs_dist(w, s[0], s[1])[g[0], g[1]]
                     for w, s, g in zip(levels["walls"], levels["start"], levels["goal"])], np.int64)


def summarize(success, steps, opt):
    """success rate, path efficiency (opt/steps over successes) and SPL (Anderson et al. 2018)."""
    success = np.asarray(success, bool)
    steps = np.asarray(steps, float)
    opt = np.asarray(opt, float)
    eff = opt / np.maximum(steps, opt)
    return dict(
        success=float(success.mean()),
        efficiency=float(eff[success].mean()) if success.any() else 0.0,
        spl=float((success * eff).mean()),
        mean_steps=float(steps.mean()),
        n=int(len(success)),
    )
