"""Run the learned policy on a fixed batch of levels (one episode per level)."""
import numpy as np

from .env import VecMaze, MAX_STEPS
from .policy import PolicyRunner


def run_policy(policy, levels, greedy=False, seed=0, keep_traj=False, chunk=1024, T=MAX_STEPS):
    n = len(levels["walls"])
    succ = np.zeros(n, bool)
    steps = np.zeros(n, np.int64)
    trajs = [None] * n
    for lo in range(0, n, chunk):
        idx = np.arange(lo, min(n, lo + chunk))
        m = len(idx)
        env = VecMaze(m, sampler=None, seed=seed, max_steps=T, auto_reset=False)
        obs = env.reset({k: levels[k][idx] for k in ("walls", "start", "goal")})
        runner = PolicyRunner(policy, m, greedy=greedy, seed=seed * 7919 + lo)
        tr = np.zeros((T + 1, m, 2), np.int64)
        tr[0] = env.pos
        for t in range(T):
            a, _, _ = runner.act(obs)
            obs, _, done, info = env.step(a)
            tr[t + 1] = env.pos
            fin = done
            succ[idx[fin]] = info["success"][fin]
            steps[idx[fin]] = info["steps"][fin]
            if env.done.all():
                break
        if keep_traj:
            for j in range(m):
                trajs[idx[j]] = tr[:steps[idx[j]] + 1, j].copy()
    out = dict(success=succ, steps=steps)
    if keep_traj:
        out["traj"] = trajs
    return out
