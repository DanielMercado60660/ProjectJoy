"""Follow-up (round 3): a mixed pool of training defenders, to test whether opponent variety fixes the
cross-play collapse of run 4 (72.8% vs the training goalie, 0.2% vs the unseen line keeper).

Pool (each training episode draws one, uniformly):
  0 goalie       the original training opponent (kept, so "cost on the goalie" compares like with like)
  1 goalie_rand  randomized-parameter goalie family, resampled per episode:
                 guard standoff U[0.08, 0.40] m, rush radius U[0.25, 0.70] m, speed scale U[0.7, 1.0]
  2 chaser       the striker that attacks the learner's goal (was unseen in run 4; seen here)
Held out, never in the pool: keeper (line keeper on its goal line, mirrors predicted ball y, never
comes out). goalie_rand never reproduces it: it always rushes inside its rush radius and guards on
the ball-goal line, not on the goal line.
"""
import numpy as np
from env import Arena, W, GOAL_HALF, R_ROBOT, R_BALL
import scripted

POOL_NAMES = ["goalie", "goalie_rand", "chaser"]


def goalie_param(arena, standoff, rush_r, speed, robot=1):
    """scripted.goalie with per-env (n,) standoff / rush radius / speed scale."""
    goal = np.array([W, 0.0])
    gb = arena.ball - goal
    dist = np.linalg.norm(gb, axis=1, keepdims=True) + 1e-9
    guard = goal + gb / dist * standoff[:, None]
    guard[:, 0] = np.minimum(guard[:, 0], W - R_ROBOT)
    guard[:, 1] = np.clip(guard[:, 1], -GOAL_HALF - 0.05, GOAL_HALF + 0.05)
    rush = dist[:, 0] < rush_r
    target = np.where(rush[:, None], arena.ball, guard)
    return scripted.drive_to(arena.pos[:, robot], arena.th[:, robot], target) * speed[:, None]


class PoolArena(Arena):
    def __init__(self, n, seed=0, pool=(0, 1, 2)):
        self.pool = np.array(pool)
        self.opp_id = np.zeros(n, dtype=np.int64)
        self.g_standoff = np.full(n, 0.22)
        self.g_rush = np.full(n, 0.45)
        self.g_speed = np.ones(n)
        super().__init__(n, seed=seed, opponent="goalie")

    def reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        super().reset(mask)
        r = self.rng
        self.opp_id[mask] = r.choice(self.pool, k)
        self.g_standoff[mask] = r.uniform(0.08, 0.40, k)
        self.g_rush[mask] = r.uniform(0.25, 0.70, k)
        self.g_speed[mask] = r.uniform(0.7, 1.0, k)
        ch = mask & (self.opp_id == 2)      # chaser start, as in Arena._sample_starts
        m = int(ch.sum())
        if m:
            p1 = np.stack([r.uniform(0.4, 0.85, m), r.uniform(-0.4, 0.4, m)], 1)
            ball = self.ball[ch]
            d = p1 - ball
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            too = dist[:, 0] < R_ROBOT + R_BALL + 0.03
            p1[too] = ball[too] + d[too] / dist[too] * (R_ROBOT + R_BALL + 0.03)
            self.pos[ch, 1] = p1


def pool_opponent(env):
    acts = scripted.goalie(env)
    gr = env.opp_id == 1
    if gr.any():
        acts[gr] = goalie_param(env, env.g_standoff, env.g_rush, env.g_speed)[gr]
    ch = env.opp_id == 2
    if ch.any():
        acts[ch] = scripted.chaser(env)[ch]
    return acts
