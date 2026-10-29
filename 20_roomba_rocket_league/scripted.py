"""Scripted robots. All return (n, 2) wheel commands in [-1, 1].

- goalie(arena):   training opponent. Guards the +x goal from a point between the ball and the goal
                   mouth; rushes the ball when it gets close to the goal.
- chaser(arena):   cross-play opponent, never seen in training. A striker: drives behind the ball and
                   pushes it toward the learner's (-x) goal.
- keeper(arena):   cross-play defender, never seen in training. Line keeper on its goal line.
- attacker(arena): baseline for the learner's seat. Gets behind the ball on the goal->ball line, then
                   drives through it toward the +x goal.
"""
import numpy as np
from env import W, GOAL_HALF, R_ROBOT, R_BALL, wrap


def drive_to(pos, th, target, gain=4.0, reverse_ok=True, slow_radius=0.05):
    """Point-to-point diff-drive controller. pos (n,2), th (n,), target (n,2)."""
    d = target - pos
    dist = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    err = wrap(ang - th)
    fwd = np.ones_like(dist)
    if reverse_ok:
        back = np.abs(err) > np.pi / 2
        fwd = np.where(back, -1.0, 1.0)
        err = np.where(back, wrap(err + np.pi), err)
    v = fwd * np.clip(dist / slow_radius, 0, 1) * np.clip(np.cos(err), 0, 1)
    w = np.clip(gain * err / np.pi, -1, 1)
    left = np.clip(v - w, -1, 1)
    right = np.clip(v + w, -1, 1)
    return np.stack([left, right], 1)


def _behind_ball(pos, th, ball, goal, standoff=0.13):
    """Shared striker logic: line up behind the ball relative to `goal`, then push through."""
    g = goal - ball
    gu = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    rb = ball - pos
    along = np.sum(rb * gu, 1)                           # >0: robot is behind the ball
    lateral = np.abs(rb[:, 0] * gu[:, 1] - rb[:, 1] * gu[:, 0])
    aligned = (along > 0.02) & (lateral < 0.05)
    behind_pt = ball - gu * standoff
    # if the robot is on the wrong side, swing around the ball sideways instead of through it
    side = np.sign(rb[:, 0] * gu[:, 1] - rb[:, 1] * gu[:, 0] + 1e-9)
    perp = np.stack([-gu[:, 1], gu[:, 0]], 1) * side[:, None]
    wrong = along < 0.0
    detour = ball - gu * 0.05 + perp * (R_ROBOT + R_BALL + 0.08)
    target = np.where(wrong[:, None], detour, behind_pt)
    target = np.where(aligned[:, None], ball + gu * 0.3, target)
    target[:, 0] = np.clip(target[:, 0], -W + R_ROBOT, W - R_ROBOT)
    target[:, 1] = np.clip(target[:, 1], -0.7 + R_ROBOT, 0.7 - R_ROBOT)
    return drive_to(pos, th, target, reverse_ok=False)


def attacker(arena, robot=0):
    goal = np.tile(np.array([W, 0.0]), (arena.n, 1))
    return _behind_ball(arena.pos[:, robot], arena.th[:, robot], arena.ball, goal)


def chaser(arena, robot=1):
    goal = np.tile(np.array([-W, 0.0]), (arena.n, 1))
    return _behind_ball(arena.pos[:, robot], arena.th[:, robot], arena.ball, goal)


def goalie(arena, robot=1):
    goal = np.array([W, 0.0])
    ball = arena.ball
    gb = ball - goal
    dist = np.linalg.norm(gb, axis=1, keepdims=True) + 1e-9
    guard = goal + gb / dist * 0.22
    guard[:, 0] = np.minimum(guard[:, 0], W - R_ROBOT)
    guard[:, 1] = np.clip(guard[:, 1], -GOAL_HALF - 0.05, GOAL_HALF + 0.05)
    rush = dist[:, 0] < 0.45
    target = np.where(rush[:, None], ball, guard)
    return drive_to(arena.pos[:, robot], arena.th[:, robot], target)


def keeper(arena, robot=1):
    """Cross-play defender #2 (unseen): a line keeper. Stays on its goal line, mirrors the ball's y
    (predicted 0.3 s ahead), never comes out."""
    pred = arena.ball + arena.bvel * 0.3
    target = np.stack([np.full(arena.n, W - R_ROBOT - 0.03),
                       np.clip(pred[:, 1], -GOAL_HALF, GOAL_HALF)], 1)
    return drive_to(arena.pos[:, robot], arena.th[:, robot], target)


def random_policy(arena, rng):
    return rng.uniform(-1, 1, (arena.n, 2))


OPPONENTS = {"goalie": goalie, "keeper": keeper, "chaser": chaser}
