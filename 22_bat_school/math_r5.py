"""Small NumPy reinforcement-learning math, independently testable."""
import numpy as np


def validation_due(update, elapsed, next_time, interval_updates=0):
    """Fixed-update selection for paired runs; retain timed selection by default."""
    if interval_updates < 0:
        raise ValueError("validation interval must be nonnegative")
    if interval_updates:
        return update == 1 or update % interval_updates == 0
    return elapsed >= next_time


def gae(reward, value, done, last, gamma=.99, lam=.95):
    advantage = np.zeros_like(reward)
    acc = np.zeros(reward.shape[1], np.float32)
    for t in range(len(reward)-1, -1, -1):
        nxt = last if t == len(reward)-1 else value[t+1]
        acc = reward[t]+gamma*nxt*(1-done[t])-value[t]+gamma*lam*(1-done[t])*acc
        advantage[t] = acc
    return advantage, advantage+value
