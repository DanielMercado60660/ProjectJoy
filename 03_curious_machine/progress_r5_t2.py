"""NumPy state for the Turn-2 error-model queue and adaptive reward scale."""
import numpy as np


class ErrorQueue:
    def __init__(self, capacity, obs_dim):
        self.capacity, self.obs_dim = capacity, obs_dim
        self.obs = np.zeros((capacity, (obs_dim + 7) // 8), np.uint8)
        self.actions = np.zeros(capacity, np.int32)
        self.targets = np.zeros(capacity, np.float32)
        self.count = self.cursor = 0

    def add(self, obs, actions, targets):
        assert obs.ndim == 2 and obs.shape[1] == self.obs_dim
        assert len(obs) == len(actions) == len(targets)
        # Binary observations pack losslessly; this queue cannot accept normalized floats.
        assert np.all((obs == 0) | (obs == 1))
        if len(obs) > self.capacity:
            obs, actions, targets = obs[-self.capacity:], actions[-self.capacity:], targets[-self.capacity:]
        ids = (np.arange(len(obs)) + self.cursor) % self.capacity
        self.obs[ids] = np.packbits(obs.astype(np.uint8), axis=1)
        self.actions[ids] = actions; self.targets[ids] = targets
        self.cursor = int((self.cursor + len(obs)) % self.capacity)
        self.count = min(self.capacity, self.count + len(obs))

    def sample(self, rng, n):
        if not self.count:
            raise ValueError('Cannot sample empty error queue')
        ids = rng.integers(self.count, size=n)
        return (np.unpackbits(self.obs[ids], axis=1, count=self.obs_dim).astype(np.float32),
                self.actions[ids], self.targets[ids])

    def state(self):
        return dict(self.__dict__)

    def load(self, state):
        assert state['capacity'] == self.capacity and state['obs_dim'] == self.obs_dim
        self.__dict__.update(state)


class RewardScale:
    """EMA variance with a floor; do not center signed rewards or erase their sign."""
    def __init__(self, decay=.95, floor=.1):
        self.decay, self.floor = decay, floor
        self.variance = None

    def normalize(self, reward, valid):
        r = np.asarray(reward, np.float32); valid = np.asarray(valid, bool)
        variance = float(np.var(r[valid])) if valid.any() else 0.
        self.variance = variance if self.variance is None else self.decay * self.variance + (1-self.decay) * variance
        scale = max(self.floor, float(np.sqrt(max(0., self.variance) + 1e-12)))
        return (np.clip(r / scale, -5., 5.) * valid).astype(np.float32), scale
