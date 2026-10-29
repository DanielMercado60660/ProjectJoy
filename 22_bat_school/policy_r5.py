"""MLX-only recurrent joint sensing/flight policy; never imported by NumPy tests."""
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from model import BatPolicy
from env_r5 import OBS_DIM, REMAIN
from io_r5 import load_checkpoint


class Policy(BatPolicy):
    def __init__(self, hidden=96):
        super().__init__(OBS_DIM, hidden)
        self.move = nn.Linear(hidden, 5)

    def step(self, obs, h):
        assert obs.shape[-1] == OBS_DIM and h.shape[-1] == self.hidden
        move, pulse, value, h = super().step(obs, h)
        # A spent budget makes the pulse action deterministic, for all methods.
        pulse = mx.stack([pulse[:, 0], mx.where(obs[:, REMAIN] > 0, pulse[:, 1], -1e9)], -1)
        return move, pulse, value, h


def as_mlx(tree):
    return tree_map(lambda x: mx.array(x) if isinstance(x, np.ndarray) else x, tree)


def restore_mlx_rng(keys):
    if len(keys) != len(mx.random.state):
        raise ValueError("MLX RNG layout changed")
    for k, saved in enumerate(keys):
        # MLX 0.32 State supports mutable key contents, not item replacement.
        mx.random.state[k][:] = mx.array(saved)


def load_policy(path):
    saved = load_checkpoint(path)
    if saved["format"] != "bat-r5-v1":
        raise ValueError("unknown checkpoint format")
    policy = Policy(saved["config"]["hidden"])
    policy.update(as_mlx(saved["weights"]))
    mx.eval(policy.parameters())
    return policy, saved


class LearnedAgent:
    def __init__(self, model, n, every=False):
        self.model, self.every = model, every
        self.h = mx.zeros((n, model.hidden))

        @partial(mx.compile, inputs=[model.state])
        def step(obs, h):
            return model.step(obs, h)
        self.step = step

    def act(self, obs):
        m, p, _, self.h = self.step(mx.array(obs), self.h)
        move = np.asarray(mx.argmax(m, -1))
        pulse = np.ones(len(obs), bool) if self.every else np.asarray(mx.argmax(p, -1)).astype(bool)
        return move, pulse & (obs[:, REMAIN] > 0)
