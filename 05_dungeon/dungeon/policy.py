"""Recurrent actor-critic in MLX: MLP encoder -> GRU cell -> policy / value heads."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .env import OBS_DIM


class GRUCell(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.x2h = nn.Linear(d_in, 3 * d_h)
        self.h2h = nn.Linear(d_h, 3 * d_h)
        self.d_h = d_h

    def __call__(self, x, h):
        xr, xz, xn = mx.split(self.x2h(x), 3, axis=-1)
        hr, hz, hn = mx.split(self.h2h(h), 3, axis=-1)
        r = mx.sigmoid(xr + hr)
        z = mx.sigmoid(xz + hz)
        n = mx.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class RecurrentPolicy(nn.Module):
    def __init__(self, hidden=128, memoryless=False):
        super().__init__()
        self.hidden = hidden
        self.memoryless = memoryless  # ablation: GRU state is zeroed before every step
        self.enc1 = nn.Linear(OBS_DIM, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.gru = GRUCell(hidden, hidden)
        self.pi1 = nn.Linear(hidden, hidden)
        self.pi2 = nn.Linear(hidden, 4)
        self.v1 = nn.Linear(hidden, hidden)
        self.v2 = nn.Linear(hidden, 1)

    def initial_state(self, n):
        return mx.zeros((n, self.hidden))

    def step(self, obs, h):
        if self.memoryless:
            h = h * 0.0
        x = nn.relu(self.enc2(nn.relu(self.enc1(obs))))
        h = self.gru(x, h)
        logits = self.pi2(nn.tanh(self.pi1(h)))
        value = self.v2(nn.tanh(self.v1(h)))[..., 0]
        return logits, value, h

    def unroll(self, obs, h0, starts):
        """obs (L, B, D), h0 (B, H), starts (L, B) 1.0 where a new episode begins at t."""
        h = h0
        logits, values = [], []
        for t in range(obs.shape[0]):
            h = h * (1.0 - starts[t])[:, None]
            lg, v, h = self.step(obs[t], h)
            logits.append(lg)
            values.append(v)
        return mx.stack(logits), mx.stack(values)


def load_policy(path, hidden=128):
    """path: .../weights.safetensors; reads policy.json next to it if present."""
    import json, os
    memoryless = False
    pj = os.path.join(os.path.dirname(path), "policy.json")
    if os.path.exists(pj):
        cfg = json.load(open(pj))
        hidden, memoryless = cfg.get("hidden", hidden), cfg.get("memoryless", False)
    p = RecurrentPolicy(hidden, memoryless)
    p.load_weights(path)
    mx.eval(p.parameters())
    return p


class PolicyRunner:
    """Batched stateful inference helper (sampling or greedy)."""

    def __init__(self, policy, n, greedy=False, seed=0):
        self.p, self.greedy = policy, greedy
        self.h = policy.initial_state(n)
        self.key = mx.random.key(seed)

        def _act(obs, h, key):
            logits, value, h2 = self.p.step(obs, h)
            a_s = mx.random.categorical(logits, key=key)
            a_g = mx.argmax(logits, axis=-1)
            return a_s, a_g, logits, value, h2

        self._act = mx.compile(_act, inputs=policy.state)

    def reset(self, mask):
        """Zero the hidden state where mask (numpy bool) is True."""
        if mask.any():
            self.h = self.h * mx.array((~mask).astype(np.float32))[:, None]

    def act(self, obs):
        self.key, sub = mx.random.split(self.key)
        a_s, a_g, logits, value, self.h = self._act(mx.array(obs), self.h, sub)
        a = a_g if self.greedy else a_s
        mx.eval(a, logits, value, self.h)
        return np.array(a), logits, value
