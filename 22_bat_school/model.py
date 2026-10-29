"""Recurrent bat policy in MLX: MLP echo encoder -> hand-written GRU -> move / pulse / value heads."""
import json

import mlx.core as mx
import mlx.nn as nn


class BatPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.enc1 = nn.Linear(obs_dim, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.wx = nn.Linear(hidden, 3 * hidden)
        self.wh = nn.Linear(hidden, 3 * hidden)
        self.move = nn.Linear(hidden, 3)
        self.pulse = nn.Linear(hidden, 2)
        self.value = nn.Linear(hidden, 1)

    def gru(self, x, h):
        H = self.hidden
        gx, gh = self.wx(x), self.wh(h)
        r = mx.sigmoid(gx[:, :H] + gh[:, :H])
        z = mx.sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = mx.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1 - z) * n + z * h

    def step(self, obs, h):
        x = nn.relu(self.enc2(nn.relu(self.enc1(obs))))
        h = self.gru(x, h)
        return self.move(h), self.pulse(h), self.value(h)[:, 0], h

    def sequence(self, obs, h0, starts):
        """obs (T,N,D); starts (T,N) = 1 where a new episode begins at t (reset h before step)."""
        h = h0
        lm, lp, vs = [], [], []
        for t in range(obs.shape[0]):
            h = h * (1 - starts[t])[:, None]
            m, p, v, h = self.step(obs[t], h)
            lm.append(m); lp.append(p); vs.append(v)
        return mx.stack(lm), mx.stack(lp), mx.stack(vs)


def save(model, path, config):
    model.save_weights(path + ".safetensors")
    with open(path + ".json", "w") as f:
        json.dump(config, f, indent=2)


def load(path):
    with open(path + ".json") as f:
        cfg = json.load(f)
    m = BatPolicy(cfg["obs_dim"], cfg["hidden"])
    m.load_weights(path + ".safetensors")
    return m, cfg
