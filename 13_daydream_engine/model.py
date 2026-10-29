"""The imagination: an MLP that maps a scene state to the next frame's state.

It predicts a normalised state *delta*; `rollout` feeds its own predictions back
in (open loop). Normalisation stats are stored with the checkpoint.
"""
import json, os
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Dynamics(nn.Module):
    def __init__(self, state_dim, hidden=512, depth=4):
        super().__init__()
        dims = [state_dim] + [hidden] * depth
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.out = nn.Linear(hidden, state_dim)
        self.out.weight = self.out.weight * 0.1
        # filled from data; frozen (not trained) because they are set via freeze below
        self.s_mu = mx.zeros(state_dim)
        self.s_sd = mx.ones(state_dim)
        self.d_mu = mx.zeros(state_dim)
        self.d_sd = mx.ones(state_dim)
        self.freeze(keys=["s_mu", "s_sd", "d_mu", "d_sd"], recurse=False)

    def __call__(self, s):
        """s: (B, D) raw state -> next raw state."""
        h = (s - self.s_mu) / self.s_sd
        for l in self.layers:
            h = nn.silu(l(h))
        d = self.out(h)
        return s + d * self.d_sd + self.d_mu

    def rollout(self, s0, n):
        """Open-loop imagination: returns (B, n+1, D) including s0."""
        out = [s0]
        s = s0
        for _ in range(n):
            s = self(s)
            out.append(s)
        return mx.stack(out, axis=1)


def set_norm(model, states):
    """states: (B, T, D) numpy training trajectories."""
    s = states.reshape(-1, states.shape[-1])
    d = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    model.s_mu, model.s_sd = mx.array(s.mean(0)), mx.array(s.std(0) + 1e-6)
    model.d_mu, model.d_sd = mx.array(d.mean(0)), mx.array(d.std(0) + 1e-6)


def save(model, path, config):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save_weights(path + ".safetensors")
    with open(path + ".json", "w") as f:
        json.dump(config, f, indent=2)


def load(path):
    with open(path + ".json") as f:
        cfg = json.load(f)
    m = Dynamics(cfg["state_dim"], cfg["hidden"], cfg["depth"])
    m.load_weights(path + ".safetensors")
    mx.eval(m.parameters())
    return m, cfg


def imagine(model, init, n, chunk=4096):
    """numpy in, numpy out, float32."""
    outs = []
    for i in range(0, len(init), chunk):
        r = model.rollout(mx.array(init[i:i + chunk].astype(np.float32)), n)
        mx.eval(r)
        outs.append(np.array(r))
    return np.concatenate(outs)
