"""Shared ant policy in MLX: one network for every ant.

Actor  : local obs -> (move logits [7], deposit logits [3])        (used at run time)
Critic : local obs + 2 colony-level scalars (time left, colony store) -> value
         The critic is only used during training (centralised critic, decentralised actors),
         so the extra scalars never reach the ants' decisions.
"""
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import sim

CRITIC_EXTRA = 2


class Policy(nn.Module):
    def __init__(self, obs_dim=sim.OBS_DIM, hid=128):
        super().__init__()
        self.hid = hid
        self.a1 = nn.Linear(obs_dim, hid)
        self.a2 = nn.Linear(hid, hid)
        self.move = nn.Linear(hid, sim.N_MOVE)
        self.dep = nn.Linear(hid, sim.N_DEP)
        self.c1 = nn.Linear(obs_dim + CRITIC_EXTRA, hid)
        self.c2 = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, 1)
        for layer in (self.move, self.dep):
            layer.weight = layer.weight * 0.01
            layer.bias = mx.zeros_like(layer.bias)

    def actor(self, x):
        h = mx.tanh(self.a1(x))
        h = mx.tanh(self.a2(h))
        return self.move(h), self.dep(h)

    def value(self, xc):
        h = mx.tanh(self.c1(xc))
        h = mx.tanh(self.c2(h))
        return self.v(h)[..., 0]


def mask_obs(obs, no_compass):
    """The 'no compass' ablation: the ant loses path integration (nest bearing + distance)."""
    if no_compass:
        obs = obs.copy()
        obs[..., [sim.I_CMP_F, sim.I_CMP_S, sim.I_NDIST]] = 0.0
    return obs


def probs(model, obs):
    """numpy obs [B,N,D] -> numpy (move probs, deposit probs)"""
    lm, ld = model.actor(mx.array(obs))
    pm, pd = mx.softmax(lm, -1), mx.softmax(ld, -1)
    return np.array(pm), np.array(pd)


def greedy_wrap(fn):
    def g(obs):
        pm, pd = fn(obs)
        return (pm == pm.max(-1, keepdims=True)).astype(np.float32) / (pm == pm.max(-1, keepdims=True)).sum(-1, keepdims=True), pd
    return g


def save_weights(model, path):
    mx.save_safetensors(path, dict(tree_flatten(model.parameters())))


def load_weights(model, path):
    model.update(tree_unflatten(list(mx.load(path).items())))
    mx.eval(model.parameters())
    return model


def export_actor_json(model, path):
    """Actor weights as plain JSON for the in-browser terrarium."""
    out = {}
    for name in ("a1", "a2", "move", "dep"):
        layer = getattr(model, name)
        out[name] = {"w": np.array(layer.weight).round(5).tolist(), "b": np.array(layer.bias).round(5).tolist()}
    with open(path, "w") as f:
        json.dump(out, f)


class NumpyActor:
    """Same actor in plain numpy (used for policy-in-the-loop eval without MLX, and to
    check the JS port)."""
    def __init__(self, model):
        self.p = {n: (np.array(getattr(model, n).weight), np.array(getattr(model, n).bias))
                  for n in ("a1", "a2", "move", "dep")}

    def __call__(self, obs):
        h = np.tanh(obs @ self.p["a1"][0].T + self.p["a1"][1])
        h = np.tanh(h @ self.p["a2"][0].T + self.p["a2"][1])
        lm = h @ self.p["move"][0].T + self.p["move"][1]
        ld = h @ self.p["dep"][0].T + self.p["dep"][1]
        sm = lambda z: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(z - z.max(-1, keepdims=True)))
        return sm(lm).astype(np.float32), sm(ld).astype(np.float32)
