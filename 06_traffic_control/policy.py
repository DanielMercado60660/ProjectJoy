"""Shared-parameter actor-critic (MLX). One small network controls all four intersections; each
intersection sees only its own lanes, its downstream links, its signal clock and a one-hot id."""
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=64, n_act=2):
        super().__init__()
        self.pi = [nn.Linear(obs_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, n_act)]
        self.v = [nn.Linear(obs_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)]

    @staticmethod
    def _mlp(layers, x):
        for l in layers[:-1]:
            x = mx.tanh(l(x))
        return layers[-1](x)

    def __call__(self, x):
        return self._mlp(self.pi, x), self._mlp(self.v, x)[..., 0]


def save_checkpoint(path, model, opt, cfg, step, extra=None):
    flat = dict(tree_flatten(model.parameters()))
    flat.update({f"opt::{k}": v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)})
    mx.savez(path + ".npz", **flat)
    with open(path + ".json", "w") as f:
        json.dump(dict(cfg=cfg, step=step, **(extra or {})), f, indent=1)


def load_policy(path):
    with open(path + ".json") as f:
        meta = json.load(f)
    cfg = meta["cfg"]
    model = ActorCritic(cfg["obs_dim"], cfg["hidden"])
    w = mx.load(path + ".npz")
    model.update(tree_unflatten([(k, v) for k, v in w.items() if not k.startswith("opt::")]))
    mx.eval(model.parameters())
    return model, meta


def greedy_controller(model):
    def ctrl(env):
        o = env.obs()
        logits, _ = model(mx.array(o))
        return np.array(mx.argmax(logits, -1))
    return ctrl
