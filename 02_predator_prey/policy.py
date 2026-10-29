"""Small MLX actor-critic used for both roles, plus numpy-friendly save/load."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from sim import OBS_DIM, N_ACT

HID = 64


class ActorCritic(nn.Module):
    def __init__(self, hid=HID):
        super().__init__()
        self.pi = [nn.Linear(OBS_DIM, hid), nn.Linear(hid, hid), nn.Linear(hid, N_ACT)]
        self.v = [nn.Linear(OBS_DIM, hid), nn.Linear(hid, hid), nn.Linear(hid, 1)]

    def logits(self, x):
        h = mx.tanh(self.pi[0](x))
        h = mx.tanh(self.pi[1](h))
        return self.pi[2](h)

    def value(self, x):
        h = mx.tanh(self.v[0](x))
        h = mx.tanh(self.v[1](h))
        return self.v[2](h)[..., 0]

    def __call__(self, x):
        return self.logits(x), self.value(x)


def new_model(seed):
    mx.random.seed(seed)
    m = ActorCritic()
    # small last policy layer -> near-uniform initial policy
    m.pi[2].weight = m.pi[2].weight * 0.01
    mx.eval(m.parameters())
    return m


def get_np(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def set_np(model, params):
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in params.items()]))
    mx.eval(model.parameters())


def act(model, obs, rng, greedy=False):
    """Sample actions for a batch of numpy observations. Returns (actions, logp, value) numpy."""
    x = mx.array(obs)
    lg, v = model(x)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    lp_np, v_np = np.array(lp), np.array(v)
    if greedy:
        a = lp_np.argmax(1)
    else:
        g = rng.gumbel(size=lp_np.shape)
        a = (lp_np + g).argmax(1)
    return a, lp_np[np.arange(len(a)), a], v_np


def save_pair(path, p_params, e_params, extra=None):
    d = {f"P/{k}": v for k, v in p_params.items()}
    d.update({f"E/{k}": v for k, v in e_params.items()})
    for k, v in (extra or {}).items():
        d[f"meta/{k}"] = np.asarray(v)
    np.savez(path, **d)


def load_pair(path):
    z = np.load(path)
    p = {k[2:]: z[k] for k in z.files if k.startswith("P/")}
    e = {k[2:]: z[k] for k in z.files if k.startswith("E/")}
    meta = {k[5:]: z[k] for k in z.files if k.startswith("meta/")}
    return p, e, meta
