"""Tiny MLP policy shared by ES and PPO: obs -> 32 -> 32 -> 2 (tanh).

Weights live in one flat float32 vector so ES can perturb them directly and PPO
can export its MLX actor mean into the same format. Inference for evaluation is
plain numpy (a 13x32x32x2 net is far too small to be worth a GPU round trip).
"""
import json

import numpy as np

from .sim import ACT_DIM, OBS_DIM

HIDDEN = (32, 32)


def shapes(obs_dim=OBS_DIM, hidden=HIDDEN, act_dim=ACT_DIM):
    dims = (obs_dim,) + tuple(hidden) + (act_dim,)
    out = []
    for a, b in zip(dims[:-1], dims[1:]):
        out += [(a, b), (b,)]
    return out


def n_params(**kw):
    return int(sum(np.prod(s) for s in shapes(**kw)))


def unflatten(theta, **kw):
    """theta: (..., P) -> list of arrays with leading batch dims."""
    lead = theta.shape[:-1]
    out, o = [], 0
    for s in shapes(**kw):
        k = int(np.prod(s))
        out.append(theta[..., o:o + k].reshape(lead + s))
        o += k
    return out


def init_theta(rng, **kw):
    parts = []
    for s in shapes(**kw):
        if len(s) == 2:
            parts.append((rng.standard_normal(s) / np.sqrt(s[0])).ravel())
        else:
            parts.append(np.zeros(s))
    theta = np.concatenate(parts).astype(np.float32)
    return theta


def forward(theta, obs):
    """Single policy: theta (P,), obs (E, D) -> actions (E, 2)."""
    ps = unflatten(theta)
    h = obs.astype(np.float32)
    for li in range(0, len(ps), 2):
        h = h @ ps[li] + ps[li + 1]
        h = np.tanh(h)  # hidden layers and output all use tanh
    return h


def forward_pop(thetas, obs):
    """Population: thetas (M, P), obs (M, K, D) -> (M, K, 2)."""
    ps = unflatten(thetas)
    h = obs.astype(np.float32)
    for li in range(0, len(ps), 2):
        h = np.tanh(np.einsum("mkd,mdh->mkh", h, ps[li]) + ps[li + 1][:, None, :])
    return h


class MLPController:
    def __init__(self, theta, name="mlp"):
        self.theta = np.asarray(theta, np.float32)
        self.name = name

    def __call__(self, obs, env=None):
        return forward(self.theta, obs)


def save_policy(path, theta, meta):
    np.savez(path, theta=np.asarray(theta, np.float32), meta=json.dumps(meta))


def load_policy(path):
    d = np.load(path, allow_pickle=False)
    return d["theta"], json.loads(str(d["meta"]))
