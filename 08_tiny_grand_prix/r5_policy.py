"""Shared NumPy/MLX functional GRU policy. Importing this file never imports MLX.

Both arms begin with EXACTLY the same old actor; added output weights start at
zero. 'ff' uses the same cell with h=0 every step (a matched memory ablation).
Weights are explicit function arguments, including inside any compiled gradient.
"""
import json
import numpy as np
from racing.policy import load_policy, unflatten, forward
from r5_common import atomic_npz
from r5_env import OBS

HIDDEN = 32
DEFAULT_WARM = 'checkpoints/ppo_dr_r0_rg_best.npz'


def init_params(seed=0, warm=DEFAULT_WARM):
    rng = np.random.default_rng(seed)
    old, _ = load_policy(warm)
    w0, b0, w1, b1, w2, b2 = unflatten(old)
    p = dict(aW0=np.vstack([w0, np.zeros((1, 32), np.float32)]), ab0=b0.copy(),
             aW1=w1.copy(), ab1=b1.copy(), aW2=w2.copy(), ab2=b2.copy())
    for gate in ('z', 'r', 'h'):
        p['W' + gate] = (rng.standard_normal((OBS, HIDDEN)) / np.sqrt(OBS)).astype(np.float32)
        p['U' + gate] = (rng.standard_normal((HIDDEN, HIDDEN)) / np.sqrt(HIDDEN)).astype(np.float32)
        p['b' + gate] = np.zeros(HIDDEN, np.float32)
    p['bz'][:] = -1  # initially keep memory for several frames
    p['out'] = np.zeros((HIDDEN, 2), np.float32)
    p['cW'] = (rng.standard_normal((OBS + HIDDEN, 64)) / np.sqrt(OBS + HIDDEN)).astype(np.float32)
    p['cb'] = np.zeros(64, np.float32)
    p['cv'] = np.zeros((64, 1), np.float32)
    p['cvb'] = np.zeros(1, np.float32)
    p['log_std'] = np.full(2, -2., np.float32)
    return p


def cell(p, x, h, xp=np):
    sig = lambda a: 1 / (1 + xp.exp(-xp.clip(a, -30, 30)))
    z = sig(x @ p['Wz'] + h @ p['Uz'] + p['bz'])
    r = sig(x @ p['Wr'] + h @ p['Ur'] + p['br'])
    c = xp.tanh(x @ p['Wh'] + (r * h) @ p['Uh'] + p['bh'])
    return (1 - z) * h + z * c


def step(p, x, h, recurrent=True, xp=np):
    h = cell(p, x, h if recurrent else xp.zeros_like(h), xp)
    a = xp.tanh(x @ p['aW0'] + p['ab0'])
    a = xp.tanh(a @ p['aW1'] + p['ab1'])
    mean = xp.tanh(a @ p['aW2'] + p['ab2'] + h @ p['out'])
    c = xp.tanh(xp.concatenate([x, h], axis=-1) @ p['cW'] + p['cb'])
    value = (c @ p['cv'] + p['cvb'])[..., 0]
    return mean, value, h


def sequence(p, obs, h, reset, recurrent=True, xp=np):
    means, values = [], []
    for t in range(obs.shape[0]):
        h = h * (1 - reset[t, :, None])
        m, v, h = step(p, obs[t], h, recurrent, xp)
        means.append(m)
        values.append(v)
    return xp.stack(means), xp.stack(values), h


class Policy:
    def __init__(self, p, recurrent=True):
        self.p, self.recurrent, self.h = p, recurrent, None

    def __call__(self, obs):
        if self.h is None or len(self.h) != len(obs):
            self.h = np.zeros((len(obs), HIDDEN), np.float32)
        out, _, self.h = step(self.p, obs, self.h, self.recurrent)
        return out


class Legacy:
    def __init__(self, path=DEFAULT_WARM):
        self.theta, _ = load_policy(path)

    def __call__(self, obs):
        return forward(self.theta, obs[:, :13])


def save_selected(path, p, meta):
    atomic_npz(path, meta=json.dumps(meta), **{'p/' + k: np.asarray(v) for k, v in p.items()})


def load_selected(path, memory_off=False):
    with np.load(path, allow_pickle=False) as z:
        p = {k[2:]: z[k].copy() for k in z.files if k.startswith('p/')}
        meta = json.loads(str(z['meta']))
    return Policy(p, meta['arm'] == 'rnn' and not memory_off), meta


def gae(reward, value, next_value, done, gamma=.995, lam=.95):
    adv = np.zeros_like(reward)
    carry = np.zeros(reward.shape[1], np.float32)
    for t in range(len(reward) - 1, -1, -1):
        nv = next_value if t == len(reward) - 1 else value[t + 1]
        nonterminal = 1 - done[t]
        delta = reward[t] + gamma * nv * nonterminal - value[t]
        carry = delta + gamma * lam * nonterminal * carry
        adv[t] = carry
    return adv, adv + value
