"""Networks and intrinsic-reward modules (MLX), plus the hashed-count bonus (numpy).

PolicyNet: shared MLP torso -> policy logits, intrinsic value head, extrinsic value head.
RND:       fixed random target net f(o) and trained predictor f^(o); bonus = ||f^(o)-f(o)||^2.
Count:     bonus = 1/sqrt(N(h(o))) with h = 64-bit random-projection hash of the binary egocentric obs.
Disagree:  ensemble of forward models predicting a fixed random embedding of o' from (o, a);
           bonus = variance across the ensemble (Pathak et al. 2019). Noise-robust in theory:
           on unpredictable transitions every member converges to the same conditional mean.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn


class RunningMeanStd:
    """Parallel-variance running statistics (numpy, float64)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        self.update_moments(x.mean(0), x.var(0), x.shape[0])

    def update_moments(self, bm, bv, bc):
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def state(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v_int = nn.Linear(hidden, 1)
        self.v_ext = nn.Linear(hidden, 1)
        self.pi.weight = self.pi.weight * 0.01

    def __call__(self, x):
        h = nn.relu(self.l1(x))
        h = nn.relu(self.l2(h))
        return self.pi(h), self.v_int(h)[:, 0], self.v_ext(h)[:, 0]


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.leaky_relu(l(x))
        return self.layers[-1](x)


class RND(nn.Module):
    def __init__(self, obs_dim, out=64, hidden=256):
        super().__init__()
        self.target = MLP([obs_dim, hidden, hidden, out])
        self.predictor = MLP([obs_dim, hidden, hidden, hidden, out])
        self.target.freeze()

    def errors(self, x):
        return mx.mean((self.predictor(x) - mx.stop_gradient(self.target(x))) ** 2, axis=1)

    def loss(self, x, mask):
        e = self.errors(x)
        return mx.sum(e * mask) / mx.maximum(mx.sum(mask), 1.0)


class Disagreement(nn.Module):
    """K forward models: (normalized obs, one-hot action) -> phi(next obs), phi a fixed random net."""

    def __init__(self, obs_dim, n_actions, k=5, out=64, hidden=256):
        super().__init__()
        self.phi = MLP([obs_dim, hidden, out])
        self.phi.freeze()
        self.models = [MLP([obs_dim + n_actions, hidden, hidden, out]) for _ in range(k)]
        self.n_actions = n_actions

    def preds(self, x, a):
        a1h = (mx.arange(self.n_actions)[None] == a[:, None]).astype(mx.float32)
        inp = mx.concatenate([x, a1h], 1)
        return mx.stack([m(inp) for m in self.models], 0)  # [k, B, out]

    def errors(self, x, a, xn):
        p = self.preds(x, a)
        return mx.mean(mx.var(p, axis=0), axis=1)

    def loss(self, x, a, xn, mask):
        """mask [k, B]: bootstrap-style per-member sample masks."""
        p = self.preds(x, a)
        tgt = mx.stop_gradient(self.phi(xn))
        e = mx.mean((p - tgt[None]) ** 2, axis=2)  # [k, B]
        return mx.sum(e * mask) / mx.maximum(mx.sum(mask), 1.0)


class HashCounter:
    """Visit counts on a 64-bit random-projection hash of the binary observation."""

    def __init__(self, obs_dim, table_bits=22, seed=0):
        rng = np.random.default_rng(seed + 12345)
        self.proj = rng.integers(0, 2 ** 63, size=obs_dim, dtype=np.uint64)
        self.mask = np.uint64(2 ** table_bits - 1)
        self.table = np.zeros(2 ** table_bits, np.int64)

    def hash(self, obs):
        b = (obs > 0.5).astype(np.uint64)
        # sum of random 64-bit words of the active bits (integer matmul wraps mod 2^64)
        h = b @ self.proj
        return (h ^ (h >> np.uint64(29))) & self.mask

    def hash64(self, obs):
        """Full 64-bit hash (no table masking): used as an episodic-memory key."""
        return (obs > 0.5).astype(np.uint64) @ self.proj

    def bonus_and_update(self, obs):
        idx = self.hash(obs)
        np.add.at(self.table, idx, 1)
        return 1.0 / np.sqrt(self.table[idx])

    def n_distinct(self):
        return int(np.count_nonzero(self.table))


class NGUAux(nn.Module):
    """NGU-style intrinsic machinery (Badia et al. 2020), simplified.

    * lifelong novelty: RND on the observation (bounded multiplier in [1, 5])
    * episodic novelty: k-nearest-neighbour similarity in a learned embedding phi(o), where phi is
      trained only through an inverse-dynamics model (predict the action from phi(o), phi(o')).
      Features that the agent's actions cannot influence (e.g. a noisy TV) carry no information
      about the action, so phi has no reason to encode them, and they stop looking novel.
    """

    def __init__(self, obs_dim, n_actions, emb=32, hidden=256):
        super().__init__()
        self.rnd = RND(obs_dim)
        self.phi = MLP([obs_dim, hidden, emb])
        self.inv = MLP([2 * emb, hidden, n_actions])

    def embed(self, x):
        return self.phi(x)

    def errors(self, x):
        return self.rnd.errors(x)

    def loss(self, x, xn, a, rnd_mask, inv_mask):
        l_rnd = self.rnd.loss(xn, rnd_mask)
        logits = self.inv(mx.concatenate([self.phi(x), self.phi(xn)], 1))
        lp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        nll = -mx.take_along_axis(lp, a[:, None], axis=1)[:, 0]
        l_inv = mx.sum(nll * inv_mask) / mx.maximum(mx.sum(inv_mask), 1.0)
        return l_rnd + l_inv


class EpisodicKNN:
    """Per-env episodic memory of embeddings; reward 1/sqrt(sum_k K(e, e_k) + 1) with the NGU kernel
    K = eps / (max(d^2 / mean_d^2 - xi, 0) + eps). (NGU uses 1/(sqrt(sum K) + c); the +1 inside keeps
    it bounded.)"""

    def __init__(self, n_envs, capacity, emb=32, k=10, eps=1e-3, xi=0.008):
        self.mem = np.zeros((n_envs, capacity, emb), np.float32)
        self.n = np.zeros(n_envs, np.int64)
        self.k, self.eps, self.xi = k, eps, xi
        self.dm2 = 1.0  # running mean of squared k-NN distances

    def reward_and_add(self, e, reset):
        self.n[reset] = 0
        N = e.shape[0]
        L = int(self.n.max())
        r = np.ones(N, np.float32)
        if L > 0:
            d2 = ((self.mem[:, :L] - e[:, None]) ** 2).sum(-1)
            d2[np.arange(L)[None, :] >= self.n[:, None]] = np.inf
            k = min(self.k, L)
            nn_d2 = np.partition(d2, k - 1, axis=1)[:, :k]
            fin = np.isfinite(nn_d2)
            if fin.any():
                self.dm2 = 0.99 * self.dm2 + 0.01 * float(nn_d2[fin].mean())
            dn = np.maximum(nn_d2 / max(self.dm2, 1e-8) - self.xi, 0.0)
            K = np.where(fin, self.eps / (dn + self.eps), 0.0)
            r = (1.0 / np.sqrt(K.sum(1) + 1.0)).astype(np.float32)
        idx = np.minimum(self.n, self.mem.shape[1] - 1)
        self.mem[np.arange(N), idx] = e
        self.n += 1
        return r

    def state(self):
        return dict(mem=self.mem.copy(), n=self.n.copy(), dm2=self.dm2)

    def load(self, s):
        self.mem[...] = s["mem"]; self.n[...] = s["n"]; self.dm2 = s["dm2"]
