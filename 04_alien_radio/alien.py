"""Alien Radio core: the world (objects + episode sampler), the two agents, and the metrics.

Game (one episode)
  * The sender sees ONE object: the target (one-hot attributes, e.g. color x shape x location).
  * The receiver sees K candidate objects in a random order and the sender's message
    (L symbols from a vocabulary of V). It must point at the target.
  * The message is the only link. The receiver never sees the target index, the sender's
    input, the sender's logits, or anything about candidate generation order.

Everything here is small and standalone on purpose (numpy for data, MLX for the nets).
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np
import mlx.core as mx
import mlx.nn as nn

ATTR_NAMES = ["color", "shape", "location"]
VALUE_NAMES = {
    "color": ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"],
    "shape": ["circle", "square", "triangle", "star"],
    "location": ["top-left", "top-right", "bottom-left", "bottom-right"],
}


# ----------------------------------------------------------------------------- world
@dataclass
class World:
    dims: tuple = (8, 4, 4)          # number of values per attribute
    holdout_frac: float = 0.2        # fraction of attribute COMBINATIONS never seen in training
    split_seed: int = 0
    objects: np.ndarray = field(init=False)   # (N, A) attribute values
    feats: np.ndarray = field(init=False)     # (N, F) concatenated one-hots
    train_ids: np.ndarray = field(init=False)
    test_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dims = tuple(self.dims)
        self.objects = np.array(list(itertools.product(*[range(d) for d in self.dims])), dtype=np.int32)
        n, a = self.objects.shape
        self.feats = np.zeros((n, sum(self.dims)), dtype=np.float32)
        off = np.cumsum([0] + list(self.dims[:-1]))
        for j in range(a):
            self.feats[np.arange(n), off[j] + self.objects[:, j]] = 1.0
        self.attr_names = ATTR_NAMES[:a]
        n_hold = int(round(self.holdout_frac * n)) if a > 1 else 0
        rng = np.random.RandomState(self.split_seed)
        while True:
            perm = rng.permutation(n)
            test, train = np.sort(perm[:n_hold]), np.sort(perm[n_hold:])
            # every single attribute value must still appear in training (only COMBINATIONS are new)
            if all(len(np.unique(self.objects[train, j])) == self.dims[j] for j in range(a)):
                break
        self.train_ids, self.test_ids = train, test
        self._index = {tuple(o): i for i, o in enumerate(self.objects.tolist())}

    @property
    def n(self):
        return len(self.objects)

    def obj_id(self, attrs):
        return self._index[tuple(int(x) for x in attrs)]

    def describe(self, i):
        return " ".join(VALUE_NAMES[nm][v] for nm, v in zip(self.attr_names, self.objects[i]))


def make_episodes(world: World, n_episodes: int, K: int, split: str, seed: int, cluster_frac: float = 0.5):
    """Pre-generate episodes. Returns cand (E,K) object ids and tpos (E,) target position.

    split='train': every candidate (target and distractors) is a training object.
    split='test' : every candidate is a held-out combination. (An earlier version drew the
                   distractors from all objects; then "pick the unfamiliar-looking one" was a
                   side channel -- a no-message receiver scored 30% instead of 20%. Now the
                   target is uniform among candidates that are all equally novel.)

    Half the episodes are "cluster" episodes: candidates are near-misses of a random centre
    (each attribute resampled with prob 0.5), so the sender has to convey every attribute.
    The target is then chosen uniformly AMONG the candidates, so the candidate set's structure
    (e.g. "the one in the middle of the cluster") carries no information about which is the target.
    The final order of candidates is a fresh random permutation.
    """
    rng = random.Random(seed)
    train_set = set(world.train_ids.tolist())
    test_set = set(world.test_ids.tolist())
    pool = world.train_ids.tolist() if split == "train" else world.test_ids.tolist()
    dims = world.dims
    objs = world.objects
    cand = np.zeros((n_episodes, K), dtype=np.int32)
    tpos = np.zeros(n_episodes, dtype=np.int32)
    pool_set = set(pool)
    e = 0
    while e < n_episodes:
        if rng.random() < cluster_frac and len(dims) > 1:
            c = objs[rng.choice(pool)].tolist()
            s = set()
            tries = 0
            while len(s) < K and tries < 200:
                tries += 1
                o = [v if rng.random() < 0.5 else rng.randrange(d) for v, d in zip(c, dims)]
                i = world._index[tuple(o)]
                if i in pool_set:
                    s.add(i)
            if len(s) < K:
                continue
            ids = list(s)
        else:
            ids = rng.sample(pool, K)
        target = rng.choice(ids)
        rng.shuffle(ids)
        cand[e] = ids
        tpos[e] = ids.index(target)
        e += 1
    assert set(np.unique(cand).tolist()) <= (train_set if split == "train" else test_set)
    return cand, tpos


def reshuffle(cand, tpos, rng: np.random.Generator):
    """Fresh random order of candidates per episode (used every training batch)."""
    E, K = cand.shape
    perm = np.argsort(rng.random((E, K)), axis=1)
    new = np.take_along_axis(cand, perm, axis=1)
    newpos = np.argmax(perm == tpos[:, None], axis=1).astype(np.int32)
    return new, newpos


# ----------------------------------------------------------------------------- agents
# Every network carries a leading "population" axis P: member p is an independent agent pair
# with its own initial weights, its own data batches and its own noise. Nothing is shared
# between members (Adam is elementwise, losses are summed), so training P members at once is
# exactly P independent runs -- just far cheaper on a busy GPU than P separate processes.
class PLinear(nn.Module):
    def __init__(self, P, n_in, n_out):
        super().__init__()
        s = 1.0 / math.sqrt(n_in)
        self.weight = mx.random.uniform(-s, s, (P, n_in, n_out))
        self.bias = mx.random.uniform(-s, s, (P, 1, n_out))

    def __call__(self, x):                      # x: (P, ..., n_in)
        shp = x.shape
        y = x.reshape(shp[0], -1, shp[-1]) @ self.weight + self.bias
        return y.reshape(*shp[:-1], y.shape[-1])


class Sender(nn.Module):
    """Target features (P,B,F) -> logits (P,B,L,V): L independent categorical distributions."""

    def __init__(self, P, n_feats, L, V, hidden=128):
        super().__init__()
        self.L, self.V = L, V
        self.l1 = PLinear(P, n_feats, hidden)
        self.l2 = PLinear(P, hidden, hidden)
        self.out = PLinear(P, hidden, L * V)

    def __call__(self, x):
        h = nn.relu(self.l1(x))
        h = nn.relu(self.l2(h))
        return self.out(h).reshape(*x.shape[:-1], self.L, self.V)


class Receiver(nn.Module):
    """(message one-hots (P,B,L,V), candidates (P,B,K,F)) -> logits over the K candidates (P,B,K).

    Note the signature: the receiver gets nothing but the message and the shuffled candidates.
    """

    def __init__(self, P, n_feats, L, V, hidden=128, emb=64):
        super().__init__()
        self.m1 = PLinear(P, L * V, hidden)
        self.m2 = PLinear(P, hidden, emb)
        self.c1 = PLinear(P, n_feats, hidden)
        self.c2 = PLinear(P, hidden, emb)

    def __call__(self, msg_onehot, cands):
        P, B = msg_onehot.shape[:2]
        m = self.m2(nn.relu(self.m1(msg_onehot.reshape(P, B, -1))))    # (P, B, emb)
        c = self.c2(nn.relu(self.c1(cands)))                            # (P, B, K, emb)
        return (c * m[:, :, None, :]).sum(-1)                           # (P, B, K)


class Game(nn.Module):
    def __init__(self, P, n_feats, L, V, hidden=128):
        super().__init__()
        self.P = P
        self.sender = Sender(P, n_feats, L, V, hidden)
        self.receiver = Receiver(P, n_feats, L, V, hidden)


# ----------------------------------------------------------------------------- fixed protocols (baselines)
def compositional_code(world: World, L: int, V: int):
    """Hand-designed protocol: symbol j = value of attribute j (padding symbol 0 if L > A)."""
    A = len(world.dims)
    assert L >= A and V >= max(world.dims)
    msgs = np.zeros((world.n, L), dtype=np.int32)
    msgs[:, :A] = world.objects
    return msgs


def holistic_code(world: World, L: int, V: int, seed: int = 0):
    """A fixed random one-to-one code: every object gets an arbitrary unique message."""
    rng = np.random.RandomState(seed)
    all_msgs = rng.choice(V ** L, size=world.n, replace=False)
    msgs = np.zeros((world.n, L), dtype=np.int32)
    for p in range(L):
        msgs[:, L - 1 - p] = (all_msgs // V ** p) % V
    return msgs


# ----------------------------------------------------------------------------- metrics
def entropy_bits(counts):
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info_bits(x, y):
    """MI between two discrete integer arrays, in bits."""
    x = np.asarray(x); y = np.asarray(y)
    joint = {}
    for a, b in zip(x.tolist(), y.tolist()):
        joint[(a, b)] = joint.get((a, b), 0) + 1
    n = len(x)
    px = np.bincount(x).astype(float) / n
    py = np.bincount(y).astype(float) / n
    mi = 0.0
    for (a, b), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / (px[a] * py[b]))
    return mi


def topographic_similarity(objects, msgs):
    """Spearman correlation between attribute Hamming distance and message Hamming distance."""
    from scipy.stats import spearmanr
    iu = np.triu_indices(len(objects), k=1)
    d_obj = (objects[:, None, :] != objects[None, :, :]).sum(-1)[iu]
    d_msg = (msgs[:, None, :] != msgs[None, :, :]).sum(-1)[iu]
    if d_msg.std() == 0:
        return 0.0
    return float(spearmanr(d_obj, d_msg).correlation)


def mi_matrix(objects, msgs):
    """MI(position p ; attribute a) in bits, shape (L, A)."""
    L, A = msgs.shape[1], objects.shape[1]
    return np.array([[mutual_info_bits(msgs[:, p], objects[:, a]) for a in range(A)] for p in range(L)])


def positional_disentanglement(objects, msgs):
    """posdis (Chaabouni et al. 2020): mean over positions of (top MI - 2nd MI) / H(position)."""
    M = mi_matrix(objects, msgs)
    scores = []
    for p in range(msgs.shape[1]):
        h = entropy_bits(np.bincount(msgs[:, p]))
        if h == 0 or M.shape[1] < 2:
            continue
        s = np.sort(M[p])[::-1]
        scores.append((s[0] - s[1]) / h)
    return float(np.mean(scores)) if scores else 0.0


def message_stats(objects, msgs, V):
    n = len(objects)
    uniq = len({tuple(m) for m in msgs.tolist()})
    msg_ids = np.array([hash(tuple(m)) for m in msgs.tolist()])
    _, counts = np.unique(msg_ids, return_counts=True)
    pos_ent = [entropy_bits(np.bincount(msgs[:, p], minlength=V)) for p in range(msgs.shape[1])]
    return {
        "unique_messages": uniq,
        "n_objects": n,
        "bits_needed": math.log2(n),
        "bits_capacity": msgs.shape[1] * math.log2(V),
        "message_entropy_bits": entropy_bits(counts),
        "per_position_entropy_bits": pos_ent,
        "symbols_used_per_position": [int(len(np.unique(msgs[:, p]))) for p in range(msgs.shape[1])],
    }
