"""Classical baselines, fit on the train split only, per style.

- BernoulliModel: every (step, instrument) cell is an independent coin with the style's train frequency.
- MarkovModel:   position-aware bigram over whole step columns (the 9-instrument chord at each 16th),
                 P(col_t | col_{t-1}, t, style), backed off to P(col_t | t, style).
- CopyModel:     returns a random training pattern of the requested style (pure memorisation).

Velocity and microtiming for the first two come from per-(style, step, instrument) train means,
so they get the same "average feel" information the VAE decoder can learn.
"""
from collections import defaultdict

import numpy as np

STEPS, N_INST, N_STYLES = 32, 9, 8
POW2 = 2 ** np.arange(N_INST)


def cols(hits):
    """[B,32,9] -> [B,32] integer code of each step column."""
    return (hits.astype(np.int64) * POW2).sum(-1)


def uncol(c):
    return ((c[..., None] // POW2) % 2).astype(np.uint8)


class _FeelMeans:
    def fit_feel(self, tr):
        H, V, O, S = tr["hits"], tr["vel"], tr["off"], tr["style"]
        self.vel = np.zeros((N_STYLES, STEPS, N_INST), np.float32)
        self.off = np.zeros((N_STYLES, STEPS, N_INST), np.float32)
        for s in range(N_STYLES):
            h = H[S == s].astype(np.float32)
            n = h.sum(0)
            inst_v = (V[S == s].sum((0, 1)) / np.maximum(h.sum((0, 1)), 1))
            self.vel[s] = np.where(n > 0, V[S == s].sum(0) / np.maximum(n, 1), inst_v[None, :])
            self.off[s] = np.where(n > 0, O[S == s].sum(0) / np.maximum(n, 1), 0)

    def feel(self, hits, style):
        return (self.vel[style] * hits).astype(np.float32), (self.off[style] * hits).astype(np.float32)


class BernoulliModel(_FeelMeans):
    name = "bernoulli"

    def fit(self, tr):
        self.p = np.zeros((N_STYLES, STEPS, N_INST))
        for s in range(N_STYLES):
            h = tr["hits"][tr["style"] == s]
            self.p[s] = (h.sum(0) + 0.5) / (len(h) + 1.0)
        self.fit_feel(tr)
        return self

    def sample(self, style, rng):
        style = np.asarray(style)
        h = (rng.random((len(style), STEPS, N_INST)) < self.p[style]).astype(np.uint8)
        return (h, *self.feel(h, style))

    def nll(self, hits, style):
        p = self.p[np.asarray(style)]
        return -(hits * np.log(p) + (1 - hits) * np.log(1 - p)).sum((1, 2))


class MarkovModel(_FeelMeans):
    name = "markov"
    START = 512

    def __init__(self, alpha=3.0, alpha_uni=0.1):  # smoothing tuned on val NLL
        self.alpha, self.alpha_uni = alpha, alpha_uni

    def fit(self, tr):
        C = cols(tr["hits"])
        self.uni = np.zeros((N_STYLES, STEPS, 512))
        self.bi = defaultdict(lambda: defaultdict(float))  # (s, t, prev) -> {cur: count}
        for c, s in zip(C, tr["style"]):
            prev = self.START
            for t in range(STEPS):
                self.uni[s, t, c[t]] += 1
                self.bi[(s, t, prev)][c[t]] += 1
                prev = c[t]
        # unigram backoff: add-alpha over all 512 columns
        self.uni_p = (self.uni + self.alpha_uni) / (self.uni.sum(-1, keepdims=True) + 512 * self.alpha_uni)
        self.fit_feel(tr)
        return self

    def _dist(self, s, t, prev):
        p = self.alpha * self.uni_p[s, t].copy()
        tot = self.alpha
        for c, n in self.bi.get((s, t, prev), {}).items():
            p[c] += n
            tot += n
        return p / tot

    def sample(self, style, rng):
        style = np.asarray(style)
        out = np.zeros((len(style), STEPS), np.int64)
        for b, s in enumerate(style):
            prev = self.START
            for t in range(STEPS):
                prev = out[b, t] = rng.choice(512, p=self._dist(s, t, prev))
        h = uncol(out)
        return (h, *self.feel(h, style))

    def nll(self, hits, style):
        C = cols(hits)
        res = np.zeros(len(C))
        for b, (c, s) in enumerate(zip(C, np.asarray(style))):
            prev = self.START
            for t in range(STEPS):
                res[b] -= np.log(self._dist(s, t, prev)[c[t]])
                prev = c[t]
        return res


class BarMarkovModel(MarkovModel):
    """Stronger n-gram: in bar 2 also condition on the column one bar earlier.

    P(col_t | col_{t-1}, col_{t-16}, t, style) for t >= 16, backed off to the plain position-aware bigram.
    Encodes the hand-known fact that drum bars repeat; a fair "classical" rival for the VAE.
    """
    name = "markov_bar"

    def __init__(self, alpha=3.0, alpha_uni=0.1, alpha_bar=1.0):
        super().__init__(alpha, alpha_uni)
        self.alpha_bar = alpha_bar

    def fit(self, tr):
        super().fit(tr)
        self.tri = defaultdict(lambda: defaultdict(float))
        for c, s in zip(cols(tr["hits"]), tr["style"]):
            for t in range(16, STEPS):
                self.tri[(s, t, c[t - 1], c[t - 16])][c[t]] += 1
        return self

    def _dist3(self, s, t, prev, bar_prev):
        p = self._dist(s, t, prev)
        if t < 16:
            return p
        p = self.alpha_bar * p
        tot = self.alpha_bar
        for c, n in self.tri.get((s, t, prev, bar_prev), {}).items():
            p[c] += n
            tot += n
        return p / tot

    def sample(self, style, rng):
        style = np.asarray(style)
        out = np.zeros((len(style), STEPS), np.int64)
        for b, s in enumerate(style):
            prev = self.START
            for t in range(STEPS):
                bp = out[b, t - 16] if t >= 16 else None
                prev = out[b, t] = rng.choice(512, p=self._dist3(s, t, prev, bp))
        h = uncol(out)
        return (h, *self.feel(h, style))

    def nll(self, hits, style):
        C = cols(hits)
        res = np.zeros(len(C))
        for b, (c, s) in enumerate(zip(C, np.asarray(style))):
            prev = self.START
            for t in range(STEPS):
                res[b] -= np.log(self._dist3(s, t, prev, c[t - 16] if t >= 16 else None)[c[t]])
                prev = c[t]
        return res


class CopyModel:
    name = "copy_train"

    def fit(self, tr):
        self.tr = tr
        return self

    def sample(self, style, rng):
        style = np.asarray(style)
        idx = np.array([rng.choice(np.flatnonzero(self.tr["style"] == s)) for s in style])
        return self.tr["hits"][idx], self.tr["vel"][idx], self.tr["off"][idx]
