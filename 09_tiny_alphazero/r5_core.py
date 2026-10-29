"""Round 5 mechanics. NumPy/numba only; importing this never imports MLX."""
import time
import numpy as np
from numba import njit
import c4
import mcts


@njit(cache=True)
def canonical_keys(pos, mask):
    out = np.empty(len(pos), np.int64)
    for i in range(len(pos)):
        key = pos[i] + mask[i]
        mirror = np.int64(0)
        for c in range(7):
            mirror |= ((key >> (7 * c)) & 127) << (7 * (6 - c))
        out[i] = min(key, mirror)
    return out


def probabilities(logits, legal):
    assert logits.shape == legal.shape and np.all(legal.any(1))
    z = np.where(legal, logits, -1e30).astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True)) * legal
    return p / np.maximum(p.sum(1, keepdims=True), 1e-12)


def choose(visits, prior, legal):
    # Deterministic ties, also handles the root-only search.
    v = np.where(legal, visits, -np.inf)
    best = v == v.max(1, keepdims=True)
    return np.argmax(np.where(best, prior, -np.inf), axis=1)


def mistakes(values, moves):
    assert values.ndim == 2 and len(values) == len(moves)
    assert np.all(np.isin(values, [-999, -1, 0, 1])), "Unsolved label cannot be scored"
    selected = values[np.arange(len(moves)), moves]
    assert np.all(selected >= -1), "Illegal action"
    return (selected < values.max(1)).astype(np.float32)


def allocation(scores, low, high, total):
    """Exact integer total; stable ranking uses no labels, including tied scores."""
    n = len(scores)
    assert n * low <= total <= n * high and high >= low
    out = np.full(n, low, np.int64)
    left = total - n * low
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        inc = min(left, high - low)
        out[i] += inc
        left -= inc
    assert left == 0 and out.sum() == total
    return out


class Search:
    """Resumable trees without changing old MCTS behavior. Root expansion is separate.

    Continuation takes a PREFIX; sorting the pool once makes adaptive search cheap.
    Every step is one simulation per active position, even at terminal leaves.
    """
    fields = ("npos", "nmask", "parent", "child0", "N", "Wv", "P", "term",
              "nnodes", "leaf", "feats", "logits", "values")

    def __init__(self, pos, mask, evaluate, capacity=128, c_puct=1.5):
        self.n = len(pos)
        self.pool = mcts.BatchMCTS(self.n, capacity, c_puct=c_puct)
        self.evaluate = evaluate
        self.capacity = capacity
        self.ids = np.arange(self.n)
        self.count = np.zeros(self.n, np.int64)
        self.legal = np.array([c4.legal_moves(x) for x in mask])
        t = self.pool
        mcts._reset(self.n, np.asarray(pos, np.int64), np.asarray(mask, np.int64),
                    t.npos, t.nmask, t.parent, t.child0, t.N, t.Wv, t.P, t.term, t.nnodes)
        self._step(self.n)  # root expansion (charged in wall time, not simulations)
        self.root_value = -t.Wv[:self.n, 0].copy()

    def _step(self, n):
        t = self.pool
        mcts._select(n, t.c_puct, t.fpu, t.npos, t.nmask, t.parent, t.child0,
                     t.N, t.Wv, t.P, t.term, t.leaf, t.feats)
        idx = np.flatnonzero(t.leaf[:n] >= 0)
        if len(idx):
            lg, v = self.evaluate(t.feats[idx])
            assert np.isfinite(lg).all() and np.isfinite(v).all()
            t.logits[idx], t.values[idx] = lg, v
            t.n_evals += len(idx)
        mcts._expand_backup(n, t.leaf, t.logits, t.values, False,
                           t.dir_alpha, t.dir_eps, t.npos, t.nmask, t.parent,
                           t.child0, t.N, t.Wv, t.P, t.term, t.nnodes)

    def advance(self, targets, deadline=float("inf")):
        """Targets must decrease down the sorted pool. Stops between GPU batches."""
        targets = np.asarray(targets)
        assert np.all(targets[:-1] >= targets[1:])
        assert np.all(targets >= self.count) and targets.max() <= self.capacity
        while np.any(self.count < targets) and time.monotonic() < deadline:
            n = int(np.count_nonzero(self.count < targets))
            assert np.all(self.count[:n] < targets[:n])
            self._step(n)
            self.count[:n] += 1

    def sort(self, scores):
        order = np.argsort(-np.asarray(scores), kind="stable")
        for field in self.fields:
            setattr(self.pool, field, getattr(self.pool, field)[order].copy())
        self.ids, self.count = self.ids[order], self.count[order]

    def stats(self):
        t = self.pool
        out = mcts._root_stats(self.n, t.child0, t.N, t.Wv, t.P, t.term)
        inv = np.argsort(self.ids)
        return tuple(x[inv] for x in out)

    def moves(self):
        visits, _, prior = self.stats()
        return choose(visits, prior, self.legal)


@njit(cache=True)
def classical_features(pos, mask, move, active_depths):
    """Tactics plus alpha-beta disagreement at 2/4/6/8 plies; deterministic ties."""
    out = np.zeros((len(pos), 6), np.float32)
    for i in range(len(pos)):
        p, m = pos[i], mask[i]
        for c in range(7):
            if c4.can_play(m, c):
                out[i, 0] += c4.winning_move(p, m, c)
                out[i, 1] += c4.winning_move(p ^ m, m, c)
        for j, depth in enumerate((2, 4, 6, 8)):
            if not active_depths[j]:
                continue
            scores = np.full(7, -1e9)
            for c in range(7):
                if not c4.can_play(m, c):
                    continue
                if c4.winning_move(p, m, c):
                    scores[c] = 1e6
                else:
                    p2, m2 = c4.play(p, m, c)
                    scores[c] = -c4.negamax(p2, m2, depth - 1, -1e9, 1e9,
                                           c4.WINDOWS, c4.CENTER_ORDER)
            out[i, 2 + j] = scores[move[i]] < scores.max()
    return out


SIGNALS = ["entropy", "uncertainty", "disagreement", "visit_entropy", "margin",
           "wins", "threats", "ab2", "ab4", "ab6", "ab8"]


def features(pos, mask, search, ab_depths=(2, 4, 6, 8)):
    visits, q, prior = search.stats()
    legal = search.legal
    move = choose(visits, prior, legal)
    vprob = visits / np.maximum(visits.sum(1, keepdims=True), 1)
    entropy = lambda p: -(p * np.log(np.maximum(p, 1e-12))).sum(1) / np.log(7)
    sortedp = np.sort(prior, axis=1)
    scalar = np.column_stack((entropy(prior), 1 - np.abs(search.root_value),
                             .5 * np.abs(prior - vprob).sum(1), entropy(vprob),
                             1 - (sortedp[:, -1] - sortedp[:, -2]),
                             classical_features(pos, mask, move,
                                                np.array([d in ab_depths for d in (2, 4, 6, 8)]))))
    # No square roots / norms in the learned loss; NumPy standardization floors std.
    x = np.concatenate((scalar, c4.features_batch(pos, mask).reshape(len(pos), -1),
                        prior, vprob, np.nan_to_num(q), search.root_value[:, None]), axis=1)
    assert x.shape == (len(pos), 117) and np.isfinite(x).all()
    return x.astype(np.float32), move, np.argmax(prior, 1)


def learner_features(x):
    """The head/linear model use board + cheap probes, not paid alpha-beta outputs."""
    out = np.asarray(x).copy()
    out[:, 7:11] = 0
    return out


def calibration(y, probability, bins=10):
    y, p = np.asarray(y), np.clip(probability, 1e-7, 1-1e-7)
    assert len(y) == len(p) and len(y) > 0 and np.isfinite(p).all()
    table, ece = [], 0.0
    for b in range(bins):
        take = np.minimum((p * bins).astype(int), bins - 1) == b
        n = int(take.sum())
        if n:
            pred, rate = float(p[take].mean()), float(y[take].mean())
            ece += n / len(y) * abs(pred - rate)
            table.append(dict(bin=b, n=n, predicted=pred, observed=rate))
    return dict(n=len(y), prevalence=float(y.mean()), brier=float(np.mean((p-y)**2)),
                nll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))), ece=ece, bins=table)


def group_masks(values, group, base_y):
    return {"all": np.ones(len(values), bool), "early": group == 0,
            "middle": group == 1, "late": group == 2,
            "drawn": values.max(1) == 0, "winnable": values.max(1) == 1,
            "lost": values.max(1) == -1, "probe_mistakes": base_y == 1}


def report_moves(values, moves, initial, sims, elapsed):
    err, base = mistakes(values, moves), mistakes(values, initial)
    rescued = int(((base == 1) & (err == 0)).sum())
    introduced = int(((base == 0) & (err == 1)).sum())
    nonlost = values.max(1) >= 0
    return dict(n=len(moves), mistakes=int(err.sum()), rescued=rescued, introduced=introduced,
                net_avoided=rescued-introduced, nonlost_n=int(nonlost.sum()),
                nonlost_error=float(err[nonlost].mean()) if nonlost.any() else None,
                simulations=int(sims), seconds=float(elapsed),
                avoided_per_ms=(rescued-introduced)/max(1000*elapsed, 1e-12),
                avoided_per_sim=(rescued-introduced)/max(sims, 1))
