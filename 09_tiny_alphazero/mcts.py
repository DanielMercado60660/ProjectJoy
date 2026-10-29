"""Game-parallel batched PUCT MCTS (AlphaZero style).

One tree per game, all trees live in flat numba arrays. One simulation step:
  1. `_select`   (numba, parallel over games): walk each tree down with PUCT to an
                 unexpanded leaf; terminal leaves are backed up immediately, the
                 others write their board features into a batch buffer.
  2. the network evaluates every pending leaf in ONE batched call (MLX on the GPU).
  3. `_expand_backup` (numba, parallel over games): softmax priors over legal moves,
                 Dirichlet noise at the root, create children, back the value up.
So with G concurrent games the GPU sees batches of ~G positions per step.

Value convention: W[node] accumulates values from the point of view of the player
who made the move INTO `node` (i.e. the player to move at its parent), so a parent
simply picks the child maximising Q + U.
"""
import numpy as np
from numba import njit, prange

import c4
from c4 import W, H

ILLEGAL, OPEN, WIN, DRAW = -1, 0, 1, 2


@njit(cache=True, parallel=True)
def _reset(n, rpos, rmask, npos, nmask, parent, child0, N, Wv, P, term, nnodes):
    for g in prange(n):
        npos[g, 0] = rpos[g]
        nmask[g, 0] = rmask[g]
        parent[g, 0] = -1
        child0[g, 0] = -1
        N[g, 0] = 0.0
        Wv[g, 0] = 0.0
        P[g, 0] = 1.0
        term[g, 0] = OPEN
        nnodes[g] = 1


@njit(cache=True, inline="always")
def _backup(g, node, val, parent, N, Wv):
    while node != -1:
        N[g, node] += 1.0
        Wv[g, node] += val
        val = -val
        node = parent[g, node]


@njit(cache=True, parallel=True)
def _select(n, c_puct, fpu, npos, nmask, parent, child0, N, Wv, P, term, leaf, feats):
    for g in prange(n):
        node = 0
        while child0[g, node] != -1:
            c0 = child0[g, node]
            sq = np.sqrt(N[g, node])
            best = -1
            bs = -1e30
            for a in range(W):
                ch = c0 + a
                if term[g, ch] == ILLEGAL:
                    continue
                nc = N[g, ch]
                q = Wv[g, ch] / nc if nc > 0 else fpu
                s = q + c_puct * P[g, ch] * sq / (1.0 + nc)
                if s > bs:
                    bs = s
                    best = ch
            node = best
        t = term[g, node]
        if t == WIN or t == DRAW:
            _backup(g, node, 1.0 if t == WIN else 0.0, parent, N, Wv)
            leaf[g] = -1
        else:
            leaf[g] = node
            c4.write_features(npos[g, node], nmask[g, node], feats[g])


@njit(cache=True, parallel=True)
def _expand_backup(n, leaf, logits, values, add_noise, alpha, eps,
                   npos, nmask, parent, child0, N, Wv, P, term, nnodes):
    for g in prange(n):
        node = leaf[g]
        if node < 0:
            continue
        p, m = npos[g, node], nmask[g, node]
        # softmax over legal moves
        mx = -1e30
        for a in range(W):
            if c4.can_play(m, a) and logits[g, a] > mx:
                mx = logits[g, a]
        pri = np.zeros(W, np.float32)
        tot = 0.0
        for a in range(W):
            if c4.can_play(m, a):
                pri[a] = np.exp(logits[g, a] - mx)
                tot += pri[a]
        for a in range(W):
            pri[a] /= tot
        if add_noise and node == 0:
            d = np.zeros(W, np.float32)
            dt = 0.0
            for a in range(W):
                if c4.can_play(m, a):
                    d[a] = np.random.gamma(alpha, 1.0)
                    dt += d[a]
            for a in range(W):
                pri[a] = (1 - eps) * pri[a] + eps * d[a] / dt
        c0 = nnodes[g]
        nnodes[g] = c0 + W
        for a in range(W):
            ch = c0 + a
            parent[g, ch] = node
            child0[g, ch] = -1
            N[g, ch] = 0.0
            Wv[g, ch] = 0.0
            if not c4.can_play(m, a):
                term[g, ch] = ILLEGAL
                P[g, ch] = 0.0
                continue
            P[g, ch] = pri[a]
            won = c4.winning_move(p, m, a)
            p2, m2 = c4.play(p, m, a)
            npos[g, ch] = p2
            nmask[g, ch] = m2
            if won:
                term[g, ch] = WIN
            elif c4.is_full(m2):
                term[g, ch] = DRAW
            else:
                term[g, ch] = OPEN
        child0[g, node] = c0
        # net value is for the player to move at `node`; store it for the mover into node
        _backup(g, node, -values[g], parent, N, Wv)


@njit(cache=True)
def _root_stats(n, child0, N, Wv, P, term):
    visits = np.zeros((n, W), np.float32)
    q = np.zeros((n, W), np.float32)
    prior = np.zeros((n, W), np.float32)
    for g in range(n):
        c0 = child0[g, 0]
        if c0 < 0:
            continue
        for a in range(W):
            ch = c0 + a
            if term[g, ch] == ILLEGAL:
                continue
            visits[g, a] = N[g, ch]
            prior[g, a] = P[g, ch]
            q[g, a] = Wv[g, ch] / N[g, ch] if N[g, ch] > 0 else np.nan
    return visits, q, prior


class BatchMCTS:
    """Holds the tree pool for up to `max_games` concurrent searches."""

    def __init__(self, max_games, sims, c_puct=1.5, fpu=0.0,
                 dir_alpha=1.0, dir_eps=0.25):
        self.G = max_games
        self.sims = sims
        self.c_puct = c_puct
        self.fpu = fpu
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        M = 1 + (sims + 1) * W
        G = max_games
        self.npos = np.zeros((G, M), np.int64)
        self.nmask = np.zeros((G, M), np.int64)
        self.parent = np.full((G, M), -1, np.int64)
        self.child0 = np.full((G, M), -1, np.int64)
        self.N = np.zeros((G, M), np.float32)
        self.Wv = np.zeros((G, M), np.float32)
        self.P = np.zeros((G, M), np.float32)
        self.term = np.zeros((G, M), np.int8)
        self.nnodes = np.zeros(G, np.int64)
        self.leaf = np.zeros(G, np.int64)
        self.feats = np.zeros((G, H, W, 2), np.float32)
        self.logits = np.zeros((G, W), np.float32)
        self.values = np.zeros(G, np.float32)
        self.n_evals = 0
        self.n_batches = 0

    def search(self, pos, mask, evaluate, add_noise=False, sims=None):
        """pos, mask: int64 arrays (n,). evaluate(feats (b,H,W,2)) -> (logits (b,W), values (b,)).
        Returns root (visits, q, prior), each (n, W)."""
        n = len(pos)
        assert n <= self.G
        sims = self.sims if sims is None else sims
        assert sims <= self.sims
        _reset(n, np.asarray(pos, np.int64), np.asarray(mask, np.int64), self.npos, self.nmask,
               self.parent, self.child0, self.N, self.Wv, self.P, self.term, self.nnodes)
        # +1: the first simulation only expands the root
        for _ in range(sims + 1):
            _select(n, self.c_puct, self.fpu, self.npos, self.nmask, self.parent, self.child0,
                    self.N, self.Wv, self.P, self.term, self.leaf, self.feats)
            idx = np.nonzero(self.leaf[:n] >= 0)[0]
            if len(idx):
                lg, v = evaluate(self.feats[idx])
                self.logits[idx] = lg
                self.values[idx] = v
                self.n_evals += len(idx)
                self.n_batches += 1
            _expand_backup(n, self.leaf, self.logits, self.values, add_noise,
                           self.dir_alpha, self.dir_eps, self.npos, self.nmask, self.parent,
                           self.child0, self.N, self.Wv, self.P, self.term, self.nnodes)
        return _root_stats(n, self.child0, self.N, self.Wv, self.P, self.term)

    def export_tree(self, g, max_depth=4, min_visits=1):
        """Nested dict of game g's tree for visualisation."""
        def rec(node, depth, move):
            d = {"move": move, "N": float(self.N[g, node]),
                 "Q": float(self.Wv[g, node] / self.N[g, node]) if self.N[g, node] > 0 else None,
                 "P": float(self.P[g, node]), "term": int(self.term[g, node]),
                 "pos": int(self.npos[g, node]), "mask": int(self.nmask[g, node]),
                 "children": []}
            c0 = self.child0[g, node]
            if c0 >= 0 and depth < max_depth:
                for a in range(W):
                    ch = c0 + a
                    if self.term[g, ch] != ILLEGAL and self.N[g, ch] >= min_visits:
                        d["children"].append(rec(ch, depth + 1, a))
            return d
        return rec(0, 0, None)


def uniform_evaluator(feats):
    """'No trained network': uniform priors, value 0. MCTS then is pure tree search
    driven only by terminal states."""
    b = feats.shape[0]
    return np.zeros((b, W), np.float32), np.zeros(b, np.float32)
