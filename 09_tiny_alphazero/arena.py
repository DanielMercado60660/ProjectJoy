"""Agents + a game-parallel arena.

All games of a match run in lock-step; whenever a network agent is to move in
several games, those positions are searched together as one batched MCTS call.

Agent specs (strings):
  random | greedy | ab<d> (alpha-beta depth d) | uct<n> (pure MCTS, n random-rollout sims)
  net:<ckpt_prefix>:<sims>   AlphaZero net + PUCT MCTS (sims=0 -> raw policy argmax)
  mcts0:<sims>               PUCT MCTS with NO trained net (uniform priors, value 0;
                             only terminal states carry information)

Openings: every match is played from a fixed list of 2-ply openings, each played
twice with colours swapped, so first-move advantage cancels out.
"""
import itertools
import time

import numpy as np

import c4
from c4 import W
from mcts import BatchMCTS, uniform_evaluator

ALL_2PLY = [list(o) for o in itertools.product(range(W), range(W))]   # 49 openings


class Agent:
    name = "agent"
    batched = False
    think_time = 0.0
    n_moves = 0

    def moves(self, pos, mask):
        raise NotImplementedError


class FnAgent(Agent):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn
        self.think_time, self.n_moves = 0.0, 0

    def moves(self, pos, mask):
        t = time.perf_counter()
        out = np.array([self.fn(p, m) for p, m in zip(pos, mask)], np.int64)
        self.think_time += time.perf_counter() - t
        self.n_moves += len(out)
        return out


class MCTSAgent(Agent):
    """Batched PUCT search with a network (or the uniform evaluator)."""
    batched = True

    def __init__(self, name, evaluate, sims, max_games=256, c_puct=1.5, seed=0):
        self.name, self.evaluate, self.sims = name, evaluate, sims
        self.mcts = BatchMCTS(max_games, max(sims, 1), c_puct=c_puct) if sims > 0 else None
        self.rng = np.random.default_rng(seed)
        self.think_time, self.n_moves = 0.0, 0
        self.last = None

    def moves(self, pos, mask):
        t = time.perf_counter()
        n = len(pos)
        out = np.zeros(n, np.int64)
        if self.sims == 0:
            from c4 import features_batch
            logits, _ = self.evaluate(features_batch(np.asarray(pos), np.asarray(mask)))
            for i in range(n):
                legal = c4.legal_moves(mask[i])
                lg = np.where(legal, logits[i], -np.inf)
                out[i] = int(np.argmax(lg))
        else:
            for s in range(0, n, self.mcts.G):
                visits, q, prior = self.mcts.search(pos[s:s + self.mcts.G], mask[s:s + self.mcts.G], self.evaluate)
                for i in range(len(visits)):
                    v = visits[i]
                    best = np.flatnonzero(v == v.max())
                    out[s + i] = self.rng.choice(best)
                self.last = (visits, q, prior)
        self.think_time += time.perf_counter() - t
        self.n_moves += n
        return out


_NET_CACHE = {}


def load_evaluator(prefix):
    if prefix not in _NET_CACHE:
        from net import load_model, make_evaluator
        model, _ = load_model(prefix)
        _NET_CACHE[prefix] = make_evaluator(model)
    return _NET_CACHE[prefix]


def make_agent(spec, max_games=256, seed=0):
    if spec == "random":
        return FnAgent(spec, c4.random_move)
    if spec == "greedy":
        return FnAgent(spec, c4.greedy_move)
    if spec.startswith("ab"):
        d = int(spec[2:])
        return FnAgent(spec, lambda p, m: c4.alphabeta_move(p, m, d, c4.WINDOWS, c4.CENTER_ORDER))
    if spec.startswith("uct"):
        n = int(spec[3:])
        return FnAgent(spec, lambda p, m: c4.uct_move(p, m, n))
    if spec.startswith("mcts0:"):
        sims = int(spec.split(":")[1])
        return MCTSAgent(spec, uniform_evaluator, sims, max_games, seed=seed)
    if spec.startswith("net:"):
        _, prefix, sims = spec.split(":")
        return MCTSAgent(spec, load_evaluator(prefix), int(sims), max_games, seed=seed)
    raise ValueError(spec)


def play_match(a, b, openings=None, seed=0, record=False):
    """Plays every opening twice (a first, then b first). Returns dict with a's score
    (win=1, draw=.5), W/D/L, per-colour scores and optionally the move lists."""
    openings = ALL_2PLY if openings is None else openings
    c4.seed_numba(seed)
    n = 2 * len(openings)
    pos = np.zeros(n, np.int64)
    mask = np.zeros(n, np.int64)
    moves = [[] for _ in range(n)]
    a_first = np.array([i % 2 == 0 for i in range(n)])
    for i in range(n):
        for c in openings[i // 2]:
            pos[i], mask[i] = c4.play(pos[i], mask[i], c)
            moves[i].append(int(c))
    ply = len(openings[0])
    result = np.full(n, -2, np.int64)   # +1 first player won, -1 second won, 0 draw
    while (result == -2).any():
        act = np.nonzero(result == -2)[0]
        first_to_move = ply % 2 == 0
        a_to_move = act[a_first[act] == first_to_move]
        b_to_move = act[a_first[act] != first_to_move]
        mv = np.zeros(n, np.int64)
        if len(a_to_move):
            mv[a_to_move] = a.moves(pos[a_to_move], mask[a_to_move])
        if len(b_to_move):
            mv[b_to_move] = b.moves(pos[b_to_move], mask[b_to_move])
        for i in act:
            c = int(mv[i])
            assert c4.can_play(mask[i], c)
            won = c4.winning_move(pos[i], mask[i], c)
            pos[i], mask[i] = c4.play(pos[i], mask[i], c)
            moves[i].append(c)
            if won:
                result[i] = 1 if first_to_move else -1
            elif c4.is_full(mask[i]):
                result[i] = 0
        ply += 1
    a_score = np.where(result == 0, 0.5, np.where((result == 1) == a_first, 1.0, 0.0))
    out = dict(a=a.name, b=b.name, games=n, score=float(a_score.mean()),
               wins=int((a_score == 1).sum()), draws=int((a_score == 0.5).sum()),
               losses=int((a_score == 0).sum()),
               score_as_first=float(a_score[a_first].mean()),
               score_as_second=float(a_score[~a_first].mean()))
    if record:
        out["moves"] = moves
        out["a_first"] = a_first.tolist()
        out["result"] = result.tolist()
    return out


def fit_elo(pairs, anchor=None, iters=2000):
    """pairs: list of (name_a, name_b, score_a, games). Bradley-Terry MLE via
    minorization-maximization on (wins + draws/2). Returns {name: elo}, anchor = 0."""
    names = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    wins = np.zeros((k, k))
    for na, nb, s, g in pairs:
        i, j = idx[na], idx[nb]
        wins[i, j] += s * g
        wins[j, i] += (1 - s) * g
    # small prior: one virtual draw against each opponent played, keeps 100% scores finite
    played = (wins + wins.T) > 0
    wins = wins + 0.5 * played
    games = wins + wins.T
    r = np.ones(k)
    for _ in range(iters):
        num = wins.sum(1)
        den = np.array([sum(games[i, j] / (r[i] + r[j]) for j in range(k) if games[i, j] > 0) for i in range(k)])
        r = num / np.maximum(den, 1e-12)
        r /= np.exp(np.mean(np.log(r)))
    elo = 400 * np.log10(r)
    if anchor is not None and anchor in idx:
        elo = elo - elo[idx[anchor]]
    return {n: float(elo[idx[n]]) for n in names}
