"""Exact Connect Four solver (7x6), after Pascal Pons' bitboard solver
(http://blog.gamesolver.org): negamax alpha-beta on bitboards, centre-first move
ordering refined by "how many winning spots does this move create", anticipation of
losing moves (only non-losing moves are searched), a transposition table with lower
and upper bounds and a mirror-canonical key, and null-window iterative search.

Uses the same (pos, mask) bitboards as c4.py: pos = stones of the player to move.

Score convention (Pons): from the view of the player to move,
  0  = draw,
  >0 = the player to move wins; score = 22 - k where k is the winner's stone count
       at the win (so +21 = win with your first stone, +1 = win with your last),
  <0 = the player to move loses, same magnitude rule for the opponent's stones.
`win_ply(n, score)` turns a score into the ply number (1-based, total stones on the
board after the winning move) at which the game ends under perfect play.

A node budget (not wall time; numba has no cheap clock) bounds every search. When it
trips, the search unwinds, no transposition-table entries are written on the way out,
and the result is reported as unsolved.
"""
import numpy as np
from numba import njit

import c4
from c4 import W, H, H1, NCELLS, BOTTOM_ROW, BOARD_MASK

MIN_SCORE = -(NCELLS // 2) + 3     # -18: fastest possible loss
MAX_SCORE = (NCELLS + 1) // 2 - 3  # +18: fastest possible win
UNSOLVED = -1000

# centre-first order, as in c4.CENTER_ORDER
ORDER = np.array(sorted(range(W), key=lambda c: abs(c - (W - 1) / 2)), dtype=np.int64)

_BOTTOM = np.int64(BOTTOM_ROW)
_BOARD = np.int64(BOARD_MASK)


@njit(cache=True, inline="always")
def column_mask(c):
    return np.int64((1 << H) - 1) << np.int64(c * H1)


@njit(cache=True, inline="always")
def tdiv2(x):
    """C-style integer division by 2 (truncate toward zero), as in Pons' code."""
    return -((-x) // 2) if x < 0 else x // 2


@njit(cache=True)
def popcount(x):
    cnt = 0
    while x:
        x &= x - 1
        cnt += 1
    return cnt


@njit(cache=True)
def winning_spots(p, mask):
    """Empty cells where stones `p` would complete four in a row (any height)."""
    # vertical
    r = (p << 1) & (p << 2) & (p << 3)
    for s in (H1, H, H + 2):         # horizontal, diagonal (\ in bit space), diagonal (/)
        t = (p << s) & (p << (2 * s))
        r |= t & (p << (3 * s))
        r |= t & (p >> s)
        t = (p >> s) & (p >> (2 * s))
        r |= t & (p << s)
        r |= t & (p >> (3 * s))
    return r & (_BOARD ^ mask)


@njit(cache=True, inline="always")
def possible(mask):
    return (mask + _BOTTOM) & _BOARD


@njit(cache=True)
def can_win_next(pos, mask):
    return (winning_spots(pos, mask) & possible(mask)) != 0


@njit(cache=True)
def non_losing_moves(pos, mask):
    """Bitmap of moves that don't let the opponent win immediately (0 if every move loses).
    Assumes the player to move cannot win immediately."""
    poss = possible(mask)
    opp_win = winning_spots(pos ^ mask, mask)
    forced = poss & opp_win
    if forced:
        if forced & (forced - 1):
            return np.int64(0)       # two threats: lost
        poss = forced
    return poss & ~(opp_win >> 1)    # never play directly below an opponent winning spot


@njit(cache=True)
def mirror_key(key):
    out = np.int64(0)
    cm = np.int64((1 << H1) - 1)
    for c in range(W):
        col = (key >> np.int64(c * H1)) & cm
        out |= col << np.int64((W - 1 - c) * H1)
    return out


@njit(cache=True)
def canon_key(pos, mask):
    k = pos + mask
    m = mirror_key(k)
    return k if k < m else m


# --------------------------------------------------------------------- the search
# stats: int64 array [nodes, node_limit, aborted]
@njit(cache=True)
def negamax(pos, mask, n_moves, alpha, beta, tt_keys, tt_vals, stats):
    stats[0] += 1
    if stats[0] > stats[1]:
        stats[2] = 1
        return 0
    nxt = non_losing_moves(pos, mask)
    if nxt == 0:
        return -tdiv2(NCELLS - n_moves)
    if n_moves >= NCELLS - 2:
        return 0
    lo = -tdiv2(NCELLS - 2 - n_moves)
    if alpha < lo:
        alpha = lo
        if alpha >= beta:
            return alpha
    hi = tdiv2(NCELLS - 1 - n_moves)
    key = canon_key(pos, mask)
    size = tt_keys.shape[0]
    idx = key % size
    if tt_keys[idx] == key:
        v = np.int64(tt_vals[idx])
        if v > MAX_SCORE - MIN_SCORE + 1:          # lower bound
            lb = v + 2 * MIN_SCORE - MAX_SCORE - 2
            if alpha < lb:
                alpha = lb
                if alpha >= beta:
                    return alpha
        else:                                      # upper bound
            ub = v + MIN_SCORE - 1
            if beta > ub:
                beta = ub
                if alpha >= beta:
                    return beta
    if beta > hi:
        beta = hi
        if alpha >= beta:
            return beta
    # move ordering: more winning spots created first, ties centre-first
    mv = np.zeros(W, np.int64)
    sc = np.zeros(W, np.int64)
    n = 0
    for i in range(W):
        m = nxt & column_mask(ORDER[i])
        if m:
            s = popcount(winning_spots(pos | m, mask))
            j = n
            while j > 0 and sc[j - 1] < s:       # stable insertion: ties keep centre order
                mv[j] = mv[j - 1]
                sc[j] = sc[j - 1]
                j -= 1
            mv[j] = m
            sc[j] = s
            n += 1
    for i in range(n):
        m = mv[i]
        score = -negamax(pos ^ mask, mask | m, n_moves + 1, -beta, -alpha, tt_keys, tt_vals, stats)
        if stats[2]:
            return 0
        if score >= beta:
            tt_keys[idx] = key
            tt_vals[idx] = np.int8(score + MAX_SCORE - 2 * MIN_SCORE + 2)
            return score
        if score > alpha:
            alpha = score
    tt_keys[idx] = key
    tt_vals[idx] = np.int8(alpha - MIN_SCORE + 1)
    return alpha


@njit(cache=True)
def solve_nb(pos, mask, weak, tt_keys, tt_vals, stats):
    """Exact score (weak=False) or its sign in {-1,0,1} (weak=True). Sets stats[2] on abort."""
    n_moves = popcount(mask)
    if can_win_next(pos, mask):
        r = tdiv2(NCELLS + 1 - n_moves)
        return 1 if weak else r
    if n_moves >= NCELLS:
        return 0
    lo = -tdiv2(NCELLS - n_moves)
    hi = tdiv2(NCELLS + 1 - n_moves)
    if weak:
        lo = -1
        hi = 1
    while lo < hi:
        med = lo + tdiv2(hi - lo)
        if med <= 0 and tdiv2(lo) < med:
            med = tdiv2(lo)
        elif med >= 0 and tdiv2(hi) > med:
            med = tdiv2(hi)
        r = negamax(pos, mask, n_moves, med, med + 1, tt_keys, tt_vals, stats)
        if stats[2]:
            return 0
        if r <= med:
            hi = r
        else:
            lo = r
    if weak:
        return 1 if lo > 0 else (-1 if lo < 0 else 0)
    return lo


def _prime_below(n):
    def isprime(k):
        if k < 2:
            return False
        i = 2
        while i * i <= k:
            if k % i == 0:
                return False
            i += 1
        return True
    while not isprime(n):
        n -= 1
    return n


class Solver:
    """Holds a transposition table. `log2_size=25` is ~300 MB (int64 key + int8 value)."""

    def __init__(self, log2_size=24):
        size = _prime_below(1 << log2_size)
        self.tt_keys = np.full(size, -1, np.int64)
        self.tt_vals = np.zeros(size, np.int8)
        self.nodes = 0

    def reset(self):
        self.tt_keys[:] = -1

    def solve(self, pos, mask, weak=False, node_limit=10**12):
        """Returns the score (or W/D/L sign if weak), or UNSOLVED if the node budget ran out."""
        stats = np.array([0, node_limit, 0], np.int64)
        r = solve_nb(np.int64(pos), np.int64(mask), weak, self.tt_keys, self.tt_vals, stats)
        self.nodes += int(stats[0])
        self.last_nodes = int(stats[0])
        return UNSOLVED if stats[2] else int(r)

    def move_values(self, pos, mask, weak=True, node_limit=10**12):
        """Value of every column from the mover's view after playing it (np.int64 (W,)):
        weak -> +1 win / 0 draw / -1 loss; strong -> exact score. Illegal columns get
        -999, unsolved ones UNSOLVED. The node budget is shared by the whole call."""
        pos, mask = np.int64(pos), np.int64(mask)
        n = popcount(mask)
        out = np.full(W, -999, np.int64)
        budget = node_limit
        for c in range(W):
            if not c4.can_play(mask, c):
                continue
            if c4.winning_move(pos, mask, c):
                out[c] = 1 if weak else tdiv2(NCELLS + 1 - n)
                continue
            p2, m2 = c4.play(pos, mask, c)
            if c4.is_full(m2):
                out[c] = 0
                continue
            r = self.solve(p2, m2, weak=weak, node_limit=max(budget, 1))
            budget -= self.last_nodes
            out[c] = UNSOLVED if r == UNSOLVED else -r
        return out


def win_ply(n_moves, score):
    """Ply (1-based count of stones on the board after the winning move) at which the game
    ends with perfect play, for a position with n_moves stones and Pons score `score`
    (from the mover's view). None for a draw."""
    if score == 0:
        return None
    k = 22 - abs(score)                       # winner's stone count at the win
    mover_is_first = n_moves % 2 == 0
    winner_is_first = mover_is_first if score > 0 else not mover_is_first
    return 2 * k - 1 if winner_is_first else 2 * k


# ------------------------------------------------------------ brute-force reference
@njit(cache=True)
def brute_force(pos, mask, n_moves):
    """Plain minimax over the full tree (no pruning, no table): exact Pons score.
    Only for positions with few empty cells."""
    best = -1000
    for c in range(W):
        if not c4.can_play(mask, c):
            continue
        if c4.winning_move(pos, mask, c):
            v = tdiv2(NCELLS + 1 - n_moves)
        else:
            p2, m2 = c4.play(pos, mask, c)
            if c4.is_full(m2):
                v = 0
            else:
                v = -brute_force(p2, m2, n_moves + 1)
        if v > best:
            best = v
    return best
