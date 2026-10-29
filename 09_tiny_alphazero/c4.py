"""Connect Four engine on int64 bitboards (numba), plus the fixed opponent suite.

Board layout (same trick as Pascal Pons' solver): column-major, each column uses
H+1 bits (one sentinel bit on top), bit index = col * (H + 1) + row, row 0 = bottom.

A position is (pos, mask):
  mask = every stone on the board
  pos  = stones of the player *to move*
After a move we flip perspective: pos ^= mask; mask |= mask + bottom(col).
So the player who just moved owns (pos ^ mask) afterwards.

Everything here is plain numba so it can be called from tight loops
(batched MCTS, arenas, the alpha-beta and rollout-MCTS baselines).
"""
import numpy as np
from numba import njit

W = 7          # columns
H = 6          # rows
H1 = H + 1     # bits per column (with sentinel)
NCELLS = W * H

BOTTOM_ROW = 0
for _c in range(W):
    BOTTOM_ROW |= 1 << (_c * H1)
BOARD_MASK = BOTTOM_ROW * ((1 << H) - 1)   # all playable cells
# center-first move ordering: 3,2,4,1,5,0,6
CENTER_ORDER = np.array(sorted(range(W), key=lambda c: abs(c - (W - 1) / 2)), dtype=np.int64)


# ----------------------------------------------------------------------------- core
@njit(cache=True, inline="always")
def top_bit(c):
    return np.int64(1) << np.int64(c * H1 + H - 1)


@njit(cache=True, inline="always")
def bottom_bit(c):
    return np.int64(1) << np.int64(c * H1)


@njit(cache=True, inline="always")
def can_play(mask, c):
    return (mask & top_bit(c)) == 0


@njit(cache=True, inline="always")
def play(pos, mask, c):
    """Returns the new (pos, mask) with perspective flipped to the next player."""
    new_pos = pos ^ mask
    new_mask = mask | (mask + bottom_bit(c))
    return new_pos, new_mask


@njit(cache=True, inline="always")
def aligned(x):
    """True if bitboard x contains 4 in a row (any direction)."""
    # horizontal
    m = x & (x >> H1)
    if m & (m >> (2 * H1)):
        return True
    # diagonal /
    m = x & (x >> (H1 + 1))
    if m & (m >> (2 * (H1 + 1))):
        return True
    # diagonal \
    m = x & (x >> (H1 - 1))
    if m & (m >> (2 * (H1 - 1))):
        return True
    # vertical
    m = x & (x >> 1)
    if m & (m >> 2):
        return True
    return False


@njit(cache=True, inline="always")
def last_mover_won(pos, mask):
    return aligned(pos ^ mask)


@njit(cache=True, inline="always")
def is_full(mask):
    return mask == np.int64(BOARD_MASK)


@njit(cache=True, inline="always")
def winning_move(pos, mask, c):
    """Would the player to move win by playing column c? (c must be legal)"""
    stone = (mask + bottom_bit(c)) & (np.int64((1 << H) - 1) << np.int64(c * H1))
    return aligned(pos | stone)


@njit(cache=True)
def legal_moves(mask):
    out = np.zeros(W, dtype=np.bool_)
    for c in range(W):
        out[c] = can_play(mask, c)
    return out


@njit(cache=True)
def n_stones(mask):
    cnt = 0
    x = mask
    while x:
        x &= x - 1
        cnt += 1
    return cnt


@njit(cache=True)
def popcount(x):
    cnt = 0
    while x:
        x &= x - 1
        cnt += 1
    return cnt


@njit(cache=True)
def status(pos, mask):
    """0 = ongoing, 1 = player who just moved has won, 2 = draw (board full)."""
    if last_mover_won(pos, mask):
        return 1
    if is_full(mask):
        return 2
    return 0


@njit(cache=True)
def write_features(pos, mask, out):
    """out: (H, W, 2) float32. plane 0 = stones of player to move, plane 1 = opponent."""
    opp = pos ^ mask
    for c in range(W):
        for r in range(H):
            b = np.int64(1) << np.int64(c * H1 + r)
            out[r, c, 0] = 1.0 if (pos & b) else 0.0
            out[r, c, 1] = 1.0 if (opp & b) else 0.0


@njit(cache=True)
def features_batch(pos_arr, mask_arr):
    n = pos_arr.shape[0]
    out = np.zeros((n, H, W, 2), dtype=np.float32)
    for i in range(n):
        write_features(pos_arr[i], mask_arr[i], out[i])
    return out


def mirror(pos, mask):
    """Left-right mirror of a position (python ints)."""
    def m(x):
        y = 0
        for c in range(W):
            col = (x >> (c * H1)) & ((1 << H1) - 1)
            y |= col << ((W - 1 - c) * H1)
        return y
    return m(pos), m(mask)


# ---------------------------------------------------------------- python helpers
def from_moves(moves):
    """Build a position from a sequence of 0-based columns. Returns (pos, mask)."""
    pos, mask = np.int64(0), np.int64(0)
    for c in moves:
        assert can_play(mask, c), f"illegal move {c}"
        assert status(pos, mask) == 0, "game already over"
        pos, mask = play(pos, mask, c)
    return pos, mask


def to_grid(pos, mask):
    """(H, W) int grid, row 0 = TOP for display. 1 = first player (X), 2 = second (O)."""
    pos, mask = int(pos), int(mask)
    first_to_move = popcount(np.int64(mask)) % 2 == 0
    me, opp = pos, pos ^ mask
    first, second = (me, opp) if first_to_move else (opp, me)
    g = np.zeros((H, W), dtype=np.int64)
    for c in range(W):
        for r in range(H):
            b = 1 << (c * H1 + r)
            if first & b:
                g[H - 1 - r, c] = 1
            elif second & b:
                g[H - 1 - r, c] = 2
    return g


def render_ascii(pos, mask):
    g = to_grid(pos, mask)
    sym = {0: ".", 1: "X", 2: "O"}
    rows = [" ".join(sym[v] for v in row) for row in g]
    return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(W))


# ------------------------------------------------------------------ opponent suite
@njit(cache=True)
def random_move(pos, mask):
    legal = np.empty(W, dtype=np.int64)
    n = 0
    for c in range(W):
        if can_play(mask, c):
            legal[n] = c
            n += 1
    return legal[np.random.randint(n)]


@njit(cache=True)
def greedy_move(pos, mask):
    """Win if you can, else block an immediate opponent win, else avoid moves that
    hand the opponent a win on top, else random."""
    for c in range(W):
        if can_play(mask, c) and winning_move(pos, mask, c):
            return c
    opp = pos ^ mask
    for c in range(W):
        if can_play(mask, c) and winning_move(opp, mask, c):
            return c
    safe = np.empty(W, dtype=np.int64)
    ns = 0
    for c in range(W):
        if can_play(mask, c):
            p2, m2 = play(pos, mask, c)
            ok = True
            if can_play(m2, c) and winning_move(p2, m2, c):
                ok = False
            if ok:
                safe[ns] = c
                ns += 1
    if ns > 0:
        return safe[np.random.randint(ns)]
    return random_move(pos, mask)


# --- alpha-beta with a window-count heuristic
def _make_windows():
    wins = []
    for c in range(W):
        for r in range(H):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= cc < W and 0 <= rr < H for cc, rr in cells):
                    m = 0
                    for cc, rr in cells:
                        m |= 1 << (cc * H1 + rr)
                    wins.append(m)
    return np.array(wins, dtype=np.int64)


WINDOWS = _make_windows()          # 69 four-cell windows on 7x6
WIN_SCORE = 100000


@njit(cache=True)
def heuristic(pos, mask, windows):
    """Score from the point of view of the player to move."""
    opp = pos ^ mask
    s = 0
    for i in range(windows.shape[0]):
        w = windows[i]
        a = popcount(pos & w)
        b = popcount(opp & w)
        if a > 0 and b == 0:
            s += 1 if a == 1 else (5 if a == 2 else 25)
        elif b > 0 and a == 0:
            s -= 1 if b == 1 else (5 if b == 2 else 25)
    return s


@njit(cache=True)
def negamax(pos, mask, depth, alpha, beta, windows, order):
    # the player to move: can they win right now?
    for c in range(W):
        if can_play(mask, c) and winning_move(pos, mask, c):
            return WIN_SCORE + depth   # prefer faster wins
    if is_full(mask):
        return 0
    if depth == 0:
        return heuristic(pos, mask, windows)
    best = -10 * WIN_SCORE
    any_move = False
    for i in range(W):
        c = order[i]
        if not can_play(mask, c):
            continue
        any_move = True
        p2, m2 = play(pos, mask, c)
        if is_full(m2):
            v = 0
        else:
            v = -negamax(p2, m2, depth - 1, -beta, -alpha, windows, order)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    if not any_move:
        return 0
    return best


@njit(cache=True)
def alphabeta_move(pos, mask, depth, windows, order):
    """Depth-limited negamax; ties at the root are broken uniformly at random so
    repeated games are not identical."""
    scores = np.full(W, -10 * WIN_SCORE, dtype=np.int64)
    best = -10 * WIN_SCORE
    for c in range(W):
        if not can_play(mask, c):
            continue
        if winning_move(pos, mask, c):
            return c
        p2, m2 = play(pos, mask, c)
        if is_full(m2):
            v = 0
        else:
            # window (best-1, inf) keeps exact scores for moves tied with the best
            v = -negamax(p2, m2, depth - 1, -10 * WIN_SCORE, -(best - 1), windows, order)
        scores[c] = v
        if v > best:
            best = v
    cand = np.empty(W, dtype=np.int64)
    n = 0
    for c in range(W):
        if can_play(mask, c) and scores[c] == best:
            cand[n] = c
            n += 1
    return cand[np.random.randint(n)]


# --- pure MCTS (UCT) with uniformly random rollouts
@njit(cache=True)
def _rollout(pos, mask):
    """Random playout. Returns result from the view of the player to move at the start:
    +1 win, -1 loss, 0 draw."""
    sign = 1.0
    while True:
        c = random_move(pos, mask)
        if winning_move(pos, mask, c):
            return sign
        pos, mask = play(pos, mask, c)
        if is_full(mask):
            return 0.0
        sign = -sign


@njit(cache=True)
def uct_search(pos, mask, budget, c_uct):
    """Plain UCT with random rollouts. Returns root child visit counts (W,)."""
    M = 1 + (budget + 1) * W
    npos = np.zeros(M, np.int64)
    nmask = np.zeros(M, np.int64)
    parent = np.full(M, -1, np.int64)
    child0 = np.full(M, -1, np.int64)
    N = np.zeros(M, np.float64)
    Wv = np.zeros(M, np.float64)
    term = np.zeros(M, np.int8)   # -1 illegal, 0 open, 1 mover won, 2 draw
    npos[0] = pos
    nmask[0] = mask
    nn = 1
    for _ in range(budget):
        node = 0
        # selection
        while child0[node] != -1:
            c0 = child0[node]
            best = -1
            bs = -1e30
            lnN = np.log(N[node] + 1.0)
            for a in range(W):
                ch = c0 + a
                if term[ch] == -1:
                    continue
                if N[ch] == 0:
                    s = 1e20 + np.random.random()   # visit unvisited children first
                else:
                    s = Wv[ch] / N[ch] + c_uct * np.sqrt(lnN / N[ch])
                if s > bs:
                    bs = s
                    best = ch
            node = best
        # evaluate
        if term[node] == 1:
            val = 1.0
        elif term[node] == 2:
            val = 0.0
        else:
            if N[node] > 0 or node == 0:
                # expand
                c0 = nn
                nn += W
                p, m = npos[node], nmask[node]
                for a in range(W):
                    ch = c0 + a
                    parent[ch] = node
                    if not can_play(m, a):
                        term[ch] = -1
                        continue
                    if winning_move(p, m, a):
                        term[ch] = 1
                    p2, m2 = play(p, m, a)
                    npos[ch] = p2
                    nmask[ch] = m2
                    if term[ch] == 0 and is_full(m2):
                        term[ch] = 2
                child0[node] = c0
                # pick a random legal child to roll out from
                while True:
                    a = np.random.randint(W)
                    if term[c0 + a] != -1:
                        break
                node = c0 + a
                if term[node] == 1:
                    val = 1.0
                elif term[node] == 2:
                    val = 0.0
                else:
                    val = -_rollout(npos[node], nmask[node])
            else:
                # value for the player who moved into node = -(result for player to move)
                val = -_rollout(npos[node], nmask[node])
        # backup (val is from the perspective of the player who moved into `node`)
        while node != -1:
            N[node] += 1.0
            Wv[node] += val
            val = -val
            node = parent[node]
    out = np.zeros(W, np.float64)
    c0 = child0[0]
    if c0 >= 0:
        for a in range(W):
            if term[c0 + a] != -1:
                out[a] = N[c0 + a]
    return out


@njit(cache=True)
def uct_move(pos, mask, budget):
    # take an immediate win if it exists (UCT finds it anyway, this just saves noise)
    for c in range(W):
        if can_play(mask, c) and winning_move(pos, mask, c):
            return c
    v = uct_search(pos, mask, budget, 1.4)
    best = -1.0
    n = 0
    cand = np.empty(W, np.int64)
    for c in range(W):
        if v[c] > best:
            best = v[c]
            n = 0
        if v[c] == best and can_play(mask, c):
            cand[n] = c
            n += 1
    return cand[np.random.randint(n)]


@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)
