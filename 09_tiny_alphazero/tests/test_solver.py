import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c4  # noqa: E402
import solver  # noqa: E402

S = solver.Solver(20)


def _random_position(rng, ply, greedy=False):
    """Uniform random (or greedy: win > block > don't give a win > random) play to `ply`
    stones, retried until nobody has won yet."""
    if greedy:
        c4.seed_numba(int(rng.integers(1 << 30)))
    while True:
        pos, mask, ok = np.int64(0), np.int64(0), True
        for _ in range(ply):
            if greedy:
                c = int(c4.greedy_move(pos, mask))
            else:
                c = int(rng.choice(np.flatnonzero(c4.legal_moves(mask))))
            if c4.winning_move(pos, mask, c):
                ok = False
                break
            pos, mask = c4.play(pos, mask, c)
        if ok:
            return pos, mask


def test_matches_brute_force_exact_and_per_move():
    """Exact scores and per-move win/draw/loss agree with full minimax (no pruning, no table)."""
    rng = np.random.default_rng(0)
    n_nontrivial = 0
    for empties in (4, 6, 8, 10, 11):
        for _ in range(25):
            pos, mask = _random_position(rng, c4.NCELLS - empties, greedy=_ % 2 == 0)
            n = c4.popcount(mask)
            ref = solver.brute_force(pos, mask, n)
            S.reset()
            assert S.solve(pos, mask) == ref
            n_nontrivial += S.last_nodes > 10
            assert S.solve(pos, mask, weak=True) == np.sign(ref)
            vals = S.move_values(pos, mask, weak=True)
            for c in range(c4.W):
                if not c4.can_play(mask, c):
                    assert vals[c] == -999
                    continue
                if c4.winning_move(pos, mask, c):
                    want = 1
                else:
                    p2, m2 = c4.play(pos, mask, c)
                    want = 0 if c4.is_full(m2) else -np.sign(solver.brute_force(p2, m2, n + 1))
                assert vals[c] == want
    assert n_nontrivial > 20      # the search really ran, not just immediate wins


def test_win_distance_decoding():
    # X: a0 b0 c0 stacked under O's replies; X to move wins now with d0 (4th X stone, ply 7)
    pos, mask = c4.from_moves([0, 0, 1, 1, 2, 2])
    s = S.solve(pos, mask)
    assert s == 22 - 4 and solver.win_ply(6, s) == 7
    # one ply earlier O is to move and loses to the same threat whatever it does... except
    # it can block c0/d0 only one at a time: X has an open three-in-a-row threat on both ends
    pos, mask = c4.from_moves([1, 1, 2, 2, 3])          # X b0 c0 d0, O b1 c1, O to move
    s = S.solve(pos, mask)
    assert s < 0 and solver.win_ply(5, s) == 7          # X wins with its 4th stone at ply 7


def test_node_budget_reports_unsolved_and_leaves_table_clean():
    pos, mask = c4.from_moves([3, 3])
    S.reset()
    assert S.solve(pos, mask, weak=True, node_limit=1000) == solver.UNSOLVED
    # after an aborted search, an exact result on an easy position is still right
    pos, mask = _random_position(np.random.default_rng(5), 32)
    assert S.solve(pos, mask) == solver.brute_force(pos, mask, c4.popcount(mask))


def test_mirror_symmetry():
    rng = np.random.default_rng(3)
    for _ in range(10):
        pos, mask = _random_position(rng, 26)
        mp, mm = c4.mirror(int(pos), int(mask))
        S.reset()
        a = S.solve(pos, mask)
        S.reset()
        assert S.solve(np.int64(mp), np.int64(mm)) == a
