import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import c4  # noqa: E402
from c4 import W, H  # noqa: E402


def test_empty_board_all_legal():
    pos, mask = c4.from_moves([])
    assert c4.legal_moves(mask).all()
    assert c4.status(pos, mask) == 0


def test_column_fills_up():
    pos, mask = c4.from_moves([3] * H)
    legal = c4.legal_moves(mask)
    assert not legal[3] and legal.sum() == W - 1
    with pytest.raises(AssertionError):
        c4.from_moves([3] * (H + 1))


def test_vertical_win():
    pos, mask = c4.from_moves([0, 1, 0, 1, 0, 1, 0])
    assert c4.status(pos, mask) == 1


def test_horizontal_win():
    pos, mask = c4.from_moves([0, 0, 1, 1, 2, 2, 3])
    assert c4.status(pos, mask) == 1


def test_no_wraparound_horizontal():
    # stones at bottom of cols 4,5,6 and top of col 0 must not count as a line
    pos, mask = c4.from_moves([4, 4, 5, 5, 6, 6])
    assert c4.status(pos, mask) == 0
    assert not c4.winning_move(pos, mask, 0)
    assert c4.winning_move(pos, mask, 3)


def test_diagonal_wins():
    # / diagonal for X: (0,0),(1,1),(2,2),(3,3)
    seq = [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
    pos, mask = c4.from_moves(seq)
    assert c4.status(pos, mask) == 1
    # \ diagonal (mirror image)
    seq_m = [W - 1 - c for c in seq]
    pos, mask = c4.from_moves(seq_m)
    assert c4.status(pos, mask) == 1


def test_no_vertical_wrap_between_columns():
    # 3 stones on top of col 0 and 1 at the bottom of col 1 are not a vertical line
    moves = [0, 1, 0, 1, 0, 2, 1, 0, 2, 0, 2, 0]
    pos, mask = c4.from_moves(moves[:-1])
    g = c4.to_grid(pos, mask)
    assert c4.status(pos, mask) == 0, g


def test_winning_move_matches_play():
    rng = np.random.default_rng(0)
    for _ in range(300):
        pos, mask = np.int64(0), np.int64(0)
        while True:
            legal = np.nonzero(c4.legal_moves(mask))[0]
            c = int(rng.choice(legal))
            wm = c4.winning_move(pos, mask, c)
            pos, mask = c4.play(pos, mask, c)
            st = c4.status(pos, mask)
            assert wm == (st == 1)
            if st:
                break


def test_draw_board():
    # play random games until one ends in a draw (full board, no four in a row)
    rng = np.random.default_rng(1)
    found = False
    for _ in range(20000):
        pos, mask = np.int64(0), np.int64(0)
        while True:
            legal = np.nonzero(c4.legal_moves(mask))[0]
            pos, mask = c4.play(pos, mask, int(rng.choice(legal)))
            st = c4.status(pos, mask)
            if st:
                break
        if st == 2:
            found = True
            assert c4.popcount(mask) == W * H
            assert not c4.legal_moves(mask).any()
            break
    assert found


def test_features_and_mirror():
    pos, mask = c4.from_moves([0, 1, 1])
    f = np.zeros((H, W, 2), np.float32)
    c4.write_features(pos, mask, f)
    # O (second player) is to move: its stone is at col 1 row 0; X has col 0 row 0 and col 1 row 1
    assert f[0, 1, 0] == 1 and f[0, 0, 1] == 1 and f[1, 1, 1] == 1 and f.sum() == 3
    mp, mm = c4.mirror(int(pos), int(mask))
    p2, m2 = c4.from_moves([W - 1, W - 2, W - 2])
    assert (mp, mm) == (int(p2), int(m2))
    f2 = np.zeros((H, W, 2), np.float32)
    c4.write_features(np.int64(mp), np.int64(mm), f2)
    assert np.array_equal(f2, f[:, ::-1])


def test_opponents_take_wins_and_block():
    c4.seed_numba(0)
    # X to move with 3 in a row in col 0 -> must win at col 0
    pos, mask = c4.from_moves([0, 1, 0, 1, 0, 2])
    for _ in range(5):
        assert c4.greedy_move(pos, mask) == 0
        assert c4.alphabeta_move(pos, mask, 2, c4.WINDOWS, c4.CENTER_ORDER) == 0
        assert c4.uct_move(pos, mask, 200) == 0
    # O to move, X threatens col 0 -> O must block
    pos, mask = c4.from_moves([0, 1, 0, 1, 0])
    for _ in range(5):
        assert c4.greedy_move(pos, mask) == 0
        assert c4.alphabeta_move(pos, mask, 2, c4.WINDOWS, c4.CENTER_ORDER) == 0
        assert c4.uct_move(pos, mask, 2000) == 0
