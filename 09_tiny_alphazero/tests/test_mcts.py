import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import c4  # noqa: E402
from mcts import BatchMCTS, uniform_evaluator  # noqa: E402


def _search(moves_list, sims=300):
    states = [c4.from_moves(m) for m in moves_list]
    pos = np.array([s[0] for s in states], np.int64)
    mask = np.array([s[1] for s in states], np.int64)
    m = BatchMCTS(len(states), sims)
    return m.search(pos, mask, uniform_evaluator)


def test_mcts_finds_win_and_block_in_batch():
    c4.seed_numba(0)
    win = [0, 1, 0, 1, 0, 2]        # X to move, col 0 wins
    block = [0, 1, 0, 1, 0]         # O to move, must block col 0
    win_h = [1, 1, 2, 2, 3, 3]      # X to move, cols 0 or 4 win
    visits, q, prior = _search([win, block, win_h])
    assert np.argmax(visits[0]) == 0
    assert np.argmax(visits[1]) == 0
    assert np.argmax(visits[2]) in (0, 4)
    assert q[0, 0] > 0.99            # proven win for the mover
    # visit counts add up to the simulation budget
    assert abs(visits[0].sum() - 300) <= 1


def test_illegal_moves_get_no_visits():
    c4.seed_numba(1)
    full_col = [3] * 6
    visits, _, prior = _search([full_col], sims=200)
    assert visits[0, 3] == 0 and prior[0, 3] == 0


def test_dirichlet_noise_changes_priors_only_at_root():
    c4.seed_numba(2)
    pos, mask = c4.from_moves([])
    m = BatchMCTS(1, 50)
    _, _, p_clean = m.search(np.array([pos]), np.array([mask]), uniform_evaluator)
    _, _, p_noisy = m.search(np.array([pos]), np.array([mask]), uniform_evaluator, add_noise=True)
    assert np.allclose(p_clean, 1 / 7)
    assert not np.allclose(p_noisy, 1 / 7)
    assert abs(p_noisy.sum() - 1) < 1e-5


def test_evaluator_sees_new_weights():
    """make_evaluator compiles a closure over the model; after a training step a NEW
    evaluator must see the updated weights (guards against a stale compiled graph)."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from net import PVNet, make_evaluator
    import train

    model = PVNet(8, 1)
    mx.eval(model.parameters())
    x = np.zeros((1, 6, 7, 2), np.float32)
    before = make_evaluator(model)(x)[1].copy()
    opt = optim.Adam(learning_rate=1e-1)
    rep = train.Replay(100)
    p, m = c4.from_moves([3])
    rep.add([p], [m], [np.full(7, 1 / 7, np.float32)], [1.0])
    train.train_steps(model, opt, rep, 5, 8, np.random.default_rng(0), True)
    after = make_evaluator(model)(x)[1]
    assert not np.allclose(before, after)


def test_pipeline_end_to_end_tiny():
    """Two tiny generations of self-play + training + checkpoint, then resume for a third."""
    import train
    run = "pytest_tiny"
    d = os.path.join(HERE, "checkpoints", run)
    shutil.rmtree(d, ignore_errors=True)
    try:
        train.main(["--run", run, "--tiny", "--gens", "2", "--eval-every", "2"])
        assert os.path.exists(os.path.join(d, "gen_002.safetensors"))
        train.main(["--run", run, "--tiny", "--gens", "3", "--eval-every", "0"])
        assert os.path.exists(os.path.join(d, "gen_003.safetensors"))
        import arena
        a = arena.make_agent(f"net:checkpoints/{run}/gen_003:8")
        r = arena.play_match(a, arena.make_agent("random"), openings=arena.ALL_2PLY[:3])
        assert r["games"] == 6
    finally:
        shutil.rmtree(d, ignore_errors=True)
        for f in (f"{run}.jsonl", f"{run}_eval.jsonl"):
            p = os.path.join(HERE, "logs", f)
            if os.path.exists(p):
                os.remove(p)
