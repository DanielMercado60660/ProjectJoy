"""Round 6 tests: mirrored observation, baseline equivalence, held-out gate, checkpoint/resume."""
import json, os, pickle, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import scripted
import r6_opponents as ro
from r6_common import R6Arena, KINDS
import r6_eval


def _rand_state(a, rng):
    a.pos = rng.uniform(-0.6, 0.6, (a.n, 2, 2))
    a.th = rng.uniform(-np.pi, np.pi, (a.n, 2))
    a.ball = rng.uniform(-0.6, 0.6, (a.n, 2))
    a.bvel = rng.normal(0, 0.5, (a.n, 2))
    a.vel = rng.normal(0, 0.3, (a.n, 2, 2))
    a.prev_act = rng.uniform(-1, 1, (a.n, 2, 2))
    a.t = rng.integers(0, 300, a.n)


def test_mirrored_obs_matches_swapped_rotated_state():
    rng = np.random.default_rng(0)
    a, b = R6Arena(16, seed=1), R6Arena(16, seed=2)
    _rand_state(a, rng)
    # b = a with robots swapped and the world rotated by 180 degrees
    b.pos = -a.pos[:, ::-1].copy()
    b.th = a.th[:, ::-1] + np.pi
    b.ball, b.bvel = -a.ball, -a.bvel
    b.vel = -a.vel[:, ::-1].copy()
    b.prev_act = a.prev_act[:, ::-1].copy()
    b.t = a.t.copy()
    np.testing.assert_allclose(a.obs_r6(1), b.obs_r6(0), atol=1e-5)


def test_obs_r6_extends_legacy_obs():
    a = R6Arena(8, seed=3)
    _rand_state(a, np.random.default_rng(1))
    np.testing.assert_allclose(a.obs_r6(0)[:, :23], a.obs(), atol=1e-6)


def test_param_attacker_defaults_equal_scripted():
    a = R6Arena(64, seed=4)
    a.curriculum = 1.0
    a.reset(np.ones(64, bool))
    for _ in range(50):
        x = scripted.attacker(a)
        y = ro.attacker_param(a, ro.ATT_DEFAULT)
        np.testing.assert_allclose(x, y, atol=1e-9)
        a.step(x, scripted.goalie(a))


def test_heldout_never_in_training_pool():
    assert not set(ro.HELDOUT) & set(KINDS)
    assert not set(ro.HELDOUT) & set(ro.POOL_FAMILIES)
    assert "HELDOUT[" not in open(os.path.join(HERE, "r6_train.py")).read()
    src = open(os.path.join(HERE, "r6_train.py")).read() + open(os.path.join(HERE, "r6_common.py")).read() \
        + open(os.path.join(HERE, "r6_tune_attacker.py")).read()
    import re
    for name in ro.HELDOUT:   # whole-word match (the arena's zone_recover state buffer is only used at eval)
        assert not re.search(rf"\b{name}\b", src), f"held-out opponent {name} referenced in training code"


def test_dev_mode_rejects_heldout(tmp_path):
    sys_argv = sys.argv
    sys.argv = ["r6_eval.py", "--mode", "dev", "--opps", "zone", "--n", "4", "--out", str(tmp_path / "x.json")]
    try:
        with pytest.raises(AssertionError):
            r6_eval.main()
    finally:
        sys.argv = sys_argv


def test_heldout_opponents_valid_actions():
    a = R6Arena(32, seed=5)
    a.curriculum = 1.0
    a.reset(np.ones(32, bool))
    for _ in range(30):
        for fn in ro.HELDOUT.values():
            act = fn(a)
            assert act.shape == (32, 2) and np.all(np.abs(act) <= 1) and np.isfinite(act).all()
        a.step(scripted.attacker(a), ro.presser(a))


@pytest.mark.parametrize("learn_def", ["0", "1"])
def test_resume_is_bit_exact(tmp_path, learn_def):
    py = sys.executable
    common = ["r6_train.py", "--learn-def", learn_def, "--total-steps", "1e6", "--envs", "64", "--horizon", "32", "--minibatch", "512",
              "--snap-every", "4096", "--ckpt-every", "2"]
    env = dict(os.environ, OMP_NUM_THREADS="2")
    run = lambda extra: subprocess.run([py] + common + extra, cwd=HERE, env=env, check=True, capture_output=True)
    run(["--max-updates", "4", "--out", str(tmp_path / "a/r"), "--log", str(tmp_path / "a/r.jsonl")])
    run(["--max-updates", "2", "--out", str(tmp_path / "b/r"), "--log", str(tmp_path / "b/r.jsonl")])
    run(["--max-updates", "4", "--out", str(tmp_path / "b/r"), "--log", str(tmp_path / "b/r.jsonl"), "--resume", "1"])
    A = pickle.load(open(tmp_path / "a/r.pkl", "rb"))
    B = pickle.load(open(tmp_path / "b/r.pkl", "rb"))
    assert A["upd"] == B["upd"] == 4 and A["steps"] == B["steps"]
    for who in ("att", "dfn") if learn_def == "1" else ("att",):
        for k in A[who]["params"]:
            np.testing.assert_array_equal(A[who]["params"][k], B[who]["params"][k])
