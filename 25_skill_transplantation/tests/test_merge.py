import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import PARAM_KEYS, Normalizer, init_params, np_mean, task_reward  # noqa: E402
import merge as M  # noqa: E402


def close(a, b, tol=1e-6):
    return all(np.allclose(a[k], b[k], atol=tol) for k in PARAM_KEYS)


def test_merge_identities():
    t0, A, B = init_params(0), init_params(1), init_params(2)
    assert close(M.interpolate(A, B, 0.0), A)
    assert close(M.interpolate(A, B, 1.0), B)
    # theta0 + 0.5 (tauA + tauB) is the plain average, whatever theta0 is
    assert close(M.task_arithmetic(t0, [A, B], 0.5), M.interpolate(A, B, 0.5), 1e-5)
    # one task vector with lambda 1 gives back the specialist
    assert close(M.task_arithmetic(t0, [A], 1.0), A, 1e-5)
    # TIES with k=1 and identical task vectors: disjoint mean == the shared vector
    assert close(M.ties(t0, [A, A], 1.0, k=1.0), A, 1e-5)


def test_mlx_matches_numpy_forward():
    from ppo import model_from_np, np_from_model
    import mlx.core as mx
    p = init_params(3)
    p["a3.weight"] = p["a3.weight"] * 100   # make the output non-trivial
    x = np.random.default_rng(0).standard_normal((64, p["a1.weight"].shape[1])).astype(np.float32)
    m = model_from_np(p)
    assert np.allclose(np.array(m.mean(mx.array(x))), np_mean(p, x), atol=5e-3)   # TF32 on GPU
    assert close(np_from_model(m), p)


def test_reward_and_normalizer():
    a = np.zeros(6)
    assert task_reward("fwd", 2.0, a) == 2.0 and task_reward("bwd", 2.0, a) == -2.0
    n = Normalizer()
    x = np.random.default_rng(0).standard_normal((1000, 17)) * 3 + 1
    n.update(x)
    assert np.allclose(n.mean, x.mean(0), atol=1e-3) and np.allclose(n.var, x.var(0), rtol=1e-2)
