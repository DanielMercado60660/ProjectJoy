"""Claude-only MLX checks: finite zero gradients, live compiled weights, RNG."""
import argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from policy_r5 import Policy, LearnedAgent, restore_mlx_rng
from env_r5 import OBS_DIM, REMAIN
from io_r5 import load_checkpoint


def check_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            check_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            check_tree(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    else:
        assert a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--replay")
    a = ap.parse_args()
    if a.before:
        before, after, replay = map(load_checkpoint, (a.before, a.after, a.replay))
        assert after["update"] == before["update"]+2
        assert int(after["optimizer"]["step"]) > int(before["optimizer"]["step"])
        assert np.isfinite(after["last_loss"])
        bweights, aweights = tree_flatten(before["weights"]), tree_flatten(after["weights"])
        assert any(not np.array_equal(x[1], y[1]) for x, y in zip(bweights, aweights))
        for key in ("weights", "optimizer", "numpy_rng", "mlx_rng", "environment", "obs", "hidden_state", "starts"):
            check_tree(after[key], replay[key])
        print(f"RESUME OK: update {before['update']} -> {after['update']}; finite loss {after['last_loss']:.6f}; deterministic replay")
        return
    mx.random.seed(91)
    model = Policy(16)
    x = np.zeros((4, 3, OBS_DIM), np.float32)
    x[..., REMAIN] = .5
    starts = np.zeros((4, 3), np.float32)
    starts[0] = 1; starts[2, 1] = 1
    m, p, v = model.sequence(mx.array(x), mx.zeros((3, 16)), mx.array(starts))
    h = mx.zeros((3, 16))
    for t in range(4):
        mm, pp, vv, h = model.step(mx.array(x[t]), h*mx.array(1-starts[t])[:, None])
        for actual, expected in ((mm, m[t]), (pp, p[t]), (vv, v[t])):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)
    zero = mx.zeros((3, OBS_DIM))
    logits = model.step(zero, mx.zeros((3, 16)))[1]
    assert np.all(np.asarray(mx.argmax(logits, -1)) == 0)
    grad = mx.grad(lambda z: mx.sqrt(mx.maximum(mx.sum(z*z), 1e-12)))(mx.zeros((8,)))
    assert np.isfinite(np.asarray(grad)).all()
    def loss(model):
        m, p, v, _ = model.step(zero, mx.zeros((3, 16)))
        return mx.mean(m*m)+mx.mean(v*v)  # spent-pulse mask intentionally excluded
    _, gradients = nn.value_and_grad(model, loss)(model)
    assert all(np.isfinite(np.asarray(g)).all() for _, g in tree_flatten(gradients))
    agent = LearnedAgent(model, 3)
    old = np.asarray(agent.step(zero, mx.zeros((3, 16)))[0]).copy()
    model.move.bias = model.move.bias+mx.array([2., 0., 0., 0., 0.])
    new = np.asarray(agent.step(zero, mx.zeros((3, 16)))[0])
    np.testing.assert_allclose(new-old, np.tile([2., 0., 0., 0., 0.], (3, 1)), atol=1e-5)
    mx.eval(mx.random.state)
    keys = [np.asarray(k).copy() for k in mx.random.state]
    draw = np.asarray(mx.random.normal((16,))).copy()
    restore_mlx_rng(keys)
    np.testing.assert_array_equal(draw, np.asarray(mx.random.normal((16,))))
    print("MLX CHECKS OK: shapes, resets, budget mask, finite zero gradient, live compiled weights, RNG restore")


if __name__ == "__main__":
    main()
