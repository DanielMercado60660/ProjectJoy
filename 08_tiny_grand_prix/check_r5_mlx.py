"""Claude-only MLX checks: NumPy parity, finite zero gradients, live compiled weights."""
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from r5_policy import HIDDEN, init_params, sequence


def main():
    p_np = init_params()
    rng = np.random.default_rng(9)
    p_np['out'][:] = rng.normal(size=p_np['out'].shape).astype(np.float32) * .01
    x = rng.uniform(-.5, .5, (4, 3, 14)).astype(np.float32)
    h = np.zeros((3, HIDDEN), np.float32)
    reset = np.zeros((4, 3), np.float32)
    reset[2, 1] = 1
    p = {k: mx.array(v) for k, v in p_np.items()}
    for recurrent in (True, False):
        expected = sequence(p_np, x, h, reset, recurrent)
        actual = sequence(p, mx.array(x), mx.array(h), mx.array(reset), recurrent, mx)
        mx.eval(actual)
        for a, b in zip(expected, actual):
            # Metal float32 matrix products may use TF32.
            np.testing.assert_allclose(a, np.array(b), atol=3e-3, rtol=3e-3)
    def loss(p, x):
        m, v, _ = sequence(p, x, mx.array(h), mx.array(reset), True, mx)
        return (m * m).mean() + (v * v).mean()
    vg = mx.compile(mx.value_and_grad(loss))
    zero = mx.zeros_like(mx.array(x))
    l, g = vg(p, zero)
    mx.eval(l, g)
    assert np.isfinite(float(l)) and all(np.isfinite(np.array(v)).all() for v in g.values())
    # Explicit dynamic p must change a compiled function's result.
    l0, _ = vg(p, mx.array(x))
    q = dict(p)
    q['ab2'] = p['ab2'] + 2
    l1, _ = vg(q, mx.array(x))
    mx.eval(l0, l1)
    assert abs(float(l0) - float(l1)) > 1e-3, 'compiled parameters were frozen'
    opt = optim.Adam(learning_rate=1e-4)
    opt.init(p)
    before = np.array(p['ab2']).copy()
    opt.update(p, g)
    mx.eval(p, opt.state)
    assert not np.array_equal(before, np.array(p['ab2']))
    # Verify the exact 0.32-compatible in-place RNG restore used by the trainer.
    saved = [np.array(v).copy() for v in mx.random.state]
    expected = np.array(mx.random.normal((4,)))
    for k in range(len(mx.random.state)):
        mx.random.state[k][:] = mx.array(saved[k])
    actual = np.array(mx.random.normal((4,)))
    np.testing.assert_array_equal(actual, expected)
    print('MLX parity, finite zero-input gradients, live compile, optimizer, RNG restore: PASS', flush=True)


if __name__ == '__main__':
    main()
