"""Artifact and static algebra checks. Never imports or runs a simulator."""
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_weights import NAMES, RAYS, forward, load, obs
from design_governor import UNITS, MARGIN_GAIN, synthetic_inputs


def main():
    expected = [(13, 32), (32,), (32, 32), (32,), (32, 2), (2,)]
    with np.load('final.npz', allow_pickle=False) as z:
        assert set(z.files) == set(NAMES)
        for name, shape in zip(NAMES, expected):
            assert z[name].shape == shape
            assert z[name].dtype == np.float32
            assert np.isfinite(z[name]).all()
    original, final = load(), load('final.npz')
    untouched = [j for j in range(32) if j not in UNITS]
    assert np.array_equal(original['W1'], final['W1'])
    assert np.array_equal(original['b1'], final['b1'])
    assert np.array_equal(original['W2'][:, untouched], final['W2'][:, untouched])
    assert np.array_equal(original['b2'][untouched], final['b2'][untouched])
    assert np.array_equal(original['W3'][untouched, 0], final['W3'][untouched, 0])
    assert original['b3'][0] == final['b3'][0]
    assert np.all(final['W3'][list(UNITS), 0] == 0.)
    assert np.all(final['W3'][list(UNITS), 1] == 2.)
    assert np.all(final['W3'][untouched, 1] == 0.)
    assert final['b3'][1] == -2.

    counts = {k: int(np.count_nonzero(original[k] != final[k])) for k in NAMES}
    changed = sum(counts.values())
    total = sum(original[k].size for k in NAMES)
    bound = float(np.sum(np.abs(original['W3'][list(UNITS), 0])))
    # These are isolated input vectors: there is no state evolution or feedback.
    vectors = np.array([obs(name, speed, ps, pp)
                        for name in RAYS
                        for speed in (0., 10., 15., 20., 28., 40.)
                        for ps in (-1., -.5, 0., .5, 1.)
                        for pp in (-1., -.5, 0., .5, 1.)])
    old_action = forward(original, vectors)[0]
    action = forward(final, vectors)[0]
    assert np.isfinite(action).all()
    observed_change = float(np.max(np.abs(old_action[:, 0]-action[:, 0])))
    assert observed_change <= bound + 1e-7
    assert np.all(action[vectors[:, 10] == 0., 1] > 0.)
    assert np.all(action[vectors[:, 10] == 40./30., 1] < 0.)

    x, target = synthetic_inputs()
    h = forward(original, x)[1]
    margins = h @ final['W2'][:, UNITS] + final['b2'][list(UNITS)]
    errors = np.abs(margins-target)/MARGIN_GAIN
    deriv = ((1-h*h)*original['W1'][10]) @ final['W2'][:, UNITS]
    assert np.all(deriv < 0.)
    summary = {
        'scope': 'File integrity and isolated synthetic-input algebra only; no simulator or driving.',
        'simulator_calls': 0,
        'strict_completion_measured': None,
        'lap_time_measured': None,
        'parameters_total': total,
        'parameters_changed': changed,
        'fraction_changed': changed/total,
        'changes_by_array': counts,
        'steering_global_absolute_change_bound': bound,
        'isolated_control_sweep_vectors': len(vectors),
        'observed_steering_change_on_control_sweep': observed_change,
        'synthetic_fit_vectors': len(x),
        'margin_error_mps_quantiles': {
            str(q): np.quantile(errors, q, axis=0).tolist()
            for q in (.5, .9, .99, 1.)
        },
        'margin_speed_derivative_min': deriv.min(axis=0).tolist(),
        'margin_speed_derivative_max': deriv.max(axis=0).tolist(),
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in ('start.npz', 'final.npz')},
    }
    Path('static_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
