"""Solve a specified static visibility/speed function in frozen h1 features.

This file never creates a track, imports racing, or advances any state.
All inputs are hand-specified ray templates and a deterministic input grid.
"""
import numpy as np
from inspect_weights import load, forward, RAYS

UNITS = (16, 23)
SPEED_INTERCEPT = 8.0
SPEED_PER_RAY = 24.0  # 0.4 m/s per meter of visible distance.
MARGIN_GAIN = 0.15
OUTPUT_GAIN = 2.0


def synthetic_inputs():
    # Fixed arithmetic sequences cover the input ranges without a random search.
    n = 65536
    k = np.arange(n, dtype=np.float64)[:, None] + .5
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                       31, 37, 41, 43, 47, 53, 59, 61])
    u = np.mod(k * np.sqrt(primes)[None, :], 1.)
    extra_templates = [
        [6, 6.9, 9, 9, 8, 8, 9, 9, 6.9, 6],
        [6, 6.8, 8, 9, 10, 14, 20, 22, 9, 6],
        [2, 2.3, 3, 4, 6, 16, 35, 20, 12, 10],
        [10, 12, 20, 35, 16, 6, 4, 3, 2.3, 2],
        [7, 8.1, 12.2, 22.7, 60, 60, 22.7, 12.2, 8.1, 7],
    ]
    templates = np.array(list(RAYS.values()) + extra_templates) / 60.
    i = np.arange(n) % len(templates)
    j = (np.arange(n) // len(templates)) % len(templates)
    mix = u[:, 0:1]
    rays = templates[i] * mix + templates[j] * (1-mix)
    rays *= (.70 + .60*u[:, 1:2])
    rays *= .85 + .30*u[:, 2:12]
    rays = np.clip(rays, .01, 1.)
    target_speed = 5. + 12.*(rays[:, 4] + rays[:, 5])
    # Focus on a generous neighborhood of the switching surface; also include
    # rest and overspeed. Previous controls vary independently.
    speed = np.clip(target_speed + 30.*(u[:, 12]-.5), 0., 40.)
    speed[::8] = 40.*u[::8, 13]
    previous_steer = 2*u[:, 14]-1
    previous_pedal = 2*u[:, 15]-1
    x = np.column_stack([rays, speed/30., previous_steer, previous_pedal])
    target = MARGIN_GAIN*(SPEED_INTERCEPT + SPEED_PER_RAY*rays[:, 4:6] - speed[:, None])
    return x, target


def fit(d):
    x, target = synthetic_inputs()
    h = forward(d, x)[1]
    a = np.column_stack([h, np.ones(len(h))])
    # Closed-form linear least squares, not gradient optimization.
    coef = np.linalg.lstsq(a, target, rcond=1e-10)[0]
    residual = a @ coef - target
    print('Specified margins: 0.15 * (8 + 24*ray[+/-6] - speed_mps)')
    print('Coefficient L2 norm', np.linalg.norm(coef))
    print('Static function residual quantiles in m/s:',
          np.quantile(np.abs(residual)/MARGIN_GAIN, [0, .5, .9, .99, 1], axis=0))
    speed_derivative = ((1-h*h) * d['W1'][10]) @ coef[:32]
    print('Static margin derivative w.r.t. normalized speed (ideal -4.5):',
          speed_derivative.min(axis=0), speed_derivative.max(axis=0))
    return coef


def edit(d, coef):
    result = {k: v.copy() for k, v in d.items()}
    result['W2'][:, UNITS] = coef[:32].astype(np.float32)
    result['b2'][list(UNITS)] = coef[32].astype(np.float32)
    result['W3'][:, 1] = 0.
    result['W3'][list(UNITS), 1] = OUTPUT_GAIN
    result['b3'][1] = -OUTPUT_GAIN
    # Disconnect the negligible original steering contributions of reused units.
    result['W3'][list(UNITS), 0] = 0.
    return result


if __name__ == '__main__':
    from inspect_weights import obs
    original = load()
    coefficients = fit(original)
    final = edit(original, coefficients)
    print('\nSTATIC PROBES ONLY: name, speed, original action, edited action')
    for name in ('straight', 'left_gentle', 'left_medium', 'left_tight',
                 'right_tight', 'near_left', 'near_right', 'aim_left'):
        for speed in (0, 10, 15, 20, 28, 36):
            x = obs(name, speed, 0., 0.)
            print(name, speed, np.round(forward(original, x)[0], 4),
                  np.round(forward(final, x)[0], 4))
    print('Changes by array:', {k: int(np.count_nonzero(final[k] != original[k])) for k in original})
    # Only the specified final file is written; there are no simulator candidates.
    np.savez('final.npz', **final)
