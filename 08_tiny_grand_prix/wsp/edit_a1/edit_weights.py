"""Explicit, deterministic weight edits. No simulator import or optimisation loop."""
import numpy as np
from inspect_weights import load, forward, handmade, NAMES


def speed_controller(path, intercept, range_gain, speed_gain=.15, output_gain=2.):
    """Reuse h1[23] and h2[28] for a forward-range-dependent speed target.

    target speed = intercept + range_gain * mean(ray[-6], ray[+6]) in m/s.
    All other units retain the source network, with prev_pedal fixed to 1
    internally so the source steering function keeps its familiar inputs.
    """
    d = load()
    x = handmade()
    a0, h1, h2, _ = forward(d, x)
    u, v = 23, 28
    # Preserve the approximately constant contribution of the removed feature.
    d['b2'] += h1[:, u].mean() * d['W2'][u, :]
    d['W2'][u, :] = 0
    d['b3'][0] += h2[:, v].mean() * d['W3'][v, 0]
    d['W3'][v, 0] = 0
    # Constant-fold the original almost-always-full previous-pedal observation.
    d['b1'] += d['W1'][12, :]
    d['W1'][12, :] = 0
    # Exact specified speed-control function, expressed as two tanh units.
    d['W1'][:, u] = 0
    d['W1'][4:6, u] = range_gain * speed_gain / 2
    d['W1'][10, u] = -30 * speed_gain
    d['b1'][u] = intercept * speed_gain
    d['W2'][:, v] = 0
    d['W2'][u, v] = 1
    d['b2'][v] = 0
    d['W3'][:, 1] = 0
    d['W3'][v, 1] = output_gain
    d['b3'][1] = 0
    np.savez(path, **{k: np.asarray(d[k], np.float32) for k in NAMES})
    aa = forward(d, x)[0]
    print(path, 'changed',sum(np.count_nonzero(d[k] != load()[k]) for k in NAMES), '/ 1570')
    print('Steering change on handmade probes, RMS/max:',np.sqrt(np.mean((aa[:,0]-a0[:,0])**2)),np.max(np.abs(aa[:,0]-a0[:,0])))


def widen_steering(source, path, mix=.5, bias=-.08):
    d = load(source)
    # Preserve narrow forward cues and add half-strength +/-35-degree cues.
    # h1[23] remains the separately programmed speed unit.
    d['W1'][2, :] += mix*d['W1'][3, :]
    d['W1'][7, :] += mix*d['W1'][6, :]
    d['b3'][0] += bias
    np.savez(path, **d)


def clear_road_bonus(source, path):
    d = load(source)
    _, h1, _, _ = forward(d, handmade())
    u = 29
    d['b2'] += h1[:,u].mean()*d['W2'][u,:]
    d['W2'][u,:] = 0
    d['W1'][:,u] = 0
    d['W1'][4:6,u] = 3
    d['b1'][u] = -3
    # Keep call 4's low-range speed envelope, adding a smooth clear-road bonus.
    d['W1'][4:6,23] = 26*.1/2
    d['W1'][10,23] = -30*.1
    d['b1'][23] = 7*.1
    d['W2'][23,28] = 3
    d['W2'][29,28] = .45
    d['b2'][28] = .45
    np.savez(path, **d)


if __name__ == '__main__':
    speed_controller('c02_range_speed.npz', intercept=11., range_gain=20.)
