"""One-shot weight edit, using only start.npz and literal sensor probes.

No racing imports, track generation, vehicle states, simulator, optimization,
rollouts, or driving. The edited network is its own complete controller.
Run from this folder with the shared Python environment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_weights import RAYS, obs

NAMES = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
SHAPES = ((13, 32), (32,), (32, 32), (32,), (32, 2), (2,))
# All constants are chosen by reasoning, not driving or simulator search.
RIGHT_LIMIT = 23
LEFT_LIMIT = 29
SPEED_LIMIT = 14
STEER_MEMORY = 6
PEDAL_UNIT = 23
STEER_MEMORY_UNIT = 16
REUSED = (RIGHT_LIMIT, LEFT_LIMIT, SPEED_LIMIT, STEER_MEMORY)
RAY_SPEED_SLOPE = 1.0       # (m/s) per m of forward ray clearance
CLEARANCE_ALLOWANCE = 1.5  # m: car radius plus 0.5 m allowance
WIDTH_COEFFICIENT = 0.5    # (m/s) per m of total side-ray clearance above 12 m
SPEED_CAP = 26.0           # m/s, with drag making equilibrium slightly lower
COMPARATOR_GAIN = 0.35    # per m/s
PEDAL_GAIN = 2.5
STEERING_NEW = 0.25
STEERING_MEMORY = 0.80
MEMORY_INPUT_GAIN = 0.25


def forward(x, weights):
    h1 = np.tanh(x @ weights['W1'] + weights['b1'])
    h2 = np.tanh(h1 @ weights['W2'] + weights['b2'])
    z = h2 @ weights['W3'] + weights['b3']
    return h1, h2, z, np.tanh(z)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    with np.load('start.npz', allow_pickle=False) as archive:
        original = {k: archive[k].copy() for k in NAMES}
    for name, shape in zip(NAMES, SHAPES):
        assert original[name].shape == shape
        assert original[name].dtype == np.float32
        assert np.isfinite(original[name]).all()
    # Work in float64, then round once to the required float32 format.
    edited = {k: a.astype(np.float64) for k, a in original.items()}

    # This network almost always used full throttle. Keep the retained steering
    # features in that familiar previous-pedal context even when the new pedal
    # controller brakes. This is exactly obs[12] = 1, folded into the biases.
    edited['b1'] += edited['W1'][12]
    edited['W1'][12] = 0

    # Literal synthetic sensors, mirrored, with independent speed/previous-steer
    # values. These are not observations from a car or a generated track.
    calibration = np.array([
        obs(rays[::direction], speed, previous_steer, 1)
        for rays in RAYS.values()
        for direction in (1, -1)
        for speed in (0, 8, 16, 24, 32)
        for previous_steer in (-0.25, 0, 0.25)
    ])
    reference_h1 = forward(calibration, original)[0]
    means = reference_h1.mean(axis=0)

    # Replace four low-steering-impact features by named control features.
    # First retain their average old contributions to every surviving h2 unit.
    edited['b2'] += means[list(REUSED)] @ edited['W2'][list(REUSED)]
    edited['W2'][list(REUSED), :] = 0
    edited['W1'][:, list(REUSED)] = 0
    edited['b1'][list(REUSED)] = 0

    # T_R/L = d_R/L - 1.5 - 0.5 * (d_side_R + d_side_L - 12).
    # h1 = tanh(0.35 * (T - v)). Sensor rays are /60 and speed is /30.
    for feature, ray in ((RIGHT_LIMIT, 4), (LEFT_LIMIT, 5)):
        edited['W1'][ray, feature] = COMPARATOR_GAIN * RAY_SPEED_SLOPE * 60
        edited['W1'][0, feature] = -COMPARATOR_GAIN * WIDTH_COEFFICIENT * 60
        edited['W1'][9, feature] = -COMPARATOR_GAIN * WIDTH_COEFFICIENT * 60
        edited['W1'][10, feature] = -COMPARATOR_GAIN * 30
        edited['b1'][feature] = COMPARATOR_GAIN * (
            -CLEARANCE_ALLOWANCE + WIDTH_COEFFICIENT * 12
        )
    edited['W1'][10, SPEED_LIMIT] = -COMPARATOR_GAIN * 30
    edited['b1'][SPEED_LIMIT] = COMPARATOR_GAIN * SPEED_CAP

    # Soft AND: throttle requires all three speed margins to be positive.
    # If any margin is <= 0, its tanh is <= 0; the other two are <= 1.
    # Therefore sum - 2 <= 0 and the pedal cannot be positive.
    edited['W2'][:, PEDAL_UNIT] = 0
    edited['W2'][[RIGHT_LIMIT, LEFT_LIMIT, SPEED_LIMIT], PEDAL_UNIT] = 1
    edited['b2'][PEDAL_UNIT] = -2
    edited['W3'][:, 1] = 0
    edited['W3'][PEDAL_UNIT, 1] = PEDAL_GAIN
    edited['b3'][1] = 0
    edited['W3'][PEDAL_UNIT, 0] = 0

    # A short steering memory reduces the original very high instantaneous
    # ray-to-steer gain, while retaining substantial sustained corner steering.
    edited['W1'][11, STEER_MEMORY] = MEMORY_INPUT_GAIN
    edited['W2'][:, STEER_MEMORY_UNIT] = 0
    edited['W2'][STEER_MEMORY, STEER_MEMORY_UNIT] = 1
    edited['b2'][STEER_MEMORY_UNIT] = 0
    edited['W3'][STEER_MEMORY_UNIT, 0] = 0

    # Report fidelity of the retained steering features before damping/recentering.
    before_logits = forward(calibration, original)[2][:, 0]
    retained_logits = forward(calibration, edited)[2][:, 0]
    retained_error = retained_logits - before_logits

    # Remove the mild left-turn offset on a symmetric, centered sensor vector.
    anchor = obs(RAYS['straight_center'], 16, 0, 1)
    center_offset = float(forward(anchor, edited)[2][0])
    edited['W3'][:, 0] *= STEERING_NEW
    edited['b3'][0] = STEERING_NEW * (edited['b3'][0] - center_offset)
    edited['W3'][STEER_MEMORY_UNIT, 0] = STEERING_MEMORY / MEMORY_INPUT_GAIN

    edited = {k: a.astype(np.float32) for k, a in edited.items()}
    np.savez('final.npz', **edited)

    # Artifact and static algebra checks only; never step a vehicle.
    with np.load('final.npz', allow_pickle=False) as archive:
        assert set(archive.files) == set(NAMES)
        reloaded = {k: archive[k].copy() for k in NAMES}
    for name, shape in zip(NAMES, SHAPES):
        assert reloaded[name].shape == shape
        assert reloaded[name].dtype == np.float32
        assert np.isfinite(reloaded[name]).all()
        assert np.array_equal(reloaded[name], edited[name])
    assert np.all(reloaded['W1'][12] == 0)

    # Exhaustive independent static sensor grid, not geometric observations.
    # Closed-form pedal identity plus monotonicity and speed-limit implications.
    checks = np.array([
        [6/60, 7/60, 10/60, 18/60, right/60, left/60, 18/60,
         10/60, 7/60, side_left/60, speed/30, 0.2, -1]
        for right in (2, 8, 12, 18, 25, 40, 60)
        for left in (2, 8, 12, 18, 25, 40, 60)
        for side_left in (4, 6, 8)
        for speed in (0, 5, 10, 15, 20, 25, 26, 30, 40)
    ])
    d_right, d_left = checks[:, 4]*60, checks[:, 5]*60
    width = (checks[:, 0]+checks[:, 9])*60
    speed = checks[:, 10]*30
    targets = np.stack([
        d_right-CLEARANCE_ALLOWANCE-WIDTH_COEFFICIENT*(width-12),
        d_left-CLEARANCE_ALLOWANCE-WIDTH_COEFFICIENT*(width-12),
        np.full(len(checks), SPEED_CAP),
    ], axis=1)
    manual_pedal = np.tanh(PEDAL_GAIN*np.tanh(
        np.tanh(COMPARATOR_GAIN*(targets-speed[:, None])).sum(axis=1)-2
    ))
    actions = forward(checks, reloaded)[3]
    identity_error = float(np.max(np.abs(actions[:, 1]-manual_pedal)))
    assert identity_error < 2e-6
    overspeed = speed >= targets.min(axis=1)
    assert np.all(actions[overspeed, 1] <= 2e-6)
    faster = checks.copy(); faster[:, 10] += 0.001
    assert np.all(forward(faster, reloaded)[3][:, 1] <= actions[:, 1]+1e-8)
    wider_forward = checks.copy(); wider_forward[:, 4:6] += 0.001
    assert np.all(forward(wider_forward, reloaded)[3][:, 1] >= actions[:, 1]-1e-8)
    other_previous_pedal = checks.copy(); other_previous_pedal[:, 12] = 1
    assert np.array_equal(actions, forward(other_previous_pedal, reloaded)[3])

    rows = []
    for name, rays in RAYS.items():
        for speed in (0, 12, 20, 30):
            x = obs(rays, speed, 0, 1)
            old_action = forward(x, original)[3]
            new_action = forward(x, reloaded)[3]
            rows.append(dict(
                case=name, speed_m_s=speed, previous_steer=0,
                original_action=old_action.tolist(), final_action=new_action.tolist(),
            ))
    counts = {k: int(np.count_nonzero(original[k] != reloaded[k])) for k in NAMES}
    total = sum(a.size for a in original.values())
    report = dict(
        scope='Static weight/hand-made-input checks only; zero simulator calls or driving.',
        start_sha256=digest('start.npz'), final_sha256=digest('final.npz'),
        total_parameters=total, changed_parameters=sum(counts.values()),
        changed_fraction=sum(counts.values())/total, changed_by_array=counts,
        repurposed_h1_mean_activations={str(i): float(means[i]) for i in REUSED},
        retained_steering_logit_rmse=float(np.sqrt(np.mean(retained_error**2))),
        retained_steering_logit_max_abs_error=float(np.max(np.abs(retained_error))),
        removed_center_steering_logit=center_offset,
        static_grid_size=len(checks), pedal_identity_max_abs_error=identity_error,
        checks_passed=[
            'six arrays, exact shapes, float32, finite, identical reload',
            'pedal agrees with the programmed formula',
            'no positive pedal at/above any of the three target speeds',
            'pedal decreases monotonically with speed on static grid',
            'pedal increases monotonically with forward clearance on static grid',
            'previous pedal has no effect on either output',
        ],
        hand_made_probes=rows,
    )
    Path('static_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'hand_made_probes'}, indent=2))
    for row in rows:
        print(f"{row['case']:24s} v={row['speed_m_s']:2d} "
              f"old={np.array(row['original_action'])} "
              f"new={np.array(row['final_action'])}")


if __name__ == '__main__':
    build()
