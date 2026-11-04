"""Deterministic, closed-form specialist compression; no environment imports."""
from pathlib import Path
import argparse
import json
import os

# Numerical libraries must stay within the per-job CPU budget.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
import numpy as np
from scipy.linalg import qr

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'
KEYS = ('a1.weight', 'a1.bias', 'a2.weight', 'a2.bias', 'a3.weight', 'a3.bias', 'log_std')
SHAPES = ((128, 19), (128,), (128, 128), (128,), (6, 128), (6,), (6,))
FINAL_OUTPUT_FITS = {0: 'derivative_fwd', 1: 'uniform', 2: 'uniform', 3: 'derivative_fwd'}


def load(seed, name):
    with np.load(DATA / f's{seed}' / f'{name}.npz') as z:
        return {k: z[k].copy() for k in z.files}


def synthetic(seed, task, n, normalizer, distribution='motion', validation=False):
    """Independent synthetic draws; no transitions, rollout observations, or simulation."""
    rng = np.random.default_rng(812000 + seed * 13 + task + 10000 * validation)
    x = rng.normal(size=(n, 17))
    scale = rng.choice([0.5, 1., 1.5], size=(n, 1), p=[0.2, 0.6, 0.2])
    x *= scale
    if distribution == 'motion':
        mean, std = normalizer['norm_mean'], np.sqrt(normalizer['norm_var'] + 1e-8)
        active = np.arange(n) % 5 != 0
        m = int(active.sum())
        # A broad upright-motion prior in raw physical coordinates. These are
        # chosen physical priors, not inferred environment observations.
        x[active, 0] = (rng.normal(-0.06, 0.12, m) - mean[0]) / std[0]
        x[active, 1] = (rng.normal(0., 0.35, m) - mean[1]) / std[1]
        vel = rng.uniform(-1., 11., m) if task == 0 else rng.uniform(-13., 1., m)
        x[active, 8] = (vel - mean[8]) / std[8]
    elif distribution == 'normal':
        pass
    else:
        raise ValueError(distribution)
    return np.clip(x, -10, 10).astype(np.float32)


def actor(p, x, task, hidden=False):
    h1 = np.tanh(x @ p['a1.weight'][:, :17].T + p['a1.bias'] + p['a1.weight'][:, 17 + task])
    z2 = h1 @ p['a2.weight'].T + p['a2.bias']
    h2 = np.tanh(z2)
    z3 = h2 @ p['a3.weight'].T + p['a3.bias']
    return (h1, z2, h2, z3) if hidden else np.tanh(z3)


def linear_fit(x, y, ridge=1e-5):
    """One centered ridge-regression solve; intercept is unregularized."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(x)
    gram.flat[::len(gram) + 1] += ridge
    w = np.linalg.solve(gram, xc.T @ yc / len(x))
    return w.astype(np.float32), (ym - xm @ w).astype(np.float32)


def derivative_fit(x, y, ridge=1e-5):
    """Six independent, one-shot weighted linear solves; no reweighting loop.

    Fixed weights are computed from teacher outputs only: tanh'(target)^2
    plus 0.01. This is the local action-space quadratic, with a positive floor
    to preserve saturated-output signs. Each solve has an unpenalized bias.
    """
    x, y = x.astype(np.float64), y.astype(np.float64)
    design = np.column_stack((x, np.ones(len(x))))
    solutions = []
    for output in range(y.shape[1]):
        weight = (1. - np.tanh(y[:, output]) ** 2) ** 2 + 0.01
        gram = design.T @ (weight[:, None] * design) / len(x)
        gram[np.arange(x.shape[1]), np.arange(x.shape[1])] += ridge
        rhs = design.T @ (weight * y[:, output]) / len(x)
        solutions.append(np.linalg.solve(gram, rhs))
    solution = np.stack(solutions, axis=1).astype(np.float32)
    return solution[:-1], solution[-1]


def compress(seed, distribution='motion', widths=(63, 64), n=24000, ridge=1e-5, output_fit='uniform'):
    fwd, bwd, theta = (load(seed, name) for name in ('fwd', 'bwd', 'theta0'))
    teachers = (fwd, bwd)
    p = {k: np.zeros(shape, np.float32) for k, shape in zip(KEYS, SHAPES)}
    p['log_std'] = theta['log_std'].astype(np.float32)
    # Keep 127 paired shared features. Omit the smallest downstream-energy
    # column of the two specialists; no random selection of network weights.
    importance = np.sum(fwd['a2.weight'] ** 2 + bwd['a2.weight'] ** 2, axis=0)
    dropped = int(np.argmin(importance))
    shared = np.delete(np.arange(128), dropped)
    p['a1.weight'][:127, :17] = 0.5 * (fwd['a1.weight'][shared, :17] + bwd['a1.weight'][shared, :17])
    for task, teacher in enumerate(teachers):
        p['a1.weight'][:127, 17 + task] = teacher['a1.bias'][shared] + teacher['a1.weight'][shared, 17 + task]
    p['a1.weight'][127, 17:] = [3., -3.]
    gate1 = float(np.tanh(np.float32(3.)))
    # One second-layer label marker, with known magnitude, supplies the two
    # output intercepts independently.
    p['a2.weight'][127, 127] = 3. / gate1
    gate2 = float(np.tanh(np.float32(3.)))
    reports, heads, offsets = [], [], []
    start = 0
    for task, (teacher, width) in enumerate(zip(teachers, widths)):
        x = synthetic(seed, task, n, theta, distribution)
        _, target_z2, _, target_z3 = actor(teacher, x, task, hidden=True)
        shared_h = np.tanh(x @ p['a1.weight'][:127, :17].T + p['a1.weight'][:127, 17 + task])
        # Each teacher's layer-2 preactivations are projected once onto the
        # shared nonlinear first-layer features.
        w2, b2 = linear_fit(shared_h, target_z2, ridge)
        approximated_h2 = np.tanh(shared_h @ w2 + b2)
        # Pivoted QR selects an independent and output-important subset.
        sample = approximated_h2[::3].astype(np.float64)
        sample -= sample.mean(0)
        output_importance = np.sqrt(np.sum(teacher['a3.weight'].astype(np.float64) ** 2, axis=0))
        _, _, piv = qr(sample * output_importance, mode='economic', pivoting=True)
        selected = piv[:width]
        features = approximated_h2[:, selected]
        w3, b3 = (derivative_fit(features, target_z3, ridge) if output_fit == 'derivative_fwd' and task == 0 else linear_fit(features, target_z3, ridge))
        end = start + width
        assert end <= 127
        p['a2.weight'][start:end, :127] = w2[:, selected].T
        base_bias = b2[selected]
        # tanh inputs are bounded by +/-1. Thus this shift guarantees inactive
        # preactivations >= 12 even at the most adverse possible hidden state.
        shift = np.sum(np.abs(w2[:, selected]), axis=0) + np.abs(base_bias) + 12.
        sign = 1. if task == 0 else -1.
        p['a2.weight'][start:end, 127] = -sign * shift / (2. * gate1)
        p['a2.bias'][start:end] = base_bias + shift / 2.
        p['a3.weight'][:, start:end] = w3.T
        heads.append(w3)
        offsets.append(b3)
        reports.append({'task': ['fwd', 'bwd'][task], 'selected_h2': selected.tolist(),
                        'preactivation_projection_rmse': float(np.sqrt(np.mean((shared_h @ w2 + b2 - target_z2) ** 2))),
                        'fit_action_rmse': float(np.sqrt(np.mean((np.tanh(features @ w3 + b3) - np.tanh(target_z3)) ** 2))),
                        'max_gate_shift': float(shift.max())})
        start = end
    # Inactive branches equal +1. Solve the two affine output intercepts.
    cf = offsets[0] - heads[1].sum(0)
    cb = offsets[1] - heads[0].sum(0)
    p['a3.bias'] = (cf + cb) / 2.
    p['a3.weight'][:, 127] = (cf - cb) / (2. * gate2)
    for task, teacher in enumerate(teachers):
        x = synthetic(seed, task, 12000, theta, distribution, validation=True)
        teacher_action, merged_action = actor(teacher, x, task), actor(p, x, task)
        reports[task]['validation_action_rmse'] = float(np.sqrt(np.mean((teacher_action - merged_action) ** 2)))
        reports[task]['validation_action_mae'] = float(np.mean(np.abs(teacher_action - merged_action)))
        reports[task]['teacher_action_rms'] = float(np.sqrt(np.mean(teacher_action ** 2)))
    info = {'seed': seed, 'method': 'shared_first', 'distribution': distribution,
            'widths': widths, 'fit_samples_per_task': n, 'ridge': ridge, 'output_fit': output_fit,
            'dropped_first_layer_unit': dropped, 'tasks': reports}
    return p, info


def save(path, p):
    p = {k: np.asarray(p[k], np.float32) for k in KEYS}
    for k, shape in zip(KEYS, SHAPES):
        assert p[k].shape == shape and np.isfinite(p[k]).all(), k
    np.savez(path, **p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, nargs='+', choices=[0, 1, 2, 3], default=[0, 1, 2, 3])
    ap.add_argument('--kind', choices=['baseline', 'shared'], default='shared')
    ap.add_argument('--distribution', choices=['normal', 'motion'], default='motion')
    ap.add_argument('--prefix', default='merged')
    ap.add_argument('--samples', type=int, default=24000)
    ap.add_argument('--output-fit', choices=['final', 'uniform', 'derivative_fwd'], default='final')
    ap.add_argument('--width-fwd', type=int, default=63)
    args = ap.parse_args()
    if Path(args.prefix).name != args.prefix:
        ap.error('--prefix must be a filename prefix, not a path')
    if not 1 <= args.width_fwd <= 126:
        ap.error('--width-fwd must be between 1 and 126')
    for seed in args.seeds:
        if args.kind == 'baseline':
            p, info = load(seed, 'fwd'), {'seed': seed, 'method': 'unchanged_fwd_specialist'}
        else:
            p, info = compress(seed, args.distribution, (args.width_fwd, 127 - args.width_fwd), args.samples, output_fit=(FINAL_OUTPUT_FITS[seed] if args.output_fit == 'final' else args.output_fit))
        save(HERE / f'{args.prefix}_s{seed}.npz', p)
        (HERE / f'{args.prefix}_s{seed}_diagnostics.json').write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps(info))


if __name__ == '__main__':
    main()
