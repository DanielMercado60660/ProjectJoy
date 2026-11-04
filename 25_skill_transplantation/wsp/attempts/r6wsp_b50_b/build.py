"""Deterministic weight-space constructions. No simulator or gradient training."""
import argparse
import json
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
import numpy as np
from scipy.linalg import qr
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / 'data'
KEYS = ['a1.weight', 'a1.bias', 'a2.weight', 'a2.bias', 'a3.weight', 'a3.bias', 'log_std']

def source(seed, task):
    return dict(np.load(DATA / f's{seed}' / f'{task}.npz'))

def hidden(p, x, task):
    h1 = np.tanh(x @ p['a1.weight'][:, :17].T + p['a1.bias'] + p['a1.weight'][:, 17 + task])
    z2 = h1 @ p['a2.weight'].T + p['a2.bias']
    h2 = np.tanh(z2)
    y = h2 @ p['a3.weight'].T + p['a3.bias']
    return h1, z2, h2, y

def synthetic(seed, n, scale=1., distribution='normal', task=0, stream=0):
    rng = np.random.default_rng(7000 + seed * 100 + task + stream*10000)
    x = rng.standard_normal((n, 17)).astype(np.float64) * scale
    if distribution in ('motion', 'mixed', 'speed'):
        p = source(seed, 'theta0')
        mean, sd = p['norm_mean'], np.sqrt(p['norm_var'] + 1e-8)
        # Synthetic upright torso and task-consistent translation. No states collected.
        raw_z = rng.normal(-.03, .06, n)
        raw_pitch = rng.normal(0, .30, n)
        raw_v = (1 if task == 0 else -1) * rng.uniform(0, 11, n)
        count = n // 2 if distribution == 'mixed' else n
        if distribution != 'speed':
            x[:count, 0] = (raw_z[:count] - mean[0]) / sd[0]
            x[:count, 1] = (raw_pitch[:count] - mean[1]) / sd[1]
        x[:count, 8] = (raw_v[:count] - mean[8]) / sd[8]
    return np.clip(x, -10, 10)

def fit(a, b, ridge=1e-7):
    # One closed-form ridge solve; intercept unpenalized.
    aa = np.column_stack((a, np.ones(len(a))))
    gram = aa.T @ aa
    reg = ridge * len(a)
    gram.flat[::len(gram)+1] += reg
    gram[-1, -1] -= reg
    c = np.linalg.solve(gram, aa.T @ b)
    return c[:-1].T, c[-1]

def sensitivity_fit(a, b, power=0., ridge=1e-7):
    if not power:
        return fit(a, b, ridge)
    aa = np.column_stack((a, np.ones(len(a))))
    result = []
    for j in range(b.shape[1]):
        # Fixed teacher-derived weights; one solve, no reweighting iterations.
        weights = np.maximum(1 - np.tanh(b[:, j])**2, .05)**(2*power)
        gram = aa.T @ (aa * weights[:, None])
        gram.flat[::len(gram)+1] += ridge * weights.sum()
        gram[-1, -1] -= ridge * weights.sum()
        result.append(np.linalg.solve(gram, aa.T @ (b[:, j]*weights)))
    c = np.array(result)
    return c[:, :-1], c[:, -1]

def select(h, wout, n, method='qr'):
    if n == h.shape[1]:
        return np.arange(n)
    hc = h - h.mean(0)
    relevance = np.linalg.norm(wout, axis=0)
    if method == 'norm':
        return np.argsort(np.std(h, axis=0) * relevance)[-n:]
    _, _, piv = qr(hc[:min(len(h), 8192)] * relevance, mode='economic', pivoting=True)
    return piv[:n]

def construct(seed, mode='gated', n=24000, scale=1., distribution='normal', allocation=64,
              feature='average', selection='qr', task_scale=1., ridge=1e-7,
              output_weight=0., hidden_weight=0., linear_units=0, linear_scale=20., gate_selection='last'):
    f, b = source(seed, 'fwd'), source(seed, 'bwd')
    if mode in ('theta0', 'fwd', 'bwd'):
        return {k: source(seed, mode)[k].copy() for k in KEYS}, {}
    ps = [f, b]
    distributions = [distribution]*2 if isinstance(distribution, str) else distribution
    scales = [scale]*2 if isinstance(scale, (int, float)) else scale
    output_weights = [output_weight]*2 if isinstance(output_weight, (int,float)) else output_weight
    hidden_weights = [hidden_weight]*2 if isinstance(hidden_weight, (int,float)) else hidden_weight
    xs = [synthetic(seed, n, scales[t], distributions[t], t) for t in range(2)]
    hs = [hidden(p, x, t) for t, (p, x) in enumerate(zip(ps, xs))]
    out = {k: np.zeros_like(f[k], dtype=np.float64) for k in KEYS}
    out['log_std'] = f['log_std'].copy()
    if feature == 'average':
        obs_weights = (f['a1.weight'][:, :17] + b['a1.weight'][:, :17]) / 2
    elif feature == 'forward75':
        obs_weights = .75*f['a1.weight'][:, :17] + .25*b['a1.weight'][:, :17]
    elif feature == 'theta0':
        obs_weights = source(seed, 'theta0')['a1.weight'][:, :17]
    elif feature in ('fwd', 'bwd'):
        obs_weights = source(seed, feature)['a1.weight'][:, :17]
    elif feature == 'union':
        obs_weights = (f['a1.weight'][:, :17] + b['a1.weight'][:, :17]) / 2
    else:
        raise ValueError(feature)
    out['a1.weight'][:, :17] = obs_weights
    for t, p in enumerate(ps):
        out['a1.weight'][:, 17+t] = task_scale * (p['a1.bias'] + p['a1.weight'][:, 17+t])
    diagnostics = {}
    if mode == 'input_gated':
        # Output-direction assignment is a discrete exact linear-assignment solve.
        wf, wb = f['a3.weight'], b['a3.weight']
        dots = wf.T @ wb
        cost = (wf*wf).sum(0)[:, None] + (wb*wb).sum(0)[None, :] - 2*np.abs(dots)
        _, perm = linear_sum_assignment(cost)
        signs = np.sign(dots[np.arange(128), perm])
        signs[signs == 0] = 1
        target_z = [hs[0][1], hs[1][1][:, perm]*signs]
        out['a1.weight'][:] = 0
        out['a1.weight'][127, 17:] = [10, -10]
        counts = [allocation, 127-allocation]
        offset = 0
        coefs, intercepts, hh2 = [], [], []
        for t, p in enumerate(ps):
            importance = np.sqrt((p['a2.weight']**2 * np.sum(p['a3.weight']**2, axis=0)[:, None]).sum(0))
            ids = select(hs[t][0], importance[None, :], counts[t], selection)
            sl = slice(offset, offset+counts[t])
            w1 = p['a1.weight'][ids, :17]
            b1 = p['a1.bias'][ids] + p['a1.weight'][ids, 17+t]
            out['a1.weight'][sl, :17] = w1
            out['a1.weight'][sl, 17+t] = b1
            out['a1.weight'][sl, 18-t] = 10*np.abs(w1).sum(1)+12
            w2, b2 = sensitivity_fit(hs[t][0][:, ids], target_z[t], hidden_weight, ridge)
            out['a2.weight'][:, sl] = w2
            coefs.append(w2.sum(1)); intercepts.append(b2)
            hh2.append(np.tanh(hs[t][0][:, ids] @ w2.T + b2))
            diagnostics[str(t)] = {'first_layer_units': ids.tolist()}
            offset += counts[t]
        out['a2.bias'] = (intercepts[0]+intercepts[1]-coefs[0]-coefs[1])/2
        out['a2.weight'][:, 127] = (intercepts[0]-intercepts[1]+coefs[0]-coefs[1])/2
        out['a3.weight'], out['a3.bias'] = sensitivity_fit(np.vstack(hh2), np.vstack([h[3] for h in hs]), output_weight, ridge)
        diagnostics['second_layer_permutation'] = perm.tolist()
        diagnostics['second_layer_signs'] = signs.tolist()
    elif mode == 'shared':
        hh = [hidden(out, x, t)[0] for t, x in enumerate(xs)]
        out['a2.weight'], out['a2.bias'] = fit(np.vstack(hh), np.vstack([h[1] for h in hs]), ridge)
        hh2 = [hidden(out, x, t)[2] for t, x in enumerate(xs)]
        out['a3.weight'], out['a3.bias'] = fit(np.vstack(hh2), np.vstack([h[3] for h in hs]), ridge)
    elif mode == 'gated':
        if gate_selection == 'saliency':
            importance = sum((p['a2.weight']**2 * np.sum(p['a3.weight']**2, axis=0)[:, None]).sum(0) for p in ps)
            dropped = int(np.argmin(importance))
            kept = np.array([i for i in range(128) if i != dropped])
            out['a1.weight'][:127] = out['a1.weight'][kept].copy()
            diagnostics['dropped_first_layer_unit'] = dropped
        if feature == 'union':
            candidates = np.vstack([p['a1.weight'][:, :17] for p in ps])
            biases = [np.tile(task_scale*(p['a1.bias']+p['a1.weight'][:, 17+t]), 2) for t,p in enumerate(ps)]
            activity = np.vstack([np.tanh(xs[t][:4096] @ candidates.T + biases[t]) for t in range(2)])
            _, _, piv = qr(activity-activity.mean(0), mode='economic', pivoting=True)
            ids1 = piv[:127]
            out['a1.weight'][:127, :17] = candidates[ids1]
            for t in range(2):
                out['a1.weight'][:127, 17+t] = biases[t][ids1]
            diagnostics['first_layer_units'] = ids1.tolist()
        # Reserve last first-layer neuron for exact float32 +/-1 task routing.
        out['a1.weight'][127] = 0
        out['a1.weight'][127, 17:] = [10, -10]
        out['a2.weight'][127, 127] = 10
        counts = [allocation, 127-allocation]
        offset = 0
        coefs, intercepts = [], []
        for t, p in enumerate(ps):
            ids = select(hs[t][2], p['a3.weight'], counts[t]-linear_units, selection)
            h = hidden(out, xs[t], t)[0][:, :127]
            w2, b2 = sensitivity_fit(h, hs[t][1][:, ids], hidden_weights[t], ridge)
            z2 = h @ w2.T + b2
            h2 = np.tanh(z2)
            if linear_units:
                assert linear_units == 6
                joint, b3 = sensitivity_fit(np.column_stack((h2, h)), hs[t][3], output_weights[t], ridge)
                nonlinear_count = h2.shape[1]
                w2 = np.vstack((w2, joint[:, nonlinear_count:]/linear_scale))
                b2 = np.concatenate((b2, np.zeros(6)))
                w3 = np.column_stack((joint[:, :nonlinear_count], np.eye(6)*linear_scale))
            else:
                w3, b3 = sensitivity_fit(h2, hs[t][3], output_weights[t], ridge)
            sl = slice(offset, offset+counts[t])
            out['a2.weight'][sl, :127] = w2
            # Bound all inactive preactivations above +10 for any h1 in [-1,1].
            gate = np.abs(w2).sum(1) + np.abs(b2) + 12
            out['a2.weight'][sl, 127] = (-1 if t == 0 else 1) * gate / 2
            out['a2.bias'][sl] = b2 + gate / 2
            out['a3.weight'][:, sl] = w3
            coefs.append(w3.sum(1)); intercepts.append(b3)
            diagnostics[str(t)] = {'units': ids.tolist(), 'preactivation_rmse': float(np.mean((z2-hs[t][1][:, ids])**2)**.5)}
            offset += counts[t]
        out['a3.bias'] = (intercepts[0]+intercepts[1]-coefs[0]-coefs[1])/2
        out['a3.weight'][:, 127] = (intercepts[0]-intercepts[1]+coefs[0]-coefs[1])/2
    else:
        raise ValueError(mode)
    out = {k: np.asarray(out[k], np.float32) for k in KEYS}
    for t in range(2):
        xx = synthetic(seed, 6000, scales[t], distributions[t], t, stream=1)
        yy = np.tanh(hidden(ps[t], xx, t)[3])
        yh = np.tanh(hidden(out, xx, t)[3])
        diagnostics.setdefault(str(t), {})['synthetic_action_rmse'] = float(np.mean((yy-yh)**2)**.5)
    return out, diagnostics

def save(seed, name, **config):
    p, diag = construct(seed, **config)
    path = HERE / name
    np.savez(path, **p)
    path.with_suffix('.json').write_text(json.dumps({'seed': seed, 'config': config, 'diagnostics': diag}, indent=2)+'\n')
    print(json.dumps({'file': name, 'config': config, 'rmse': [diag.get(str(t), {}).get('synthetic_action_rmse') for t in range(2)]}), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int)
    ap.add_argument('--name')
    ap.add_argument('--config', default='{}')
    args = ap.parse_args()
    if args.seed is not None:
        save(args.seed, args.name, **json.loads(args.config))
    else:
        # Fixed after exactly 50 evaluations; rebuilding never invokes an evaluator.
        for seed, config in enumerate(FINAL_CONFIGS):
            save(seed, f'merged_s{seed}.npz', **config)

FINAL_CONFIGS = [
    {'mode': 'gated', 'allocation': 48,
     'hidden_weight': [0., 1.], 'output_weight': [0., 1.]},
    {'mode': 'gated', 'distribution': 'motion'},
    {'mode': 'gated', 'feature': 'fwd', 'distribution': ['normal', 'motion'],
     'hidden_weight': [0., 1.], 'output_weight': [0., 1.]},
    {'mode': 'gated', 'feature': 'forward75', 'distribution': ['speed', 'normal'],
     'hidden_weight': [0., 1.], 'output_weight': [0., 1.]},
]

if __name__ == '__main__':
    main()
