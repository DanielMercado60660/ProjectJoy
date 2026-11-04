"""Deterministic weight-space surgery. No simulator or gradient optimizer imports."""
from pathlib import Path
import argparse
import json
import os
for _thread_env in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_env] = '4'
import numpy as np
from scipy.linalg import qr

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'
KEYS = ['a1.weight', 'a1.bias', 'a2.weight', 'a2.bias', 'a3.weight', 'a3.bias', 'log_std']
# Frozen after exactly 50 evaluations. Rebuilding reads no attempt artifacts.
FINAL_CONFIGS = {
    0: {'mode':'shared','scale':1.5,'profile':'upright','mix':1.0},
    1: {'mode':'splice','branches':[
        {'mode':'shared'},
        {'mode':'shared','scale':1.5,'mixture':True}]},
    2: {'mode':'splice','branches':[
        {'mode':'shared','scale':1.5,'profile':'upright','basis':'pool','ranking':'weight'},
        {'mode':'shared','scale':1.5,'profile':'upright','basis':'pool'}]},
    3: {'mode':'splice','branches':[
        {'mode':'shared','scale':1.5,'profile':'transition'},
        {'mode':'shared','scale':1.5,'profile':'upright','weighted3':True}]},
}


def source(seed, name):
    with np.load(DATA / f's{seed}' / f'{name}.npz') as z:
        return {k: z[k].copy() for k in z.files}


def synthetic(seed, task, n, scale=1., velocity=3., mixture=False, profile='normal'):
    rng = np.random.default_rng(70231 + seed * 17 + task)
    x = rng.standard_normal((n, 17))
    if mixture:
        x *= rng.choice([0.5, 1., 1.5], size=(n, 1))
    x *= scale
    x[:, 8] += velocity if task == 0 else -velocity
    if profile in ('transition', 'upright'):
        x[:,8] = rng.uniform(-0.5, 2*velocity, n) * (1 if task == 0 else -1)
    if profile == 'upright':
        q = source(seed,'theta0')
        x[:,1] = x[:,1]*.25 - q['norm_mean'][1]/np.sqrt(q['norm_var'][1])
        x[:,0] *= .6
    return np.clip(x, -10, 10)


def affine_fit(x, y, ridge=1e-5):
    a = np.column_stack([x, np.ones(len(x))])
    g = a.T @ a / len(a)
    reg = np.eye(a.shape[1]) * ridge
    reg[-1, -1] = 0.
    return np.linalg.solve(g + reg, a.T @ y / len(a))


def weighted_affine_fit(x, y, ridge=1e-5, floor=.1):
    """One closed-form solve per output, weighted by teacher tanh sensitivity."""
    a = np.column_stack([x, np.ones(len(x))])
    reg = np.eye(a.shape[1])*ridge
    reg[-1,-1] = 0
    answer = np.empty((a.shape[1],y.shape[1]))
    for j in range(y.shape[1]):
        w = floor + (1-np.tanh(y[:,j])**2)**2
        weighted = a*w[:,None]
        answer[:,j] = np.linalg.solve(a.T@weighted/len(a)+reg, weighted.T@y[:,j]/len(a))
    return answer


def select(h, wout, count, ranking='qr'):
    if ranking == 'weight':
        merit = np.std(h, axis=0) * np.linalg.norm(wout, axis=0)
        return np.argsort(-merit, kind='stable')[:count]
    # Rank independent activation columns, scaled by their output influence.
    a = h[:4096] - h[:4096].mean(axis=0)
    a *= np.linalg.norm(wout, axis=0)[None, :]
    _, _, piv = qr(a, mode='economic', pivoting=True)
    return piv[:count]


def construct(seed, cfg):
    if cfg['mode'] == 'splice':
        configs = cfg['branches']
        p0, m0 = construct(seed,configs[0])
        p1, m1 = construct(seed,configs[1])
        assert np.array_equal(p0['a1.weight'],p1['a1.weight'])
        assert np.array_equal(p0['a1.bias'],p1['a1.bias'])
        cut0 = configs[0].get('count_f',63)+1
        cut1 = configs[1].get('count_f',63)+1
        assert cut0 == cut1
        p = {k:v.copy() for k,v in p0.items()}
        p['a2.weight'][cut0:] = p1['a2.weight'][cut1:]
        p['a2.bias'][cut0:] = p1['a2.bias'][cut1:]
        p['a3.weight'][:,cut0:] = p1['a3.weight'][:,cut1:]
        p['a3.bias'] = (np.array(m0['0']['intercept'])+np.array(m1['1']['intercept'])).astype(np.float32)
        return p, {'0':m0['0'],'1':m1['1']}
    f, b = source(seed, 'fwd'), source(seed, 'bwd')
    if cfg['mode'] in ('fwd', 'theta0'):
        p = f if cfg['mode'] == 'fwd' else source(seed, 'theta0')
        return {k: p[k].astype(np.float32) for k in KEYS}, {}
    n = cfg.get('n', 16384)
    count_f = cfg.get('count_f', 63)
    counts = [count_f, 126-count_f]
    p = {k: np.zeros_like(f[k], dtype=np.float64) for k in KEYS}
    p['log_std'] = f['log_std'].copy()
    mix = cfg.get('mix', .5)
    w1 = mix * f['a1.weight'][:, :17] + (1-mix) * b['a1.weight'][:, :17]
    # Remove the common first-layer feature with least downstream path magnitude.
    influence = sum(np.linalg.norm(q['a3.weight'] @ q['a2.weight'], axis=0) for q in (f,b))
    keep = np.argsort(-influence, kind='stable')[:127]
    p['a1.weight'][:127, :17] = w1[keep]
    for t, q in enumerate((f,b)):
        p['a1.weight'][:127, 17+t] = q['a1.bias'][keep] + q['a1.weight'][keep, 17+t]
    basis = cfg.get('basis','mean')
    if basis == 'pool':
        pool = np.concatenate([f['a1.weight'][:,:17],b['a1.weight'][:,:17]])
        activations = []
        biases = []
        for t,q in enumerate((f,b)):
            bias = np.tile(q['a1.bias']+q['a1.weight'][:,17+t],2)
            biases.append(bias)
            x = synthetic(seed,t,2048,cfg.get('scale',1.),cfg.get('velocity',3.),cfg.get('mixture',False),cfg.get('profile','normal'))
            activations.append(np.tanh(x@pool.T+bias))
        values = np.concatenate(activations)
        values -= values.mean(axis=0)
        values *= np.tile(influence,2)
        _,_,order = qr(values,mode='economic',pivoting=True)
        ids = order[:127]
        p['a1.weight'][:127,:17] = pool[ids]
        for t in range(2):
            p['a1.weight'][:127,17+t] = biases[t][ids]
    elif basis == 'linear':
        p['a1.weight'][:17] = 0
        p['a1.weight'][:17,:17] = np.eye(17)*.1
        p['a1.weight'][17:127,:17] = w1[keep[:110]]
        for t,q in enumerate((f,b)):
            p['a1.weight'][17:127,17+t] = q['a1.bias'][keep[:110]]+q['a1.weight'][keep[:110],17+t]
    p['a1.weight'][127, 17:] = [3., -3.]
    gate = np.tanh(3.)
    metrics = {}
    offset = 0
    for t, (q, count) in enumerate(zip((f,b), counts)):
        x = synthetic(seed, t, n, cfg.get('scale', 1.), cfg.get('velocity',3.), cfg.get('mixture',False), cfg.get('profile','normal'))
        hteacher = np.tanh(x @ q['a1.weight'][:, :17].T + q['a1.bias'] + q['a1.weight'][:, 17+t])
        zteacher = hteacher @ q['a2.weight'].T + q['a2.bias']
        h2teacher = np.tanh(zteacher)
        logits = h2teacher @ q['a3.weight'].T + q['a3.bias']
        h = np.tanh(x @ p['a1.weight'][:127,:17].T + p['a1.weight'][:127,17+t])
        fit2 = affine_fit(h, zteacher, cfg.get('ridge2',1e-5))
        z2 = h @ fit2[:-1] + fit2[-1]
        h2 = np.tanh(z2)
        inds = select(h2, q['a3.weight'], count, cfg.get('ranking','qr'))
        if cfg.get('weighted2',False):
            fit2[:,inds] = weighted_affine_fit(h, zteacher[:,inds], cfg.get('ridge2',1e-5), cfg.get('floor2',.1))
            h2[:,inds] = np.tanh(h@fit2[:-1,inds]+fit2[-1,inds])
        if cfg.get('weighted3',False):
            fit3 = weighted_affine_fit(h2[:,inds], logits, cfg.get('ridge3',1e-5), cfg.get('floor3',.1))
        else:
            fit3 = affine_fit(h2[:, inds], logits, cfg.get('ridge3',1e-5))
        rows = slice(offset, offset+count)
        p['a2.weight'][rows,:127] = fit2[:-1,inds].T
        active_bias = fit2[-1,inds]
        # For every h1 in [-1,1], off-task preactivation is at least +12.
        d = 12 + np.abs(active_bias) + np.abs(fit2[:-1,inds]).sum(axis=0)
        sign = -1 if t == 0 else 1
        p['a2.weight'][rows,127] = sign * d/(2*gate)
        p['a2.bias'][rows] = active_bias+d/2
        p['a3.weight'][:,rows] = fit3[:-1].T
        # One constant unit per branch permits an arbitrary active intercept.
        c = offset+count
        p['a2.weight'][c,127] = sign*12/(2*gate)
        p['a2.bias'][c] = 6
        p['a3.weight'][:,c] = -fit3[-1] - fit3[:-1].sum(axis=0)
        p['a3.bias'] += fit3[-1]
        pred = h2[:,inds] @ fit3[:-1] + fit3[-1]
        metrics[str(t)] = {'action_mse':float(np.mean((np.tanh(pred)-np.tanh(logits))**2)),
                           'logit_mse':float(np.mean((pred-logits)**2)), 'h2_indices':inds.tolist(),
                           'intercept':fit3[-1].tolist()}
        offset += count+1
    return {k: v.astype(np.float32) for k,v in p.items()}, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str)
    ap.add_argument('--seed',type=int)
    ap.add_argument('--out',type=Path)
    args = ap.parse_args()
    if args.config:
        cfg = json.loads(args.config)
        p, metrics = construct(args.seed, cfg)
        np.savez(args.out, **p)
        print(json.dumps(metrics))
    else:
        for seed in range(4):
            p, metrics = construct(seed, FINAL_CONFIGS[seed])
            np.savez(HERE/f'merged_s{seed}.npz', **p)
            print(seed, json.dumps(metrics))


if __name__ == '__main__':
    main()
