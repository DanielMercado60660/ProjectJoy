"""Round-5 turn-2 reviewer checks (numpy/MuJoCo only, no MLX, ~1 min, 1 thread).

1. Aggregate results/r5/t2_s{0,1,2}/{eval,training,qualification}.json across seeds.
2. Re-run each seed's unchanged CPG on its fresh test bank (must reproduce eval.json exactly).
3. Pooled-CPG check: per skill, choose among the three independent turn-1 CEM picks
   (main_s0/1/2) on each seed's VALIDATION bank only, re-pick phase reset on validation,
   then score once on the fresh test bank. Search cost = 3 x 1,536,000 = 4,608,000 env steps.
Writes results/r5/review_t2/review.json.
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from r5_core import TASKS, bank, cpg_policy, episode, gait, make_env, orders, reward

R = Path('results/r5')
OUT = R / 'review_t2'
SEEDS = (0, 1, 2)
METHODS = ('adapter', 'dagger', 'joint', 'oracle', 'cpg', 'zero')


def summarize(rows):
    iso = [r for r in rows if r['kind'] == 'isolated']
    sw = [r for r in rows if r['kind'] == 'transition' and r['switch']]
    m = lambda L, k: float(np.mean([r[k] for r in L]))
    return dict(tracking=m(iso, 'tracking_fraction'), speed_mae=m(iso, 'speed_mae'), return_=m(iso, 'return_'),
                per_task={t: m([r for r in iso if r['task'] == t], 'tracking_fraction') for t in TASKS},
                switches=len(sw), failures=int(sum(r['failure'] for r in sw)),
                falls=int(sum(r['fall_proxy'] for r in sw)), stalls=int(sum(r['stall'] for r in sw)),
                unreached=int(sum(not r['reached'] for r in sw)),
                switch_failure=m(sw, 'failure'), capped_settle_s=m(sw, 'capped_settle_s'))


def aggregate():
    out = dict(per_seed={}, pooled={}, retention={}, cost={}, memory={}, qualification={}, failing_transitions={})
    for s in SEEDS:
        d = json.loads((R / f't2_s{s}/eval.json').read_text())
        rows = [r for r in d['rows'] if r['split'] == 'test' and r['stage'] == 3]
        out['per_seed'][s] = {m: summarize([r for r in rows if r['method'] == m]) for m in METHODS}
        out['retention'][s] = d['retention']
        out['memory'][s] = d['sizes']
        fails = {}
        for r in rows:
            if r['kind'] == 'transition' and r['switch'] and r['failure']:
                key = f"{r['method']}:{r['source']}->{r['task']}"
                fails[key] = fails.get(key, 0) + 1
        out['failing_transitions'][s] = fails
        t = json.loads((R / f't2_s{s}/training.json').read_text())
        out['cost'][s] = {m: L[-1] for m, L in t['ledger'].items()}
        q = json.loads((R / f't2_s{s}/qualification.json').read_text())
        out['qualification'][s] = {k: dict(tracking=v['tracking'], qualified=v['qualified'], selected_step=v['selected_step'])
                                   for k, v in q.items()}
    for m in METHODS:
        vals = [out['per_seed'][s][m] for s in SEEDS]
        out['pooled'][m] = dict(
            tracking_mean=float(np.mean([v['tracking'] for v in vals])),
            tracking_sd=float(np.std([v['tracking'] for v in vals], ddof=1)),
            speed_mae_mean=float(np.mean([v['speed_mae'] for v in vals])),
            switch_failures=f"{sum(v['failures'] for v in vals)}/{sum(v['switches'] for v in vals)}",
            falls=sum(v['falls'] for v in vals), stalls=sum(v['stalls'] for v in vals),
            unreached=sum(v['unreached'] for v in vals),
            capped_settle_mean=float(np.mean([v['capped_settle_s'] for v in vals])),
            capped_settle_sd=float(np.std([v['capped_settle_s'] for v in vals], ddof=1)))
    fails = {m: sum(out['per_seed'][s][m]['failures'] for s in SEEDS) for m in METHODS}
    n = 144
    out['fisher_pooled_switch_failures'] = {
        f'{a}_vs_{b}': float(fisher_exact([[fails[a], n - fails[a]], [fails[b], n - fails[b]]])[1])
        for a, b in (('adapter', 'joint'), ('adapter', 'dagger'), ('adapter', 'oracle'), ('dagger', 'joint'), ('dagger', 'oracle'))}
    out['fisher_note'] = ('Treats 144 switches as independent; they are not (4 near-identical resets x 12 switches/seed, '
                          'failures cluster on one transition type per seed), so p-values are optimistic.')
    return out


def cpg_val_score(theta, tid, seed, horizon=500, env=None):
    env.reset(seed=int(seed))
    score = 0.
    for t in range(horizon):
        a = gait(theta, t)
        _, _, _, _, info = env.step(a)
        score += reward(tid, info['x_velocity'], a)
    return score / horizon


def cpg_test(theta, reset, s):
    rows = []
    for seed in bank('test_t2', s, 4):
        for tid in range(3):
            rows += [dict(r, kind='isolated') for r in episode(cpg_policy(theta, reset), seed, (tid,), 600)['segments']]
        for order in orders(3):
            rows += [dict(r, kind='transition') for r in episode(cpg_policy(theta, reset), seed, order, 200)['segments']]
    return summarize(rows)


def pooled_cpg():
    cpgs = {s: json.loads((R / f'main_s{s}/cpg.json').read_text()) for s in SEEDS}
    env = make_env(500)
    out = {}
    for s in SEEDS:
        own = cpgs[s]
        rec = dict(original=cpg_test(np.array(own['theta']), own['reset_phase'], s))
        theta, choice, val = [], [], []
        for tid in range(3):
            scores = [float(np.mean([cpg_val_score(np.array(cpgs[src]['theta'][tid]), tid, v, env=env)
                                     for v in bank('val', s, 3)])) for src in SEEDS]
            choice.append(int(np.argmax(scores)))
            val.append(scores)
            theta.append(cpgs[choice[-1]]['theta'][tid])
        theta = np.array(theta)
        phase = {}
        for reset in (False, True):
            phase[str(reset)] = float(np.mean([np.mean([r['tracking_fraction'] for r in
                                    episode(cpg_policy(theta, reset), int(v), o, 200)['segments']])
                                    for o in orders(3) for v in bank('val', s, 3)]))
        reset = phase['True'] > phase['False']
        rec.update(pooled_choice_source_seed=choice, pooled_val_scores=val, pooled_phase_val=phase,
                   pooled_reset_phase=bool(reset), pooled=cpg_test(theta, reset, s))
        out[s] = rec
    env.close()
    return out


if __name__ == '__main__':
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = aggregate()
    res['cpg_check'] = pooled_cpg()
    for s in SEEDS:  # harness must reproduce the stored CPG test numbers
        a, b = res['cpg_check'][s]['original'], res['per_seed'][s]['cpg']
        assert abs(a['tracking'] - b['tracking']) < 1e-9 and a['failures'] == b['failures'], (s, a, b)
    res['cpg_check_note'] = ('pooled = per skill, best of the three independent turn-1 CEM picks on this seed\'s '
                             'validation bank; 3x the per-seed CPG search budget (4,608,000 env steps); test used once.')
    res['wall_s'] = time.time() - t0
    (OUT / 'review.json').write_text(json.dumps(res, indent=1, default=float))
    for m, v in res['pooled'].items():
        print(m, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    print('fisher', res['fisher_pooled_switch_failures'])
    for s in SEEDS:
        c = res['cpg_check'][s]
        print('cpg seed', s, 'orig', round(c['original']['tracking'], 4), c['original']['failures'],
              'pooled', round(c['pooled']['tracking'], 4), c['pooled']['failures'], c['pooled']['per_task'],
              'choice', c['pooled_choice_source_seed'], 'reset', c['pooled_reset_phase'])
    print(f'R5_REVIEW_T2_OK {res["wall_s"]:.1f}s')
