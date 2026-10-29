"""Round-5 entry points; only `train` imports MLX. See RUNME_R5.md."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import time

# Limit CPU libraries before NumPy/MLX import. One process per job.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

from r5_core import (CPG_HI, CPG_LO, TASKS, Deadline, add_adapter, atomic_pickle,
                     bank, budget, cpg_policy, episode, gait, init_params, make_env,
                     network_policy, orders, read_pickle, retention, reward,
                     size_counts, write_json)

PROFILES = {
    'tiny': dict(base=64, teacher=64, student=64, hidden=16, rank=2, envs=2, rollout=16,
                 epochs=1, minibatch=32, dagger_batches=2, horizon=48, switch_steps=16,
                 replay_capacity=64, cpg_iters=2, cpg_pop=4, cpg_restarts=2,
                 cpg_horizon=40, eval_segment=24, eval_seeds=1),
    'battery': dict(base=262144, teacher=393216, student=65536, hidden=128, rank=8,
                    envs=8, rollout=128, epochs=4, minibatch=512, dagger_batches=20,
                    horizon=600, switch_steps=200, replay_capacity=8192,
                    cpg_iters=32, cpg_pop=16, cpg_restarts=2, cpg_horizon=500,
                    eval_segment=200, eval_seeds=4),
}


def locations(tag):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('tag must contain only letters, digits, underscore or hyphen')
    ck, res = Path('checkpoints') / f'r5_{tag}', Path('results/r5') / tag
    ck.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return ck, res


def manifest(args, z, ck, res):
    doc = dict(schema=1, tag=args.tag, seed=args.seed, profile=args.profile, config=z,
               targets=[3., -3., 1.], budget=budget(z['base'], z['teacher'], z['student']),
               unit='completed training environment steps + teacher query examples; not equal FLOPs',
               no_legacy_warm_start=True,
               source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                              for name in ('r5.py', 'r5_core.py', 'r5_train.py', 'r5_report.py')})
    path = res / 'manifest.json'
    if path.exists():
        import json
        if json.loads(path.read_text()) != doc:
            raise ValueError('tag already has a different manifest; use a fresh tag')
    else:
        write_json(path, doc)
    return doc


def run_train(args, z, ck, res, deadline):
    from r5_train import train
    costs = budget(z['base'], z['teacher'], z['student'])
    start = init_params(args.seed, z['hidden'])
    shared = dict(envs=z['envs'], rollout=z['rollout'], epochs=z['epochs'],
                  minibatch=z['minibatch'], dagger_batches=z['dagger_batches'],
                  horizon=z['horizon'], switch_steps=z['switch_steps'],
                  replay_capacity=z['replay_capacity'], experiment_seed=args.seed)

    def go(name, params, method, seen, steps, rng_offset, teachers=None, replay=None):
        cfg = dict(shared, method=method, seen=seen, steps=steps,
                   seed=100*args.seed+rng_offset, lr=1e-3 if method == 'dagger' else 3e-4)
        return train(params, ck / f'{name}.pkl', cfg, deadline, teachers, replay,
                     pause=args.pause_after_checkpoint)

    base = go('base', start, 'ppo', [0], z['base'], 0)
    teachers = {0: base['params']}
    previous = {name: base for name in ('adapter', 'dagger', 'joint')}
    records = {name: dict(stage1='base.pkl') for name in previous}
    ledger = {name: [dict(stage=1, components=['base.pkl'])] for name in previous}
    # Stage-wise interleaving ensures a timeout leaves all methods' earlier stages intact.
    for stage in (2, 3):
        tid = stage - 1
        teacher = go(f'teacher_{TASKS[tid]}', base['params'], 'ppo', [tid], z['teacher'], 10+tid)
        teachers[tid] = teacher['params']
        for method in ('adapter', 'dagger', 'joint'):
            before = previous[method]
            p = before['params']
            if method == 'adapter':
                p = add_adapter(p, tid, z['rank'], 100*args.seed+30+tid)
            result = go(f'{method}_s{stage}', p, method if method != 'joint' else 'ppo',
                        [tid] if method == 'adapter' else list(range(stage)),
                        z['student'] if method == 'dagger' else costs['addition'],
                        40+tid, teachers=teachers if method == 'dagger' else None,
                        replay=before['replay'] if method == 'dagger' else None)
            if method == 'adapter':
                # Structural invariant: neither trunk nor any older route changes.
                assert all(np.array_equal(before['params'][k], result['params'][k]) for k in before['params'])
            previous[method] = result
            name = f'{method}_s{stage}.pkl'
            records[method][f'stage{stage}'] = name
            components = ledger[method][-1]['components'] + [name]
            if method == 'dagger':
                components = components + [f'teacher_{TASKS[tid]}.pkl']
            ledger[method].append(dict(stage=stage, components=components))
    for method, stages in ledger.items():
        for row in stages:
            states = [read_pickle(ck / f) for f in row['components']]
            keys = states[0]['cost'].keys()
            row['cost'] = {k: sum(s['cost'][k] for s in states) for k in keys}
            row['credits'] = row['cost']['env_steps'] + row['cost']['teacher_queries']
            row['expected_credits'] = z['base'] + (row['stage'] - 1) * costs['addition']
            assert row['credits'] == row['expected_credits']
            row['cost']['specialist_env_steps_included'] = sum(s['cost']['env_steps'] for f, s in
                                                               zip(row['components'], states)
                                                               if f.startswith('teacher_'))
    write_json(res / 'training.json', dict(complete=True, checkpoints=records, ledger=ledger))
    print('R5_TRAIN_COMPLETE exact per-stage credits agree; teachers included', flush=True)


def fast_cpg_score(theta, tid, seed, horizon, env=None):
    owned = env is None
    if owned:
        env = make_env(horizon)
    try:
        env.reset(seed=int(seed))
        score = 0.
        for t in range(horizon):
            a = gait(theta, t)
            _, _, _, _, info = env.step(a)
            score += reward(tid, info['x_velocity'], a)
        return float(score / horizon)
    finally:
        if owned:
            env.close()


def run_cpg(args, z, ck, res, deadline, env):
    config = {k: v for k, v in z.items() if k.startswith('cpg_') or k.startswith('eval_')}
    config['seed'] = args.seed
    path = ck / 'cpg_state.pkl'
    rng = np.random.default_rng(1000 + args.seed)
    st = dict(config=config, jobs={}, picks={}, phase_scores={}, train_steps=0, val_steps=0,
              rng=rng.bit_generator.state, complete=False, wall_s=0.)
    if path.exists():
        st = read_pickle(path)
        assert st['config'] == config, 'CPG configuration changed; use fresh tag'
        rng.bit_generator.state = st['rng']
    if st['complete']:
        print('R5_CPG_COMPLETE (already saved)', flush=True)
        return
    t0, old_wall = time.monotonic(), st['wall_s']

    def save():
        st['rng'] = rng.bit_generator.state
        st['wall_s'] = old_wall + time.monotonic() - t0
        atomic_pickle(path, st)

    for tid in range(3):
        for restart in range(z['cpg_restarts']):
            key = f'{tid}_{restart}'
            job = st['jobs'].setdefault(key, dict(iter=0, mu=(CPG_HI+CPG_LO)/2,
                       sd=(CPG_HI-CPG_LO)/3, best=None, best_score=-1e30, curve=[], scores=[], candidates=None))
            while job['iter'] < z['cpg_iters']:
                deadline.check()
                if job['candidates'] is None:
                    job['candidates'] = np.clip(job['mu'] + job['sd'] * rng.normal(size=(z['cpg_pop'], 19)), CPG_LO, CPG_HI)
                    job['reset_seed'] = int(rng.choice(bank('cpg', args.seed, 10000)))
                    save()
                for j in range(len(job['scores']), z['cpg_pop']):
                    deadline.check()
                    job['scores'].append(fast_cpg_score(job['candidates'][j], tid, job['reset_seed'], z['cpg_horizon'], env))
                    st['train_steps'] += z['cpg_horizon']
                    save()  # candidate boundary: no repeated completed simulation on resume
                scores = np.array(job['scores'])
                elite = job['candidates'][np.argsort(scores)[-max(2, z['cpg_pop']//4):]]
                job['mu'], job['sd'] = elite.mean(0), elite.std(0) + .03*(CPG_HI-CPG_LO)
                if scores.max() > job['best_score']:
                    job['best_score'] = float(scores.max())
                    job['best'] = job['candidates'][scores.argmax()].copy()
                job['curve'].append(float(scores.max()))
                job['iter'] += 1
                job['candidates'], job['scores'] = None, []
                save()
            print(f'[r5 cpg] {TASKS[tid]} restart={restart} generations={job["iter"]}', flush=True)
        if str(tid) not in st['picks']:
            candidates = [st['jobs'][f'{tid}_{r}'][key] for r in range(z['cpg_restarts']) for key in ('mu', 'best')]
            selection = st.setdefault('selection', {}).setdefault(str(tid), {})
            scores = []
            for index, candidate in enumerate(candidates):
                vals = selection.setdefault(str(index), [])
                seeds = bank('val', args.seed, min(3, z['eval_seeds']))
                for seed in seeds[len(vals):]:
                    deadline.check()
                    vals.append(fast_cpg_score(candidate, tid, seed, z['cpg_horizon'], env))
                    st['val_steps'] += z['cpg_horizon']
                    save()
                scores.append(float(np.mean(vals)))
            theta = candidates[int(np.argmax(scores))]
            st['picks'][str(tid)] = dict(theta=theta, candidate_scores=scores,
                                       near_box_edge=(np.minimum(theta-CPG_LO, CPG_HI-theta) < .02*(CPG_HI-CPG_LO)))
            save()
    theta = np.array([st['picks'][str(t)]['theta'] for t in range(3)])
    # Tune switch phase handling on validation only, identically for all three gaits.
    for reset in (False, True):
        scores = st['phase_scores'].setdefault(str(reset), [])
        cases = [(o, int(s)) for o in orders(3) for s in bank('val', args.seed, min(3,z['eval_seeds']))]
        for order, seed in cases[len(scores):]:
            deadline.check()
            ep = episode(cpg_policy(theta, reset), seed, order, z['eval_segment'])
            scores.append(float(np.mean([s['tracking_fraction'] for s in ep['segments']])))
            st['val_steps'] += len(order) * z['eval_segment']
            save()
    reset = np.mean(st['phase_scores']['True']) > np.mean(st['phase_scores']['False'])
    st['wall_s'] = old_wall + time.monotonic() - t0
    write_json(res / 'cpg.json', dict(theta=theta, reset_phase=bool(reset),
               picks=st['picks'], phase_scores=st['phase_scores'], train_steps=st['train_steps'],
               val_steps=st['val_steps'], wall_s=st['wall_s'], config=config,
               generosity='CPG gets its own reported search budget; not an equal-compute comparison'))
    st['complete'] = True
    save()
    print('R5_CPG_COMPLETE all 3 tasks equally tuned, validation-only choices', flush=True)


def load_complete(path):
    st = read_pickle(path)
    if not st['complete']:
        raise ValueError(f'incomplete checkpoint: {path}')
    return st


def run_eval(args, z, ck, res, deadline):
    import json
    training = json.loads((res / 'training.json').read_text())
    cpg = json.loads((res / 'cpg.json').read_text())
    snapshots = training['checkpoints']
    models, sizes = {}, {}
    for method, mapping in snapshots.items():
        for stage in (1, 2, 3):
            st = load_complete(ck / mapping[f'stage{stage}'])
            models[(method, stage)] = network_policy(st['params'])
            sizes[f'{method}_s{stage}'] = dict(size_counts(st['params'], stage),
                   checkpoint_bytes=(ck / mapping[f'stage{stage}']).stat().st_size,
                   trainable_parameters=sum(st['params'][k].size for k in st['trainable_keys']))
    teachers = {0: load_complete(ck/'base.pkl')['params']}
    for tid in (1, 2):
        teachers[tid] = load_complete(ck/f'teacher_{TASKS[tid]}.pkl')['params']
    for stage in (1, 2, 3):
        models[('oracle', stage)] = network_policy(None, teachers)
    # Switch phase was selected with all three skills, so CPG is a final-stage baseline.
    models[('cpg', 3)] = cpg_policy(np.array(cpg['theta']), cpg['reset_phase'])
    sizes['oracle_s3'] = dict(actor_parameters=sum(size_counts(p, 1)['actor_parameters'] for p in teachers.values()),
                              actor_bytes=sum(size_counts(p, 1)['actor_bytes'] for p in teachers.values()))
    sizes['cpg_s3'] = dict(actor_parameters=57, actor_bytes=np.array(cpg['theta'], dtype=np.float64).nbytes)
    models[('zero', 3)] = lambda obs, tid, t, age: np.zeros(6)
    path = res / 'eval_state.pkl'
    st = dict(rows=[], done=[], examples={}, env_steps=0, wall_s=0.)
    if path.exists():
        st = read_pickle(path)
    old_wall, t0 = st['wall_s'], time.monotonic()
    for split in ('test', 'legacy'):
        seeds = bank(split, args.seed, z['eval_seeds'] if split == 'test' else min(2,z['eval_seeds']))
        for (method, stage), policy in models.items():
            if split == 'legacy' and stage != 3:
                continue
            cases = [('isolated', (tid,), 3*z['eval_segment']) for tid in range(stage)]
            if stage > 1:
                cases += [('transition', order, z['eval_segment']) for order in orders(stage)]
            for kind, order, segment in cases:
                for seed in seeds:
                    key = f'{split}_{method}_{stage}_{kind}_{order}_{seed}'
                    if key in st['done']:
                        continue
                    deadline.check()
                    ep = episode(policy, seed, order, segment, deadline)
                    for row in ep['segments']:
                        st['rows'].append(dict(row, method=method, stage=stage, kind=kind, split=split,
                                               reset_seed=int(seed), order=list(order)))
                    st['env_steps'] += segment*len(order)
                    if stage == 3 and kind == 'transition' and split == 'test':
                        severity = (sum(r['failure'] for r in ep['segments'][1:]),
                                    float(np.mean([r['speed_mae'] for r in ep['segments']])))
                        previous = st['examples'].get(method)
                        if previous is None or severity > previous['severity']:
                            st['examples'][method] = dict(severity=severity, episode=ep)
                    st['done'].append(key)
                    st['wall_s'] = old_wall + time.monotonic() - t0
                    atomic_pickle(path, st)
    write_json(res / 'eval.json', dict(complete=True, rows=st['rows'], sizes=sizes,
                  retention=retention([r for r in st['rows'] if r['split']=='test']),
                  env_steps=st['env_steps'], wall_s=st['wall_s'],
                  note='fall is a torso-height/pitch proxy; capped latency includes non-settlers; test used only after fitting'))
    write_json(res / 'demo_traces.json', st['examples'])
    from r5_report import report
    report(args.tag, res, deadline)
    print('R5_EVAL_COMPLETE fresh-test and legacy rows separate; worst transitions saved', flush=True)


def replay_demo(args, z, ck, res, deadline):
    """Rerun the saved worst cases from checkpoints, independent of eval resume cache."""
    import json
    traces = json.loads((res/'demo_traces.json').read_text())
    cpg = json.loads((res/'cpg.json').read_text())
    teachers = {0: load_complete(ck/'base.pkl')['params']}
    for tid in (1, 2):
        teachers[tid] = load_complete(ck/f'teacher_{TASKS[tid]}.pkl')['params']
    out = {}
    for method, example in traces.items():
        if method in ('adapter', 'dagger', 'joint'):
            policy = network_policy(load_complete(ck/f'{method}_s3.pkl')['params'])
        elif method == 'oracle':
            policy = network_policy(None, teachers)
        elif method == 'cpg':
            policy = cpg_policy(np.array(cpg['theta']), cpg['reset_phase'])
        else:
            assert method == 'zero'
            policy = lambda obs, tid, t, age: np.zeros(6)
        saved = example['episode']
        fresh = episode(policy, saved['seed'], saved['order'], z['eval_segment'], deadline)
        errors = {key: float(np.max(np.abs(np.array(saved['trace'][key])-np.array(fresh['trace'][key]))))
                  for key in ('v', 'body', 'reward')}
        assert max(errors.values()) < 1e-6, (method, errors)
        out[method] = errors
    write_json(res/'demo_reload.json', out)
    print('R5_DEMO_RELOAD_OK all checkpoint trajectories agree within 1e-6', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['train', 'cpg', 'evaluate', 'report', 'replay-demo'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--profile', choices=PROFILES, default='battery')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-minutes', type=float, required=True)
    ap.add_argument('--pause-after-checkpoint', action='store_true', help='smoke SIGKILL probe only')
    args = ap.parse_args()
    z = PROFILES[args.profile]
    ck, res = locations(args.tag)
    manifest(args, z, ck, res)
    deadline = Deadline(args.max_minutes)
    try:
        if args.command == 'report':
            from r5_report import report
            report(args.tag, res, deadline)
        elif args.command == 'cpg':
            with make_env(z['cpg_horizon']) as env:
                run_cpg(args, z, ck, res, deadline, env)
        elif args.command == 'replay-demo':
            replay_demo(args, z, ck, res, deadline)
        else:
            {'train': run_train, 'evaluate': run_eval}[args.command](args, z, ck, res, deadline)
    except TimeoutError as exc:
        print(f'R5_PARTIAL {exc}; repeat the exact command to resume', flush=True)
        return 75
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
