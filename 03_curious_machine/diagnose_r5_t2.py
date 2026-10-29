"""NumPy receipt audit and oracle gate; --mlx-probe is Claude-only, never Astra.
Reports diagnostic witnesses separately from all held-out headline results.
"""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
from env import VecRooms
from r5_core import TASKS, ReplayGraph
from planner_r5_t2 import Knowledge, CompositionalPlanner
from evaluate_r5 import task_success


def oracle_check():
    rows = []
    for world in ('plain', 'tv'):
        k = Knowledge.oracle(VecRooms(1, noisy_tv=world == 'tv'))
        for task in TASKS:
            plan, expansions = CompositionalPlanner(k).plan(task)
            env = VecRooms(1, max_steps=129, noisy_tv=world == 'tv', seed=64600)
            if plan:
                for action in plan:
                    env.step([action])
            rows.append(dict(world=world, task=list(task), length=len(plan) if plan is not None else None,
                             expansions=expansions, success=bool(task_success(env, [task])[0])))
    assert all(r['success'] and r['length'] <= 128 for r in rows), rows
    return rows


def audit():
    root = Path('results/r5/r5_ac'); out = {'oracle': oracle_check(), 'frontier_witnesses': [], 'lp_logs': [], 'training_seconds': {}}
    for world in ('plain', 'tv'):
        data = np.load(root / f'frontier_{world}_s510.npz')
        old = ReplayGraph(data['observations'], data['actions'])
        knowledge = Knowledge.from_replay(data['observations'], data['actions'])
        for task in TASKS:
            record = dict(world=world, task=list(task), knowledge=knowledge.stats())
            for label, planner, budget in (('old128', old, 128), ('old512', old, 512),
                                            ('new128', CompositionalPlanner(knowledge), 128)):
                plan, expanded = planner.plan(task, budget, 100000 if label == 'new128' else 4096)
                record[label] = dict(length=None if plan is None else len(plan), expansions=expanded)
                if label == 'new128':
                    env = VecRooms(1, max_steps=129, noisy_tv=world == 'tv', seed=64600)
                    if plan:
                        for action in plan: env.step([action])
                    record['new_success'] = bool(task_success(env, [task])[0])
                    assert record['new_success']
            out['frontier_witnesses'].append(record)
        for seed in (510, 511, 512):
            path = Path(f'logs/r5/r5_ac/lp_{world}_s{seed}/updates.jsonl')
            logs = [json.loads(x) for x in path.read_text().splitlines()]
            ck = pickle.loads(Path(f'checkpoints/r5_r5_ac/lp_{world}_s{seed}.pkl').read_bytes())
            scale = float(np.sqrt(ck['ret_rms']['var'] + 1e-8))
            fields = ('reward_mean','reward_std','progress_positive','progress_negative','progress_old','progress_new')
            early = {k: float(np.mean([r[k] for r in logs[1:65]])) for k in fields}
            late = {k: float(np.mean([r[k] for r in logs[-64:]])) for k in fields}
            out['lp_logs'].append(dict(world=world,seed=seed,source=str(path),early64=early,late64=late,
                                       final_reward_scale=scale,late_raw_std_over_final_scale=late['reward_std']/scale,
                                       entropy='not logged; --mlx-probe required'))
    for seed in (510,511,512):
        paths = list(Path('logs/r5/r5_ac').glob(f'*_s{seed}/outcome.json'))
        out['training_seconds'][seed] = {p.parent.name: json.loads(p.read_text())['invocation_seconds'] for p in paths}
    a, b = (json.loads((root/p).read_text()) for p in ('lp_tv_s510.json','lp_tv_s510_reload.json'))
    out['reload_exact'] = {k:a[k] == b[k] for k in ('episodes','tasks','transfer','empty_replay')}
    # Recompute every table cell from individual per-seed files (never trust printed rounding).
    summary = {}
    for world in ('plain','tv'):
        for method in ('lp','rnd_ep','disagree','ngu','frontier','random'):
            paths = [root/f'{method}_{world}_s{s}.json' for s in (510,511,512)]
            rows = [json.loads(p.read_text()) for p in paths]
            fields = [('exploration','rooms'),('exploration','interactions'),('exploration','tv_steps'),
                      ('transfer','success_rate'),('transfer','plan_coverage'),('empty_replay','success_rate')]
            summary[f'{world}/{method}'] = {'sources':[str(p) for p in paths], **{
                f'{a}/{b}':dict(mean=float(np.mean([r[a][b] for r in rows])),std=float(np.std([r[a][b] for r in rows]))) for a,b in fields}}
    out['recomputed_summary'] = summary
    return out


def mlx_probe(deadline):
    import mlx.core as mx
    from train_r5 import R5Explorer
    rows = []
    for world in ('plain','tv'):
        for seed in (510,511,512):
            if time.monotonic() >= deadline: raise TimeoutError('MLX probe guard')
            ck = f'checkpoints/r5_r5_ac/lp_{world}_s{seed}.pkl'
            ex, _ = R5Explorer.load(ck)
            data = np.load(f'results/r5/r5_ac/lp_{world}_s{seed}.npz')
            x, xn = mx.array(data['observations'][:-1]), mx.array(data['observations'][1:])
            a = mx.array(data['actions'].astype(np.int32))
            logits, _, _ = ex.policy(x)
            lp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
            entropy = -mx.sum(mx.exp(lp) * lp, axis=1)
            old, new = ex.aux.errors(x,a,xn,old=True), ex.aux.errors(x,a,xn)
            mx.eval(entropy, old, new)
            from env import VIEW,CELL_FEAT,TV
            tv=(data['observations'][1:,:-5].reshape(-1,VIEW*VIEW,CELL_FEAT)[:,:,TV]>.5).any(1)
            delta=np.array(old-new)
            rows.append(dict(world=world,seed=seed,checkpoint=ck,entropy_mean=float(mx.mean(entropy)),
                             snapshot_progress_mean=float(mx.mean(old-new)), snapshot_progress_std=float(mx.std(old-new)),
                             progress_tv=float(delta[tv].mean()) if tv.any() else None,
                             progress_non_tv=float(delta[~tv].mean()) if (~tv).any() else None))
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mlx-probe',action='store_true')
    p.add_argument('--max-minutes',type=float,default=1);a=p.parse_args()
    if not 0 < a.max_minutes <= 2:p.error('probe max-minutes in (0,2]')
    dest=Path(a.out)
    if dest.exists():raise FileExistsError(dest)
    start=time.monotonic();result=mlx_probe(start+a.max_minutes*60) if a.mlx_probe else audit()
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(kind='MLX entropy probe' if a.mlx_probe else 'NumPy diagnostics, not headline',
                                    elapsed_seconds=time.monotonic()-start,results=result),indent=2,allow_nan=False))
    print('DIAGNOSTICS_OK',dest)
