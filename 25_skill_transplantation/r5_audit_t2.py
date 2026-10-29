"""Read-only turn-1 audit and validation-only diagnostics. No MLX import."""
from pathlib import Path
import json
import time
import numpy as np
from r5_core import (TASKS, bank, episode, network_policy, read_pickle, write_json, Deadline)


def main():
    deadline = Deadline(1.5)
    out = dict(seeds={}, source='completed main_s0..2; old test metrics are retrospective only',
               validation='existing policies on validation reset seeds; no new test access')
    for seed in range(3):
        res, ck = Path(f'results/r5/main_s{seed}'), Path(f'checkpoints/r5_main_s{seed}')
        ev = json.loads((res/'eval.json').read_text())
        cpg = json.loads((res/'cpg.json').read_text())
        stats, timings, edges, curves = {}, {}, {}, {}
        for method in ('adapter','dagger','joint','oracle','cpg'):
            r = [r for r in ev['rows'] if r['split']=='test' and r['stage']==3 and r['method']==method]
            iso = [x for x in r if x['kind']=='isolated']
            sw = [x for x in r if x['kind']=='transition' and x['switch']]
            stats[method] = dict(tracking=float(np.mean([x['tracking_fraction'] for x in iso])),
                                 switch_failure=float(np.mean([x['failure'] for x in sw])))
        for path in sorted(ck.glob('*.pkl')):
            if path.name=='cpg_state.pkl':
                continue
            st = read_pickle(path)
            timings[path.name] = dict(st['cost'])
        names = ['frequency'] + [f'amplitude_{i}' for i in range(6)] + [f'phase_{i}' for i in range(6)] + [f'bias_{i}' for i in range(6)]
        for task, pick in cpg['picks'].items():
            edges[TASKS[int(task)]] = {names[i]:float(pick['theta'][i]) for i in np.flatnonzero(pick['near_box_edge'])}
        cstate = read_pickle(ck/'cpg_state.pkl')
        for key, job in cstate['jobs'].items():
            curves[key] = dict(last5_mean=float(np.mean(job['curve'][-5:])),
                               prior5_mean=float(np.mean(job['curve'][-10:-5])),
                               best_generation=int(np.argmax(job['curve']))+1)
        validation = {}
        for tid, name in enumerate(('base','teacher_bwd','teacher_slow')):
            p = read_pickle(ck/f'{name}.pkl')['params']
            rows = [episode(network_policy(p), int(s), (tid,), 600, deadline)['segments'][0]
                    for s in bank('val',seed,3)]
            validation[TASKS[tid]] = dict(tracking=float(np.mean([r['tracking_fraction'] for r in rows])),
                                         speed_mae=float(np.mean([r['speed_mae'] for r in rows])),
                                         fall_proxy=float(np.mean([r['fall_proxy'] for r in rows])))
        out['seeds'][str(seed)] = dict(final=stats, acquisition=[r for r in ev['retention'] if r['stage']==TASKS.index(r['task'])+1],
                                        timing=timings, unique_train_wall_s=sum(c['wall_s'] for c in timings.values()),
                                        validation_teachers=validation,cpg_edges=edges,cpg_curves=curves)
    write_json('results/r5/turn2_audit/turn1.json',out)
    for s, r in out['seeds'].items():
        print('seed',s,'train seconds',round(r['unique_train_wall_s'],3),'validation teachers',r['validation_teachers'],'CPG edges',r['cpg_edges'])
    print('R5_T2_AUDIT_OK validation only; no MLX imported')


if __name__=='__main__':
    main()
