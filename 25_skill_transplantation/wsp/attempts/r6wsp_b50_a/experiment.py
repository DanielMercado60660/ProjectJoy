"""Sequential, budget-limited evaluation with a human-specified reason for each edit."""
from pathlib import Path
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from build import HERE, construct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2,3')
    ap.add_argument('--name', required=True)
    ap.add_argument('--reason', required=True)
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    config = json.loads(a.config)
    sp = HERE/'selected.json'
    selected = json.loads(sp.read_text()) if sp.exists() else {}
    hp = HERE/'history.json'
    history = json.loads(hp.read_text()) if hp.exists() else []
    for seed in map(int,a.seeds.split(',')):
        if len(history) >= 50:
            raise RuntimeError('Local 50-call cap reached')
        p, metrics = construct(seed, config)
        candidate = HERE/f'{a.name}_s{seed}.npz'
        np.savez(candidate, **p)
        with (HERE/'NOTES.md').open('a') as out:
            out.write(f'\n### Call {len(history)+1}: seed {seed}, {a.name}\n\n{a.reason}\n\nConfig: `{json.dumps(config,sort_keys=True)}`.\n')
        cmd = [sys.executable, 'wsp/tools/eval_budget.py', '--attempt','r6wsp_b50_a','--seed',str(seed),'--cand',str(candidate)]
        result = subprocess.run(cmd, cwd=HERE.parents[2], text=True, capture_output=True)
        (HERE/f'{a.name}_s{seed}_eval.txt').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        data = json.JSONDecoder().raw_decode(result.stdout)[0]
        old = selected.get(str(seed))
        kept = old is None or data['mean_retained'] > old['score']
        entry = dict(name=a.name, config=config, reason=a.reason, metrics=metrics, evaluation=data, kept=kept)
        history.append(entry)
        hp.write_text(json.dumps(history,indent=2)+'\n')
        if kept:
            selected[str(seed)] = dict(config=config, score=data['mean_retained'], name=a.name, call=data['call'])
            shutil.copyfile(candidate,HERE/f'merged_s{seed}.npz')
            sp.write_text(json.dumps(selected,indent=2)+'\n')
        with (HERE/'NOTES.md').open('a') as out:
            out.write(f"\nTraining resets: fwd {data['fwd']['mean_return']}, bwd {data['bwd']['mean_return']}; mean retained {data['mean_retained']:.4f}. **{'KEPT' if kept else 'REJECTED'}**")
            out.write(f"; prior best {old['score']:.4f}.\n" if old else '; initial measured baseline.\n')
        with (HERE/'STATUS.md').open('a') as out:
            out.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} call {data['call']}/50 s{seed} {a.name}: {data['mean_retained']:.4f}; {'kept' if kept else 'rejected'}; next: reasoned construction comparison.\n")
        compact = {k:{m:v for m,v in val.items() if m not in ('h2_indices','intercept')} for k,val in metrics.items()}
        print(json.dumps(dict(call=data['call'], seed=seed, name=a.name, score=data['mean_retained'], fwd=data['fwd']['mean_return'], bwd=data['bwd']['mean_return'], kept=kept, synth=compact)),flush=True)

if __name__ == '__main__':
    main()
