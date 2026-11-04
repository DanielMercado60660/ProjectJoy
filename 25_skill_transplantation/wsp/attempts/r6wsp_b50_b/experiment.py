"""Bookkeeping and only the authorized evaluator as a subprocess."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from build import HERE, save

def run(seed, label, config, reason):
    ledger = HERE / 'ledger.jsonl'
    used = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    if used >= 50:
        raise RuntimeError('50-call budget exhausted')
    name = f'c{used+1:02d}_s{seed}_{label}.npz'
    save(seed, name, **config)
    # Write the intended rationale before evaluating; record failure as well.
    with (HERE/'NOTES.md').open('a') as f:
        f.write(f'### Call {used+1}: seed {seed}, {label}\n\n{reason}\n\n')
        f.write(f'Configuration: `{json.dumps(config, sort_keys=True)}`.\n\n')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4')
    result = subprocess.run([sys.executable, str(HERE.parent.parent/'tools/eval_budget.py'),
                             '--attempt', HERE.name, '--seed', str(seed), '--cand', str(HERE/name)],
                            env=env, text=True, capture_output=True)
    (HERE/f'c{used+1:02d}_eval.txt').write_text(result.stdout+result.stderr)
    if result.returncode:
        with (HERE/'NOTES.md').open('a') as f:
            f.write(f'Evaluator failed with return code {result.returncode}; see c{used+1:02d}_eval.txt. No retention.\n\n')
        print(result.stdout, result.stderr)
        raise RuntimeError('evaluation failed')
    data = json.JSONDecoder().raw_decode(result.stdout[result.stdout.index('{'):])[0]
    bestfile = HERE/'best.json'
    best = json.loads(bestfile.read_text()) if bestfile.exists() else {}
    previous = best.get(str(seed))
    keep = previous is None or data['mean_retained'] > previous['score']
    decision = 'KEEP' if keep else 'REJECT'
    if keep:
        best[str(seed)] = {'score': data['mean_retained'], 'config': config, 'candidate': name, 'call': data['call']}
        bestfile.write_text(json.dumps(best, indent=2)+'\n')
        shutil.copyfile(HERE/name, HERE/f'merged_s{seed}.npz')
    with (HERE/'NOTES.md').open('a') as f:
        f.write(f"Returns fwd/bwd: {data['fwd']['mean_return']}, {data['bwd']['mean_return']}; retained: "
                f"{data['fwd']['retained_vs_specialist']}, {data['bwd']['retained_vs_specialist']}; "
                f"mean retained **{data['mean_retained']}**. **{decision}**")
        f.write(' (initial baseline).\n\n' if previous is None else f"; prior best {previous['score']} from call {previous['call']}.\n\n")
    print(json.dumps({'call':data['call'], 'seed':seed, 'label':label, 'fwd':data['fwd']['mean_return'],
                     'bwd':data['bwd']['mean_return'], 'mean':data['mean_retained'], 'decision':decision}), flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2,3')
    ap.add_argument('--label', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--reason', required=True)
    a=ap.parse_args()
    for s in map(int, a.seeds.split(',')):
        run(s, a.label, json.loads(a.config), a.reason)
