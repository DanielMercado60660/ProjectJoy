"""Synthetic-only artifact/routing verification. Never imports an environment."""
import json
import hashlib
from pathlib import Path
import numpy as np
from build import HERE, FINAL_RECIPES, forward, inputs

SHAPES={'a1.weight':(128,19),'a1.bias':(128,),
        'a2.weight':(128,128),'a2.bias':(128,),
        'a3.weight':(6,128),'a3.bias':(6,),'log_std':(6,)}

def main():
    ledger=[json.loads(line) for line in (HERE/'ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==10, 'Expected exactly ten authorized evaluation records'
    report={'evaluation_calls':len(ledger),'environment_steps':sum(e['env_steps'] for e in ledger),'seeds':[]}
    for seed in range(4):
        path=HERE/f'merged_s{seed}.npz'
        with np.load(path) as z:
            assert set(z.files)==set(SHAPES)
            p={k:z[k] for k in z.files}
        for key,shape in SHAPES.items():
            assert p[key].shape==shape and p[key].dtype==np.float32
            assert np.isfinite(p[key]).all()
        best=max((e for e in ledger if e['seed']==seed),key=lambda e:e['mean_retained'])
        cand=Path(best['cand'])
        assert hashlib.sha256(cand.read_bytes()).hexdigest()[:16]==best['cand_sha256']
        with np.load(cand) as z:
            for key in p:
                assert np.array_equal(p[key],z[key]), (seed,key,'rebuild differs from evaluated incumbent')
        metadata=json.loads(path.with_suffix('.json').read_text())
        inactive_error=0.
        for task in ['fwd','bwd']:
            # Includes box corners, far beyond invented calibration distribution.
            rng=np.random.default_rng(97531+seed)
            x=np.concatenate([inputs(seed,task,n=1024,bank=2),rng.choice([-10.,10.],size=(1024,17))]).astype(np.float32)
            h1=np.tanh(x@p['a1.weight'][:,:17].T+p['a1.bias']+p['a1.weight'][:,17+(task=='bwd')])
            h2=np.tanh(h1@p['a2.weight'].T+p['a2.bias'])
            other='bwd' if task=='fwd' else 'fwd'
            lo,hi=metadata['branches'][other]['second_slice']
            err=float(np.max(np.abs(h2[:,lo:hi]-1)))
            inactive_error=max(inactive_error,err)
            assert err==0.,(seed,task,'inactive branch did not saturate')
            a=forward(p,x,task)
            assert a.shape==(len(x),6) and np.isfinite(a).all() and np.max(np.abs(a))<=1
        report['seeds'].append({'seed':seed,'recipe':FINAL_RECIPES[seed],
            'best_call':best['call'],'mean_retained':best['mean_retained'],
            'fwd_return':best['fwd']['mean_return'],'bwd_return':best['bwd']['mean_return'],
            'array_identical_to_evaluated_candidate':True,'inactive_branch_max_error':inactive_error,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report['mean_retained']=float(np.mean([r['mean_retained'] for r in report['seeds']]))
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
