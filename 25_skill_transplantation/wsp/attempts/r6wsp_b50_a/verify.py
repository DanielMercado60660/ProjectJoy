"""Synthetic-only artifact and deterministic-rebuild verification; no env calls."""
import hashlib
import json
from build import HERE, KEYS, FINAL_CONFIGS, construct, source, np

SHAPES = {'a1.weight':(128,19), 'a1.bias':(128,), 'a2.weight':(128,128),
          'a2.bias':(128,), 'a3.weight':(6,128), 'a3.bias':(6,), 'log_std':(6,)}

def forward(p,x):
    h1=np.tanh(x@p['a1.weight'].T+p['a1.bias'])
    h2=np.tanh(h1@p['a2.weight'].T+p['a2.bias'])
    return np.tanh(h2@p['a3.weight'].T+p['a3.bias']),h2


def main():
    history = json.loads((HERE/'history.json').read_text())
    ledger = [json.loads(line) for line in (HERE/'ledger.jsonl').read_text().splitlines()]
    assert len(history)==len(ledger)==50
    assert [r['call'] for r in ledger]==list(range(1,51))
    selected=json.loads((HERE/'selected.json').read_text())
    report={'evaluation_calls':50,'environment_steps':300000,'seeds':{}}
    for seed in range(4):
        path=HERE/f'merged_s{seed}.npz'
        p=dict(np.load(path))
        rebuilt,metrics=construct(seed,FINAL_CONFIGS[seed])
        assert set(p)==set(KEYS)
        for k in KEYS:
            assert p[k].dtype==np.float32 and p[k].shape==SHAPES[k]
            assert np.isfinite(p[k]).all()
            assert np.array_equal(p[k],rebuilt[k]),(seed,k,'rebuild changed an array')
        receipt=ledger[selected[str(seed)]['call']-1]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest[:16]==receipt['cand_sha256'],(seed,'artifact bytes differ from evaluated candidate')
        trials=[r for r in ledger if r['seed']==seed]
        assert receipt['mean_retained']==max(r['mean_retained'] for r in trials)
        task_report={}
        # New synthetic samples, never simulator states, with broad coverage.
        for task,name in enumerate(('fwd','bwd')):
            rng=np.random.default_rng(918273+seed*2+task)
            x=np.zeros((32768,19),np.float32)
            x[:,:17]=np.clip(rng.normal(size=(len(x),17))*1.5,-10,10)
            x[:,8]=rng.uniform(-.5,6,len(x))*(1 if task==0 else -1)
            ref=source(seed,name)
            x[:,1]=x[:,1]*.25-ref['norm_mean'][1]/np.sqrt(ref['norm_var'][1])
            x[:,0]*=.6
            x[:,17+task]=1
            a,h2=forward(p,x)
            teacher,_=forward(ref,x)
            assert np.isfinite(a).all() and np.max(np.abs(a))<=1
            inactive=h2[:,64:] if task==0 else h2[:,:64]
            assert np.array_equal(inactive,np.ones_like(inactive))
            # A worst-case box bound includes every possible h1 in [-1,1].
            rows=slice(64,128) if task==0 else slice(0,64)
            g=np.tanh(np.float32(3))*(1 if task==0 else -1)
            lower=(p['a2.bias'][rows].astype(float)+p['a2.weight'][rows,127].astype(float)*g
                   -np.abs(p['a2.weight'][rows,:127].astype(float)).sum(axis=1))
            assert lower.min()>11.99
            task_report[name]={'fresh_synthetic_action_mse':float(np.mean((a-teacher)**2)),
                               'inactive_units_exactly_one':True,'off_preactivation_lower_bound':float(lower.min())}
        report['seeds'][str(seed)]={'sha256':digest,'selected_call':receipt['call'],
            'training_mean_retained':receipt['mean_retained'],'training_fwd_return':receipt['fwd']['mean_return'],
            'training_bwd_return':receipt['bwd']['mean_return'],'array_rebuild_bitwise_equal':True,
            'evaluated_artifact_hash_matches':True,'synthetic_checks':task_report}
    report['training_mean_retained']=float(np.mean([s['training_mean_retained'] for s in report['seeds'].values()]))
    report['held_out_reset_score']='not measured; no held-out environment access'
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
