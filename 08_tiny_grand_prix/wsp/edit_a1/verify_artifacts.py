"""Check saved weights and evaluator receipts without driving or changing them."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from inspect_weights import load, forward, handmade


if __name__=='__main__':
    records=[json.loads(line) for line in Path('eval_log.jsonl').read_text().splitlines()]
    assert len(records)==12
    assert [r['call'] for r in records]==list(range(1,13))
    assert len(records)<=25
    shape={'W1':(13,32),'b1':(32,),'W2':(32,32),'b2':(32,),
           'W3':(32,2),'b3':(2,)}
    start=load()
    checked=[]
    for r in records:
        path=Path(r['weights'])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest[:12]==r['sha'], path
        d=load(path)
        with np.load(path,allow_pickle=False) as z:
            for k,s in shape.items():
                assert z[k].shape==s and z[k].dtype==np.float32 and np.isfinite(z[k]).all()
        result=json.loads(Path(f"call{r['call']:02d}.json").read_text())
        assert result['call']==r['call'] and result['score_key']==r['key']
        checked.append({'call':r['call'],'file':str(path),'sha256':digest})
    best=json.loads(Path('best.json').read_text())
    maximum=max(records,key=lambda r:r['key'])
    assert best['call']==maximum['call']==12
    assert best['key']==maximum['key']
    d=load('best.npz');candidate=load(maximum['weights'])
    reproduced=load('reproduced_best.npz')
    for k in d:
        assert np.array_equal(d[k],candidate[k])
        assert np.array_equal(d[k],reproduced[k])
    changed={k:int(np.count_nonzero(d[k]!=start[k])) for k in d}
    assert sum(changed.values())==280
    assert sum(v.size for v in d.values())==1570
    a=forward(d,handmade())[0]
    assert a.shape==(441,2) and np.isfinite(a).all() and np.max(abs(a))<=1
    receipt=json.loads(Path('call12.json').read_text())
    for name in ('strict_g1.0','soft_g1.0'):
        assert all(t['finished'] for t in receipt['results'][name]['per_track'])
    assert all(t['wall_contact_steps']==0 for t in receipt['results']['soft_g1.0']['per_track'])
    report={'verified_at':datetime.now().astimezone().isoformat(),
            'calls_used':len(records),'budget':25,'best_call':12,
            'best_candidate':maximum['weights'],'summaries':maximum['summaries'],
            'changed_by_array':changed,'changed':sum(changed.values()),
            'total':1570,'fraction_changed':sum(changed.values())/1570,
            'reproduction_arrays_bitwise_equal':True,'handmade_probe_count':len(a),
            'evaluated_candidates_verified':checked}
    Path('verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evaluated_candidates_verified'},indent=2))
