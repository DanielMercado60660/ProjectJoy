"""Deterministic closed-form weight-space merge; no environment imports or training."""
import os
# Match the bounded-thread construction used for every evaluated candidate.
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import qr

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'
KEYS = ['a1.weight','a1.bias','a2.weight','a2.bias','a3.weight','a3.bias','log_std']

def load(seed, task):
    with np.load(DATA / f's{seed}' / f'{task}.npz') as z:
        return {k: z[k].astype(np.float64) for k in z.files}

def inputs(seed, task, n=24000, bank=0, distribution='motion'):
    """Invented independent observations, never simulator states."""
    z = load(seed, 'theta0')
    rng = np.random.default_rng(30100 + 101*seed + 13*(task=='bwd') + 7919*bank)
    x = rng.normal(size=(n,17))
    scales = rng.choice([0.6,1.,1.5], size=(n,1), p=[0.2,0.5,0.3])
    x *= scales
    if distribution == 'motion':
        moving = rng.random(n) > 0.2
        speed = rng.normal(5.0 if task=='fwd' else -7.0, 2.5, n)
        speed[~moving] = rng.normal(0,1.,np.sum(~moving))
        x[:,8] = (speed-z['norm_mean'][8])/np.sqrt(z['norm_var'][8]+1e-8)
        upright = rng.random(n) > 0.25
        x[upright,0] = (rng.normal(0.,.12,np.sum(upright))-z['norm_mean'][0])/np.sqrt(z['norm_var'][0]+1e-8)
        x[upright,1] = (rng.normal(0.,.45,np.sum(upright))-z['norm_mean'][1])/np.sqrt(z['norm_var'][1]+1e-8)
    return np.clip(x,-10,10)

def teacher(p,x,task):
    h1=np.tanh(x@p['a1.weight'][:,:17].T+p['a1.bias']+p['a1.weight'][:,17+(task=='bwd')])
    z2=h1@p['a2.weight'].T+p['a2.bias']
    h2=np.tanh(z2)
    y=h2@p['a3.weight'].T+p['a3.bias']
    return h1,z2,h2,y

def fit(x,y,weighted=False):
    """Single affine least-squares solution, small fixed ridge for conditioning."""
    a=np.column_stack([x,np.ones(len(x))])
    if weighted:
        weights=(1-np.tanh(y)**2)**2+0.02
        coefs=[]
        for j in range(y.shape[1]):
            w=weights[:,j]
            gram=a.T@(a*w[:,None])
            ridge=1e-6*w.sum()
            gram.flat[::len(gram)+1]+=ridge
            gram[-1,-1]-=ridge
            coefs.append(np.linalg.solve(gram,a.T@(w*y[:,j])))
        coef=np.stack(coefs,axis=1)
        return coef[:-1].T,coef[-1]
    gram=a.T@a
    ridge=1e-6*len(a)
    gram.flat[::len(gram)+1]+=ridge
    gram[-1,-1]-=ridge
    coef=np.linalg.solve(gram,a.T@y)
    return coef[:-1].T,coef[-1]

def select(h,importance,count):
    # Column-pivoted QR is a deterministic rank-revealing pruning heuristic.
    # sqrt importance balances outgoing relevance against activation diversity.
    centered=h-h.mean(0)
    scaled=centered*np.sqrt(np.maximum(importance,1e-8))[None,:]
    _,_,piv=qr(scaled[:8000],mode='economic',pivoting=True,check_finite=False)
    return np.sort(piv[:count])

def forward(p,x,task):
    h=np.tanh(x@p['a1.weight'][:,:17].T+p['a1.bias']+p['a1.weight'][:,17+(task=='bwd')])
    h=np.tanh(h@p['a2.weight'].T+p['a2.bias'])
    return np.tanh(h@p['a3.weight'].T+p['a3.bias'])

def build(seed, mode='shared', distribution='motion', nf=63, alpha=0.5, weighted=False, center=False):
    ps={t:load(seed,t) for t in ['fwd','bwd']}
    xs={t:inputs(seed,t,distribution=distribution) for t in ps}
    teachers={t:teacher(ps[t],xs[t],t) for t in ps}
    p={k:np.zeros_like(ps['fwd'][k]) for k in KEYS}
    p['log_std']=ps['fwd']['log_std'].copy()
    # First-layer unit 127 is an exact +/-1 task bit in float32.
    p['a1.weight'][127,17:]=[20.,-20.]
    metadata={'seed':seed,'mode':mode,'distribution':distribution,'nf':nf,'alpha':alpha,'weighted':weighted,'center':center,'branches':{}}
    if mode=='shared':
        # Preserve ancestor row alignment. The one dropped row is least influential.
        influence=sum(np.linalg.norm(ps[t]['a2.weight'],axis=0) for t in ps)
        rows=np.sort(np.argsort(influence)[-127:])
        p['a1.weight'][:127,:17]=alpha*ps['fwd']['a1.weight'][rows,:17]+(1-alpha)*ps['bwd']['a1.weight'][rows,:17]
        for task in ps:
            col=17+(task=='bwd')
            p['a1.weight'][:127,col]=ps[task]['a1.bias'][rows]+ps[task]['a1.weight'][rows,col]
            if center:
                p['a1.weight'][:127,col]+=(ps[task]['a1.weight'][rows,:17]-p['a1.weight'][:127,:17])@xs[task].mean(0)
        metadata['shared_first_rows']=rows.tolist()
    else:
        first_start=0
        for task,count in [('fwd',nf),('bwd',127-nf)]:
            pt=ps[task]
            h1=teachers[task][0]
            importance=np.linalg.norm(pt['a2.weight'],axis=0)
            rows=select(h1,importance,count)
            sl=slice(first_start,first_start+count)
            p['a1.weight'][sl,:17]=pt['a1.weight'][rows,:17]
            col=17+(task=='bwd')
            p['a1.weight'][sl,col]=pt['a1.bias'][rows]+pt['a1.weight'][rows,col]
            # Saturate every inactive first-layer unit for ALL clipped observations.
            p['a1.weight'][sl,35-col]=12+10*np.abs(pt['a1.weight'][rows,:17]).sum(1)
            metadata['branches'][task]={'first_rows':rows.tolist(),'first_slice':[first_start,first_start+count]}
            first_start+=count
    out_fits={}
    second_start=0
    for task,count in [('fwd',nf),('bwd',127-nf)]:
        pt=ps[task]; x=xs[task]; h1,z2,h2,y=teachers[task]
        col=17+(task=='bwd')
        merged_h=np.tanh(x@p['a1.weight'][:,:17].T+p['a1.weight'][:,col]+p['a1.bias'])
        if mode=='shared':
            source=merged_h[:,:127]
            first_slice=slice(0,127)
        else:
            lo,hi=metadata['branches'][task]['first_slice']
            first_slice=slice(lo,hi); source=merged_h[:,first_slice]
        # Reconstruct all teacher second-layer preactivations from retained first units.
        w2,b2=fit(source,z2,weighted)
        approximated_h2=np.tanh(source@w2.T+b2)
        rows=select(approximated_h2,np.linalg.norm(pt['a3.weight'],axis=0),count)
        w2=w2[rows]; b2=b2[rows]
        compressed_h2=np.tanh(source@w2.T+b2)
        w3,b3=fit(compressed_h2,y,weighted)
        sl=slice(second_start,second_start+count)
        p['a2.weight'][sl,first_slice]=w2
        sign=1 if task=='fwd' else -1
        if mode=='shared':
            # Inactive preactivation >=12 for any shared hidden vector in [-1,1].
            gate=12+np.abs(w2).sum(1)+np.abs(b2)
            p['a2.bias'][sl]=b2+gate/2
            p['a2.weight'][sl,127]=-sign*gate/2
        else:
            # Inactive h1=1; inactive h2=0 by exact affine cancellation.
            inactive_bias=-w2.sum(1)
            p['a2.bias'][sl]=(b2+inactive_bias)/2
            p['a2.weight'][sl,127]=sign*(b2-inactive_bias)/2
        p['a3.weight'][:,sl]=w3
        out_fits[task]=(w3,b3)
        info=metadata['branches'].setdefault(task,{})
        info.update({'second_rows':rows.tolist(),'second_slice':[second_start,second_start+count],
                     'train_action_rmse':float(np.sqrt(np.mean((np.tanh(compressed_h2@w3.T+b3)-np.tanh(y))**2)))})
        second_start+=count
    # Second-layer unit 127 carries the task bit to the output bias.
    p['a2.weight'][127,127]=20
    wf,bf=out_fits['fwd']; wb,bb=out_fits['bwd']
    if mode=='shared': bf=bf-wb.sum(1); bb=bb-wf.sum(1)
    p['a3.bias']=(bf+bb)/2
    p['a3.weight'][:,127]=(bf-bb)/2
    p={k:v.astype(np.float32) for k,v in p.items()}
    for task in ps:
        xv=inputs(seed,task,n=12000,bank=1,distribution=distribution)
        target=np.tanh(teacher(ps[task],xv,task)[-1])
        pred=forward(p,xv,task)
        metadata['branches'][task]['validation_action_rmse']=float(np.sqrt(np.mean((pred-target)**2)))
    return p,metadata

# These fixed recipes are chosen from the ten authorized training-reset evaluations.
# Rebuilding uses no ledger, candidate file, simulator, or external weights.
RECIPES = {
    'shared': dict(mode='shared', distribution='motion', nf=63, alpha=0.5, weighted=False, center=False),
    'forward_capacity': dict(mode='shared', distribution='motion', nf=80, alpha=0.75, weighted=False, center=False),
    'balanced': dict(mode='shared', distribution='motion', nf=72, alpha=0.625, weighted=True, center=True),
    'split': dict(mode='split', distribution='motion', nf=63, alpha=0.5, weighted=False, center=False),
    'derivative': dict(mode='shared', distribution='motion', nf=63, alpha=0.5, weighted=True, center=False),
    'centered': dict(mode='shared', distribution='motion', nf=63, alpha=0.5, weighted=True, center=True),
}
FINAL_RECIPES = {0: 'balanced', 1: 'shared', 2: 'balanced', 3: 'balanced'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,nargs='+',choices=range(4),default=list(range(4)))
    ap.add_argument('--recipe',choices=['final',*RECIPES],default='final')
    a=ap.parse_args()
    for seed in a.seed:
        recipe=FINAL_RECIPES[seed] if a.recipe=='final' else a.recipe
        p,m=build(seed,**RECIPES[recipe])
        prefix='merged' if a.recipe=='final' else recipe
        dest=HERE/f'{prefix}_s{seed}.npz'
        np.savez(dest,**p)
        dest.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n')
        print(json.dumps({'path':str(dest),'recipe':recipe,'synthetic_action_rmse':{
            task:info['validation_action_rmse'] for task,info in m['branches'].items()
        }}),flush=True)

if __name__=='__main__': main()
