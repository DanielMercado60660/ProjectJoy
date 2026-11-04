"""Try expressing the wide-ray correction in the readout, restoring W1 rows.

Fit only specified functions on deterministic hand-made observation grids.
No evaluator use from this script.
"""
import json
import numpy as np
from inspect_weights import load, forward
from symmetric_readout import probe_grid

if __name__=='__main__':
    teacher=load('c11_drag_feedforward.npz')
    d=load('c11_drag_feedforward.npz')
    start=load()
    units=np.array([j for j in range(32) if j!=28])
    cols=np.array([j for j in range(32) if j not in (23,29)])
    for row in (2,7): d['W1'][row,cols]=start['W1'][row,cols]
    train=probe_grid([.6,1.,1.4],[0.,.25,.5,.75,1.,1.2,1.3],[-.8,0.,.8])
    valid=probe_grid([.8,1.2],[.15,.4,.65,.9,1.1,1.25],[-.5,.5])
    h=forward(d,train)[2]
    H=np.c_[h[:,units],np.ones(len(h))]
    old=np.r_[d['W3'][units,0],d['b3'][0]]
    y=forward(teacher,train)[3][:,0]
    reg=1e-5*len(H)*np.eye(H.shape[1]);reg[-1,-1]=1e-8
    delta=np.linalg.solve(H.T@H+reg,H.T@(y-H@old))
    d['W3'][units,0]=(old+delta)[:-1]
    d['b3'][0]=(old+delta)[-1]
    # Recompile the same explicit drag surrogate into the restored features.
    co=np.linalg.solve(H.T@H+reg,H.T@(.8*train[:,10]-.25))
    d['W3'][units,1]=co[:-1]
    d['b3'][1]=co[-1]
    metrics={}
    for name,x in [('fit_handmade',train),('validation_handmade',valid)]:
        new=forward(d,x)[3];desired=forward(teacher,x)[3]
        err=new-desired
        metrics[name]={'n':len(x),'logit_rmse':np.sqrt(np.mean(err**2,axis=0)).tolist(),
                       'logit_max_error':np.max(abs(err),axis=0).tolist()}
    print(json.dumps(metrics,indent=2))
    print('Steer max readout change',float(np.max(abs(delta))))
    print('Changed scalars',sum(np.count_nonzero(d[k]!=start[k]) for k in d))
    np.savez('c12_compact_readout.npz',**d)
    with open('compact_probe_metrics.json','w') as f:json.dump(metrics,f,indent=2)
