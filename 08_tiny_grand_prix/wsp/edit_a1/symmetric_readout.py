"""Closed-form readout edit against a specified reflection-symmetric function.

All fitting/validation observations are hand-specified sensor vectors and grids.
No simulator imports, steps, gradient descent, or simulator score fitting.
"""
import json
import numpy as np
from inspect_weights import load, forward


def probe_grid(scales, speeds, prev_steers):
    shapes = np.array([
        [.10,.116,.174,.324,.957,.957,.324,.174,.116,.10],
        [.10,.12,.17,.25,.34,.48,.62,.30,.13,.10],
        [.13,.15,.18,.22,.27,.34,.45,.50,.16,.09],
        [.15,.20,.30,.50,.80,.30,.16,.09,.06,.05],
        [.08,.11,.30,.40,.28,.23,.19,.16,.12,.10],
        [.16,.25,.48,.20,.12,.09,.07,.05,.04,.04],
        [.10,.15,.20,.35,.65,.90,.50,.20,.13,.10],
    ])
    rows=[]
    for shape in shapes:
        for sr in scales:
            for sl in scales:
                rays=np.minimum(shape*np.array([sr]*5+[sl]*5),1.)
                for speed in speeds:
                    for steer in prev_steers:
                        row=np.r_[rays,speed,steer,1.]
                        rows.append(row)
                        mirror=row.copy();mirror[:10]=row[9::-1];mirror[11]*=-1
                        rows.append(mirror)
    return np.array(rows)


def target(d,x):
    mirror=x.copy();mirror[:,:10]=x[:,9::-1];mirror[:,11]*=-1
    _,_,h,z=forward(d,x)
    _,_,_,zm=forward(d,mirror)
    desired=.5*(z[:,0]-zm[:,0])
    return h,z[:,0],desired


if __name__=='__main__':
    d=load('c08_medium_visibility.npz')
    train=probe_grid([.6,1.,1.4],[0.,.25,.5,.75,1.,1.25],[-.8,0.,.8])
    valid=probe_grid([.8,1.2],[.15,.4,.65,.9,1.1],[-.5,.5])
    h,z,y=target(d,train)
    # Keep the programmed pedal unit disconnected from steering.
    units=np.array([j for j in range(32) if j!=28])
    H=np.c_[h[:,units],np.ones(len(h))]
    regularizer=.001*len(H)*np.eye(H.shape[1])
    regularizer[-1,-1]=1e-8
    delta=np.linalg.solve(H.T@H+regularizer,H.T@(y-z))
    d['W3'][units,0]+=delta[:-1].astype(np.float32)
    d['b3'][0]+=delta[-1]
    metrics={}
    for name,x in [('fit_handmade',train),('validation_handmade',valid)]:
        _,z0,desired=target(load('c08_medium_visibility.npz'),x)
        new=forward(d,x)[3][:,0]
        metrics[name]={
            'n':len(x),
            'old_logit_rmse':float(np.sqrt(np.mean((z0-desired)**2))),
            'new_logit_rmse':float(np.sqrt(np.mean((new-desired)**2))),
            'new_logit_max_error':float(np.max(abs(new-desired))),
        }
    print(json.dumps(metrics,indent=2))
    print('Largest readout change',float(np.max(abs(delta))))
    np.savez('c09_symmetric_readout.npz',**d)
    with open('symmetric_probe_metrics.json','w') as f:
        json.dump(metrics,f,indent=2)
