"""Fit a specified drag feedforward term to existing neural features, CPU only."""
import json
import numpy as np
from inspect_weights import load, forward
from symmetric_readout import probe_grid

if __name__=='__main__':
    d=load('c10_broad_bend_speed.npz')
    train=probe_grid([.6,1.,1.4],[0.,.25,.5,.75,1.,1.2,1.3],[-.8,0.,.8])
    valid=probe_grid([.8,1.2],[.15,.4,.65,.9,1.1,1.25],[-.5,.5])
    units=np.array([j for j in range(32) if j!=28])
    h=forward(d,train)[2]
    H=np.c_[h[:,units],np.ones(len(h))]
    # Pedal to balance drag is v^2/1600 = .5625*speed_obs^2.
    # Use a linear approximation to its logit over the main 15–25 m/s band.
    # A quadratic target was probed first, but the old features approximated
    # it poorly (validation RMSE .0718, max .2069); it was not driven.
    y=.8*train[:,10]-.25
    reg=1e-5*len(H)*np.eye(H.shape[1]);reg[-1,-1]=1e-8
    co=np.linalg.solve(H.T@H+reg,H.T@y)
    d['W3'][units,1]+=co[:-1].astype(np.float32)
    d['b3'][1]+=co[-1]
    metrics={}
    for name,x in [('fit_handmade',train),('validation_handmade',valid)]:
        h=forward(d,x)[2]
        prediction=np.c_[h[:,units],np.ones(len(h))]@co
        error=prediction-(.8*x[:,10]-.25)
        metrics[name]={'n':len(x),'rmse':float(np.sqrt(np.mean(error**2))),
                       'max_error':float(np.max(abs(error)))}
    print(json.dumps(metrics,indent=2))
    print('Coefficient max:',np.max(abs(co)))
    np.savez('c11_drag_feedforward.npz',**d)
    with open('drag_probe_metrics.json','w') as f:json.dump(metrics,f,indent=2)
