"""Exact saturating distance-speed circuit; preserve steering on handmade probes."""
import argparse,json
from pathlib import Path
import numpy as np
from probe import load,save,forward
from edit_readout import handmade

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output')
    p.add_argument('--top-speed',type=float,default=32)
    p.add_argument('--front-slope',type=float,default=1.8)
    p.add_argument('--inner-gain',type=float,default=.15)
    p.add_argument('--outer-gain',type=float,default=3)
    p.add_argument('--r35',type=float,default=.8)
    p.add_argument('--r60',type=float,default=.4)
    p.add_argument('--r90',type=float,default=1)
    args=p.parse_args()
    orig=load();d=load();x=handmade(True)
    old_z=forward(orig,x)[3]
    means=forward(orig,handmade())[1].mean(axis=0)
    for j in (14,28):
        d['b2']+=means[j]*d['W2'][j]
        d['W2'][j]=0;d['W1'][:,j]=0;d['b1'][j]=0
    d['W1'][4,14]=d['W1'][5,14]=args.front_slope/2
    d['W1'][10,28]=.1
    k=16
    d['W2'][:,k]=0;d['b2'][k]=0
    d['W2'][14,k]=args.inner_gain*args.top_speed
    d['W2'][28,k]=-args.inner_gain*300
    d['W3'][:,1]=0;d['W3'][k,1]=args.outer_gain;d['b3'][1]=0
    d['W3'][k,0]=0
    broad=args.r35*(x[:,7]-x[:,2])+args.r60*(x[:,8]-x[:,1])+args.r90*(x[:,9]-x[:,0])
    target_delta=old_z[:,0]+broad-forward(d,x)[3][:,0]
    indices=np.array([i for i in range(32) if i!=k])
    H=np.c_[forward(d,x)[2][:,indices],np.ones(len(x))]
    reg=np.eye(32)*.001;reg[-1,-1]=0
    delta=np.linalg.solve(H.T@H+reg,H.T@target_delta)
    d['W3'][indices,0]+=delta[:-1];d['b3'][0]+=delta[-1]
    save(args.output,d)
    err=forward(d,x)[3][:,0]-old_z[:,0]-broad
    target_speed=args.top_speed*np.tanh(args.front_slope*(x[:,4]+x[:,5])/2)
    v_hat=300*np.tanh(.1*x[:,10])
    prescribed=np.tanh(args.outer_gain*np.tanh(args.inner_gain*(target_speed-v_hat)))
    meta={**vars(args),'steer_rmse_logit':float(np.sqrt(np.mean(err**2))),
          'max_steer_abs_logit_error':float(abs(err).max()),
          'max_pedal_error':float(abs(forward(d,x)[0][:,1]-prescribed).max()),
          'changed_from_start':sum(int(np.count_nonzero(d[n]!=orig[n])) for n in d)}
    Path(args.output+'.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
