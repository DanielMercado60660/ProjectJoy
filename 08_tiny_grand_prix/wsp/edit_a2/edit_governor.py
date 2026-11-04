"""Wire an exact speed-clearance pedal circuit into two weak steering units.

No simulator import or optimization. The only solve preserves the original
steering plus a prescribed broad-ray correction on hand-made static inputs.
"""
import argparse,json
from pathlib import Path
import numpy as np
from probe import load,save,forward
from edit_readout import handmade

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output')
    p.add_argument('--base',type=float,default=9)
    p.add_argument('--clearance',type=float,default=27)
    p.add_argument('--inner-gain',type=float,default=.1)
    p.add_argument('--outer-gain',type=float,default=3)
    p.add_argument('--r35',type=float,default=.8)
    p.add_argument('--r60',type=float,default=.4)
    p.add_argument('--r90',type=float,default=1)
    args=p.parse_args()
    original=load();d=load();x=handmade(True)
    old_a,old_h1,old_h2,old_z=forward(original,x)
    j,k=14,16
    old_mean=float(forward(original,handmade())[1][:,j].mean())
    d['b2']+=old_mean*d['W2'][j]
    d['W2'][j]=0
    d['W1'][:,j]=0
    d['W1'][4,j]=d['W1'][5,j]=args.inner_gain*args.clearance/2
    d['W1'][10,j]=-30*args.inner_gain
    d['b1'][j]=args.inner_gain*args.base
    d['W2'][:,k]=0;d['W2'][j,k]=1;d['b2'][k]=0
    d['W3'][:,1]=0;d['W3'][k,1]=args.outer_gain;d['b3'][1]=0
    d['W3'][k,0]=0
    broad=args.r35*(x[:,7]-x[:,2])+args.r60*(x[:,8]-x[:,1])+args.r90*(x[:,9]-x[:,0])
    new_z=forward(d,x)[3]
    target_delta=old_z[:,0]+broad-new_z[:,0]
    indices=np.array([i for i in range(32) if i!=k])
    H=np.c_[forward(d,x)[2][:,indices],np.ones(len(x))]
    reg=np.eye(32)*.001;reg[-1,-1]=0
    delta=np.linalg.solve(H.T@H+reg,H.T@target_delta)
    d['W3'][indices,0]+=delta[:-1];d['b3'][0]+=delta[-1]
    save(args.output,d)
    err=forward(d,x)[3][:,0]-old_z[:,0]-broad
    prescribed=np.tanh(args.outer_gain*np.tanh(np.tanh(args.inner_gain*(args.base+args.clearance*(x[:,4]+x[:,5])/2-30*x[:,10]))))
    meta={**vars(args),'unit_h1':j,'unit_h2':k,'compensated_h1_mean':old_mean,
          'steer_rmse_logit':float(np.sqrt(np.mean(err**2))),
          'max_steer_abs_logit_error':float(abs(err).max()),
          'max_pedal_error':float(abs(forward(d,x)[0][:,1]-prescribed).max()),
          'changed_from_start':sum(int(np.count_nonzero(d[n]!=original[n])) for n in d)}
    Path(args.output+'.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
