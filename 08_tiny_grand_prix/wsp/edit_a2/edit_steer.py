"""Add an explicit wider-angle/side-clearance correction with output-weight algebra."""
import argparse, json
from pathlib import Path
import numpy as np
from probe import load,save,forward
from edit_readout import handmade

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--r35',type=float,default=.8)
    p.add_argument('--r60',type=float,default=.4)
    p.add_argument('--r90',type=float,default=1.)
    p.add_argument('--bias',type=float,default=0.)
    p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--wide-openings',action='store_true')
    args=p.parse_args()
    d=load(args.input); x=handmade(args.wide_openings)
    target=args.r35*(x[:,7]-x[:,2])+args.r60*(x[:,8]-x[:,1])+args.r90*(x[:,9]-x[:,0])+args.bias
    H=np.c_[forward(d,x)[2],np.ones(len(x))]
    reg=np.eye(33)*.001; reg[-1,-1]=0
    delta=np.linalg.solve(H.T@H+reg,H.T@target)
    d['W3'][:,0]=args.gain*d['W3'][:,0]+delta[:32]
    d['b3'][0]=args.gain*d['b3'][0]+delta[32]
    save(args.output,d)
    err=H@delta-target
    meta={**vars(args),'rmse_logit':float(np.sqrt(np.mean(err**2))),
          'max_abs_logit_error':float(abs(err).max()),
          'changed_from_start':sum(int(np.count_nonzero(d[n]!=load()[n])) for n in d)}
    Path(args.output+'.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta)
