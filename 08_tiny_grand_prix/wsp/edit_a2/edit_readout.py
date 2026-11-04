"""Fit a specified pedal function on a deterministic, handmade sensor lattice.

This is direct linear algebra, not gradient training or simulator optimization.
No evaluation is launched here. Each candidate is chosen and evaluated manually.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from probe import load, save, forward, straight

def handmade(wide_openings=False):
    templates = [straight(width=w, offset=o, heading=h)[:10]
                 for w in (5,6,7) for o in (-3,0,3) for h in (-.2,0,.2)]
    # Explicit illustrative corner/opening/closing sensor patterns; no tracks or dynamics.
    corners = np.array([
        [.10,.12,.17,.25,.38,.56,.70,.25,.13,.10],
        [.10,.12,.17,.25,.38,.50,.32,.18,.12,.10],
        [.10,.11,.14,.18,.24,.32,.45,.60,.14,.10],
        [.14,.16,.21,.30,.45,.65,.30,.15,.08,.06],
        [.06,.07,.09,.12,.18,.28,.48,.40,.20,.14],
        [.03,.04,.05,.07,.10,.14,.23,.40,.28,.18],
        [.10,.11,.14,.20,.40,.80,.40,.20,.12,.10],
        [.10,.12,.16,.24,.60,.60,.24,.16,.12,.10],
        [.08,.10,.12,.15,.20,.20,.15,.12,.10,.08],
    ])
    for c in corners:
        for scale in (.8,1,1.2):
            templates.extend([np.minimum(1,c*scale),np.minimum(1,c[::-1]*scale)])
    if wide_openings:
        originals=list(templates)
        for rays in originals:
            for j in (1,2,3,6,7,8):
                extra=rays.copy();extra[j]=1.
                templates.append(extra)
    rows=[]
    for rays in templates:
        for v in (0,.25,.5,.7,.9,1.1,1.3):
            for st in (-.6,0,.6):
                for pd in (-1,0,1):
                    rows.append(np.r_[rays,v,st,pd])
    return np.asarray(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output')
    p.add_argument('--base',type=float,default=7)
    p.add_argument('--clearance',type=float,default=27)
    p.add_argument('--gain',type=float,default=.25)
    p.add_argument('--pin-pedal',action='store_true')
    p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--wide-openings',action='store_true')
    args=p.parse_args()
    d=load(); x=handmade(args.wide_openings)
    if args.pin_pedal:
        d['b1']+=d['W1'][12]
        d['W1'][12]=0
    target=args.gain*(args.base+args.clearance*(x[:,4]+x[:,5])/2-30*x[:,10])
    h2=forward(d,x)[2]
    H=np.c_[h2,np.ones(len(h2))]
    reg=np.eye(33)*args.ridge; reg[-1,-1]=0
    w=np.linalg.solve(H.T@H+reg,H.T@target)
    d['W3'][:,1]=w[:32]; d['b3'][1]=w[32]
    err=H@w-target
    save(args.output,d)
    meta={**vars(args),'samples':len(x),'rmse_logit':float(np.sqrt(np.mean(err**2))),
          'max_abs_logit_error':float(abs(err).max()),'max_weight':float(abs(w).max()),
          'changed':sum(int(np.count_nonzero(d[n]!=load()[n])) for n in d)}
    Path(args.output+'.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    for speed in (0,15,25,35):
        print('straight',speed,forward(d,straight(speed=speed))[0])
