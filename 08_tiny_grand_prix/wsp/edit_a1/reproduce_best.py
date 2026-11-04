"""Reproduce call 12 from start.npz using only CPU weight arithmetic.

This does not import the simulator or spend an evaluation call. Existing
evaluated candidates and evaluator-owned files are never written.
"""
import numpy as np
from inspect_weights import load, forward, handmade
from symmetric_readout import probe_grid


def reproduce():
    original=load()
    d={k:v.copy() for k,v in original.items()}
    _,h1,h2,_=forward(d,handmade())
    d['b2']+=h1[:,23].mean()*d['W2'][23,:]
    d['W2'][23,:]=0
    d['b3'][0]+=h2[:,28].mean()*d['W3'][28,0]
    d['W3'][28,0]=0
    d['b1']+=d['W1'][12,:]
    d['W1'][12,:]=0
    d['W1'][:,23]=0
    d['W1'][4:6,23]=26*.20/2
    d['W1'][10,23]=-30*.20
    d['b1'][23]=7*.20
    d['W2'][:,28]=0
    d['W2'][23,28]=1
    d['b2'][28]=0
    d['W3'][:,1]=0
    d['W3'][28,1]=2
    d['b3'][1]=0
    # Call 4: wide-ray features and reduction of left bias.
    d['W1'][2,:]+=.5*d['W1'][3,:]
    d['W1'][7,:]+=.5*d['W1'][6,:]
    d['b3'][0]-=.08
    # Calls 6–8: clear-road bonus and stronger pedal response.
    h1=forward(d,handmade())[1]
    d['b2']+=h1[:,29].mean()*d['W2'][29,:]
    d['W2'][29,:]=0
    d['W1'][:,29]=0
    d['W1'][4:6,29]=3
    d['b1'][29]=-2.4
    d['W1'][4:6,23]=26*.1/2
    d['W1'][10,23]=-30*.1
    d['b1'][23]=7*.1
    d['W2'][23,28]=3
    d['W2'][29,28]=.45
    d['b2'][28]=.45
    d['W3'][28,1]=3
    # Call 9: approximate reflection symmetry via a specified target function.
    units=np.array([j for j in range(32) if j!=28])
    x=probe_grid([.6,1.,1.4],[0.,.25,.5,.75,1.,1.25],[-.8,0.,.8])
    mirror=x.copy();mirror[:,:10]=x[:,9::-1];mirror[:,11]*=-1
    _,_,h,z=forward(d,x)
    zm=forward(d,mirror)[3]
    desired=.5*(z[:,0]-zm[:,0])
    H=np.c_[h[:,units],np.ones(len(x))]
    reg=.001*len(H)*np.eye(H.shape[1]);reg[-1,-1]=1e-8
    delta=np.linalg.solve(H.T@H+reg,H.T@(desired-z[:,0]))
    d['W3'][units,0]+=delta[:-1].astype(np.float32)
    d['b3'][0]+=delta[-1]
    # Call 10: larger bonus on broad bends, unchanged low-range base.
    d['W2'][29,28]=.60
    d['b2'][28]=.60
    # Call 11: approximate drag compensation in the existing readout.
    x=probe_grid([.6,1.,1.4],[0.,.25,.5,.75,1.,1.2,1.3],[-.8,0.,.8])
    H=np.c_[forward(d,x)[2][:,units],np.ones(len(x))]
    reg=1e-5*len(H)*np.eye(H.shape[1]);reg[-1,-1]=1e-8
    co=np.linalg.solve(H.T@H+reg,H.T@(.8*x[:,10]-.25))
    d['W3'][units,1]+=co[:-1].astype(np.float32)
    d['b3'][1]+=co[-1]
    # Call 12: restore 60 original W1 entries and re-express the same functions.
    teacher={k:v.copy() for k,v in d.items()}
    cols=np.array([j for j in range(32) if j not in (23,29)])
    for row in (2,7):d['W1'][row,cols]=original['W1'][row,cols]
    H=np.c_[forward(d,x)[2][:,units],np.ones(len(x))]
    old=np.r_[d['W3'][units,0],d['b3'][0]]
    y=forward(teacher,x)[3][:,0]
    delta=np.linalg.solve(H.T@H+reg,H.T@(y-H@old))
    d['W3'][units,0]=(old+delta)[:-1]
    d['b3'][0]=(old+delta)[-1]
    co=np.linalg.solve(H.T@H+reg,H.T@(.8*x[:,10]-.25))
    d['W3'][units,1]=co[:-1]
    d['b3'][1]=co[-1]
    return d


if __name__=='__main__':
    d=reproduce()
    np.savez('reproduced_best.npz',**d)
    expected=load('c12_compact_readout.npz')
    for k in d:
        assert np.array_equal(d[k],expected[k]), f'{k} differs from call 12'
    print('Reproduced call 12 bit-for-bit at the float32 array level from start.npz.')
