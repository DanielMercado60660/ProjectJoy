"""CPU-only algebra on weights and hand-made observations; never imports a simulator."""
import json
from pathlib import Path
import numpy as np

NAMES = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')
def load(path='start.npz'):
    with np.load(path, allow_pickle=False) as f:
        return {n: f[n].copy() for n in NAMES}

def save(path, d):
    np.savez(path, **{n: np.asarray(d[n], np.float32) for n in NAMES})

def forward(d, x):
    h1 = np.tanh(np.asarray(x) @ d['W1'] + d['b1'])
    h2 = np.tanh(h1 @ d['W2'] + d['b2'])
    z = h2 @ d['W3'] + d['b3']
    return np.tanh(z), h1, h2, z

def jacobian(d, x):
    a, h1, h2, z = forward(d, x)
    return d['W1'] @ ((1-h1*h1)[:, None] * d['W2']) @ ((1-h2*h2)[:, None]*d['W3'])

def straight(width=6, offset=0, heading=0, speed=20, prev_steer=0, prev_pedal=1):
    # Analytic distances to two infinite parallel lines, only a static sensor probe.
    angles = np.deg2rad(np.array([-90,-60,-35,-18,-6,6,18,35,60,90])) + heading
    s = np.sin(angles)
    rays = np.minimum(60, np.where(s>0, width-offset, width+offset)/np.maximum(abs(s),1e-8))/60
    return np.r_[rays, speed/30, prev_steer, prev_pedal]

def summarize(path):
    r = json.loads(Path(path).read_text())
    print('keys:', list(r))
    for key, val in r.items():
        if isinstance(val, dict):
            print(key, {k:v for k,v in val.items() if not isinstance(v,(list,dict))})
        elif not isinstance(val,list):
            print(key, val)
    return r

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=170)
    d=load()
    print('W3:\n', d['W3'], '\nb3:', d['b3'])
    for speed in [0,15,25,35]:
        for pp in [-1,0,1]:
            x=straight(speed=speed, prev_pedal=pp)
            a,h1,h2,z=forward(d,x)
            J=jacobian(d,x)
            print('straight speed, prev_pedal:', speed,pp,'action',a,'logits',z)
            print('logit derivative speed, prev_steer, prev_pedal:',J[10:])
    x=straight(speed=25)
    a,h1,h2,z=forward(d,x)
    print('ray logit derivatives:',jacobian(d,x)[:10])
    print('h2 contributions [unit, activation, steer, pedal]:')
    print(np.c_[np.arange(32),h2,h2[:,None]*d['W3']])
    print('h1 pedal logit sensitivity:')
    print(d['W2'] @ ((1-h2*h2)*d['W3'][:,1]))
