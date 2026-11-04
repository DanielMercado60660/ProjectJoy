"""Static analysis of start.npz only; no simulator, trajectories, or track data."""
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=180)
w = {k: v.copy() for k, v in np.load('start.npz', allow_pickle=False).items()}
NAMES = ['right90','right60','right35','right18','right6','left6','left18','left35','left60','left90','speed','prev_steer','prev_pedal']
RAYS = {
    'straight_center': [6, 6.93, 10.46, 19.42, 57.4, 57.4, 19.42, 10.46, 6.93, 6],
    'straight_near_left': [9, 10.39, 15.69, 29.12, 60, 28.7, 9.71, 5.23, 3.46, 3],
    'straight_at_left': [10.8, 12.47, 18.83, 34.95, 60, 11.48, 3.88, 2.09, 1.39, 1.2],
    'heading_left': [6.06, 7.61, 13.22, 34.55, 60, 24.80, 13.69, 8.80, 6.47, 6.06],
    'gentle_left': [6, 7, 10, 16, 26, 45, 32, 12, 7, 6],
    'sharp_left': [6, 7, 10, 13, 17, 24, 35, 14, 8, 6],
    'very_sharp_left': [6, 7, 9, 11, 14, 18, 24, 19, 8, 6],
    'approach_left': [6, 7, 10.5, 18, 32, 40, 22, 11, 7, 6],
}

def forward(x, weights=w):
    a = np.tanh(x @ weights['W1'] + weights['b1'])
    b = np.tanh(a @ weights['W2'] + weights['b2'])
    z = b @ weights['W3'] + weights['b3']
    return a,b,z,np.tanh(z)

def obs(rays, speed=20, steer=0, pedal=1):
    return np.r_[np.array(rays)/60, speed/30, steer,pedal]

def jacobian(x):
    h1,h2,z,y = forward(x)
    # Derivatives of logits, not final tanh, to expose saturated outputs.
    j2 = (1-h2*h2)[:,None] * w['W3']
    j1 = (1-h1*h1)[:,None] * (w['W2'] @ j2)
    return w['W1'] @ j1, j1, j2

if __name__ == '__main__':
    print('HAND-MADE OBSERVATION PROBES; NO DRIVING OR SIMULATOR')
    for name, rays in RAYS.items():
        print('\n', name)
        for speed in [0, 10, 20, 30]:
            x = obs(rays,speed)
            a,b,z,y=forward(x)
            xr=x.copy(); xr[:10]=x[:10][::-1]; xr[11]*=-1
            yr=forward(xr)[-1]
            print(f'v={speed:2}: action={y} logits={z}, mirror action={yr}')
    for name in ['straight_center','sharp_left','straight_at_left']:
        x=obs(RAYS[name],20)
        a,b,z,y=forward(x)
        j,j1,j2=jacobian(x)
        print('\nJACOBIAN',name)
        for label,v in zip(NAMES,j): print(label,v)
        print('H2 contributions (unit, activation, steer logit, pedal logit)')
        c=b[:,None]*w['W3']
        for k in np.argsort(-np.abs(c[:,1]))[:12]: print(k,b[k],c[k])
        print('H1 speed derivative contributions (unit, speed weight, steer, pedal)')
        c=w['W1'][10,:,None]*j1
        for k in np.argsort(-np.abs(c[:,1]))[:12]: print(k,w['W1'][10,k],c[k])
    print('\nPREVIOUS ACTION DEPENDENCE')
    for pp in [-1,0,1]:
        for ps in [-.3,0,.3]:
            print(pp,ps,forward(obs(RAYS['sharp_left'],20,ps,pp))[-1])
