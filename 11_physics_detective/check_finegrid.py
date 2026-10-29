"""Re-check the active detective's Bayes coverage on a 20x finer grid (30,001 points) to rule out
grid resolution as the cause of 0.882 vs 0.90.  ../.venv/bin/python check_finegrid.py  (~3 min)"""
import sys; sys.path.insert(0, '.')
import numpy as np
from inference import GridPosterior
D = np.load('results/active.npz'); mu = D['mu']
fine = np.linspace(0.05, 0.8, 30001)
for strat in ['active']:
    des, Y, C = D[f'{strat}_designs'], D[f'{strat}_y'], D[f'{strat}_cens']
    pit = np.zeros((len(mu), 15)); pit_c = np.zeros((len(mu), 15))
    for i in range(len(mu)):
        p = GridPosterior(fine); pc = GridPosterior()
        for j in range(15):
            p.update(des[i,j,0], des[i,j,1], Y[i,j], C[i,j]); pc.update(des[i,j,0], des[i,j,1], Y[i,j], C[i,j])
            pit[i,j] = np.interp(mu[i], fine, np.cumsum(p.w)); pit_c[i,j] = np.interp(mu[i], pc.grid, np.cumsum(pc.w))
    for name, P in [('fine', pit), ('coarse', pit_c)]:
        c90 = (np.abs(P-0.5) <= 0.45).mean(0); c50 = (np.abs(P-0.5) <= 0.25).mean(0)
        print(strat, name, 'cov50 all %.4f cov90 all %.4f' % ((np.abs(P-0.5)<=0.25).mean(), (np.abs(P-0.5)<=0.45).mean()), 'cov90 by n', c90.round(3))
    np.save('results/active_pit_finegrid.npy', pit)
