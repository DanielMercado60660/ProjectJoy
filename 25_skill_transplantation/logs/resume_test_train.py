import sys; sys.path.insert(0,'/Users/danielmercado/ProjectJoy/25_skill_transplantation')
from common import *; from ppo import train
p,h = None,None
p, n, h = train(init_params(0), Normalizer(), ['fwd']*16, 4096*40, 0, sys.argv[1], update_norm=True, resume=True, ckpt_every=2, log_every=1)
print('updates logged', [r['update'] for r in h['log']])
