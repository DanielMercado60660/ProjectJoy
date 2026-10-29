"""Probe: how often does the learner drive backwards / touch the ball with its rear?
    ../.venv/bin/python probe_reverse.py checkpoints/run4_tanh_relp.npz
500 held-out goalie matches (seed 1001), first 150 steps."""
import sys
import numpy as np
from env import Arena
from ppo import load_ckpt, make_policy
import scripted

for f in sys.argv[1:]:
    m, n, _ = load_ckpt(f)
    p = make_policy(m, n)
    a = Arena(500, seed=1001, opponent="goalie")
    a.reset(np.ones(500, bool))
    rev, rear = [], []
    for t in range(150):
        act = np.clip(p(a), -1, 1)
        rev.append(((act[:, 0] + act[:, 1]) / 2 < -0.1).mean())
        d = a.ball - a.pos[:, 0]
        fwd = d[:, 0] * np.cos(a.th[:, 0]) + d[:, 1] * np.sin(a.th[:, 0])
        near = np.linalg.norm(d, axis=1) < 0.13
        rear.append(((fwd < 0) & near).sum() / max(near.sum(), 1))
        a.step(act, scripted.goalie(a))
    print(f"{f}: steps reversing {np.mean(rev):.2f}, near-ball steps with ball behind robot {np.mean(rear):.2f}")
