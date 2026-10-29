"""Round-5 data and views. NumPy only; existing artifacts are read-only.

The only privileged observation is the explicitly declared frame-0 cue (sigma .25 px).
No later state, visibility, collision event or turn flag is passed to a tracker.
"""
from pathlib import Path
import numpy as np
import sim

ROOT = Path(__file__).resolve().parent
CUE_SIGMA = 0.25
FRESH_SEED = 51034000  # reserved headline seeds; never used by smoke checks
SMOKE_SEED = 51024000
LOOK_FIRST, LOOK_LAST = 4, 31


def read_split(name, n=None):
    if name in ('fresh', 'smoke_fresh'):
        if n is None:
            raise ValueError('fresh requires n')
        seed = FRESH_SEED if name == 'fresh' else SMOKE_SEED
        rows = [sim.simulate_one(np.random.default_rng(seed + i),
                                dict(sim.BASE, **sim.HARD)) for i in range(n)]
        pos, present, occ, colors, turned = map(np.stack, zip(*rows))
        vis = np.stack([sim.occ_visible_frac(p[..., 0], o[o[:, 0] >= 0])
                        for p, o in zip(pos, occ)])
        return dict(pos=pos, present=present, occ=occ, colors=colors, turned=turned,
                    vis=vis, noise_seed=np.int64(seed))
    if name not in ('hard_train', 'hard_val', 'hard_test'):
        raise ValueError(name)
    with np.load(ROOT / 'data' / (name + '.npz')) as z:
        return {k: (z[k][:n] if z[k].ndim else z[k]) for k in z.files}


def observation_seed(d, index):
    return int(d['noise_seed']) * 100003 + int(index)


def cues(d, indices):
    return np.stack([d['pos'][i, 0] + np.random.default_rng(observation_seed(d, i) + 71)
                     .normal(0, CUE_SIGMA, (sim.K, 2)) for i in indices]).astype(np.float32)


def view_occ(occ, wall=-1):
    out = occ.copy()
    if wall >= 0:
        if wall >= len(occ) or occ[wall, 0] < 0:
            raise ValueError('peek must name a real wall')
        out[wall] = -1
    return out


def frames(d, index, look=None):
    """look=(time, wall); replace that view only, sharing sensor noise exactly."""
    pos, present, occ, colors = (d[k][index] for k in ('pos', 'present', 'occ', 'colors'))
    out = sim.render(pos[None], present[None], occ[None], colors[None], noise=0)[0]
    if look is not None:
        t, wall = look
        out[t] = sim.render(pos[None, t:t+1], present[None], view_occ(occ, wall)[None],
                            colors[None], noise=0)[0, 0]
    noise = np.random.default_rng(observation_seed(d, index)).normal(0, .02, out.shape)
    return np.clip(out + noise, 0, 1).astype(np.float32)


def look_flags(length, look=None):
    flags = np.zeros((length, 2), np.float32)
    if look is not None:
        flags[look[0], look[1]] = 1
    return flags


def random_look(occ, seed, length=sim.T):
    rng = np.random.default_rng(seed)
    return (int(rng.integers(LOOK_FIRST, min(LOOK_LAST, length - 2) + 1)),
            int(rng.choice(np.flatnonzero(occ[:, 0] >= 0))))
