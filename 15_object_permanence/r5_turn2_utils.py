"""Turn-2 pure NumPy helpers; old split names and training code remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import sim
from r5_data import read_split as read_old_split

TURN2_FRESH_SEED = 52024000


def read_split(name, n):
    if name != 'turn2_fresh':
        return read_old_split(name, n)
    if n < 1:
        raise ValueError('n must be positive')
    rows = [sim.simulate_one(np.random.default_rng(TURN2_FRESH_SEED+i),
                            dict(sim.BASE, **sim.HARD)) for i in range(n)]
    pos, present, occ, colors, turned = map(np.stack, zip(*rows))
    vis = np.stack([sim.occ_visible_frac(p[..., 0], o[o[:, 0] >= 0]) for p, o in zip(pos, occ)])
    return dict(pos=pos, present=present, occ=occ, colors=colors, turned=turned,
                vis=vis, noise_seed=np.int64(TURN2_FRESH_SEED))


def permute_queries(pos, present, cue, order):
    """Change query/label order ONLY. Rendered images always retain their original order."""
    assert pos.ndim == 4 and pos.shape[2:] == (3, 2)
    assert order.shape == present.shape == cue.shape[:2] == (len(pos), 3)
    assert np.all(np.sort(order, axis=1) == np.arange(3))
    p = np.take_along_axis(pos, order[:, None, :, None], axis=2)
    pr = np.take_along_axis(present, order, axis=1)
    c = np.take_along_axis(cue, order[:, :, None], axis=1)
    return p, pr, c


def query_features(cue, present, occ):
    """NumPy specification of the shared per-query 8-vector; no slot index or sorting."""
    own = (cue-32)/32 * present[..., None]
    geometry = np.where(occ >= 0, (occ-32)/32, -2).reshape(len(cue), 4)
    count = np.broadcast_to(present.sum(1)[:, None, None]/3, (len(cue), 3, 1))
    return np.concatenate([own, present[..., None], count,
                           np.broadcast_to(geometry[:, None], (len(cue), 3, 4))], axis=-1)


def learning_rate(step, steps, peak=3e-4):
    """Warm 1000 steps, hold through first third, cosine to 5% by total step budget."""
    warm = min(1000, max(1, steps//20))
    hold = max(warm, steps//3)
    if step < warm:
        return float(peak*(.05+.95*step/warm))
    fraction = np.clip((step-hold)/max(1, steps-hold), 0, 1)
    return float(peak*(.05+.95*.5*(1+np.cos(np.pi*fraction))))


def baseline_grids(particles=512, extended=False):
    import itertools
    qs = (.02, .15, .5, 1., 2.) if extended else (.02, .15, .5)
    misses = (.05, .2, .35, .5, .7) if extended else (.05, .2, .35)
    kalman = [dict(q=q, r=r) for q, r in itertools.product((.035, .15, .5, 1.2), (.07, .5, 2., 7., 14., 30.))]
    particle = [dict(q=q, r=r, miss=miss, particles=particles, bandwidth=bw)
                for q, r, miss, bw in itertools.product(qs, (.3, 1., 3., 6.), misses, (.5, 1.5, 4., 8.))]
    return kalman, particle


def audit_checkpoint(checkpoint, out):
    """Weights-only diagnostic, no neural forward pass and no MLX import."""
    path = Path(checkpoint)
    with path.open('rb') as f:
        state = pickle.load(f)
    report = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  paired_step=state['step'], active_seconds=state['active_seconds'],
                  paired_steps_per_second=state['step']/state['active_seconds'], models={})
    for name, weights in state['models'].items():
        w = weights['cue_projection.weight']
        report['models'][name] = dict(cue_projection_norm=float(np.linalg.norm(w)),
                                     position_columns_norm=float(np.linalg.norm(w[:, :6])),
                                     nonzero_position_weights=int(np.count_nonzero(w[:, :6])))
    target = Path(out)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/r5_ac/last.pkl')
    p.add_argument('--out', default='results/r5/turn2_cue_audit.json')
    args = p.parse_args()
    audit_checkpoint(args.checkpoint, args.out)
