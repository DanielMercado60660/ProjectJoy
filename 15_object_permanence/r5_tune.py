"""Val-only classical search, bounded in time. Never reads a test split."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from r5_data import read_split, frames, cues
from r5_classical import kalman_track, particle_track
from r5_metrics import nll
from r5_turn2_utils import baseline_grids


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def measure(method, config, data, ids, deadline):
    losses, err = [], []
    fn = particle_track if method == 'particle' else kalman_track
    for i in ids:
        if time.monotonic() >= deadline:
            return None
        dist, diag = fn(frames(data, i), cues(data, [i])[0], data['present'][i], data['occ'][i], **config)
        losses.append(float(nll(dist, data['pos'][i])[:, data['present'][i]].mean()))
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/r5/tuning.json')
    p.add_argument('--max-minutes', type=float, required=True)
    p.add_argument('--screen-n', type=int, default=8)
    p.add_argument('--confirm-n', type=int, default=24)
    p.add_argument('--particles', type=int, default=256)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--turn2', action='store_true', help='extend PF q/miss beyond turn-1 boundary winners')
    args = p.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if not 0 < args.max_minutes <= 45:
        p.error('max-minutes must be in (0,45]')
    start = time.monotonic()
    deadline = start+60*args.max_minutes-3
    screen_deadline = start+.7*(deadline-start)
    d = read_split('hard_val', max(128, args.screen_n+args.confirm_n))
    screen = list(range(args.screen_n))
    confirm = list(range(64, 64+args.confirm_n))
    if args.screen_n > 64 or args.confirm_n > 64:
        p.error('screen-n and confirm-n must be <=64 (disjoint validation blocks)')
    kgrid, pgrid = baseline_grids(args.particles, args.turn2)
    # Reproducible shuffled search; first candidates include sane interior settings.
    rng = np.random.default_rng(5024)
    rng.shuffle(kgrid)
    rng.shuffle(pgrid)
    kanchor = dict(q=.5, r=7.)
    panchor = dict(q=.15, r=1., miss=.2, particles=args.particles, bandwidth=1.5)
    if args.turn2:
        panchor = dict(q=.5, r=1., miss=.35, particles=args.particles, bandwidth=4.)
    kgrid = [kanchor]+[c for c in kgrid if c != kanchor]
    pgrid = [panchor]+[c for c in pgrid if c != panchor]
    if args.tiny:
        kgrid, pgrid = kgrid[:1], pgrid[:1]
    rows = []
    output = dict(version=2 if args.turn2 else 1, split='hard_val', screen_indices=screen, confirm_indices=confirm,
                  objective='mean per-sequence all-frame pixel NLL',
                  search_space={'kalman': kgrid, 'particle': pgrid}, rows=rows, selected={},
                  complete=False, elapsed_seconds=0.)
    for pair in itertools.zip_longest(pgrid, kgrid):
        for method, cfg in zip(('particle', 'kalman'), pair):
            if cfg is None:
                continue
            candidate_start = time.monotonic()
            loss = measure(method, cfg, d, screen, screen_deadline)
            if loss is None:
                break
            row = dict(method=method, config=cfg, screen_nll=loss,
                       screen_seconds=time.monotonic()-candidate_start)
            rows.append(row)
            print(f'TUNE {method} cfg={cfg} val_nll={loss:.4f}', flush=True)
            write_json(args.out, output)
        if time.monotonic() >= screen_deadline:
            break
    # Confirm up to 3 finalists per method on a distinct validation block. Select
    # ONLY after all finalists for that method finish, otherwise use screen winner.
    for method in ('particle', 'kalman'):
        candidates = sorted([r for r in rows if r['method'] == method], key=lambda r:r['screen_nll'])[:3]
        if not candidates:
            output['failure'] = f'no complete {method} candidate; increase tuning budget'
            write_json(args.out, output)
            raise RuntimeError(output['failure'])
        winner = candidates[0]
        complete = True
        for row in candidates:
            candidate_start = time.monotonic()
            loss = measure(method, row['config'], d, confirm, deadline)
            if loss is None:
                complete = False
                break
            row['confirm_nll'] = loss
            row['confirm_seconds'] = time.monotonic()-candidate_start
        if complete:
            winner = min(candidates, key=lambda r:r['confirm_nll'])
        output['selected'][method] = winner['config']
        output.setdefault('selection_stage', {})[method] = 'confirm' if complete else 'screen_only'
        space = output['search_space'][method]
        edges = []
        for key, value in winner['config'].items():
            values = sorted(set(c[key] for c in space))
            if len(values) > 1 and value in (values[0], values[-1]):
                edges.append(key)
        output.setdefault('boundary_parameters', {})[method] = edges
    output['elapsed_seconds'] = time.monotonic()-start
    output['complete'] = True
    output['grid_complete'] = len(rows) == len(pgrid)+len(kgrid)
    write_json(args.out, output)
    print(f'TUNE_DONE candidates={len(rows)} selected={output["selected"]} elapsed_s={output["elapsed_seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
