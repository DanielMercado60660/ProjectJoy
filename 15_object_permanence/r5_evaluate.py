"""Matched causal peeks; fresh headline / legacy secondary. MLX imports are lazy."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from r5_data import frames, cues, look_flags, random_look, observation_seed
from r5_turn2_utils import read_split
from r5_classical import particle_track, kalman_track
from r5_metrics import (score_sequence, moments, wall_scores, uncertainty_look,
                        peek_value, aggregate, paired_ci)
from r5_tune import write_json


def neural_track(model, f, cue, present, occ, look=None):
    import mlx.core as mx
    from r5_model import distribution
    inputs = [mx.array(x) for x in (f[None], cue[None], present[None].astype(np.float32),
                                    occ[None], look_flags(len(f), look)[None])]
    return tuple(x[0] for x in distribution(model(*inputs))), {}


def runner(models, tuning, name, d, i, look):
    started = time.monotonic()
    f, cue = frames(d, i, look), cues(d, [i])[0]
    pres, occ = d['present'][i], d['occ'][i]
    inference_started = time.monotonic()
    if name in models:
        dist, diag = neural_track(models[name], f, cue, pres, occ, look)
    elif name == 'particle':
        dist, diag = particle_track(f, cue, pres, occ, look, seed=observation_seed(d, i)+900,
                                    **tuning['selected'][name])
    else:
        dist, diag = kalman_track(f, cue, pres, occ, look, **tuning['selected'][name])
    diag['inference_seconds'] = time.monotonic()-inference_started
    diag['render_seconds'] = inference_started-started
    return dist, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--tuning', required=True)
    p.add_argument('--thresholds', required=True)
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--split', choices=['fresh', 'turn2_fresh', 'smoke_fresh', 'hard_test'], default='fresh')
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--out', default='results/r5/eval_fresh.json')
    p.add_argument('--max-minutes', type=float, required=True)
    args = p.parse_args()
    if not 0 < args.max_minutes <= 45:
        p.error('max-minutes must be in (0,45]')
    start = time.monotonic()
    deadline = start+60*args.max_minutes-5
    models, state = {}, None
    if args.checkpoint:
        from r5_load import load_pair
        models, state = load_pair(args.checkpoint)
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None
    tuning = json.loads(Path(args.tuning).read_text())
    assert tuning['complete'], 'tuning did not complete'
    names = list(models) + ['kalman', 'particle']
    if args.calibrate:
        if Path(args.thresholds).exists():
            raise FileExistsError(args.thresholds)
        d = read_split('hard_val', 128+args.n)
        thresholds, counts = {}, {}
        for name in names:
            scores = []
            for i in range(128, 128+args.n):
                if time.monotonic() >= deadline:
                    raise RuntimeError('threshold calibration guard: incomplete; no thresholds published')
                dist, _ = runner(models, tuning, name, d, i, None)
                s = wall_scores(dist, d['present'][i], d['occ'][i])
                scores.extend(s[3:31].max(-1).tolist())
            thresholds[name] = float(np.quantile(scores, .75))
            counts[name] = len(scores)
            print(f'THRESHOLD {name} val_q75={thresholds[name]:.4f}', flush=True)
        write_json(args.thresholds, dict(thresholds=thresholds, split='hard_val',
                   indices=list(range(128, 128+args.n)), scores_per_method=counts,
                   checkpoint=args.checkpoint, step=state['step'] if state else None,
                   checkpoint_sha256=checkpoint_hash,
                   tuning=tuning['selected'], elapsed_seconds=time.monotonic()-start))
        return
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    th = json.loads(Path(args.thresholds).read_text())
    if th['checkpoint'] != args.checkpoint or th['step'] != (state['step'] if state else None):
        raise ValueError('thresholds must be calibrated for this exact checkpoint step')
    if th['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('checkpoint bytes changed after threshold calibration')
    if th['tuning'] != tuning['selected']:
        raise ValueError('thresholds belong to different classical settings')
    d = read_split(args.split, args.n)
    rows = {name: {policy: [] for policy in ('never', 'random', 'uncertainty')} for name in names}
    traces = {f'{name}_{policy}_{kind}': [] for name in names
              for policy in ('never', 'random', 'uncertainty') for kind in ('mean', 'cov', 'look')}
    if 'mixture' in names:
        for policy in ('never', 'random', 'uncertainty'):
            for kind in ('weights', 'modes', 'modecov'):
                traces[f'mixture_{policy}_{kind}'] = []
    completed = 0
    for i in range(args.n):
        if time.monotonic() >= deadline:
            break
        local = {}
        for name in names:
            if time.monotonic() >= deadline:
                break
            never, diag = runner(models, tuning, name, d, i, None)
            looks = dict(never=None, random=random_look(d['occ'][i], observation_seed(d, i)+81),
                         uncertainty=uncertainty_look(never, d['present'][i], d['occ'][i], th['thresholds'][name]))
            for policy, look in looks.items():
                if time.monotonic() >= deadline:
                    break
                dist, diagnostics = (never, diag) if look is None else runner(models, tuning, name, d, i, look)
                score = score_sequence(dist, d['pos'][i], d['vis'][i], d['present'][i])
                score['diagnostics'] = diagnostics
                if look is not None:
                    score['peek'] = peek_value(never, dist, d['pos'][i], d['present'][i], look)
                else:
                    score['cost_views'] = 0
                local[name, policy] = (score, moments(dist), look, dist if name == 'mixture' else None)
        if len(local) != len(names)*3:
            break
        # Commit only complete matched sequences across EVERY method/policy.
        for (name, policy), (score, (mean, cov), look, components) in local.items():
            rows[name][policy].append(score)
            prefix = f'{name}_{policy}'
            traces[prefix+'_mean'].append(mean)
            traces[prefix+'_cov'].append(cov)
            traces[prefix+'_look'].append((-1, -1) if look is None else look)
            if components is not None:
                for kind, values in zip(('weights', 'modes', 'modecov'), components):
                    traces[prefix+'_'+kind].append(values)
        completed += 1
        print(f'EVAL split={args.split} complete_sequences={completed}/{args.n} elapsed_s={time.monotonic()-start:.1f}', flush=True)
    if not completed:
        raise RuntimeError('guard expired before one matched sequence')
    summary = {name: {policy: aggregate(rs) for policy, rs in policies.items()} for name, policies in rows.items()}
    contrasts = {}
    for name in names:
        for policy in ('random', 'uncertainty'):
            contrasts[f'{name}/{policy}/later_nll_gain'] = paired_ci([r['peek']['later_nll_gain'] for r in rows[name][policy]])
            contrasts[f'{name}/{policy}/later_error_gain_px'] = paired_ci([r['peek']['later_error_gain_px'] for r in rows[name][policy]])
        contrasts[f'{name}/uncertainty_minus_random/all_nll'] = paired_ci([
            a['all']['nll']-b['all']['nll'] for a, b in zip(rows[name]['uncertainty'], rows[name]['random'])])
    if models:
        contrasts['mixture_minus_gaussian/never/hidden_nll'] = paired_ci([
            a['hidden']['nll']-b['hidden']['nll'] for a, b in zip(rows['mixture']['never'], rows['gaussian']['never'])
            if a['hidden']['n']])
    result = dict(split=args.split, requested_n=args.n, completed_n=completed, complete=completed==args.n,
                  checkpoint=args.checkpoint, paired_step=state['step'] if state else None,
                  checkpoint_sha256=checkpoint_hash,
                  training_config=state['config'] if state else None, tuning=tuning,
                  thresholds=th, elapsed_seconds=time.monotonic()-start,
                  calibration='Rosenblatt x-then-y chi2(2); nominal .393469/.864665',
                  summary=summary, contrasts=contrasts, per_sequence=rows)
    write_json(args.out, result)
    trace_path = str(Path(args.out).with_suffix('.npz'))
    np.savez_compressed(trace_path, **{k: np.array(v) for k, v in traces.items()},
                        pos=d['pos'][:completed], vis=d['vis'][:completed], present=d['present'][:completed],
                        occ=d['occ'][:completed], colors=d['colors'][:completed], noise_seed=d['noise_seed'])
    print(f'EVAL_DONE matched_n={completed} requested_n={args.n} out={args.out}', flush=True)


if __name__ == '__main__':
    main()
