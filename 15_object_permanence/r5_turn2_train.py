"""Turn-2 matched training: per-query binding, fixed-step LR, paired val selection.

Preserves the original trainer and checkpoint format reader. All jobs are guarded
at <=35 minutes. Resume has no schedule restart and restores both Adam/RNG states.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import signal
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_unflatten
from r5_train import atomic_pickle, array_tree
from r5_model import training_loss, distribution
from r5_turn2_model import BoundCueTracker
from r5_turn2_utils import permute_queries, learning_rate
from r5_data import read_split, frames, cues, random_look, look_flags
from r5_metrics import nll, moments
from r5_tune import write_json
from train import render_mx

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('signal received; save after complete paired step', flush=True)


def make_models(cfg):
    return {name: BoundCueTracker(components=k, **cfg['architecture'])
            for name, k in [('gaussian', 1), ('mixture', 3)]}


def batch(data, idx, rng):
    pos, pres, occ, colors = (data[k][idx] for k in ('pos', 'present', 'occ', 'colors'))
    cue = pos[:, 0]+rng.normal(0, .25, pos[:, 0].shape).astype(np.float32)
    order = np.stack([rng.permutation(3) for _ in idx])
    target_pos, query_present, query_cue = permute_queries(pos, pres, cue, order)
    flags = np.zeros((len(idx), pos.shape[1], 2), np.float32)
    # Render original object order: query permutation changes neither pixels nor physics.
    f = render_mx(mx.array(pos), mx.array(pres.astype(np.float32)), mx.array(occ), mx.array(colors), noise=0)
    for b in range(len(idx)):
        if rng.random() < 2/3:
            t, wall = random_look(occ[b], int(rng.integers(2**31)))
            ro = occ[b:b+1].copy()
            ro[0, wall] = -1
            reveal = render_mx(mx.array(pos[b:b+1, t:t+1]), mx.array(pres[b:b+1].astype(np.float32)),
                               mx.array(ro), mx.array(colors[b:b+1]), noise=0)
            f = f.at[b, t].add(reveal[0, 0]-f[b, t])
            flags[b, t, wall] = 1
    f = mx.clip(f+.02*mx.random.normal(f.shape), 0, 1)
    return (f, mx.array(query_cue), mx.array(query_present.astype(np.float32)), mx.array(occ),
            mx.array(flags), mx.array((target_pos-32)/32))


def validate_models(models, data, ids, deadline):
    values = {name: [] for name in models}
    for i in ids:
        if time.monotonic() >= deadline:
            return None
        look = random_look(data['occ'][i], 72000+i) if i % 2 else None
        inp = [mx.array(x) for x in (frames(data, i, look)[None], cues(data, [i]),
                data['present'][i:i+1].astype(np.float32), data['occ'][i:i+1], look_flags(40, look)[None])]
        present = data['present'][i]
        visible = (data['vis'][i] >= .999) & present
        gap = float(np.diff(np.sort(data['pos'][i, 0, present, 0])).min())
        for name, model in models.items():
            dist = tuple(x[0] for x in distribution(model(*inp)))
            mean, _ = moments(dist)
            err = np.linalg.norm(mean-data['pos'][i], axis=-1)
            loss = nll(dist, data['pos'][i])
            values[name].append(dict(nll=float(loss[:, present].mean()), gap=gap,
                                     visible_n=int(visible.sum()), visible_sum=float(err[visible].sum()),
                                     frame0_error=float(err[0, present].mean())))
    summary = {}
    for name, rows in values.items():
        groups = {}
        for tag, select in [('<1', lambda r:r['gap'] < 1), ('>=8', lambda r:r['gap'] >= 8)]:
            rs = [r for r in rows if select(r)]
            count = sum(r['visible_n'] for r in rs)
            groups[tag] = dict(sequences=len(rs), visible_error_px=sum(r['visible_sum'] for r in rs)/count if count else None)
        summary[name] = dict(nll=float(np.mean([r['nll'] for r in rows])),
                             visible_error_px=sum(r['visible_sum'] for r in rows)/sum(r['visible_n'] for r in rows),
                             frame0_error_px=float(np.mean([r['frame0_error'] for r in rows])), by_xgap=groups)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='checkpoints/r5_t2_ac')
    p.add_argument('--warm', default='checkpoints/hard/final')
    p.add_argument('--steps', type=int, default=120000)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--max-minutes', type=float, required=True)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=52024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--val-count', type=int, default=128)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pause-at', type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 < args.max_minutes <= 35 or min(args.steps, args.batch, args.save_every) <= 0:
        p.error('positive steps/batch/save interval; max-minutes must be in (0,35]')
    if not 0 <= args.val_count <= 128:
        p.error('val-count must be 0..128 (reserved block 256:384)')
    start = time.monotonic()
    out = Path(args.out)
    last = out/'last.pkl'
    if last.exists() != args.resume:
        raise ValueError('existing last.pkl requires --resume; a new run requires a new output path')
    arch, warm_hash = dict(enc_dim=32, hid=16, layers=1), None
    if not args.tiny:
        with open(args.warm+'.state.pkl', 'rb') as f:
            old = pickle.load(f)
        arch = {k: old['cfg'][k] for k in ('enc_dim', 'hid', 'layers')}
        warm_hash = hashlib.sha256(Path(args.warm+'.weights.npz').read_bytes()).hexdigest()
    cfg = dict(version=2, cue_mode='bound_shared', architecture=arch, steps=args.steps,
               batch=args.batch, seed=args.seed, lr=args.lr, tiny=args.tiny, warm_sha256=warm_hash,
               training_split='hard_train', cue_sigma_px=.25, peek_probability=2/3,
               query_permutation=True, frame0_mean='observed noisy cue', beta=.5,
               aux_steps=min(8000, max(1, args.steps//5)),
               validation_indices=list(range(256, 256+args.val_count)),
               selection='lowest mean of both heads val pixel NLL at same step; half random peeks',
               schedule='linear warmup min(1000,steps/20), hold to steps/3, cosine to .05*lr')
    cfg['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                           ('r5_turn2_train.py', 'r5_turn2_model.py', 'r5_turn2_utils.py',
                            'r5_train.py', 'r5_model.py', 'r5_data.py', 'model.py', 'train.py', 'sim.py')}
    mx.random.seed(args.seed)
    models = make_models(cfg)
    if args.tiny:
        from model import Tracker
        original = Tracker(**arch)
        mx.eval(original.parameters())
        warm = array_tree(original.parameters())
    else:
        with np.load(args.warm+'.weights.npz') as z:
            warm = dict(z)
    for model in models.values():
        model.warm_start(warm)
    opts = {name: optim.AdamW(learning_rate=args.lr, weight_decay=1e-4) for name in models}
    rng = np.random.default_rng(args.seed)
    step, history, active_seconds, best_score, best_step = 0, [], 0., None, None
    if args.resume:
        from r5_load import load_pair
        models, state = load_pair(last)
        if state['config'] != cfg:
            raise ValueError('resume configuration/source mismatch; keep schedule, data, seed and warm source fixed')
        for name, opt in opts.items():
            opt.state = tree_unflatten([(k, mx.array(v)) for k, v in state['optimizers'][name].items()])
        step, history, active_seconds = state['step'], state['history'], state['active_seconds']
        best_score, best_step = state['best_pair_nll'], state['best_step']
        rng.bit_generator.state = state['numpy_rng']
        print(f'RESUMED paired_step={step} optimizer_steps={[int(o.step.item()) for o in opts.values()]}', flush=True)
    tr = read_split('hard_train', 8 if args.tiny else None)
    va = read_split('hard_val', 256+args.val_count) if args.val_count else None
    grads = {name: nn.value_and_grad(model, training_loss) for name, model in models.items()}
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save(is_best=False):
        mx.eval([m.parameters() for m in models.values()], [o.state for o in opts.values()])
        state = dict(config=cfg, step=step, history=history,
                     models={n: array_tree(m.parameters()) for n, m in models.items()},
                     optimizers={n: array_tree(o.state) for n, o in opts.items()},
                     numpy_rng=rng.bit_generator.state, next_mlx_seed=args.seed+step,
                     mlx_rng_rule='reseed seed+paired step before batch',
                     active_seconds=active_seconds+time.monotonic()-start,
                     best_pair_nll=best_score, best_step=best_step)
        atomic_pickle(last, state)
        if is_best:
            atomic_pickle(out/'best.pkl', state)
        write_json(out/'selection.json', dict(last_step=step, best_step=best_step, best_pair_nll=best_score,
                   validation_indices=cfg['validation_indices'], selected_checkpoint=str(out/'best.pkl') if best_step else None,
                   active_seconds=state['active_seconds']))
        print(f'SAVED {last} paired_step={step} best_step={best_step}', flush=True)

    print(f'TURN2_PAIR_START step={step} warm_sha256={warm_hash} cfg={cfg}', flush=True)
    while step < args.steps and not STOP:
        if time.monotonic()-start >= max(0., args.max_minutes*60-10):
            print('WALL_GUARD reached', flush=True)
            break
        mx.random.seed(args.seed+step)
        inputs = batch(tr, rng.integers(0, len(tr['pos']), args.batch), rng)
        mx.eval(*inputs)
        losses = {}
        rate = learning_rate(step, args.steps, args.lr)
        for name, model in models.items():
            opts[name].learning_rate = rate
            loss, gradient = grads[name](model, *inputs, max(0., 1-step/cfg['aux_steps']))
            gradient, _ = optim.clip_grad_norm(gradient, 5.)
            opts[name].update(model, gradient)
            mx.eval(model.parameters(), opts[name].state, loss)
            losses[name] = float(loss.item())
            if not np.isfinite(losses[name]):
                raise FloatingPointError(f'{name} nonfinite loss; prior paired checkpoint retained')
        step += 1
        history.append(dict(step=step, loss=losses))
        if step <= 2 or step % 100 == 0:
            print(f'TURN2_PAIR step={step} loss={losses} lr={rate:.6g} elapsed_s={time.monotonic()-start:.1f}', flush=True)
        if step % args.save_every == 0 or step == args.steps or (args.stop_after and step >= args.stop_after):
            is_best = False
            if args.val_count:
                val = validate_models(models, va, cfg['validation_indices'], start+args.max_minutes*60-10)
                if val is not None:
                    score = float(np.mean([v['nll'] for v in val.values()]))
                    history[-1]['validation'] = val
                    if best_score is None or score < best_score:
                        best_score, best_step, is_best = score, step, True
                    print(f'TURN2_VAL step={step} pair_nll={score:.6f} details={json.dumps(val)} best_step={best_step}', flush=True)
            save(is_best)
            if step == args.pause_at:
                print(f'PAUSED_FOR_KILL_TEST step={step}', flush=True)
                while not STOP and time.monotonic()-start < args.max_minutes*60-10:
                    time.sleep(.05)
            if args.stop_after and step >= args.stop_after:
                break
    save()
    print(f'TURN2_PAIR_DONE paired_step={step} best_step={best_step} elapsed_s={time.monotonic()-start:.1f}', flush=True)


if __name__ == '__main__':
    main()
