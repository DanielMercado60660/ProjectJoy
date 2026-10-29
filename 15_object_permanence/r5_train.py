"""Matched Gaussian/mixture fine-tuning in a single guarded job (MLX runner only).

Every completed step updates BOTH models on the same batch, frames, cue and peek.
One atomic checkpoint contains both networks, both optimizers, config and RNG state.
No mx.compile is used. Resume configuration changes are rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from r5_model import CueTracker, training_loss, distribution
from r5_data import read_split, frames, cues, random_look, look_flags
from r5_metrics import nll
from train import render_mx

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('signal received; saving after the current paired step', flush=True)


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        pickle.dump(value, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def array_tree(tree):
    return {k: np.array(v) for k, v in tree_flatten(tree)}


def make_models(cfg):
    return {name: CueTracker(components=k, **cfg['architecture'])
            for name, k in [('gaussian', 1), ('mixture', 3)]}


def load_pair(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    models = make_models(state['config'])
    for name, model in models.items():
        model.load_weights([(k, mx.array(v)) for k, v in state['models'][name].items()])
    mx.eval([m.parameters() for m in models.values()])
    return models, state


def batch(data, idx, rng):
    pos = data['pos'][idx]
    pres = data['present'][idx].astype(np.float32)
    occ = data['occ'][idx]
    colors = data['colors'][idx]
    cue = pos[:, 0] + rng.normal(0, .25, pos[:, 0].shape).astype(np.float32)
    flags = np.zeros((len(idx), pos.shape[1], 2), np.float32)
    # Render with zero noise first, then one shared noise tensor. A revealed frame
    # must not receive independent sensor noise or expose the second wall.
    f = render_mx(mx.array(pos), mx.array(pres), mx.array(occ), mx.array(colors), noise=0)
    for b in range(len(idx)):
        if rng.random() < 2/3:
            t, wall = random_look(occ[b], int(rng.integers(2**31)))
            ro = occ[b:b+1].copy()
            ro[0, wall] = -1
            reveal = render_mx(mx.array(pos[b:b+1, t:t+1]), mx.array(pres[b:b+1]),
                               mx.array(ro), mx.array(colors[b:b+1]), noise=0)
            f = f.at[b, t].add(reveal[0, 0]-f[b, t])
            flags[b, t, wall] = 1
    f = mx.clip(f + .02*mx.random.normal(f.shape), 0, 1)
    return f, mx.array(cue), mx.array(pres), mx.array(occ), mx.array(flags), mx.array((pos-32)/32)


def validate_models(models, data, count, deadline):
    values = {name: [] for name in models}
    for i in range(count):
        if time.monotonic() >= deadline:
            return None
        look = random_look(data['occ'][i], 72000+i) if i % 3 else None
        args = [mx.array(x) for x in (frames(data, i, look)[None], cues(data, [i]),
                data['present'][i:i+1].astype(np.float32), data['occ'][i:i+1],
                look_flags(40, look)[None])]
        for name, model in models.items():
            dist = distribution(model(*args))[0:3]
            loss = nll(tuple(x[0] for x in dist), data['pos'][i])
            values[name].append(float(loss[:, data['present'][i]].mean()))
    return {k: float(np.mean(v)) for k, v in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='checkpoints/r5_pair')
    parser.add_argument('--warm', default='checkpoints/hard/final')
    parser.add_argument('--steps', type=int, default=1600, help='fixed total schedule; keep constant on resume')
    parser.add_argument('--stop-after', type=int, default=0, help='pause at this total step; keep --steps unchanged')
    parser.add_argument('--max-minutes', type=float, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=51024)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--val-count', type=int, default=8)
    parser.add_argument('--tiny', action='store_true', help='small scratch architecture, tests only; no warm start')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pause-at', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 45:
        parser.error('--max-minutes must be in (0,45]')
    if args.steps <= 0 or args.batch <= 0 or args.save_every <= 0:
        parser.error('steps, batch, save-every must be positive')
    started = time.monotonic()
    path = Path(args.out) / 'last.pkl'
    if path.exists() and not args.resume:
        raise FileExistsError(f'{path} exists; use --resume or a fresh output folder')
    if args.resume and not path.exists():
        raise FileNotFoundError(path)
    source = None
    arch = dict(enc_dim=32, hid=16, layers=1)
    if not args.tiny:
        with open(args.warm+'.state.pkl', 'rb') as f:
            old = pickle.load(f)
        arch = {k: old['cfg'][k] for k in ('enc_dim', 'hid', 'layers')}
        source = hashlib.sha256(Path(args.warm+'.weights.npz').read_bytes()).hexdigest()
    cfg = dict(architecture=arch, steps=args.steps, batch=args.batch, seed=args.seed, lr=args.lr,
               tiny=args.tiny, warm_sha256=source, version=1, training_split='hard_train',
               peek_probability=2/3, cue_sigma_px=.25, beta=.5, aux_fraction=.25)
    cfg['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                           for name in ('r5_train.py', 'r5_model.py', 'r5_data.py',
                                        'model.py', 'train.py', 'sim.py')}
    mx.random.seed(args.seed)
    models = make_models(cfg)
    if not args.tiny:
        with np.load(args.warm+'.weights.npz') as z:
            warm = dict(z)
        for model in models.values():
            model.warm_start(warm)
    else:
        # Even smoke starts from a common random original-format network.
        from model import Tracker
        original = Tracker(**arch)
        mx.eval(original.parameters())
        warm = array_tree(original.parameters())
        for model in models.values():
            model.warm_start(warm)
    opts = {name: optim.AdamW(learning_rate=args.lr, weight_decay=1e-4) for name in models}
    rng = np.random.default_rng(args.seed)
    step, history, active_seconds = 0, [], 0.
    if args.resume:
        loaded, state = load_pair(path)
        if state['config'] != cfg:
            raise ValueError('resume config mismatch; keep architecture, schedule, seed, batch, warm source fixed')
        models = loaded
        for name, opt in opts.items():
            opt.state = tree_unflatten([(k, mx.array(v)) for k, v in state['optimizers'][name].items()])
        step, history, active_seconds = state['step'], state['history'], state['active_seconds']
        rng.bit_generator.state = state['numpy_rng']
        print(f'RESUMED paired_step={step} optimizer_steps={[int(o.step.item()) for o in opts.values()]}', flush=True)
    tr = read_split('hard_train', 8 if args.tiny else None)
    va = read_split('hard_val', args.val_count)
    grads = {name: nn.value_and_grad(model, training_loss) for name, model in models.items()}
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save():
        mx.eval([m.parameters() for m in models.values()], [o.state for o in opts.values()])
        state = dict(config=cfg, step=step, history=history,
                     models={n: array_tree(m.parameters()) for n, m in models.items()},
                     optimizers={n: array_tree(o.state) for n, o in opts.items()},
                     numpy_rng=rng.bit_generator.state,
                     mlx_rng_rule='reseed with seed + paired step before every batch',
                     next_mlx_seed=args.seed+step,
                     active_seconds=active_seconds+time.monotonic()-started)
        atomic_pickle(path, state)
        print(f'SAVED {path} paired_step={step}', flush=True)

    print(f'PAIR_START step={step} warm_sha256={source} cfg={cfg}', flush=True)
    # Reserve 10 seconds for atomic save. Guard includes setup and validation.
    while step < args.steps and not STOP:
        if time.monotonic()-started >= max(0., args.max_minutes*60-10):
            print('WALL_GUARD reached', flush=True)
            break
        mx.random.seed(args.seed+step)
        inputs = batch(tr, rng.integers(0, len(tr['pos']), args.batch), rng)
        mx.eval(*inputs)
        losses = {}
        for name, model in models.items():
            opt = opts[name]
            opt.learning_rate = args.lr*(.1 + .9*.5*(1+np.cos(np.pi*step/args.steps)))
            loss, g = grads[name](model, *inputs, max(0., 1-step/max(1, .25*args.steps)))
            g, norm = optim.clip_grad_norm(g, 5.)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, loss)
            losses[name] = float(loss.item())
            if not np.isfinite(losses[name]):
                raise FloatingPointError(f'{name} nonfinite loss; prior checkpoint preserved')
        step += 1
        history.append(dict(step=step, loss=losses))
        if step % 10 == 0 or step <= 2:
            print(f'PAIR step={step} loss={losses} elapsed_s={time.monotonic()-started:.1f}', flush=True)
        if step % args.save_every == 0 or step == args.steps or (args.stop_after and step >= args.stop_after):
            save()
            if step == args.pause_at:
                print(f'PAUSED_FOR_KILL_TEST step={step}', flush=True)
                while not STOP and time.monotonic()-started < args.max_minutes*60-10:
                    time.sleep(.05)
            if args.val_count and time.monotonic()-started < args.max_minutes*60-20:
                val = validate_models(models, va, args.val_count, started+args.max_minutes*60-10)
                if val is not None:
                    history[-1]['val_nll_px'] = val
                    print(f'VAL step={step} pixel_nll={val}', flush=True)
            if args.stop_after and step >= args.stop_after:
                break
    save()
    print(f'PAIR_DONE paired_step={step} elapsed_s={time.monotonic()-started:.1f}', flush=True)


if __name__ == '__main__':
    main()
