"""Paired recurrent / memoryless PPO in MLX, with complete resumable simulator state.

All MLX imports are inside main/runtime functions: NumPy tests can import helpers.
The main loop is TIME-budgeted. --max-updates is only a smoke/debug ceiling.
"""
import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
import numpy as np

from racing.sim import DT
from r5_common import Budget, atomic_npz, emit
from r5_env import SurfaceRace, finish_flags, make_bank
from r5_eval import validate
from r5_policy import DEFAULT_WARM, HIDDEN, Policy, gae, init_params, save_selected, sequence, step


def chunks(a, size):
    """[time, env, ...] -> [unroll, chunk*env, ...], retaining temporal order."""
    t, e = a.shape[:2]
    assert t % size == 0
    tail = a.shape[2:]
    return a.reshape((t // size, size, e) + tail).transpose((1, 0, 2) + tuple(range(3, a.ndim + 1))).reshape((size, t // size * e) + tail)


def rollout(p, env, rng, h, horizon, episode_steps, recurrent):
    obs = env.observe()
    e = env.E
    store = {k: [] for k in ('obs', 'act', 'logp', 'rew', 'done', 'value', 'h', 'reset')}
    reset = np.zeros(e, np.float32)
    logstd = p['log_std']
    std = np.exp(logstd)
    crashes = finishes = 0
    for _ in range(horizon):
        store['obs'].append(obs.copy())
        store['h'].append(h.copy())
        store['reset'].append(reset.copy())
        mean, v, h = step(p, obs, h, recurrent)
        a = mean + std * rng.standard_normal((e, 2)).astype(np.float32)
        logp = (-.5 * ((a - mean) / std) ** 2 - logstd - .5 * np.log(2 * np.pi)).sum(-1)
        obs, dp, _ = env.step(a)
        crash = env.base.crashed.copy()
        finish = finish_flags(env.base.prog, env.base.bank.length[env.base.track_id], crash)
        end = crash | finish | (env.base.t >= episode_steps)
        # Time limit is a real finite-horizon terminal for all methods, not a truncation.
        reward = .1 * dp - 20. * crash + 10. * finish - .01
        for k, val in [('act', a), ('logp', logp), ('value', v), ('rew', reward), ('done', end)]:
            store[k].append(val.copy())
        crashes += int(crash.sum())
        finishes += int(finish.sum())
        if end.any():
            obs = env.reset(rng, mask=end)
            h[end] = 0
        reset = end.astype(np.float32)
    next_v = step(p, obs, h, recurrent)[1]
    b = {k: np.asarray(v, np.float32) for k, v in store.items()}
    b['adv'], b['ret'] = gae(b['rew'], b['value'], next_v, b['done'])
    b['adv'] = (b['adv'] - b['adv'].mean()) / max(float(b['adv'].std()), 1e-6)
    return b, h, dict(crashes=crashes, finishes=finishes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r5_seed0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-minutes', type=float, default=18)
    ap.add_argument('--max-updates', type=int, default=1000000000)
    ap.add_argument('--envs', type=int, default=128)
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--unroll', type=int, default=16)
    ap.add_argument('--minibatch-seqs', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n-train', type=int, default=64, help='tracks per geometry')
    ap.add_argument('--n-val', type=int, default=16, help='full validation = six groups times this count')
    ap.add_argument('--episode-steps', type=int, default=600)
    ap.add_argument('--eval-every', type=int, default=0, help='0 = calibrate to about 60 seconds')
    ap.add_argument('--save-every', type=int, default=0, help='0 = about 30 seconds; use 1 for kill test')
    ap.add_argument('--warm', default=DEFAULT_WARM)
    ap.add_argument('--resume')
    ap.add_argument('--no-compile', action='store_true', help='diagnostic fallback; timing will change')
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    if not a.tag.startswith('r5_') or '/' in a.tag:
        raise ValueError('tag must begin r5_ and contain no path separators')
    if a.horizon % a.unroll or (a.horizon // a.unroll * a.envs) % a.minibatch_seqs:
        raise ValueError('unroll must divide horizon; minibatch must divide sequence count')
    if min(a.envs, a.horizon, a.unroll, a.epochs, a.n_train, a.n_val, a.episode_steps) <= 0:
        raise ValueError('positive sizes required')
    state_path = Path(f'checkpoints/{a.tag}_state.npz')
    if state_path.exists() and not a.resume:
        raise ValueError('state exists: use --resume or a fresh tag')
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    mx.random.seed(a.seed)
    cfg = {k: v for k, v in vars(a).items() if k not in ('max_minutes', 'max_updates', 'resume', 'tag', 'eval_every', 'save_every')}
    cfg['warm_sha256'] = hashlib.sha256(Path(a.warm).read_bytes()).hexdigest()
    bank, val_bank = make_bank('train', a.n_train), make_bank('val', a.n_val)
    arms = {}
    for name in ('rnn', 'ff'):
        rng = np.random.default_rng(a.seed + 61001)
        env = SurfaceRace(bank, a.envs)
        env.reset(rng)
        p = {k: mx.array(v) for k, v in init_params(a.seed, a.warm).items()}
        opt = optim.Adam(learning_rate=a.lr)
        opt.init(p)
        arms[name] = dict(p=p, opt=opt, rng=rng, env=env, h=np.zeros((a.envs, HIDDEN), np.float32),
                          best=-1e30, best_step=-1, active_seconds=0., gradient_steps=0,
                          selected_p={}, selected_meta={})
    update = 0
    if a.resume:
        with np.load(a.resume, allow_pickle=False) as z:
            if json.loads(str(z['config'])) != cfg:
                raise ValueError('resume configuration mismatch (saved config is authoritative)')
            update = int(z['update'])
            for name, arm in arms.items():
                prefix = name + '/'
                arm['p'] = {k[len(prefix)+2:]: mx.array(z[k]) for k in z.files if k.startswith(prefix + 'p/')}
                arm['opt'].state = tree_unflatten([(k[len(prefix)+2:], mx.array(z[k])) for k in z.files if k.startswith(prefix + 'o/')])
                arm['rng'].bit_generator.state = json.loads(str(z[prefix + 'rng']))
                arm['h'] = z[prefix + 'h'].copy()
                arm['best'] = float(z[prefix + 'best'])
                arm['best_step'] = int(z[prefix + 'best_step'])
                arm['active_seconds'] = float(z[prefix + 'active_seconds'])
                arm['gradient_steps'] = int(z[prefix + 'gradient_steps'])
                arm['selected_p'] = {k[len(prefix)+7:]: z[k].copy() for k in z.files if k.startswith(prefix + 'best_p/')}
                arm['selected_meta'] = json.loads(str(z[prefix + 'selected_meta']))
                save_selected(f'checkpoints/{a.tag}_{name}_selected.npz', arm['selected_p'], arm['selected_meta'])
                arm['env'].restore({k[len(prefix)+4:]: z[k] for k in z.files if k.startswith(prefix + 'env/')})
            # MLX 0.32: the state container has no item assignment. Mutate each key.
            for k in range(len(mx.random.state)):
                mx.random.state[k][:] = mx.array(z[f'mx_rng/{k}'])
        emit(dict(event='resumed', update=update, state=str(a.resume)))
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    history = f'logs/r5/{a.tag}_train.jsonl'

    def loss(p, obs, h, reset, act, old_lp, adv, ret, recurrent):
        mean, value, _ = sequence(p, obs, h, reset, recurrent, mx)
        ls = p['log_std']
        logp = (-.5 * ((act - mean) / mx.exp(ls)) ** 2 - ls - .5 * np.log(2 * np.pi)).sum(-1)
        ratio = mx.exp(mx.clip(logp - old_lp, -20, 20))
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, .8, 1.2) * adv).mean()
        vl = ((value - ret) ** 2).mean()
        entropy = (ls + .5 * np.log(2 * np.pi * np.e)).sum()
        return pg + .25 * vl - .001 * entropy

    # p is an explicit dynamic input, never a captured model. No frozen-weight compile.
    gradients = {}
    for name in arms:
        recurrent = name == 'rnn'
        def lf(p, obs, h, reset, act, old_lp, adv, ret, recurrent=recurrent):
            return loss(p, obs, h, reset, act, old_lp, adv, ret, recurrent)
        vg = mx.value_and_grad(lf)
        gradients[name] = vg if a.no_compile else mx.compile(vg)

    def save():
        saved = dict(config=json.dumps(cfg), update=update, env_steps_per_arm=update * a.envs * a.horizon)
        for name, arm in arms.items():
            mx.eval(arm['p'], arm['opt'].state)
            prefix = name + '/'
            saved.update({prefix + 'p/' + k: np.array(v) for k, v in arm['p'].items()})
            saved.update({prefix + 'o/' + k: np.array(v) for k, v in tree_flatten(arm['opt'].state)})
            saved.update({prefix + 'env/' + k: v for k, v in arm['env'].state().items()})
            for k in ('h', 'best', 'best_step', 'active_seconds', 'gradient_steps'):
                saved[prefix + k] = arm[k]
            saved[prefix + 'rng'] = json.dumps(arm['rng'].bit_generator.state)
            saved[prefix + 'selected_meta'] = json.dumps(arm['selected_meta'])
            saved.update({prefix + 'best_p/' + k: v for k, v in arm['selected_p'].items()})
        saved.update({f'mx_rng/{k}': np.array(v) for k, v in enumerate(mx.random.state)})
        atomic_npz(state_path, **saved)

    def selection():
        results = {}
        for name, arm in arms.items():
            start = time.monotonic()
            p = {k: np.array(v) for k, v in arm['p'].items()}
            results[name] = validate(Policy(p, name == 'rnn'), val_bank, a.episode_steps, budget)
            arm['active_seconds'] += time.monotonic() - start
        # Commit selection only after BOTH full validation sets complete.
        for name, v in results.items():
            arm = arms[name]
            if v['score'] > arm['best']:
                arm['best'], arm['best_step'] = v['score'], update
                arm['selected_p'] = {k: np.array(x).copy() for k, x in arm['p'].items()}
                arm['selected_meta'] = dict(arm=name, step=update, config=cfg, validation=v,
                                   prior_data='shared existing PPO+DR warm start; see DESIGN_R5.md')
                save_selected(f'checkpoints/{a.tag}_{name}_selected.npz', arm['selected_p'], arm['selected_meta'])
            emit(dict(event='validation', arm=name, update=update, best_step=arm['best_step'], **v), history)

    val_seconds = 2.
    if update == 0:
        start = time.monotonic()
        selection()
        val_seconds = time.monotonic() - start
        save()
    pair_seconds = 3.
    eval_every = a.eval_every or 50
    save_every = a.save_every or 25
    start_update = update
    last_save = update
    while update < a.max_updates and not stopped[0] and budget.fits(pair_seconds * 1.3, reserve=3):
        begin = time.monotonic()
        metrics = {}
        for name in (('rnn', 'ff') if update % 2 == 0 else ('ff', 'rnn')):
            arm = arms[name]
            tick = time.monotonic()
            p_np = {k: np.array(v) for k, v in arm['p'].items()}
            b, arm['h'], rec = rollout(p_np, arm['env'], arm['rng'], arm['h'], a.horizon, a.episode_steps, name == 'rnn')
            data = {k: chunks(b[k], a.unroll) for k in ('obs', 'reset', 'act', 'logp', 'adv', 'ret')}
            h0 = b['h'][::a.unroll].reshape(-1, HIDDEN)
            nseq = len(h0)
            for _ in range(a.epochs):
                perm = arm['rng'].permutation(nseq)
                for off in range(0, nseq, a.minibatch_seqs):
                    ix = perm[off:off + a.minibatch_seqs]
                    l, g = gradients[name](arm['p'], mx.array(data['obs'][:, ix]), mx.array(h0[ix]),
                        mx.array(data['reset'][:, ix]), mx.array(data['act'][:, ix]), mx.array(data['logp'][:, ix]),
                        mx.array(data['adv'][:, ix]), mx.array(data['ret'][:, ix]))
                    # Epsilon INSIDE sqrt: finite derivatives even at all-zero gradient.
                    norm = mx.sqrt(sum((v * v).sum() for _, v in tree_flatten(g)) + 1e-12)
                    scale = mx.minimum(1., .5 / norm)
                    g = {k: v * scale for k, v in g.items()}
                    arm['opt'].update(arm['p'], g)
                    arm['p']['log_std'] = mx.clip(arm['p']['log_std'], -3.5, -.7)
                    mx.eval(l, norm, arm['p'], arm['opt'].state)
                    if not np.isfinite(float(l)) or not np.isfinite(float(norm)):
                        raise FloatingPointError(f'{name} nonfinite loss/gradient at update {update}')
                    arm['gradient_steps'] += 1
            elapsed = time.monotonic() - tick
            arm['active_seconds'] += elapsed
            metrics[name] = dict(**rec, loss=float(l), grad_norm=float(norm), seconds=elapsed,
                                 env_steps_s=a.envs * a.horizon / elapsed,
                                 active_seconds=arm['active_seconds'], gradient_steps=arm['gradient_steps'])
        update += 1
        measured = time.monotonic() - begin
        pair_seconds = max(measured, .8 * pair_seconds + .2 * measured)
        if update - start_update in (1, 5, 10):
            eval_every = a.eval_every or max(1, round(60 / pair_seconds))
            save_every = a.save_every or max(1, round(30 / pair_seconds))
            emit(dict(event='rate_plan', paired_update_s=pair_seconds,
                      projected_additional_updates=max(0, int(budget.remaining / (pair_seconds + val_seconds / eval_every))),
                      env_steps_per_pair=2 * a.envs * a.horizon, eval_every=eval_every,
                      max_updates_is_only_ceiling=a.max_updates), history)
        emit(dict(event='update', update=update, env_steps_per_arm=update*a.envs*a.horizon,
                  wall_seconds=budget.elapsed, arms=metrics), history)
        if update % eval_every == 0 and budget.fits(val_seconds * 1.5, reserve=4):
            ts = time.monotonic()
            try:
                selection()
            except TimeoutError:
                emit(dict(event='validation_incomplete_no_selection', update=update), history)
            val_seconds = max(val_seconds, time.monotonic() - ts)
        if update - last_save >= save_every:
            save()
            last_save = update
    # Last-generation selection is optional; partial validation can never win.
    if budget.fits(val_seconds * 1.5, reserve=3) and update > start_update:
        try:
            selection()
        except TimeoutError:
            emit(dict(event='final_validation_incomplete_no_selection', update=update), history)
    save()
    emit(dict(event='finished', update=update, wall_seconds=budget.elapsed,
              allowance_seconds=budget.seconds, stop='signal' if stopped[0] else ('ceiling' if update >= a.max_updates else 'wall_guard'),
              selected={k: v['best_step'] for k, v in arms.items()},
              arm_active_seconds={k: v['active_seconds'] for k, v in arms.items()},
              state=str(state_path)), history)
    if update == start_update:
        raise SystemExit('No optimizer updates: setup exceeded allowance; use measured timings before main run.')


if __name__ == '__main__':
    main()
