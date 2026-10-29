"""Round-5 MLX updates and resumable PPO / replay DAgger. Runner only.

All randomness is NumPy-owned, including initialization, actions and batches.
Checkpoints contain simulator integration state as well as optimizer/RNG/replay.
"""
import hashlib
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

from r5_core import (ACT, INPUT, Replay, atomic_pickle, bank, forward, gae, inputs,
                     logstd, make_env, pack_envs, read_pickle, restore_envs, reward, train_keys)


def fingerprint(p):
    digest = hashlib.sha256()
    for k in sorted(p):
        digest.update(k.encode())
        digest.update(p[k].tobytes())
    return digest.hexdigest()


def logp(a, mu, ls):
    return (-.5 * ((a - mu) / mx.exp(ls)) ** 2 - ls - .5 * np.log(2*np.pi)).sum(-1)


def safe_clip(grads, limit=.5):
    # Epsilon is INSIDE sqrt; zero gradients must have finite derivatives too.
    norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()) + 1e-12)
    scale = mx.minimum(1., limit / norm)
    return {k: g * scale for k, g in grads.items()}


def train(start, path, cfg, deadline, teachers=None, replay=None, pause=False):
    """Exact rollout-multiple budget. Returns complete checkpoint; guard raises TimeoutError.

    Rollout+optimizer update is the atomic unit. An interrupted update replays from
    the last saved simulator/RNG state. Max wall overshoot is one such unit + save.
    """
    path = Path(path)
    expected = dict(cfg, start_sha256=fingerprint(start), schema=1)
    method, seen = cfg['method'], cfg['seen']
    n, T, steps = cfg['envs'], cfg['rollout'], cfg['steps']
    per_update = n * T
    assert steps > 0 and steps % per_update == 0
    rng = np.random.default_rng(cfg['seed'])
    keys = train_keys(start, method, seen[-1])
    frozen = {k: mx.array(v) for k, v in start.items() if k not in keys}
    variables = {k: mx.array(start[k]) for k in keys}
    opt = optim.Adam(learning_rate=cfg['lr'])
    opt.init(variables)
    update, history = 0, []
    cost = dict(env_steps=0, teacher_queries=0, gradient_steps=0, gradient_examples=0,
                actor_forward_examples=0, critic_forward_examples=0, wall_s=0.)
    state = None
    if path.exists():
        state = read_pickle(path)
        if state['config'] != expected:
            raise ValueError(f'checkpoint config/input changed: {path}; use a fresh tag')
        if state['complete']:
            print(f'[r5 skip] {path.name} complete', flush=True)
            return state
        variables = {k: mx.array(state['params'][k]) for k in keys}
        opt.state = tree_map(mx.array, state['optimizer'])
        rng.bit_generator.state = state['rng']
        update, history, cost = state['update'], state['history'], state['cost']
        replay = state['replay']
        print(f'[r5 resume] {path.name} update={update} step={cost["env_steps"]}', flush=True)
    if method == 'dagger' and replay is None:
        replay = Replay(cfg['replay_capacity'])
    envs = [make_env(cfg['horizon']) for _ in range(n)]
    train_seeds = bank('train', cfg['experiment_seed'], 10000)
    obs = np.stack([e.reset(seed=int(train_seeds[rng.integers(len(train_seeds))]))[0] for e in envs])
    age = np.zeros(n, int)
    if state:
        # Initial reset above is only wrapper initialization: restore the RNG as well.
        rng.bit_generator.state = state['rng']
        restore_envs(envs, state['env_state'])
        obs, age = state['obs'].copy(), state['age'].copy()
    phase = np.arange(n)
    if state is None:
        # Every worker has a different initial command, switching cyclically.
        phase = np.arange(n)
    t0, previous_wall = time.monotonic(), cost['wall_s']

    def merged(v=None):
        return dict(frozen, **(variables if v is None else v))

    def save(complete=False):
        p = {k: np.array(v) for k, v in merged().items()}
        if not all(np.isfinite(v).all() for v in p.values()):
            raise FloatingPointError('nonfinite parameters; previous checkpoint retained')
        cost['wall_s'] = previous_wall + time.monotonic() - t0
        st = dict(params=p, optimizer=tree_map(np.array, opt.state), config=expected,
                  rng=rng.bit_generator.state, mlx_seed=cfg['seed'],
                  randomness='all stochastic operations use saved NumPy RNG; MLX has no random ops',
                  update=update, step=cost['env_steps'], complete=complete, history=history,
                  cost=cost.copy(), replay=replay, env_state=pack_envs(envs), obs=obs.copy(), age=age.copy(),
                  trainable_keys=keys)
        atomic_pickle(path, st)
        return st

    def ppo_loss(v, x, tids, actions, old_lp, adv, returns):
        p = merged(v)
        lp = logp(actions, forward(p, x, tids, mx), logstd(p, tids, mx))
        ratio = mx.exp(mx.clip(lp - old_lp, -20., 20.))
        pg = -mx.minimum(ratio * adv, mx.clip(ratio, .8, 1.2) * adv).mean()
        vl = ((forward(p, x, tids, mx, value=True) - returns) ** 2).mean()
        kl = ((ratio - 1) - (lp - old_lp)).mean()
        return pg + .5 * vl, (pg, vl, kl)

    def imitation_loss(v, x, tids, targets):
        return mx.mean((forward(merged(v), x, tids, mx) - targets) ** 2)

    grad = mx.value_and_grad(imitation_loss if method == 'dagger' else ppo_loss)
    try:
        while update < steps // per_update:
            deadline.check()
            p = {k: np.array(v) for k, v in merged().items()}
            X = np.zeros((T, n, INPUT), np.float32)
            A = np.zeros((T, n, ACT), np.float32)
            R, D = np.zeros((T, n), np.float32), np.zeros((T, n), np.float32)
            V = np.zeros((T + 1, n), np.float32)
            IDS = np.zeros((T, n), np.int32)
            for t in range(T):
                ids = np.array(seen)[(age // cfg['switch_steps'] + phase) % len(seen)]
                x = inputs(obs, ids)
                mu = forward(p, x, ids)
                sigma = .1 if method == 'dagger' else np.exp(logstd(p, ids))
                a = (mu + sigma * rng.normal(size=mu.shape)).astype(np.float32)
                X[t], A[t], IDS[t] = x, a, ids
                cost['actor_forward_examples'] += n
                if method == 'dagger':
                    y = np.empty_like(a)
                    for tid in seen:
                        ix = ids == tid
                        if ix.any():
                            y[ix] = forward(teachers[tid], x[ix], ids[ix])
                    replay.add(x, y, ids)
                    cost['teacher_queries'] += n  # one teacher, not all teachers, per state
                else:
                    V[t] = forward(p, x, ids, value=True)
                    cost['critic_forward_examples'] += n
                for i, e in enumerate(envs):
                    ac = np.clip(a[i], -1, 1)
                    o, _, term, trunc, info = e.step(ac)
                    R[t, i] = reward(ids[i], info['x_velocity'], ac)
                    age[i] += 1
                    if term or trunc:
                        if method != 'dagger' and trunc and not term:
                            # Hypothetical continuation command at truncation, not reset command.
                            nextid = seen[(age[i] // cfg['switch_steps'] + phase[i]) % len(seen)]
                            R[t, i] += .99 * forward(p, inputs(o[None], [nextid]), np.array([nextid]), value=True)[0]
                            cost['critic_forward_examples'] += 1
                        D[t, i] = 1
                        o, _ = e.reset(seed=int(train_seeds[rng.integers(len(train_seeds))]))
                        age[i] = 0
                    obs[i] = o
            cost['env_steps'] += per_update
            opt.learning_rate = cfg['lr']  # constant schedule across resumptions and methods
            losses = []
            if method == 'dagger':
                for _ in range(cfg['dagger_batches']):
                    xx, yy, ii = replay.sample(rng, cfg['minibatch'])
                    loss, g = grad(variables, mx.array(xx), mx.array(ii), mx.array(yy))
                    variables = opt.apply_gradients(safe_clip(g), variables)
                    mx.eval(variables, opt.state, loss)
                    losses.append(float(loss))
                    cost['gradient_steps'] += 1
                    cost['gradient_examples'] += len(xx)
                    cost['actor_forward_examples'] += len(xx)
                kl = None
            else:
                ids = np.array(seen)[(age // cfg['switch_steps'] + phase) % len(seen)]
                V[T] = forward(p, inputs(obs, ids), ids, value=True)
                cost['critic_forward_examples'] += n
                adv, ret = gae(R, V, D)
                adv = adv.reshape(-1)
                adv = (adv - adv.mean()) / np.sqrt(adv.var() + 1e-8)
                xx, ii, aa = mx.array(X.reshape(-1, INPUT)), mx.array(IDS.reshape(-1)), mx.array(A.reshape(-1, ACT))
                old_lp = mx.stop_gradient(logp(aa, forward(merged(), xx, ii, mx), logstd(merged(), ii, mx)))
                mx.eval(old_lp)
                cost['actor_forward_examples'] += per_update
                kl = 0.
                for _ in range(cfg['epochs']):
                    perm = rng.permutation(per_update)
                    for j in range(0, per_update, cfg['minibatch']):
                        ix = perm[j:j+cfg['minibatch']]
                        ix_mx = mx.array(ix)
                        (loss, aux), g = grad(variables, xx[ix_mx], ii[ix_mx], aa[ix_mx], old_lp[ix_mx],
                                             mx.array(adv[ix]), mx.array(ret.reshape(-1)[ix]))
                        variables = opt.apply_gradients(safe_clip(g), variables)
                        mx.eval(variables, opt.state, loss, aux)
                        losses.append(float(loss))
                        kl += float(aux[2])
                        cost['gradient_steps'] += 1
                        cost['gradient_examples'] += len(ix)
                        cost['actor_forward_examples'] += len(ix)
                        cost['critic_forward_examples'] += len(ix)
                kl /= len(losses)
            if not np.isfinite(losses).all():
                raise FloatingPointError('nonfinite loss; last checkpoint retained')
            update += 1
            rec = dict(update=update, env_steps=cost['env_steps'], loss=float(np.mean(losses)),
                       reward_mean=float(np.mean(R)), kl=kl,
                       replay_counts=None if replay is None else replay.count.tolist())
            history.append(rec)
            st = save(update == steps // per_update)
            if update == 1 or update % 10 == 0 or st['complete']:
                print(f'[r5 train] {path.name} {rec}', flush=True)
            if pause:
                print('R5_PAUSED_AFTER_CHECKPOINT', flush=True)
                while True:
                    deadline.check()
                    time.sleep(.1)
        return st
    finally:
        for e in envs:
            e.close()
