"""Round 5 numerics/protocol. NumPy only at import; MuJoCo is loaded lazily.

MLX trains the networks; NumPy mirrors their forward pass for CPU simulation,
following the original exhibit. No R1-R4 modules or artifacts are modified.
"""
from itertools import permutations
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

TASKS = ('fwd', 'bwd', 'slow')
TARGETS = np.array([3., -3., 1.], dtype=np.float32)
OBS, ACT, INPUT, DT = 17, 6, 20, .05
# Fixed physical scaling, shared by every arm. No fitted/pretrained normalizer.
SCALE = np.array([1.] * 8 + [5.] * 9, dtype=np.float32)
CPG_LO = np.array([.5] + [0.] * 6 + [-np.pi] * 6 + [-1.] * 6)
CPG_HI = np.array([8.] + [2.] * 6 + [np.pi] * 6 + [1.] * 6)


def bank(split, seed=0, n=4):
    if not 0 <= seed < 100 or not 1 <= n <= 10000:
        raise ValueError('seed must be 0..99 and bank size 1..10000')
    bases = dict(train=1_000_000, cpg=3_000_000, val=5_000_000,
                 test=7_000_000, test_t2=9_000_000, smoke=13_000_000, legacy=10_000)
    if split == 'legacy':
        return np.arange(10000, 10000 + min(n, 10))
    return bases[split] + seed * 10000 + np.arange(n)


def inputs(obs, tids):
    obs, tids = np.asarray(obs), np.asarray(tids, dtype=int)
    assert obs.shape == (len(tids), OBS) and np.all((tids >= 0) & (tids < 3))
    return np.concatenate((np.clip(obs / SCALE, -10, 10), np.eye(3)[tids]), -1).astype(np.float32)


def reward(tid, velocity, action):
    return np.exp(-.5 * (np.asarray(velocity) - TARGETS[tid]) ** 2) - .03 * np.square(action).sum(-1)


def training_reward(tid, velocity, action, dense_weight=0.):
    """Evaluation reward stays fixed. Dense L1 error supplies a reversal signal far away.

    At v=+3 with a -3 target, the Gaussian slope is ~9e-8; the additional
    0.2/m/s training term retains a progress signal. No gradients flow through it.
    """
    return reward(tid, velocity, action) - dense_weight * np.abs(np.asarray(velocity)-TARGETS[tid])


def scheduled_lr(initial, update, updates, final_fraction=None):
    if final_fraction is None:
        return initial  # preserve turn-1 behavior
    fraction = min(1., update / max(1, updates-1))
    return initial * (1. - (1.-final_fraction)*fraction)


def deployment_params(state):
    return state['selection']['params'] if state.get('selection') is not None else state['params']


def reset_exploration(p, minimum, adapter_tid=None):
    """Reset only training exploration; deterministic actor outputs stay identical."""
    p = {k:v.copy() for k,v in p.items()}
    if adapter_tid is None:
        p['ls'] = np.maximum(p['ls'], minimum).astype(np.float32)
    else:
        key = f'd{adapter_tid}_ls'
        p[key] = (np.maximum(p['ls']+p[key], minimum)-p['ls']).astype(np.float32)
    return p


def init_params(seed, hidden=128):
    rng = np.random.default_rng(seed)
    p = {}
    for pre, out in [('a', ACT), ('c', 1)]:
        for layer, (ni, no) in enumerate([(INPUT, hidden), (hidden, hidden), (hidden, out)], 1):
            a = rng.normal(size=(max(ni, no), min(ni, no)))
            q, r = np.linalg.qr(a)
            q *= np.where(np.diag(r) >= 0, 1., -1.)
            if no < ni:
                q = q.T
            gain = .01 if layer == 3 and pre == 'a' else (1. if layer == 3 else np.sqrt(2.))
            p[f'{pre}{layer}w'] = (q[:no, :ni] * gain).astype(np.float32)
            p[f'{pre}{layer}b'] = np.zeros(no, np.float32)
    p['ls'] = np.full(ACT, -.5, np.float32)
    return p


def add_adapter(p, tid, rank, seed):
    assert tid in (1, 2) and rank > 0
    p = {k: v.copy() for k, v in p.items()}
    rng = np.random.default_rng(seed)
    for pre in ('a', 'c'):
        for i in (1, 2, 3):
            no, ni = p[f'{pre}{i}w'].shape
            prefix = f'd{tid}_{pre}{i}'
            # A random, B zero: initially identical policy, B has a nonzero gradient.
            p[prefix + 'A'] = (rng.normal(size=(rank, ni)) / np.sqrt(ni)).astype(np.float32)
            p[prefix + 'B'] = np.zeros((no, rank), np.float32)
            p[prefix + 'b'] = np.zeros(no, np.float32)
    p[f'd{tid}_ls'] = np.zeros(ACT, np.float32)
    return p


def forward(p, x, tids, xp=np, value=False):
    """Same algebra on NumPy and MLX. LoRA scale=1, explicit task routing."""
    pre = 'c' if value else 'a'
    h = x
    for i in (1, 2, 3):
        z = h @ p[f'{pre}{i}w'].T + p[f'{pre}{i}b']
        for tid in (1, 2):
            k = f'd{tid}_{pre}{i}'
            if k + 'A' in p:
                delta = (h @ p[k + 'A'].T) @ p[k + 'B'].T + p[k + 'b']
                z = z + delta * (tids == tid)[:, None]
        h = z if value and i == 3 else xp.tanh(z)
    return h[:, 0] if value else h


def logstd(p, tids, xp=np):
    out = xp.broadcast_to(p['ls'], (len(tids), ACT))
    for tid in (1, 2):
        if f'd{tid}_ls' in p:
            out = out + (tids == tid)[:, None] * p[f'd{tid}_ls']
    return xp.clip(out, -3., 0.)


def train_keys(p, method, tid):
    if method == 'adapter':
        return sorted(k for k in p if k.startswith(f'd{tid}_'))
    if method == 'dagger':
        return sorted(k for k in p if k.startswith('a'))
    return sorted(p)


def gae(rewards, values, done, gamma=.99, lam=.95):
    assert values.shape == (rewards.shape[0] + 1, rewards.shape[1])
    adv, last = np.zeros_like(rewards), np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        keep = 1. - done[t]
        delta = rewards[t] + gamma * values[t + 1] * keep - values[t]
        last = delta + gamma * lam * keep * last
        adv[t] = last
    return adv, adv + values[:-1]


def normalize_advantages(adv):
    return (adv-adv.mean()) / np.sqrt(adv.var()+1e-8)


class Replay:
    """Bounded per-task FIFO + balanced task sampling; persisted across additions."""
    def __init__(self, capacity=8192):
        self.x = np.zeros((3, capacity, INPUT), np.float32)
        self.y = np.zeros((3, capacity, ACT), np.float32)
        self.count, self.ptr = np.zeros(3, int), np.zeros(3, int)

    def add(self, x, y, tids):
        for a, b, tid in zip(x, y, tids):
            j = self.ptr[tid]
            self.x[tid, j], self.y[tid, j] = a, b
            self.ptr[tid] = (j + 1) % self.x.shape[1]
            self.count[tid] = min(self.count[tid] + 1, self.x.shape[1])

    def sample(self, rng, n):
        active = np.flatnonzero(self.count)
        assert len(active)
        tids = rng.choice(active, n)
        ix = np.array([rng.integers(self.count[t]) for t in tids])
        return self.x[tids, ix], self.y[tids, ix], tids.astype(np.int32)


def budget(base, teacher, student):
    # Each queried example costs one additional credit, including old-skill queries.
    return dict(base=base, teacher=teacher, student=student,
                addition=teacher + 2 * student, final=base + 2 * (teacher + 2 * student))


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.tmp', 'wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(path) + '.tmp', path)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)  # Only trusted, locally generated checkpoints.


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.tmp', 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())
    os.replace(str(path) + '.tmp', path)


class Deadline:
    def __init__(self, minutes):
        if not 0 < minutes <= 45:
            raise ValueError('--max-minutes must be in (0, 45]')
        self.start = time.monotonic()
        self.end = self.start + minutes * 60

    def check(self):
        if time.monotonic() >= self.end - min(2., (self.end - self.start) / 10):
            raise TimeoutError('wall guard: resumable work saved at previous atomic boundary')


def make_env(horizon=600):
    import gymnasium as gym
    return gym.make('HalfCheetah-v5', max_episode_steps=horizon)


def pack_envs(envs):
    import mujoco
    out = []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for e in envs:
        u = e.unwrapped
        state = np.empty(mujoco.mj_stateSize(u.model, spec))
        mujoco.mj_getState(u.model, u.data, state, spec)
        out.append(dict(state=state, elapsed=e._elapsed_steps, rng=u.np_random.bit_generator.state))
    return out


def restore_envs(envs, states):
    import mujoco
    for e, st in zip(envs, states):
        u = e.unwrapped
        mujoco.mj_setState(u.model, u.data, st['state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(u.model, u.data)
        e._elapsed_steps = st['elapsed']
        u.np_random.bit_generator.state = st['rng']


def schedule(t, order, segment):
    return order[min(t // segment, len(order) - 1)]


def sustained(mask, length):
    mask = np.asarray(mask, bool)
    if len(mask) < length:
        return None
    hits = np.flatnonzero(np.convolve(mask.astype(int), np.ones(length, int), 'valid') == length)
    return int(hits[0]) if len(hits) else None


def segment_metrics(v, heights, pitch, target):
    v, heights, pitch = map(np.asarray, (v, heights, pitch))
    good = np.abs(v - target) <= .5
    start = sustained(good, 10)
    angle = np.arctan2(np.sin(pitch), np.cos(pitch))
    fall = sustained((heights < .3) | (np.abs(angle) > 1.2), 5) is not None
    # Allow 0.5s for ordinary zero crossing, then require >=1s at near-zero speed.
    stall = sustained(np.abs(v[10:]) < .1, 20) is not None
    reached = start is not None
    return dict(tracking_fraction=float(good.mean()), speed_mae=float(np.abs(v - target).mean()),
                fall_proxy=bool(fall), stall=bool(stall), reached=bool(reached),
                settle_s=None if start is None else (start + 10) * DT,
                capped_settle_s=(len(v) if start is None else start + 10) * DT,
                failure=bool(fall or stall or not reached))


def episode(policy, seed, order, segment=200, deadline=None):
    """No reset at a command switch. Always executes every step; no abstention."""
    env = make_env(segment * len(order))
    records = dict(v=[], height=[], pitch=[], reward=[], task=[], body=[])
    try:
        obs, _ = env.reset(seed=int(seed))
        parents = env.unwrapped.model.body_parentid.tolist()
        for t in range(segment * len(order)):
            if deadline and t % 32 == 0:
                deadline.check()
            tid = schedule(t, order, segment)
            a = np.asarray(policy(obs, tid, t, t % segment))
            assert a.shape == (ACT,) and np.isfinite(a).all()
            obs, _, term, trunc, info = env.step(np.clip(a, -1, 1))
            data = env.unwrapped.data
            records['v'].append(float(info['x_velocity']))
            records['height'].append(float(data.xpos[1, 2]))
            records['pitch'].append(float(data.qpos[2]))
            records['reward'].append(float(reward(tid, info['x_velocity'], np.clip(a, -1, 1))))
            records['task'].append(tid)
            records['body'].append(data.xpos[:, [0, 2]].tolist())
            if (term or trunc) and t + 1 < segment * len(order):
                raise RuntimeError('unexpected early episode termination')
        rows = []
        for i, tid in enumerate(order):
            sl = slice(i * segment, (i + 1) * segment)
            row = segment_metrics(records['v'][sl], records['height'][sl], records['pitch'][sl], TARGETS[tid])
            row.update(task=TASKS[tid], switch=i > 0, source=None if i == 0 else TASKS[order[i-1]],
                       return_=float(np.sum(records['reward'][sl])))
            rows.append(row)
        return dict(seed=int(seed), order=list(order), segments=rows, trace=records, parents=parents)
    finally:
        env.close()


def network_policy(params, teachers=None):
    def policy(obs, tid, t, age):
        p = params if teachers is None else teachers[tid]
        return forward(p, inputs(obs[None], [tid]), np.array([tid]))[0]
    return policy


def validation_score(params, seen, seeds, horizon, deadline):
    """Equal task weight; lexicographic worst-task, mean-task, then negative MAE."""
    policy = network_policy(params)
    tasks = {}
    for tid in seen:
        rows = [episode(policy, int(s), (tid,), horizon, deadline)['segments'][0] for s in seeds]
        tasks[TASKS[tid]] = dict(tracking=float(np.mean([r['tracking_fraction'] for r in rows])),
                                mae=float(np.mean([r['speed_mae'] for r in rows])),
                                fall_proxy=float(np.mean([r['fall_proxy'] for r in rows])))
    scores = [r['tracking'] for r in tasks.values()]
    score = (min(scores), float(np.mean(scores)), -float(np.mean([r['mae'] for r in tasks.values()])))
    return dict(score=score, tasks=tasks, seeds=[int(s) for s in seeds], env_steps=len(seen)*len(seeds)*horizon)


def gait(theta, t):
    return np.clip(theta[13:19] + theta[1:7] * np.sin(2 * np.pi * theta[0] * t * DT + theta[7:13]), -1, 1)


def cpg_policy(thetas, reset_phase=False):
    return lambda obs, tid, t, age: gait(thetas[tid], age if reset_phase else t)


def orders(n):
    return list(permutations(range(n))) if n > 1 else [(0,)]


def size_counts(p, nskills):
    actor = {k: v for k, v in p.items() if k.startswith('a') or ('_a' in k)}
    # Ignore actor exploration std at deterministic deployment; include it in training.
    return dict(actor_parameters=sum(v.size for v in actor.values()),
                actor_bytes=sum(v.nbytes for v in actor.values()),
                training_parameters=sum(v.size for v in p.values()),
                training_parameter_bytes=sum(v.nbytes for v in p.values()), skills=nskills)


def retention(rows, minimum_acquisition=.05):
    """Track competence loss from each skill's acquisition snapshot, not a reward ratio."""
    result = []
    for method in ('adapter', 'dagger', 'joint'):
        for task_id, task in enumerate(TASKS):
            anchor = [r['tracking_fraction'] for r in rows if r['method'] == method and
                      r['stage'] == task_id + 1 and r['task'] == task and r['kind'] == 'isolated']
            if not anchor:
                continue
            base = float(np.mean(anchor))
            for stage in range(task_id + 1, 4):
                vals = [r['tracking_fraction'] for r in rows if r['method'] == method and
                        r['stage'] == stage and r['task'] == task and r['kind'] == 'isolated']
                if vals:
                    score = float(np.mean(vals))
                    result.append(dict(method=method, task=task, stage=stage, acquisition=base,
                                       competence=score, loss=base-score,
                                       qualified=base >= minimum_acquisition,
                                       qualification_threshold=minimum_acquisition,
                                       retained_ratio=None if base < minimum_acquisition else score/base))
    return result
