"""Matched strict-lap evaluation, complete six-group validation, event accounting."""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from racing.sim import DT
from r5_baseline import AdaptiveRay
from r5_common import Budget, GROUPS, emit, write_json
from r5_env import SurfaceRace, finish_flags, fixed_surfaces, make_bank, normal_envelope, suite
from r5_policy import Legacy, load_selected

ADAPT_CAP = 5.0
STABLE_STEPS = 5


class Events:
    """Pre-register 2 patch transitions / 1 switch; never drop an unreached event."""
    def __init__(self, modes):
        n = len(modes)
        self.planned = np.arange(2)[None, :] < np.where(modes == 0, 2, np.where(modes == 1, 1, 0))[:, None]
        self.status = np.full((n, 2), 'unreached', dtype='<U16')
        self.start = np.full((n, 2), -1, int)
        self.latency = np.full((n, 2), ADAPT_CAP)
        self.good = np.zeros((n, 2), int)
        self.count = np.zeros(n, int)
        self.direction = np.zeros((n, 2), int)

    def update(self, t, changed, direction, normal, terminal):
        for e in np.flatnonzero(changed):
            self.status[e, self.status[e] == 'pending'] = 'next_change'
            j = self.count[e]
            if j < 2 and self.planned[e, j]:
                self.start[e, j] = t
                self.status[e, j] = 'pending'
                self.direction[e, j] = int(direction[e])
                self.count[e] += 1
        pending = self.status == 'pending'
        self.good = np.where(pending & normal[:, None], self.good + 1, 0)
        # Confirm a sustained half-second within the envelope; report its onset.
        recovered = pending & (self.good >= STABLE_STEPS) & ~terminal[:, None]
        delay = (t - self.start - STABLE_STEPS + 2) * DT
        self.latency[recovered] = np.clip(delay[recovered], 0, ADAPT_CAP)
        self.status[recovered] = 'recovered'
        pending = self.status == 'pending'
        self.status[pending & terminal[:, None]] = 'terminated'
        self.status[(self.status == 'pending') & ((t - self.start + 1) * DT >= ADAPT_CAP)] = 'window_expired'

    def finish(self):
        self.status[self.status == 'pending'] = 'censored'

    def rows(self, e):
        return [dict(slot=j, status=str(self.status[e, j]), exposed=bool(self.start[e, j] >= 0),
                     change_step=int(self.start[e, j]), direction=int(self.direction[e, j]),
                     capped_latency_s=float(self.latency[e, j])) for j in range(2) if self.planned[e, j]]


def run(controller, bank, ids, modes, scenario_seed, max_steps=600, record=False, deadline=None, repeat_scenarios=0):
    n = len(ids)
    env = SurfaceRace(bank, n)
    obs = env.reset(np.random.default_rng(scenario_seed), ids=ids, modes=modes)
    if repeat_scenarios and n % repeat_scenarios:
        raise ValueError('candidate batch must be a multiple of scenario count')
    fixed_surfaces(env, ids, modes, scenario_seed)
    active_steps = 0
    active = np.ones(n, bool)
    finished = np.zeros(n, bool)
    crashed = np.zeros(n, bool)
    times = np.full(n, max_steps * DT)
    lengths = bank.length[ids]
    prev_mu = env.last_grip.copy()
    events = Events(modes)
    history = []
    for t in range(max_steps):
        if deadline is not None and t % 16 == 0 and not deadline.fits(0, reserve=.5):
            raise TimeoutError('incomplete evaluation discarded; wall guard reached')
        mu = env.grip_now()
        changed = active & (np.abs(mu - prev_mu) > 1e-7)
        direction = np.sign(mu - prev_mu)
        action = controller(obs)
        if action.shape != (n, 2) or not np.isfinite(action).all():
            raise ValueError('controller produced nonfinite or misshaped actions')
        env.base.done[~active] = True
        active_steps += int(active.sum())
        obs, _, _ = env.step(action)
        f = active & finish_flags(env.base.prog, lengths, env.base.crashed)
        c = active & env.base.crashed
        terminal = f | c | ((t + 1) == max_steps)
        events.update(t, changed, direction, normal_envelope(env) if (events.status == 'pending').any() or changed.any() else np.zeros(n, bool), terminal)
        finished |= f
        crashed |= c
        times[f] = (t + 1) * DT
        if record:
            history.append(np.column_stack([env.base.snapshot(), env.last_grip,
                                            np.abs(env.base.lat) / (bank.half_width[ids] - 1), active]))
        active &= ~(f | c)
        prev_mu = mu.copy()
        if not active.any():
            break
    events.finish()
    progress = np.clip(env.base.prog / lengths, 0, 1)
    result = dict(finished=finished, crashed=crashed, lap_penalty_s=times, progress=progress,
                  utility=2 * finished - 2 * crashed + progress - times / (max_steps * DT),
                  events=events, steps=active_steps, slot_steps=(t + 1) * n)
    if record:
        result['trajectory'] = np.stack(history)
    return result


def summarize(r, mask=None):
    mask = np.ones(len(r['finished']), bool) if mask is None else mask
    f, c = r['finished'][mask], r['crashed'][mask]
    ev = r['events']
    planned = ev.planned & mask[:, None]
    exposed = planned & (ev.start >= 0)
    recovered = planned & (ev.status == 'recovered')
    n = int(mask.sum())
    return dict(n=n, completion=float(f.mean()), crash_rate=float(c.mean()),
                collisions_per_attempt=float(c.mean()), timeout_rate=float((~f & ~c).mean()),
                mean_finished_lap_s=float(r['lap_penalty_s'][mask][f].mean()) if f.any() else None,
                mean_dnf_penalized_lap_s=float(r['lap_penalty_s'][mask].mean()),
                mean_progress=float(r['progress'][mask].mean()), utility=float(r['utility'][mask].mean()),
                events_planned=int(planned.sum()), events_exposed=int(exposed.sum()),
                events_recovered=int(recovered.sum()),
                event_exposure_coverage=float(exposed.sum() / planned.sum()) if planned.any() else None,
                event_recovery_coverage=float(recovered.sum() / planned.sum()) if planned.any() else None,
                adaptation_capped_s=float(ev.latency[planned].mean()) if planned.any() else None,
                adaptation_recovered_s=float(ev.latency[recovered].mean()) if recovered.any() else None,
                event_status_counts={s: int(((ev.status == s) & planned).sum()) for s in np.unique(ev.status[planned])})


def validate(controller, bank, max_steps=600, deadline=None):
    ids, modes, labels, seed = suite(bank, 'val')
    r = run(controller, bank, ids, modes, seed, max_steps, deadline=deadline)
    groups = {g: summarize(r, labels == g) for g in GROUPS}
    assert all(v['n'] == len(bank.tracks) // 2 for v in groups.values())
    return dict(score=float(np.mean([g['utility'] for g in groups.values()])), groups=groups,
                env_steps=r['steps'])


def controllers(tag):
    for arm in ('rnn', 'ff'):
        path = f'checkpoints/{tag}_{arm}_selected.npz'
        yield arm, lambda path=path: load_selected(path)[0], path
    path = f'checkpoints/{tag}_rnn_selected.npz'
    yield 'rnn_memory_off', lambda: load_selected(path, memory_off=True)[0], path
    yield 'legacy_ff_frozen_extra_data', Legacy, 'checkpoints/ppo_dr_r0_rg_best.npz'
    bpath = f'checkpoints/{tag}_baseline_selected.json'
    p = json.loads(Path(bpath).read_text())['params']
    yield 'adaptive_ray', lambda: AdaptiveRay(p), bpath
    yield 'adaptive_ray_frozen_estimator', lambda: AdaptiveRay(p, frozen=True), bpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r5_seed0')
    ap.add_argument('--split', choices=['test', 'legacy', 'val'], default='test')
    ap.add_argument('--n', type=int, default=64, help='tracks/geometry; legacy uses 2*n regular + n hard')
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--max-minutes', type=float, default=2.5)
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    out = Path(f'results/r5/{a.tag}_{a.split}.json')
    # An interrupted evaluation resumes only missing METHODS, never silently reduces groups.
    from hashlib import sha256
    specs = list(controllers(a.tag))
    config = dict(split=a.split, n=a.n, max_steps=a.max_steps,
                  source_sha256={p: sha256(Path(p).read_bytes()).hexdigest() for p in
                                 ('r5_eval.py', 'r5_env.py', 'r5_policy.py', 'r5_baseline.py', 'racing/sim.py')},
                  checkpoints={name: sha256(Path(path).read_bytes()).hexdigest() for name, _, path in specs})
    doc = json.loads(out.read_text()) if out.exists() else dict(config=config, complete=False, methods={},
        comparison_notes={
            'rnn_and_ff': 'Matched fine-tuning from the same historical PPO+DR checkpoint; extra prior training.',
            'adaptive_ray': 'CEM also includes historical DR ray knobs; prior-data/compute history is unmatched.',
            'legacy_ff_frozen_extra_data': 'Frozen historical 13-sensor policy; no R5 fine-tuning or gyro input.',
            'ablations': 'Memory-off and frozen-estimator variants are not retuned.'})
    if doc['config'] != config:
        raise ValueError('existing eval differs: use a fresh tag/output; preserve prior receipt')
    bank = make_bank(a.split, a.n)
    doc['track_manifest'] = bank.r5_track_manifest
    ids, modes, labels, seed = suite(bank, a.split)
    for name, factory, path in specs:
        if name in doc['methods']:
            continue
        start = time.monotonic()
        try:
            r = run(factory(), bank, ids, modes, seed, a.max_steps, deadline=budget)
        except TimeoutError:
            break
        rows = [dict(track_seed=bank.tracks[ids[i]].seed, group=str(labels[i]),
                     finished=bool(r['finished'][i]), crashed=bool(r['crashed'][i]),
                     lap_penalty_s=float(r['lap_penalty_s'][i]), progress=float(r['progress'][i]),
                     events=r['events'].rows(i)) for i in range(len(ids))]
        doc['methods'][name] = dict(checkpoint=path, overall=summarize(r),
            groups={g: summarize(r, labels == g) for g in GROUPS}, episodes=rows,
            seconds=time.monotonic() - start)
        write_json(out, doc)
        emit(dict(method=name, **doc['methods'][name]['overall']))
    doc['complete'] = len(doc['methods']) == len(specs)
    doc['paired'] = {}
    if doc['complete']:
        for other in ('ff', 'adaptive_ray', 'legacy_ff_frozen_extra_data', 'rnn_memory_off'):
            left = doc['methods']['rnn']['episodes']
            right = doc['methods'][other]['episodes']
            assert [(e['track_seed'], e['group']) for e in left] == [(e['track_seed'], e['group']) for e in right]
            common = [i for i, (x, y) in enumerate(zip(left, right)) if x['finished'] and y['finished']]
            delta = np.array([x['lap_penalty_s'] - y['lap_penalty_s'] for x, y in zip(left, right)])
            doc['paired']['rnn_minus_' + other] = dict(
                n=len(left), jointly_finished=len(common),
                mean_dnf_penalized_delta_s=float(delta.mean()),
                mean_joint_finish_delta_s=float(delta[common].mean()) if common else None)
    doc['wall_seconds_this_invocation'] = budget.elapsed
    write_json(out, doc)
    emit(dict(output=str(out), complete=doc['complete'], seconds=budget.elapsed))
    if not doc['complete']:
        raise SystemExit('PARTIAL: rerun exactly this command within the remaining project allowance')


if __name__ == '__main__':
    main()
