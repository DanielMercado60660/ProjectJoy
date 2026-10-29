"""Post-run R5 audit: saved outcomes, selection, and sensor observability (no MLX).

Replays frozen policies only; never changes selection or fits anything on test.
The single-packet slip probe uses a fixed .05 rad/s margin and no latent state.
Ground-truth grip / old speed are used ONLY to score that diagnostic.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from racing.sim import CAR_R, DT, G, MAX_STEER, V_SCALE, WHEELBASE
from r5_common import Budget, GROUPS, write_json
from r5_env import SurfaceRace, finish_flags, fixed_surfaces, make_bank, normal_envelope, suite
from r5_eval import Events, controllers, validate
from r5_policy import Policy, load_selected


def single_packet_probe(obs, margin=.05):
    """Approximate saturation and mu from only current speed, last steer, gyro.

    Exact physics uses midpoint speed, unavailable in one packet. Unexcited
    frames supply no equality estimate; a returned number is NOT a grip label
    unless the flag is true. All operations remain finite at rest.
    """
    v = obs[:, 10] * V_SCALE
    omega = np.abs(obs[:, 13])
    demand = np.abs(v * np.tan(obs[:, 11] * MAX_STEER) / WHEELBASE)
    flag = (v > 3) & (demand > omega + margin)
    mu = omega * np.maximum(v, 1) / G
    return flag, mu


def event_curve(rows, thresholds):
    """All planned events in denominator; a failure cap is never a recovery."""
    events = [ev for row in rows for ev in row['events']]
    recovered = [ev for ev in events if ev['status'] == 'recovered']
    return dict(planned=len(events), statuses=dict(Counter(e['status'] for e in events)),
                thresholds_s=list(map(float, thresholds)),
                recovered_fraction=[sum(e['capped_latency_s'] <= t + 1e-9 for e in recovered) / len(events)
                                    for t in thresholds] if events else [],
                capped_mean_s=float(np.mean([e['capped_latency_s'] for e in events])) if events else None)


def first_packet_estimates(samples, rows, mask):
    """Time until the current packet flags slip, counting unexposed events too.

    Stop at the next surface change or five seconds. A transition at simulation
    step t can first be observed by the controller at t+1, hence minimum .1 s.
    """
    planned, delays, errors = 0, [], []
    for i in np.flatnonzero(mask):
        for ev in rows[i]['events']:
            planned += 1
            start = ev['change_step']
            if start < 0:
                continue
            next_changes = [e['change_step'] for e in rows[i]['events'] if e['change_step'] > start]
            stop = min([start + 50, len(samples), *next_changes])
            s = samples[start:stop, i]
            candidates = np.flatnonzero((s[:, 0] > 0) & (s[:, 1] > 0))
            if len(candidates):
                first = candidates[0]
                delays.append((first + 1) * DT)
                errors.append(s[first, 3])
    return dict(planned=planned, flagged_within_window=len(delays),
                thresholds_s=[.1, .5, 1., 5.],
                flagged_fraction=[sum(d <= t + 1e-9 for d in delays) / planned for t in (.1, .5, 1., 5.)] if planned else [],
                first_flag_grip_mae=float(np.mean(errors)) if errors else None)


def aggregate(rows):
    f = np.array([r['finished'] for r in rows], bool)
    c = np.array([r['crashed'] for r in rows], bool)
    lap = np.array([r['lap_penalty_s'] for r in rows])
    ev = [e for r in rows for e in r['events']]
    good = [e for e in ev if e['status'] == 'recovered']
    return dict(n=len(rows), finishes=int(f.sum()), crashes=int(c.sum()),
                completion=float(f.mean()), crash_rate=float(c.mean()), timeout_rate=float((~f & ~c).mean()),
                mean_finished_lap_s=float(lap[f].mean()) if f.any() else None,
                mean_dnf_penalized_lap_s=float(lap.mean()), events_planned=len(ev),
                events_exposed=sum(e['exposed'] for e in ev), events_recovered=len(good),
                event_recovery_coverage=len(good) / len(ev) if ev else None,
                adaptation_capped_s=float(np.mean([e['capped_latency_s'] for e in ev])) if ev else None,
                adaptation_recovered_s=float(np.mean([e['capped_latency_s'] for e in good])) if good else None)


def audit_receipt(doc):
    assert doc['complete'], 'partial evaluation cannot support headline results'
    out = {}
    keys = None
    for name, m in doc['methods'].items():
        rows = m['episodes']
        order = [(r['track_seed'], r['group']) for r in rows]
        assert keys is None or keys == order
        keys = order
        stats = {'overall': aggregate(rows)}
        stats['groups'] = {g: aggregate([r for r in rows if r['group'] == g]) for g in GROUPS}
        for scope, values in [('overall', stats['overall']), *stats['groups'].items()]:
            saved = m['overall'] if scope == 'overall' else m['groups'][scope]
            for k, v in values.items():
                if k in saved:
                    if v is None:
                        assert saved[k] is None
                    else:
                        np.testing.assert_allclose(v, saved[k], rtol=0, atol=1e-12, err_msg=f'{name}/{scope}/{k}')
        thresholds = np.arange(51) / 10
        stats['curves'] = {g: event_curve(rows if g == 'overall' else [r for r in rows if r['group'] == g], thresholds)
                           for g in ('overall', 'normal_patch', 'hard_patch', 'normal_switch', 'hard_switch')}
        stats['direction_curves'] = {}
        for direction in (-1, 1):
            directional = [dict(events=[e for e in r['events'] if e['direction'] == direction]) for r in rows]
            stats['direction_curves'][str(direction)] = event_curve(directional, thresholds)
        out[name] = stats
    return out


def replay_probe(controller, bank, ids, modes, seed, labels, saved_rows, budget):
    """Instrument the unchanged evaluation physics; assert every episode/event matches."""
    n = len(ids)
    env = SurfaceRace(bank, n)
    obs = env.reset(np.random.default_rng(seed), ids=ids, modes=modes)
    fixed_surfaces(env, ids, modes, seed)
    active, finished, crashed = np.ones(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    times = np.full(n, 60.)
    previous_mu = env.last_grip.copy()
    events = Events(modes)
    samples, bad_envelope = [], []
    for t in range(600):
        if not budget.fits(0, reserve=2):
            raise TimeoutError('analysis wall guard; no partial report written')
        mu = env.grip_now()
        changed = active & (np.abs(mu - previous_mu) > 1e-7)
        direction = np.sign(mu - previous_mu)
        v0 = env.base.v.copy()
        action = controller(obs)
        env.base.done[~active] = True
        obs, _, _ = env.step(action)
        b = env.base
        f = active & finish_flags(b.prog, bank.length[ids], b.crashed)
        c = active & b.crashed
        terminal = f | c | (t == 599)
        normal = normal_envelope(env)
        events.update(t, changed, direction, normal, terminal)
        finished |= f
        crashed |= c
        times[f] = (t + 1) * DT

        valid = active & ~terminal & (v0 > 3) & (b.v > 3)
        vm = (v0 + b.v) / 2
        actual_demand = np.abs(vm * np.tan(b.steer_prev * MAX_STEER) / WHEELBASE)
        true_sat = actual_demand > mu * G / np.maximum(vm, 1) + 1e-7
        detected, estimate = single_packet_probe(obs)
        # columns: valid, detected, true saturation, absolute mu error, wet/dry correct
        samples.append(np.column_stack([valid, detected, true_sat, np.abs(estimate - mu),
                                        (estimate < .775) == (mu < .775)]))
        k = np.array([abs(bank.tracks[tid].curvature[i]) for tid, i in zip(ids, b.idx)])
        vref = np.clip(np.sqrt(np.maximum(mu * G / np.maximum(k, .01), 1e-8)), 6, 30)
        bad = np.column_stack([np.abs(b.lat) / (bank.half_width[ids] - CAR_R) > .5,
                               b.v < .6 * vref, b.v > 1.3 * vref])
        np.testing.assert_array_equal(normal, ~bad.any(1) & ~b.crashed)
        bad_envelope.append(bad)
        active &= ~(f | c)
        previous_mu = mu.copy()
        if not active.any():
            break
    events.finish()
    for i, row in enumerate(saved_rows):
        assert finished[i] == row['finished'] and crashed[i] == row['crashed']
        np.testing.assert_allclose(times[i], row['lap_penalty_s'], rtol=0, atol=1e-12)
        assert events.rows(i) == row['events'], f'event replay mismatch: {i}'
    samples = np.stack(samples)
    bad_envelope = np.stack(bad_envelope)
    result = {}
    for group in ('overall', *GROUPS):
        mask = np.ones(n, bool) if group == 'overall' else labels == group
        s = samples[:, mask].reshape(-1, 5)
        s = s[s[:, 0] > 0]
        flagged = s[:, 1] > 0
        truth = s[:, 2] > 0
        err = s[flagged, 3]
        expired = []
        for i in np.flatnonzero(mask):
            for ev in saved_rows[i]['events']:
                if ev['status'] == 'window_expired':
                    start = ev['change_step']
                    expired.append(bad_envelope[start:start + 50, i].mean(0))
        result[group] = dict(first_packet_after_change=first_packet_estimates(samples, saved_rows, mask),
            valid_frames=len(s), flagged_frames=int(flagged.sum()),
            flagged_fraction=float(flagged.mean()), true_saturation_fraction=float(truth.mean()),
            saturation_precision=float(truth[flagged].mean()) if flagged.any() else None,
            saturation_recall=float(flagged[truth].mean()) if truth.any() else None,
            grip_mae=float(err.mean()) if len(err) else None,
            grip_absolute_error_p95=float(np.quantile(err, .95)) if len(err) else None,
            wet_dry_accuracy=float(s[flagged, 4].mean()) if flagged.any() else None,
            expired_windows=len(expired),
            expired_mean_violation_fraction=dict(zip(('lateral', 'underspeed', 'overspeed'),
                map(float, np.mean(expired, axis=0)))) if expired else None)
    return dict(episode_and_event_replay_verified=True, groups=result)


def selection_audit(tag, budget):
    history = [json.loads(s) for s in Path(f'logs/r5/{tag}_train.jsonl').read_text().splitlines()]
    updates = [r for r in history if r['event'] == 'update']
    final = [r for r in history if r['event'] == 'finished'][-1]
    assert updates[-1]['update'] == final['update']
    assert updates[-1]['arms']['rnn']['gradient_steps'] == updates[-1]['arms']['ff']['gradient_steps']
    bank = make_bank('val', 16)
    result = dict(finished=final, last_update=updates[-1], arms={})
    with np.load(f'checkpoints/{tag}_state.npz', allow_pickle=False) as state:
        assert int(state['update']) == final['update']
        for arm in ('rnn', 'ff'):
            curve = [r for r in history if r['event'] == 'validation' and r['arm'] == arm]
            assert all(set(r['groups']) == set(GROUPS) and all(g['n'] == 16 for g in r['groups'].values()) for r in curve)
            best = max(curve, key=lambda r: r['score'])
            _, meta = load_selected(f'checkpoints/{tag}_{arm}_selected.npz')
            assert best['update'] == meta['step'] == final['selected'][arm]
            assert best['score'] == meta['validation']['score']
            prefix = arm + '/p/'
            params = {k[len(prefix):]: state[k].copy() for k in state.files if k.startswith(prefix)}
            last_val = validate(Policy(params, arm == 'rnn'), bank, deadline=budget)
            result['arms'][arm] = dict(validation_count=len(curve), initial=curve[0], best=best,
                last_logged_validation=curve[-1], final_update_validation=last_val,
                curve=[dict(update=r['update'], score=r['score'],
                            finishes=sum(round(g['n'] * g['completion']) for g in r['groups'].values()),
                            **{g: r['groups'][g]['completion'] for g in GROUPS}) for r in curve])
    rows = [json.loads(s) for s in Path(f'logs/r5/{tag}_baseline.jsonl').read_text().splitlines()]
    selected = json.loads(Path(f'checkpoints/{tag}_baseline_selected.json').read_text())
    vals = [r for r in rows if r['event'] == 'baseline_validation']
    assert all(set(r['groups']) == set(GROUPS) and all(g['n'] == 16 for g in r['groups'].values()) for r in vals)
    best = max(vals, key=lambda r: r['score'])
    assert best['generation'] == selected['generation'] and best['score'] == selected['validation']['score']
    bounds = {k: dict(value=v, low=selected['ranges'][k][0], high=selected['ranges'][k][1],
                     fractional_distance_to_nearest_bound=float(min(v - selected['ranges'][k][0], selected['ranges'][k][1] - v)
                                                                 / np.ptp(selected['ranges'][k])))
              for k, v in selected['params'].items()}
    result['baseline'] = dict(finished=[r for r in rows if r['event'] == 'baseline_finished'][-1],
        config=selected['config'], validation_count=len(vals), initial=vals[:2], best=best, bounds=bounds,
        curve=[dict(generation=r['generation'], score=r['score']) for r in vals])
    return result


def plots(report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    styles = [('ff', 'FF'), ('rnn', 'RNN'), ('rnn_memory_off', 'RNN memory off'), ('adaptive_ray', 'Adaptive ray')]
    for ax, group in zip(axes, ('overall', 'hard_patch', 'hard_switch')):
        for name, label in styles:
            c = report['test'][name]['curves'][group]
            ax.step(c['thresholds_s'], c['recovered_fraction'], where='post', label=label)
        ax.set(title=group.replace('_', ' '), xlabel='Sustained-envelope onset by (s)', ylim=(0, 1.03), xlim=(0, 5))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Fraction of all planned events recovered')
    axes[-1].legend(fontsize=8, loc='lower right')
    fig.savefig(prefix + '_recovery.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for arm in ('rnn', 'ff'):
        a = report['selection']['arms'][arm]
        curve = a['curve']
        axes[0].plot([r['update'] for r in curve], [r['score'] for r in curve], label=arm)
        axes[0].scatter([a['best']['update']], [a['best']['score']], marker='*', s=90)
        axes[1].plot([r['update'] for r in curve], [r['finishes'] / 96 for r in curve], label=arm)
    curve = report['selection']['baseline']['curve']
    axes[2].scatter([r['generation'] for r in curve], [r['score'] for r in curve], s=10, alpha=.55, label='Winner / mean')
    axes[2].plot([r['generation'] for r in curve], np.maximum.accumulate([r['score'] for r in curve]), label='Incumbent')
    for ax, title, xlabel in zip(axes, ('PPO full validation utility', 'PPO full validation completion', 'CEM full validation utility'),
                                ('Paired update', 'Paired update', 'Generation')):
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(prefix + '_validation.png', dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r5_seed0')
    ap.add_argument('--output', default='results/r5/r5_turn2_analysis.json')
    ap.add_argument('--media-prefix', default='media/r5_turn2')
    ap.add_argument('--max-minutes', type=float, default=1)
    a = ap.parse_args()
    for path in (a.output, a.media_prefix + '_recovery.png', a.media_prefix + '_validation.png'):
        if Path(path).exists():
            raise FileExistsError(f'preserve prior analysis: choose fresh output / media prefix: {path}')
    budget = Budget(a.max_minutes)
    inputs = [f'results/r5/{a.tag}_{split}.json' for split in ('test', 'legacy')]
    docs = [json.loads(Path(p).read_text()) for p in inputs]
    assert docs[0]['config']['n'] == 64 and docs[0]['config']['max_steps'] == 600
    assert docs[1]['config']['n'] == 50 and docs[1]['config']['max_steps'] == 600
    for doc in docs:
        for path, expected in doc['config']['source_sha256'].items():
            assert sha256(Path(path).read_bytes()).hexdigest() == expected, path
        for name, _, path in controllers(a.tag):
            assert sha256(Path(path).read_bytes()).hexdigest() == doc['config']['checkpoints'][name], path
    report = dict(tag=a.tag, diagnostic_only=True, test=audit_receipt(docs[0]), legacy=audit_receipt(docs[1]))
    report['selection'] = selection_audit(a.tag, budget)
    bank = make_bank('test', 64)
    ids, modes, labels, seed = suite(bank, 'test')
    report['sensor_probe'] = {}
    for name, factory, _ in controllers(a.tag):
        if name not in ('rnn', 'ff', 'adaptive_ray'):
            continue
        start = time.monotonic()
        report['sensor_probe'][name] = replay_probe(factory(), bank, ids, modes, seed, labels,
                                                   docs[0]['methods'][name]['episodes'], budget)
        print(json.dumps(dict(event='replay_verified', method=name, episodes=len(ids), seconds=time.monotonic() - start)), flush=True)
    inputs += [f'logs/r5/{a.tag}_{s}.jsonl' for s in ('train', 'baseline')]
    inputs += [path for _, _, path in controllers(a.tag)] + [f'checkpoints/{a.tag}_state.npz', __file__]
    report['input_sha256'] = {p: sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    plots(report, a.media_prefix)
    report['seconds'] = budget.elapsed
    assert budget.fits(0, reserve=0), 'analysis exceeded allowance'
    write_json(a.output, report)
    print(json.dumps(dict(event='ANALYSIS_PASS', output=a.output, seconds=budget.elapsed)), flush=True)


if __name__ == '__main__':
    main()
