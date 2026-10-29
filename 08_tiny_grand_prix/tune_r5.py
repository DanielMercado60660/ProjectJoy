"""Time-budgeted CEM tuning of ALL 13 adaptive-controller knobs, train/val only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from r5_baseline import AdaptiveRay, DEFAULT, HI, KEYS, LO, RANGES, as_dict, vector
from r5_common import Budget, atomic_npz, emit, write_json
from r5_env import make_bank
from r5_eval import run, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r5_seed0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-minutes', type=float, default=12)
    ap.add_argument('--max-generations', type=int, default=1000000000)
    ap.add_argument('--population', type=int, default=48)
    ap.add_argument('--episodes', type=int, default=24, help='per candidate, divisible by 6')
    ap.add_argument('--n-train', type=int, default=64)
    ap.add_argument('--n-val', type=int, default=16)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--eval-every', type=int, default=0)
    ap.add_argument('--resume')
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    if a.episodes % 6 or a.population < 4 or not a.tag.startswith('r5_') or '/' in a.tag:
        raise ValueError('episodes divisible by 6; population >=4; local r5_ tag required')
    state_path = Path(f'checkpoints/{a.tag}_baseline_state.npz')
    selected_path = Path(f'checkpoints/{a.tag}_baseline_selected.json')
    history = f'logs/r5/{a.tag}_baseline.jsonl'
    if state_path.exists() and not a.resume:
        raise ValueError('state exists: resume or use a fresh tag')
    config = {k: v for k, v in vars(a).items() if k not in ('tag', 'max_minutes', 'max_generations', 'resume', 'eval_every')}
    rng = np.random.default_rng(a.seed + 71001)
    mean, std = (vector() - LO) / (HI - LO), np.full(len(KEYS), .28)
    gen, best, best_gen, steps = 0, -1e30, -1, 0
    incumbent = vector()
    selected_info = {}
    if a.resume:
        with np.load(a.resume, allow_pickle=False) as z:
            if json.loads(str(z['config'])) != config:
                raise ValueError('resume config mismatch')
            mean, std, incumbent = z['mean'], z['std'], z['incumbent']
            gen, best, best_gen, steps = int(z['generation']), float(z['best']), int(z['best_gen']), int(z['steps'])
            rng.bit_generator.state = json.loads(str(z['rng']))
            selected_info = json.loads(str(z['selected_info']))
            write_json(selected_path, selected_info)
        emit(dict(event='baseline_resumed', generation=gen))
    bank, val_bank = make_bank('train', a.n_train), make_bank('val', a.n_val)

    def save():
        atomic_npz(state_path, config=json.dumps(config), generation=gen, mean=mean, std=std,
                   incumbent=incumbent, best=best, best_gen=best_gen, steps=steps,
                   rng=json.dumps(rng.bit_generator.state), selected_info=json.dumps(selected_info))

    def select(candidate):
        nonlocal best, best_gen, incumbent, selected_info
        v = validate(AdaptiveRay(candidate), val_bank, a.max_steps, budget)
        if v['score'] > best:
            best, best_gen, incumbent = v['score'], gen, candidate.copy()
            selected_info = dict(params=as_dict(candidate), generation=gen,
                       validation=v, config=config, ranges=RANGES)
            write_json(selected_path, selected_info)
        emit(dict(event='baseline_validation', generation=gen, best_generation=best_gen, **v), history)

    val_seconds = 1.
    if gen == 0:
        ts = time.monotonic()
        select(incumbent)
        # Include the earlier carefully tuned DR follower as a strong starting point.
        old = json.loads(Path('checkpoints/ray_follower_dr.json').read_text())
        old = old.get('params', old)
        seedp = dict(DEFAULT)
        seedp.update({k: old[k] for k in ('k_steer', 'k_center', 'margin', 'v_max', 'v_min')})
        seedp['power'] = old['p']
        seedp['brake_fraction'] = np.clip(old['brake'] / (9.81 * seedp['prior'] * seedp['safety_mu']), LO[3], HI[3])
        select(vector(seedp))
        mean = (incumbent - LO) / (HI - LO)
        val_seconds = time.monotonic() - ts
        save()
    generation_seconds = 3.
    eval_every = a.eval_every or 10
    first_gen = gen
    while gen < a.max_generations and budget.fits(generation_seconds * 1.3, reserve=3):
        begin = time.monotonic()
        rng_before = json.dumps(rng.bit_generator.state)
        x = np.clip(rng.normal(mean, std, (a.population, len(KEYS))), 0, 1)
        # Repeated broad exploration prevents an early local optimum exhausting the budget.
        x[-max(1, a.population // 4):] = rng.random((max(1, a.population // 4), len(KEYS)))
        candidates = LO + x * (HI - LO)
        candidates[0] = incumbent
        x[0] = (incumbent - LO) / (HI - LO)
        k = a.episodes
        ids = rng.integers(0, a.n_train, k)
        hard = np.tile(np.array([0, 1, 0, 1, 0, 1]), k // 6)
        ids += hard * a.n_train
        modes = np.tile(np.array([0, 0, 1, 1, 2, 2]), k // 6)
        seed = int(rng.integers(1, 2**31))
        try:
            r = run(AdaptiveRay(np.repeat(candidates, k, axis=0)), bank,
                    np.tile(ids, a.population), np.tile(modes, a.population), seed,
                    a.max_steps, deadline=budget, repeat_scenarios=k)
        except TimeoutError:
            # Discard the WHOLE generation, including its random draws.
            rng.bit_generator.state = json.loads(rng_before)
            break
        scores = r['utility'].reshape(a.population, k).mean(1)
        elites = np.argsort(scores)[-max(2, a.population // 4):]
        mean = .3 * mean + .7 * x[elites].mean(0)
        std = np.maximum(.08, .3 * std + .7 * x[elites].std(0))
        winner = candidates[int(np.argmax(scores))]
        gen += 1
        steps += r['steps']
        measured = time.monotonic() - begin
        generation_seconds = max(measured, .8 * generation_seconds + .2 * measured)
        if gen - first_gen in (1, 3, 5):
            eval_every = a.eval_every or max(1, round(30 / generation_seconds))
            emit(dict(event='baseline_rate_plan', generation_s=generation_seconds,
                      env_steps_s=r['steps'] / measured,
                      projected_additional_generations=max(0, int(budget.remaining / (generation_seconds + val_seconds/eval_every))),
                      eval_every=eval_every), history)
        emit(dict(event='generation', generation=gen, train_best=float(scores.max()),
                  train_mean=float(scores.mean()), env_steps=steps, wall_seconds=budget.elapsed), history)
        if gen % eval_every == 0 and budget.fits(val_seconds * 1.5, reserve=3):
            ts = time.monotonic()
            try:
                # Full validation of winner AND CEM mean (including every hard group).
                select(winner)
                select(LO + mean * (HI - LO))
            except TimeoutError:
                emit(dict(event='baseline_validation_incomplete', generation=gen), history)
            val_seconds = max(val_seconds, time.monotonic() - ts)
        save()  # atomic fixed filename, no per-generation snapshots
    if gen > first_gen and budget.fits(val_seconds * 1.5, reserve=2):
        try:
            select(LO + mean * (HI - LO))
        except TimeoutError:
            pass
    save()
    emit(dict(event='baseline_finished', generation=gen, selected_generation=best_gen,
              env_steps=steps, wall_seconds=budget.elapsed, allowance_seconds=budget.seconds,
              stop='ceiling' if gen >= a.max_generations else 'wall_guard'), history)
    if gen == first_gen:
        raise SystemExit('No completed generation; use measured setup timing to size allowance.')


if __name__ == '__main__':
    main()
