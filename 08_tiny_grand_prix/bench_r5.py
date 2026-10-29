"""Short NumPy-only throughput receipt; does not measure MLX update speed."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from r5_baseline import AdaptiveRay
from r5_common import Budget, emit, write_json
from r5_env import SurfaceRace, make_bank
from r5_eval import run, validate
from r5_policy import HIDDEN, init_params
from train_r5 import rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-minutes', type=float, default=.5)
    ap.add_argument('--out', default='results/r5/rate_receipt.json')
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    bank, val = make_bank('train', 8), make_bank('val', 16)
    rng = np.random.default_rng(81)
    env = SurfaceRace(bank, 128)
    env.reset(rng)
    p = init_params()
    h = np.zeros((128, HIDDEN), np.float32)
    # Warm the numba signatures and numpy BLAS before timing.
    _, h, _ = rollout(p, env, rng, h, 4, 600, True)
    tick = time.monotonic()
    count = 0
    while count < 20 and budget.fits(2) and (time.monotonic() - tick < 3 or count == 0):
        _, h, _ = rollout(p, env, rng, h, 64, 600, True)
        count += 1
    dt = time.monotonic() - tick
    receipt = dict(note='CPU-only smoke timing, no MLX import or optimizer step',
                   recurrent_rollout_env_steps=count * 128 * 64,
                   recurrent_rollout_seconds=dt,
                   recurrent_rollout_env_steps_s=count * 128 * 64 / dt,
                   hardware_contention='shared machine; recalibrate in train_r5.py and tune_r5.py')
    if budget.fits(5):
        k, pop = 24, 48
        ids = np.tile(np.arange(k) % len(bank.tracks), pop)
        modes = np.tile(np.array([0, 0, 1, 1, 2, 2]), pop * k // 6)
        tick = time.monotonic()
        r = run(AdaptiveRay(), bank, ids, modes, 81, 600, deadline=budget, repeat_scenarios=k)
        sec = time.monotonic() - tick
        receipt.update(baseline_population_seconds=sec, baseline_env_steps=r['steps'],
                       baseline_env_steps_s=r['steps']/sec, baseline_population=pop, baseline_episodes=k, baseline_slot_steps=r['slot_steps'])
    if budget.fits(5):
        tick = time.monotonic()
        v = validate(AdaptiveRay(), val, deadline=budget)
        receipt['full_validation_seconds'] = time.monotonic() - tick
        receipt['full_validation_groups'] = {g: row['n'] for g, row in v['groups'].items()}
    # Existing measured run is a useful upper context, NOT a new-model prediction.
    old = Path('logs/ppo_dr_history.jsonl')
    if old.exists():
        last = json.loads(old.read_text().splitlines()[-1])
        receipt['historical_ppo'] = dict(source=str(old), env_steps=last['env_steps'], minutes=last['minutes'],
                                        env_steps_s=last['env_steps']/(60*last['minutes']))
    receipt['total_wall_seconds'] = budget.elapsed
    write_json(a.out, receipt)
    emit(receipt)


if __name__ == '__main__':
    main()
