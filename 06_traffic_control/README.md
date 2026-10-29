# 06 · Traffic Control From Hell

A tiny city with deeply questionable traffic. Four signalised intersections on a 2×2 grid, a
queue-level simulator, cheap signal rules, and one learned controller (PPO in MLX, one network
shared by all four intersections). The question for the first version: **can a learned controller
cut waiting without starving the quiet side street?**

```
            Elm St (medium)        Quiet Lane (~54 veh/h)
   1st Ave  I0 ------------------- I1      <- heavy east-west arterial (~540 veh/h per direction,
            |                      |          ~1030 at the rush-hour peak)
   2nd Ave  I2 ------------------- I3      <- heavy east-west arterial
```

## The simulator (`sim.py`)
- Batched numpy, 2-s steps, one hour per episode (1800 steps). Vehicles drive straight through.
- Poisson arrivals on the 8 edge lanes. Demand patterns: `steady`, `rush` (arterials ×1.9 at
  17:30, Elm St ×1.3, Quiet Lane unchanged), and `train` (random scale, random rush peak 1.0–2.1,
  random peak time, random east/west asymmetry ±30%).
- Two phases per intersection (NS green or EW green). **Non-conflicting phases are enforced by the
  simulator**: it only ever serves one phase, so the policy cannot choose an unsafe state.
  Switching costs 2 steps (4 s) of all-red.
- Saturation flow 1 veh/step per green lane. Internal links take 10 s to drive and hold 16
  vehicles; a full link blocks its upstream green ("spillback").
- Controllers decide every 10 s (5 steps), baselines and PPO alike.
- Metrics per episode: throughput, mean / p95 / max wait per signal crossing, Quiet Lane mean and
  worst wait, worst approach mean wait, Jain fairness over the 16 approaches' mean waits,
  spillback-blocked steps, fraction of time links are full. Cars still waiting at the end of the
  hour count with their wait so far in the *worst* waits, so a starved street cannot hide; mean
  and p95 wait cover completed crossings only (counting the unfinished cars changes hell-hour
  means by ≤ 3 s, see RESULTS.md).

## Controllers
- `baselines.py`: fixed-time (cycle and split tuned per street on training demand),
  longest-queue-first, max-pressure, and max-pressure with a max-wait guard (threshold tuned on
  training demand). These rules have no minimum green: they may switch at every 10-s decision.
- `review_check.py` (added by the reviewer): LQF with a tuned minimum green and max-pressure with a
  tuned switching hysteresis (both tuned on training demand), plus mean wait counting unfinished
  trips. `../.venv/bin/python review_check.py` (~10 s) → `runs/review_check.json`.
- `policy.py`, `train.py`: shared-parameter actor-critic (2×64 tanh MLPs, policy and value),
  PPO with GAE. Each intersection observes its 4 lanes (stop-line queue, cars still driving in,
  front-car wait, downstream link occupancy), its phase, all-red flag, time since switch, and a
  one-hot intersection id. Reward per intersection per decision:
  `-0.1 * (queue/10 + 0.5 * Σ_lanes min(front_wait/60 s, 5)^2)`, 75% local + 25% mean of all
  four. The squared front-wait term is the anti-starvation pressure.

## Commands (run from this folder)
```bash
# smoke test (< 1 min): pytest invariants + tiny train/reload + quick eval + demo build, temp dir only
./smoke.sh

# main training run (15 min wall clock, through the shared slot lock)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --minutes 15 \
    --out checkpoints/ppo --log runs/train_log.jsonl

# tune baselines on training demand, evaluate everything on held-out seeds (writes runs/)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python evaluate.py \
    --ckpt checkpoints/ppo_best --episodes 20

# ablation (run after the write-up): no squared-wait fairness term
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --minutes 10 --beta 0 \
    --out checkpoints/ablation_beta0 --log runs/ablation_beta0_log.jsonl
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python evaluate.py \
    --ckpt checkpoints/ablation_beta0_best --episodes 20 --out runs/ablation_beta0

# follow-up 1 (round 3): more training seeds (one gpu_run job per group; 4 single-thread procs each)
MIN=11 SEEDS="1 2 3" B0SEEDS="1" ../bin/gpu_run ./train_seeds.sh   # -> checkpoints/seeds/, runs/seeds/train_*.jsonl
MIN=11 SEEDS="4 5" B0SEEDS="2 3" ../bin/gpu_run ./train_seeds.sh
# evaluate every seed + the tuned rules on the same 20 held-out seeds per pattern (~10-25 s)
../.venv/bin/python eval_seeds.py                                   # -> runs/seeds/eval_seeds.{json,md}

# rebuild the demo page and the summary chart from runs/
../.venv/bin/python make_demo.py
```
Open `media/demo.html` in any browser (offline, no network). Append `#t=900` to the URL to start
paused at a given step (e.g. the rush-hour peak).

## File layout
```
sim.py              batched queue simulator, demand patterns, metrics
baselines.py        fixed-time, longest-queue-first, max-pressure, max-pressure + guard
policy.py           MLX actor-critic, checkpoint save/load, greedy controller
train.py            PPO training loop (validation on training-distribution seeds picks *_best)
evaluate.py         baseline tuning + held-out evaluation -> runs/eval_results.json, eval_table.md, demo_traces.npz
make_demo.py        -> media/demo.html, media/summary.png
review_check.py     reviewer's extra baselines + censoring check -> runs/review_check.json
train_seeds.sh      follow-up 1: trains extra PPO seeds (beta=0.5 and beta=0) in one gpu_run job
eval_seeds.py       follow-up 1: per-seed + across-seed held-out eval vs tuned rules -> runs/seeds/
tests/test_pipeline.py  invariants (vehicle conservation, no conflicting greens, link caps), tiny train + reload
smoke.sh            end-to-end smoke check
checkpoints/        ppo_best.{npz,json} (selected on validation), ppo_last.{npz,json}, ablation_beta0_*;
                    seeds/ppo_s{1..5}_*, seeds/beta0_s{1..3}_* (follow-up 1)
runs/               training logs, eval results, demo traces; runs/ablation_beta0/ for the beta=0 ablation
media/              demo.html, summary.png, snapshot PNGs
RESULTS.md          numbers, baseline comparison, what worked / didn't
```
