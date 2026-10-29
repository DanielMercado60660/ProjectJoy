# 02 · The Predator–Prey Arms Race

One pursuer (red) and one evader (blue) learn against each other in a 4×4 arena with three round
obstacles. Both are point masses with acceleration control, a cruise speed cap and a sprint that
drains stamina. The pursuer is faster in a straight line (cruise 1.0, sprint 1.5) and the evader turns
harder (acceleration 6.0 vs 2.5, cruise 0.9, sprint 1.35). Sprint lasts 2.5 s from a full tank and takes
8 s to refill. Capture is a centre distance under 0.2. Episodes end at 15 s (300 ticks of 0.05 s).

Both sides are small MLX actor-critics (two 64-unit tanh layers for the policy and two for the value
head; 17 discrete actions: drift, 8 directions, 8 directions while sprinting) trained with competitive
self-play PPO. The pursuer and evader are trained at the same time. Each iteration plays three batches
of matches: current vs current, current pursuer vs two archived evaders, and two archived pursuers vs
the current evader. The archived opponents are drawn uniformly from every snapshot so far, saved every
10 iterations starting from the random init. The reward is zero-sum: +1 for a capture, plus 0.2 × the
distance closed each tick. Each side also pays a small cost for its own thrust and sprint.

**Scripted opponents are never used in training.** They form the held-out evaluation suite:
`random` (a random heading held about 20 ticks), `greedy` (sprints straight at the evader when
close), `lead` (aims at the evader's predicted position), `flee` (runs straight away), and `smartflee`
(runs away with wall and obstacle repulsion and slides along the wall when cornered).

## Run

```sh
cd 02_predator_prey
./smoke.sh                                   # < 1 min: pytest + 3-iteration train + eval + render, all to $TMPDIR

# main run (18 min wall clock, MLX on CPU: under load it ran ~4x faster than the shared GPU for this tiny net)
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python train.py --minutes 18 --device cpu

# held-out evaluation: suite matrix, energy proxy, cross-play, figures (a few minutes)
../bin/gpu_run ../.venv/bin/python evaluate.py

# seed-1 replicate
../bin/gpu_run ../.venv/bin/python train.py --minutes 18 --max-iters 710 --device cpu --seed 1 --out checkpoints/seed1 --log results/seed1
../bin/gpu_run ../.venv/bin/python evaluate.py --ckpt checkpoints/seed1 --results results/seed1 --media results/seed1/media

# demo videos
../.venv/bin/python render.py --gif                       # media/matches.mp4 + .gif (2x2 tiles)
../.venv/bin/python render.py --tiles final/final --ticks 900 --seed 777 --name selfplay_standoff_and_orbit --gif
../.venv/bin/python render.py --tiles final/flee greedy/flee --ticks 330 --seed 111 --name failure_learned_P_vs_naive_flee --gif
```

## Layout

| path | what |
|---|---|
| `sim.py` | vectorized numpy arena, physics, rewards, scripted opponent suite |
| `policy.py` | MLX actor-critic, sampling, save/load |
| `train.py` | self-play PPO with opponent archive; logs `results/train_log.csv`, `results/train_eval.csv` |
| `evaluate.py` | held-out suite (seed 12345), energy proxy, cross-play matrix → `results/eval.json`, `results/eval_tables.md`, `media/*.png` |
| `render.py` | MP4/GIF renderer for any pursuer/evader pairing (`final`, `it<N>` archive snapshot, or scripted) |
| `checkpoints/final.npz` | final pursuer + evader weights and run metadata |
| `checkpoints/archive.npz` | every archived snapshot (`<iter>/P/...`, `<iter>/E/...`), i.e. the opponent pool |
| `tests/test_sim.py`, `smoke.sh` | unit tests (speed caps, stamina, walls, obstacles, capture, save/load) and end-to-end smoke |
| `media/matches.mp4`, `.gif` | demo: 2×2 matches (learned vs learned, learned P vs smartflee, lead vs learned E, early P vs final E) |
| `media/selfplay_standoff_and_orbit.*` | final pursuer vs final evader: wall sliding, standoff, obstacle orbit |
| `media/failure_learned_P_vs_naive_flee.*` | good failure: learned pursuer parks next to a cornered naive evader; greedy catches it |
| `media/crossplay.png`, `arms_race_curves.png`, `occupancy.png` | cross-play heatmap, training/transfer curves, evader occupancy |
| `checkpoints/seed1/`, `results/seed1/` | seed-1 replicate (710 iterations), its eval tables and figures (`results/seed1/media/`) |
| `results/run1_killed/` | logs of the first main run, killed externally at iteration 230 (identical trajectory, same seed) |
| `RESULTS.md` | numbers and write-up |
| `review/` | cross-review: stronger scripted (herding) pursuer `review/stronger_pursuers.py` (`tune` on seed 777, `heldout 0.35 0 0.6 1.0` on seed 12345 → `stronger_pursuers.json`), corner-stall re-check `flee_stall_check.py` |

In the clips, red is the pursuer and blue the evader. The bars along the bottom are stamina (pursuer on the left, evader on the right), and the arena flashes pink on a capture.
