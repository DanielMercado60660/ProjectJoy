# 21 Duck Hunt: The Waterfowl Arms Race (first satisfying version)

Two halves, each with its own cheap, tuned classical baseline:

1. **Flight school.** A 2D point-mass duck (gravity, quadratic drag, 2 g thrust limit with a
   0.1 s actuator lag, unobserved Ornstein-Uhlenbeck wind gusts) must pass 6 waypoints
   (radius 1 m) in order within 25 s. A PPO policy (MLX, tanh-squashed Gaussian, 2x128 MLP)
   is compared with a PD/pursuit-guidance controller on the same thrust limits whose 8 gains
   were random-searched on the same training courses. Measured on 500 held-out courses.
2. **Duck tracking.** An arcade scene renderer (sky gradient, drifting clouds, hills, trees,
   crows/gulls, the flapping mallard sprite flying a flight-sim trajectory) produces 80x80
   frames. A small MLX CNN (heatmap + sub-cell offset, single frame) is compared with a
   classical tracker (causal background subtraction + duck-colour chromaticity score + blob +
   constant-velocity Kalman filter with gating) whose 15 parameters were random-searched on
   training scenes. Both see exactly the same frames. Measured on held-out scenes with new
   backgrounds (sunset, desert, snow), dusk lighting with more noise, new distractors
   (balloons, duck-coloured hawks), and all three at once.

Round 6 (2026-11-04) built the hunter/evasion stage ("Go deeper"), see below.

**Cross-review (2026-10-24):** a reviewer-added stronger flight baseline, PD+ (`pd_plus.py`: aim at
the ring edge along the shortest-path bisector, braking-distance speed profile, no slow-down at
the last waypoint, direction-preserving thrust clip; tuned with the same 2801-configuration
budget), completes 100% of the held-out courses in 6.48 s vs PPO's 6.62 s. The learned flyer
does not beat a good heuristic. Details in RESULTS.md.

## Round 6 (2026-11-04): flight rematch, robust tracker, hunter game

All on fresh held-out sets whose seeds were fixed after each method was frozen. Summary at the top
of RESULTS.md; full write-ups:

- **Flight** ([RESULTS_R6_flight.md](RESULTS_R6_flight.md)). A learned residual on top of PD+ beats PD+
  at twice its tuning budget by 0.11 s (CI excludes 0) at both wind levels, faster on ~70% of
  courses. Pure PPO at 2.3e8 steps beats PD+ in its training wind (−0.047 s) but loses in stronger
  unseen wind (+0.073 s).
- **Tracker** ([RESULTS_R6_tracker.md](RESULTS_R6_tracker.md)). A CNN trained on procedurally
  randomized scenes (palettes, textures, lighting, haze, decoy birds) loses 0–1.3% of frames on every
  held-out shift (was 54–99.7%) and 5.5% when everything shifts at once. The classical tracker,
  given matched help, loses 2.5–36%.
- **Hunter game** ([HUNTER_DESIGN.md](HUNTER_DESIGN.md), [RESULTS_R6_hunter.md](RESULTS_R6_hunter.md),
  [media/hunter_demo.html](media/hunter_demo.html)). A duck on a timed course vs a crosshair hunter
  with 3 shells. After 8 generations of self-play the learned hunter out-hits a tuned lead-pursuit
  hunter by 9–14 points on training-style evaders and ties it on a held-out evader. The hunter wins
  the arms race: the learned duck dodges the lead-pursuit hunters better than a straight-flying PD+
  duck (hit 51% vs 62% by the tuned one) but not a held-out constant-acceleration hunter, and the
  final hunter still hits the final duck 70% of the time.

Commands: "Reproduce" in RESULTS_R6_flight.md, "How to run" in RESULTS_R6_tracker.md, "Files and
commands" in RESULTS_R6_hunter.md. New files:
`train_flight_r6.py`, `pd_plus_r6.py`, `eval_r6_flight.py`, `r6_flight_figs.py`, `hunter.py`,
`train_hunter.py`, `tune_lead_r6h.py`, `eval_hunter.py`, `r6h_*.py`, `make_hunter_demo.py`,
`scene_r6.py`, `train_tracker_r6.py`, `track_classical_r6.py`, `eval_tracker_r6.py`, `select_r6.py`,
`diagnose_r6.py`, `replicate_r6.py`, `demo_tracker_r6.py`, `tests/test_r6*.py`.

## Commands (run from this folder)

```sh
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
./smoke.sh                                                        # tiny end-to-end check (pytest, < 2 min)

# flight
../bin/gpu_run ../.venv/bin/python train_flight.py --steps 30000000   # PPO, ~4 min on CPU-MLX
../bin/gpu_run ../.venv/bin/python tune_pd.py                         # PD baseline random search
../.venv/bin/python eval_flight.py                                    # held-out courses -> results/flight_eval.json
../.venv/bin/python analyze_flight.py                                 # pass distance/speed, thrust saturation
../bin/gpu_run ../.venv/bin/python pd_plus.py --tune                  # reviewer's PD+ baseline, same budget (~75 s)
../.venv/bin/python pd_plus.py --eval --style                         # -> results/pd_plus_eval.json, pd_plus_style.json, media/flight_heldout_pdplus.png

# tracking
../bin/gpu_run ../.venv/bin/python track_classical.py --tune          # classical tracker search
../bin/gpu_run ../.venv/bin/python train_tracker.py --scenes 500 --steps 6000
../bin/gpu_run ../.venv/bin/python eval_tracker.py                    # held-out scenes -> results/tracker_eval.json
../.venv/bin/python diagnose.py                                       # failure mechanisms -> results/failure_diagnosis.json

# follow-up: photometric-jitter variants of BOTH trackers (separate *_aug files)
../bin/gpu_run ../.venv/bin/python train_tracker.py --aug --scenes 500 --steps 6000 --out checkpoints/ducknet_aug --log results/tracker_train_log_aug.jsonl
../bin/gpu_run ../.venv/bin/python track_classical.py --tune --aug
../bin/gpu_run ../.venv/bin/python eval_tracker.py aug                # -> results/tracker_eval_aug.json
../.venv/bin/python diagnose.py aug
../bin/gpu_run ../.venv/bin/python retune_classical_wide.py           # reviewer: thr range widened to 0.99, 150 local steps (~2.5 min)

# demo (reloads checkpoints/)
../.venv/bin/python demo.py          # media/duck_hunt_demo.gif + media/duck_hunt_demo_strip.png
../.venv/bin/python demo.py --failures [aug]   # media/failures[_aug].png from saved eval tracks
```

## Layout

| file | what |
|---|---|
| `flight.py` | vectorized flight sim, course generator, PD/pursuit controller, episode runner, metrics |
| `train_flight.py` | PPO in MLX (CPU device so rollout and update arithmetic match) |
| `tune_pd.py` | random search + local refinement of the PD gains on training courses |
| `eval_flight.py` | held-out course evaluation (3 wind conditions) + `media/flight_heldout.png` |
| `scene.py` | arcade scene renderer and the scene conditions (train / iid / bg / light / distract / all) |
| `track_classical.py` | classical tracker, shared Kalman/gating filter, metrics, tuning |
| `train_tracker.py` | DuckNet CNN (MLX) training |
| `eval_tracker.py` | tracker evaluation; tunes the CNN's Kalman post-filter on val scenes |
| `analyze_flight.py` | how each flyer passes waypoints (closest approach, speed, thrust saturation) |
| `pd_plus.py` | reviewer-added stronger flight baseline (PD+), its tuning, held-out eval and style/plot |
| `retune_classical_wide.py` | reviewer check: aug classical tracker re-refined with the `thr` cap widened |
| `diagnose.py` | where lost trackers point (distractor / terrain / parked) and classical-tracker internals |
| `demo.py` | GIF demo: learned flyer on a held-out course, both trackers overlaid; `--failures` strips |
| `tests/test_smoke.py`, `smoke.sh` | smoke test |
| `checkpoints/` | `flight_ppo.npz` (+opt, json), `pd_params.json`, `pd_plus_params.json`, `ducknet.safetensors`, `classical_tracker_params.json`, `cnn_kf_params.json`; `*_aug` versions for the follow-up |
| `results/` | logs, tuning records, evaluation JSONs, per-frame tracks |
| `media/` | demo GIF, strips, flight plot, scene preview |
