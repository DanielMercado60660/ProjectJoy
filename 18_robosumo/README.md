# 18 RoboSumo: the reinforcement learning fight club (first satisfying version)

Two differential-drive wheeled robots in a MuJoCo sumo ring. A self-play PPO policy (MLX, 64×64 tanh MLP,
tanh-bounded action mean) trains against itself, an archive of its own past snapshots, and three tuned
scripted opponents; a fourth script (the CounterPusher) is **held out** and only met at evaluation.
A tournament then measures win/loss/draw with 95% Wilson intervals, cross-play against archived snapshots,
falls (tips) and ring-outs, policy diversity across the archive, and robustness to ring sizes and friction
values outside the training range. Results and caveats: [RESULTS.md](RESULTS.md).

## The game
- Robot: 0.28 × 0.20 m box chassis (2 kg), two velocity-servo wheels (r = 0.05 m, ≤ 20 rad/s ≈ 1 m/s,
  torque ≤ 0.8 N·m) and two frictionless casters. All pushing, traction and tipping are MuJoCo contacts.
- Ring: radius R (training 0.9–1.1 m, uniform); floor/wheel friction scale (training 0.8–1.2).
- A robot loses when its chassis centre leaves the ring, or when it tips (chassis up-vector z < 0.35).
  Both at once, or 10 s (250 control steps at 25 Hz) without a decision, is a draw.
- Action: left/right wheel speed in [-1, 1]. Observation (15 numbers, egocentric): own velocity and yaw
  rate, ring centre and own edge margin, opponent relative position/distance/velocity/heading, opponent
  edge margin, own uprightness. Scripts read exactly the same observation.

## Scripted opponents (`sumo_env.py`)
| name | idea | used in training? |
|---|---|---|
| `charger` | turn toward the opponent and ram | yes |
| `flanker` | get to the ring-centre side of the opponent, then push outward | yes |
| `circler` | orbit the opponent, cut in when its side is exposed, steer away from the edge | yes |
| `counter` | hold the centre, sidestep a charging opponent, then push | **no (held out)** |

Each script's parameters were grid-searched on training seeds only (`tune_scripts.py`, see RESULTS.md).

Added in cross-review (`review_scripts.py`, original scripts unchanged): `corner` aims beside the opponent's centre so the
noses meet corner to corner; tuned only against charger/flanker/circler (seeds 300, 302), evaluated on seeds 6000+.
It ties PPO head to head and beats the held-out counter by more than PPO does (RESULTS.md, review addendum).

## Commands (run from this folder)
```bash
./smoke.sh                                                    # < 30 s: train/resume/tournament/render at tiny size
../bin/gpu_run ../.venv/bin/python tune_scripts.py            # ~2.5 min: grid-search the 4 scripts -> results/tuned_scripts.json
OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python ppo.py --minutes 20 --draw_pen 0.5 --approach 1.0 \
    --p_easy 0.15 --p_self 0.25 --p_archive 0.25 --out checkpoints/main     # main run (run 3)
../bin/gpu_run ../.venv/bin/python tournament.py --ckpt checkpoints/main.pkl --n 200   # -> results/tournament.json
../.venv/bin/python make_report.py                            # -> results/tables.md, media/train_curve.png
../.venv/bin/python render_demo.py --ckpt checkpoints/main.pkl --opponents circler counter charger flanker   # -> media/demo.mp4/.gif/.txt
../.venv/bin/python render_demo.py --ckpt checkpoints/run1_passive/main.pkl --opponents charger flanker --out media/run1_passive_turtle
../.venv/bin/python render_tip_clip.py                        # -> media/exploit_tip_highfriction.mp4/.gif
```
Cross-review checks (each well under a minute; results/review_*.json):
```bash
../.venv/bin/python review_eval.py probe                      # push physics probe (head-on / side / offset)
../.venv/bin/python review_eval.py select                     # tune the corner script vs the 3 training scripts
../bin/gpu_run ../.venv/bin/python review_eval.py rr          # corner vs every entrant, n=200
../bin/gpu_run ../.venv/bin/python review_eval.py contact     # replay round robin, classify ring-outs by contact
../bin/gpu_run ../.venv/bin/python review_eval.py physics     # chassis height / tilt / speed / MuJoCo warnings
```
`ppo.py --resume` continues from `<out>.pkl` (weights, Adam state, update count, archive, RNG state, config).

## Layout
```
sumo_env.py        MuJoCo model, SumoEnv (one bout), egocentric observations, scripted opponents, play_bout
tune_scripts.py    grid search of script parameters on training seeds
ppo.py             self-play PPO in MLX with a historical-opponent archive (checkpoint/resume)
tournament.py      round robin, cross-play, robustness grid, diversity probes (4 worker processes)
make_report.py     markdown tables + training-curve plot
render_demo.py     reload a checkpoint, render bouts offscreen with MuJoCo to MP4/GIF
render_tip_clip.py find + render a high-friction self-tip bout (out-of-distribution failure)
review_scripts.py  reviewer-added angled scripts (corner = the one kept; hooker, leaner, rehit = discarded attempts)
review_eval.py     reviewer checks: corner tuning + round robin, contact replay, physics sanity, push probe
smoke.sh           end-to-end smoke test
checkpoints/       main.pkl (+ main_log.jsonl), run1_passive/, run2_passive/ (killed early runs)
results/           tuned_scripts.json, tournament.json, tables.md; review_corner_select.json, review_corner_rr.json,
                   review_contact.json, review_physics.json (cross-review)
media/             demo.mp4/.gif (PPO vs circler, counter, charger, flanker; eval seeds 7000-7003, not cherry-picked),
                   run1_passive_turtle.mp4/.gif (aborted run 1), exploit_tip_highfriction.mp4/.gif, train_curve.png
logs/              stdout of every run
```
