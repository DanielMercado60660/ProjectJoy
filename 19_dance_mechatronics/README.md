# 19 Dance Dance Mechatronics: a planar robot learns a kick-dip groove

A simulated **Walker2d** (gymnasium `Walker2d-v5` body: planar, 6 actuated joints, ±100 Nm) dances a
short synthetic clip at a fixed tempo: **right heel-kick on beat 1, dip on beat 2, left heel-kick on
beat 3, dip on beat 4**, at 90 BPM (2.67 s loop), in place. Three controllers track it:

* **PD-only**: joint PD servos on the reference, tuned gains. This is also exactly what the RL policy
  does with a zero action.
* **Baseline**: tuned PD + reference inverse-dynamics feed-forward + a balance term (stance-ankle
  torque on COM error, stance-hip torque on torso-pitch error). Variants: foot-relative COM error, and
  push-tuned. 12 parameters, tuned by random search
  plus an evolution strategy (about 4000 evaluations) on the same reward the RL maximises, training seeds only.
* **Anchored baseline** (added in review, `baseline_anchor.py`): the foot-relative baseline plus a bounded
  COM-target shift and swing-foot placement on the root-x drift, so it holds its spot without the original's
  runaway. Same tuning protocol (`tune_anchor.py`). It narrows PPO's foot-error edge from 5.6× to about 2×
  but not its push edge (RESULTS.md).
* **PPO (MLX)**: DeepMimic-style motion imitation. Phase input, reference-state initialisation (random
  start phase), early termination on falls, DeepMimic tracking reward. The policy outputs a residual on
  the PD targets (`target = q_ref + 0.1 rad * a`), with the same PD gains as the baseline.

The body is **planar**, so the "weight shift" is fore-aft over the stance foot, not side-to-side.
Walker2d's hips can only extend (thigh range -150..0 deg), so the moves are heel kicks and knee dips,
not knee lifts.

See **RESULTS.md** for numbers, the feasibility story, and what was surprising.

## Why the clip is physically plausible
There is no mocap, so the clip is keyframed (`clip.py`). Only the joint channels are hand-set. Per frame
the stance foot is pinned flat on the floor, and the whole-body lean is solved so that the **centre of
pressure from inverse dynamics** (`mj_inverse` on the finite-differenced trajectory) sits at mid-foot. A
cart-table correction is iterated in the Fourier domain. `feasibility()` then checks that the CoP stays
inside the sole, the normal force stays positive, the friction cone holds, and joint torques stay
under 100 Nm. The first draft (100 BPM, one-beat kicks) failed this check badly. The shipped clip
passes, and a tuned controller tracks it without falling (RESULTS.md).

## How to run (from this folder)
```bash
PY=../.venv/bin/python
./smoke.sh                                          # end-to-end at tiny size, < 2 min
$PY clip.py                                         # build the clip, print the feasibility report
$PY feasibility.py                                  # clip sweep -> results/feasibility_sweep.json, media/feasibility.png
# baseline tuning (training seeds 100+); --push 60 = tune under the same random pushes as PPO run 2
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 ../bin/gpu_run $PY tune_baseline.py --n_random 3000 --gens 40 --out results/baseline_tuned.json
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 ../bin/gpu_run $PY tune_baseline.py --n_random 3000 --gens 40 --push 60 --train_seeds 8 --out results/baseline_tuned_push8.json
# PPO run 1: nominal training (4 rollout workers, MLX update on CPU)
OMP_NUM_THREADS=1 ../bin/gpu_run $PY ppo.py --residual pd --minutes 40 --act_scale 0.1 --init_logstd -1.6 --out checkpoints/ppo_pd
OMP_NUM_THREADS=1 ../bin/gpu_run $PY ppo.py --residual pd --minutes 18 --act_scale 0.1 --init_logstd -1.6 --out checkpoints/ppo_pd --resume
# PPO run 2: continue from run 1 with random training pushes U(0, 60) N
OMP_NUM_THREADS=1 ../bin/gpu_run $PY ppo.py --residual pd --minutes 40 --act_scale 0.1 --push 60 --init_from checkpoints/ppo_pd.pkl --out checkpoints/ppo_push
# PPO run 3: continue from run 2, residual range 0.1 -> 0.25 rad (output rescaled), pushes U(0, 100) N
OMP_NUM_THREADS=1 ../bin/gpu_run $PY ppo.py --residual pd --minutes 40 --act_scale 0.25 --push 100 --lr 3e-5 --target_kl 0.02 --init_from checkpoints/ppo_push.pkl --out checkpoints/ppo_push100
# foot-relative-COM baseline variant
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 ../bin/gpu_run $PY tune_baseline.py --n_random 3000 --gens 40 --com_rel --out results/baseline_tuned_rel.json
# review: anchored baseline (nominal and push-tuned), evaluated on the same held-out seeds
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 ../bin/gpu_run $PY tune_anchor.py --n_random 3000 --gens 40 --out results/baseline_tuned_anchor.json
NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 ../bin/gpu_run $PY tune_anchor.py --n_random 3000 --gens 40 --push 60 --train_seeds 8 --warm results/baseline_tuned_anchor.json --out results/baseline_tuned_anchor_push8.json
OMP_NUM_THREADS=1 $PY eval.py --seeds 16 --loops 10 --controllers baseline_anchor --ckpt baseline_anchor=results/baseline_tuned_anchor.json \
    --conditions nominal push40 push50 push60 push70 push80 push100 push120 tempo0.9 tempo1.1 friction0.3 mass1.3 --out results/eval_review_anchor.json
$PY render_demo.py --ckpt checkpoints/ppo_pd.pkl --cond nominal --loops 2 --baseline results/baseline_tuned_anchor.json --out media/review_anchor_nominal
# evaluation on held-out seeds 1000+ (16 seeds x 10 loops per condition), ~3 min with 4 workers
OMP_NUM_THREADS=1 ../bin/gpu_run $PY eval.py --seeds 16 --loops 10 \
    --conditions nominal push40 push50 push60 push70 push80 push100 push120 tempo0.9 tempo1.1 friction0.3 mass1.3 \
    --ckpt baseline_rel=results/baseline_tuned_rel.json baseline_push=results/baseline_tuned_push8.json \
           ppo_pd=checkpoints/ppo_pd.pkl ppo_push=checkpoints/ppo_push.pkl ppo_push100=checkpoints/ppo_push100.pkl \
    --out results/eval_final.json
$PY make_plots.py --eval results/eval_final.json --ckpts checkpoints/ppo_pd.pkl checkpoints/ppo_push.pkl checkpoints/ppo_push100.pkl --baseline_score 0.9883
# demo videos (reload checkpoints; baseline panel always first)
$PY render_demo.py --ckpt checkpoints/ppo_pd.pkl --cond nominal --loops 2 --out media/demo_nominal
$PY render_demo.py --ckpt checkpoints/ppo_pd.pkl checkpoints/ppo_push.pkl --cond push60 --loops 3 --seed 1005 --gif_every 3 --out media/demo_push60_seed1005
$PY render_demo.py --ckpt checkpoints/ppo_push.pkl --cond push80 --loops 2 --seed 1002 --out media/demo_push80_failure
```
The exact commands and outputs used for RESULTS.md are listed in RESULTS.md ("Provenance").

## File layout
| file | what |
|---|---|
| `walker2d_dance.xml` | Walker2d-v5 body unchanged; motors replaced by PD servos with the same ±100 Nm limit |
| `clip.py` | keyframed reference, CoP-based lean solve, inverse-dynamics feasibility check |
| `dance_env.py` | 50 Hz tracking env (500 Hz RK4 sim), DeepMimic reward, baseline controller, metrics |
| `tune_baseline.py` | random search + ES tuning of the baseline on training seeds |
| `baseline_anchor.py`, `tune_anchor.py` | review addition: anchored baseline and its tuner (same protocol) |
| `ppo.py` | MLX PPO, rollout workers, checkpoint/resume (`--resume`, `--init_from`) |
| `eval.py` | matched evaluation across conditions -> `results/eval_*.json` |
| `render_demo.py` | offscreen MuJoCo render with reference ghost and beat light -> `media/` |
| `feasibility.py` | clip sweep and feasibility plot |
| `make_plots.py` | training curves and robustness bars |
| `smoke.sh` | < 2 min end-to-end check |
| `checkpoints/` | `ppo_pd.pkl` (run 1, nominal), `ppo_push.pkl` (run 2, pushes ≤60 N), `ppo_push100.pkl` (run 3, pushes ≤100 N); weights, Adam state, obs stats, RNG, config, full history |
| `results/` | tuned baseline params, eval tables, feasibility sweep |
| `logs/` | training / tuning logs |
| `media/` | demo MP4/GIF, beat contact sheets, plots |
