# S2 option: a small latent world-action model (DreamZero idea, Microduck scale)

Status: planned arm for LeWM S2, added 2026-11-04 at the owner's request. Run only if S1's gates pass
(design/LEWM_FEASIBILITY.md, results/lewm/protocol.json).

## Where the idea comes from
DreamZero (NVIDIA, arXiv 2602.15922, code Apache-2.0 at github.com/dreamzero0/dreamzero): a "world
action model" that denoises future video and the next action chunk together with one flow-matching
loss, built on a 14B image-to-video backbone (Wan2.1-I2V-14B). Trained on ~500 h of AgiBot teleop plus
DROID, it roughly doubles VLA task progress on unseen tasks and environments and adapts to a new robot
from ~30 min of play data. Stated limits: heavy compute (≥2 datacenter GPUs; 0.15–3 s per chunk),
**short horizon (~6 s), memory-dependent tasks not evaluated**, most failures come from bad video
predictions.

We cannot run DreamZero here (size, CUDA, MLX-only shelf). What transfers is the training signal:
predicting what happens next and what to do next in one model, so the action head has to model the
scene's consequences.

## The Microduck version
- Inputs: the frozen S1 LeWM encoder's latents for the last K frames (plus IMU/proprio and the task
  spec the current learner already gets), and the policy's recurrent memory state.
- Output, jointly denoised by one flow-matching head: the next H LeWM latents **and** the next H
  10 Hz commands (vx, wz, head, press/done heads as in R3/H; discrete heads stay discrete, as a
  classifier on the same trunk).
- Size: tens of millions of parameters at most. Training data: the same 2,729 + 616 noisy training
  episodes (training rooms only).
- Execution: run the first few commands of each chunk, then re-plan with real observations
  (DreamZero's trick of replacing predicted frames with real ones, which avoids compounding error).

## Comparison (S2, same budget, 2 seeds each, 80 confirmation rooms looked at once)
1. R3/H policy with its own encoder (current best, 0–1 press-needed successes).
2. R3/H policy with the frozen LeWM encoder swapped in (the main S2 arm).
3. **WAM arm:** same trunk and encoder as 2, with joint latent+action flow matching.
4. Ablation of 3: identical, but the future-latent loss is switched off (action chunks only), to
   separate "predicting the future" from "predicting action chunks".

Bars as in S2: ≥2 press-needed successes per seed and ≥5 in total, with no rise in done-without-press.
Also report the latent prediction error against copy-last, per horizon step.

## Expected limits (say so up front)
A ~6 s look-ahead does not cover the 10–50 s go-to-plate and return. If the bottleneck is task stage and
memory rather than perception, arms 2–4 will tie; that is a valid result and points back to skill
selection / a stage memory. The skill-selection plan (STATUS "Where we stopped", 2026-10-27) stays the
fallback.
