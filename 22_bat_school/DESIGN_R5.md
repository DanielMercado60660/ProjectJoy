# Round 5 — When is another echo worth its budget?

**Question.** Does learned sensing help when the old event rule is insufficient?
Compare a joint recurrent flight/pulse policy with seriously tuned event and
uncertainty schedulers, plus pulse-every-step references. A classical win or a
learned bat that waits until timeout is an acceptable, reportable result.

**World and information.** New files preserve the original exhibit. A grid bat
can advance, reverse, turn either way, or wait. A pulse returns after the move;
there is no free arrival pulse. Four cardinal beams give interpolated range bins
for walls/shutters and a line-of-sight moth. Sliding shutters block reciprocal
edges with periods 6–14 steps; actions encounter their *arrival-time* state.
Toy multipath adds delayed returns and viewpoint-dependent near ghosts, plus
noise/dropout. This is a coarse acoustic simulator, not realistic wave physics.
All policies receive exactly the same 107 floats: echoes, heard/bump flags,
previous action/pulse, remaining and initial budget, and a fixed-horizon clock.
No map, position, heading, goal bearing, group or obstacle phase is exposed.

**Cases.** Static, moving, ambiguous and hard moving+ambiguous groups are all
present in training and full validation. First three use 5×5 caves; hard uses
6×6, more shutters/noise/dropout. Budgets are **8, 16, 32, 64**; horizon is 320.
Train/validation/fresh-test counts are 256/32/128 caves **per group**. Independent
seeds alone duplicated two layouts in the first test; geometry-hash partitioning
now enforces separation and excludes all original train/test layouts. Fixed
evaluation consumes identical noise at each cave/time even on silent steps.
Original 5×5/6×6 test-layout prefixes remain a labelled secondary comparison
under R5 physics; they are not comparable to original physics scores.

**Methods and selection.** Train a fresh MLX GRU(96), joint five-move/two-pulse
heads and critic with recurrent PPO; no privileged critic or warm start. Reward:
goal +3, new cell +.08, step −.005, collision −.08, pulse −.004, timeout −.25.
Classical controllers maintain odometry, visit memory, static wall evidence and
decaying dynamic occupancy. Both get paired random-search candidates over echo
threshold .16–.60, wall threshold .30–.65, temperature .05–.24, decay .02–.30,
blend .3–1, interval 1–6, uncertainty .25–.95, rationing .05–1, caution .2–.7,
repeat allowance 0–3, exploration .2–4, moth threshold .12–.5, route rule and
handedness. Every candidate and neural selection uses **all 16 group×budget
cells**. Completion ranks first; a bounded collision/time/pulse term only breaks
completion ties, so refusing all completions cannot win. Test data never select.
Every-step classical uses the best
validated controller; learned-every overrides only the learned pulse head.

**Evidence.** Save episode rows, completion, collisions/step and /episode,
pulses (all episodes and successes), steps, timeout/coverage, waits, consecutive
listens, and turn/translation-listen frequencies. Repeat/turn Brier improvements
use contemporaneous truth and an independent fixed decoder. Translation compares
only the shared edge from opposite sides. These are sensor-quality proxies,
not evidence of intention or the network's beliefs. Preserve the first hard-case
failure and success clips, explicitly recording any absent category.

**Budget and custody.** Battery plan: smoke ≤1.9 min, baseline search 5, PPO 24,
fresh evaluation ≤1.5, legacy ≤1, demo ≤.5: ≤33.9 min. Measured full-validation
candidate pair: 2.67 s (`results/r5/delivery_speed.json`), about 111 pairs in five
minutes. This replaces a tiny fixed sweep. Earlier PPO measured ~19k env steps/s;
new MLX speed is unmeasured. Main training has **no update cap**: measure after
four updates, log projected work, and fill 24 minutes. Signal/resume and finite
gradient checks must pass first. Atomic working NPZ holds weights, optimizer,
step, config, both RNGs, environment and recurrent state; only one selected NPZ
needs committing. AC may add two equally budgeted seeds; report all seeds.
