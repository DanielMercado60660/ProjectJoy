# 21 Duck Hunt, round 6: the hunter game ("Go deeper")

State-based arcade duel built on the flight sim, as designed in `HUNTER_DESIGN.md` (written before
coding). An evasive duck must pass 6 rings in order within 11 s; hiding, timing out and leaving the
arena all count as losses. The hunter moves a crosshair at up to 14 m/s and sees the duck's position
with 0.3 m noise. It has 3 shells; each shot lands 0.25 s after the trigger at the crosshair's
position, with a 0.4 s cooldown and a 0.6 m hit radius. No pixels are involved, so this does not
depend on the tracker work.
**Demo:** `media/hunter_demo.html` (self-contained, opens offline). It replays 35 held-out episodes
across 9 matchups, including a held-out opponent and a preserved failure.

## Headline (500 held-out courses, new course seed 5,777,000, play seed 7; `results/r6h_eval.json`)

| finding | numbers (hit rate; paired 95% bootstrap CI over courses, `results/r6h_ci.json`) |
|---|---|
| **Learned hunter (after self-play) beats the best scripted hunter** on evaders like those it trained against | `hunter_g8` vs the *tuned* lead-KF hunter: straight 0.748 vs 0.620 (+0.128 [+0.072, +0.184]), jink 0.826 vs 0.688 (+0.138 [+0.086, +0.190]), dodge 0.686 vs 0.594 (+0.092 [+0.032, +0.152]) |
| ...but **only ties it on the held-out evader** | dodge_jink 0.614 vs 0.626 (−0.012 [−0.070, +0.046]) |
| The stage-1 hunter (trained vs scripted only) does **not** beat the tuned lead-KF; self-play is what made it strong | `hunter_s` − lead_tuned: −0.026 / −0.052 / −0.046 (straight / jink / dodge, all CIs include 0), −0.148 [−0.206, −0.088] on dodge_jink |
| **Learned duck evades lead-KF hunters**: its training opponent (default lead-KF) and the tuned lead-KF it never saw | hit by tuned lead-KF: duck_g8 0.508 vs straight 0.620 (−0.112 [−0.166, −0.056]); by default lead-KF: 0.358 vs 0.512 (−0.154 [−0.208, −0.098]) |
| ...but **not the held-out constant-acceleration KF hunter** | hit by lead_ca: 0.498 vs 0.524 (−0.026 [−0.084, +0.030]) |
| **The arms race favours the hunter** | final duck vs final hunter: 70.0% hit, only 30% survive; final duck vs stage-1 duck against the final hunter: −0.052 [−0.104, +0.000] |
| Course discipline holds | learned ducks' worst crash + timeout rate against any hunter is ≤ 2.0%; survivors' course time 6.75–7.20 s vs 6.47 s for the racing line |

Without tuning, the scripted lead-KF hunter is a weak baseline. Its hand-set parameters gave a
0.54 hit rate on training courses; 1001 tuning configurations raised that to 0.73
(`tune_lead_r6h.py`, `results/r6h_lead_tuning.json`). Before the tuning, the learned hunter looked
far better than "classical" (0.75 vs 0.51 on straight); against the tuned baseline the margin
roughly halves (+0.13). Both evaluations are kept: `results/r6h_eval_pre_leadtune.json` and
`results/r6h_eval.json`.

## Agents

| agent | what | budget |
|---|---|---|
| `straight` | PD+ racing line (R5's `pd_plus.py` params), ignores the hunter | — |
| `jink` | PD+ + random-telegraph lateral thrust (0.25 × AMAX, flips every 0.2–0.6 s); magnitude cut from 0.7 because at 0.7 it could not finish within 11 s | hand-set |
| `dodge` | PD+; while a shot is pending within 1.5 m, full thrust away from the aim point | hand-set |
| `dodge_jink` | **held out**: both of the above | hand-set |
| `naive` | crosshair chases the latest noisy position; fires when within 0.5 m | hand-set |
| `lead` | constant-velocity Kalman filter; aims at the predicted position at impact time; fires when on target and the filter has settled | hand-set |
| `lead_ca` | **held out**: constant-acceleration KF, tighter fire rule | hand-set |
| `lead_tuned` | lead hunter, 6 params tuned (filter order, process noise, fire tolerance, min observations, max std, lead-horizon multiplier); tuned on training courses against straight/jink/dodge | 1001 configs × 3 evaders × 128–256 courses (≈ 1e8 sim steps at most) |
| learned duck | PPO; action = PD+ + 1.0 × policy (residual, zero-init, so it starts as `straight`); sees its flight obs + crosshair, ammo, cooldown and the pending shot | stage 1: 12M steps; self-play: 8 × 5M |
| learned hunter | PPO; sees the last 6 noisy positions relative to the crosshair (no filter), crosshair, ammo, cooldown, time; actions = crosshair velocity + trigger | stage 1: 12M steps; self-play: 8 × 5M |

**Rewards.**
- Duck: +1 survive, −1 hit, −1 crash or timeout, plus 0.1-weighted potential shaping on the
  remaining path.
- Hunter: +1 hit, −1 if the duck survives, −0.02 per shot, plus potential shaping on the
  crosshair-to-duck distance. The shaping uses the true position, at training time only.

**Self-play.** Each generation trains the duck for 5M steps against the hunter archive, then the
hunter for 5M steps against the duck archive (which now includes the new duck). Each opponent is
drawn per episode: the newest with p = 0.5, the rest uniformly. Scripted agents stay in the
archives.
- Learner weights and Adam state carry over between generations.
- State is saved after every generation, and resume was verified at tiny size.
- Total self-play time: 30.0 min in one process (generations took 96–323 s each).
- Stage 1: 135 s (duck) and 73 s (hunter). The PPO update runs on the MLX GPU stream.

## Results

### Ducks vs scripted hunters (survival = completed and not hit; hit rate in brackets)

| duck | naive | lead (KF) | lead_ca (held out) | lead_tuned | course time of survivors vs lead |
|---|---|---|---|---|---|
| straight | 0.806 (0.194) | 0.488 (0.512) | 0.476 (0.524) | 0.380 (0.620) | 6.47 s |
| jink | 0.746 (0.246) | 0.456 (0.540) | 0.468 (0.530) | 0.310 (0.688) | 7.35 s |
| dodge | 0.694 (0.124) | 0.430 (0.568) | 0.598 (0.400) | 0.388 (0.594) | 6.94 s |
| dodge_jink (held out) | 0.606 (0.136) | 0.414 (0.572) | 0.626 (0.352) | 0.350 (0.626) | 7.83 s |
| duck_s (stage 1) | 0.840 (0.156) | 0.624 (0.376) | 0.496 (0.504) | 0.434 (0.566) | 6.75 s |
| duck_g8 (self-play) | 0.800 (0.194) | 0.640 (0.358) | 0.502 (0.498) | 0.492 (0.508) | 7.20 s |

- Random jinking never helps against a Kalman hunter.
- Scripted dodging helps against the naive and constant-acceleration hunters, not against the
  lead-KF hunters (default or tuned). Dodging costs the duck crashes: dodge 18% and dodge_jink 26% worst-case
  crash + timeout, because dodging fully away from a low shot can fly the duck into the ground
  (preserved in the demo as "FAILURE: ... crashes while dodging").
- The learned ducks keep their crash + timeout rate at ≤ 2%.

### Hunters vs scripted ducks (hit rate; shots per course in brackets)

| hunter | straight | jink | dodge | dodge_jink (held out) |
|---|---|---|---|---|
| naive | 0.194 (2.81) | 0.246 (2.77) | 0.124 (2.85) | 0.136 (2.86) |
| lead | 0.512 (2.43) | 0.540 (2.35) | 0.568 (2.36) | 0.572 (2.31) |
| lead_ca (held out) | 0.524 (2.44) | 0.530 (2.42) | 0.400 (2.56) | 0.352 (2.58) |
| lead_tuned | 0.620 (2.31) | 0.688 (2.21) | 0.594 (2.31) | 0.626 (2.27) |
| hunter_s (stage 1) | 0.594 (2.37) | 0.636 (2.30) | 0.548 (2.44) | 0.478 (2.47) |
| hunter_g8 (self-play) | **0.748** (2.06) | **0.826** (1.95) | **0.686** (2.19) | 0.614 (2.30) |

The learned hunter uses fewer shells per course for more hits, i.e. it waits for better shots.

### Self-play generations (hit rate; HO = held-out opponent)

| gen | duck_g vs lead | vs lead_ca (HO) | vs lead_tuned | vs final hunter | vs all learned hunters (mean) | hunter_g vs straight | vs dodge_jink (HO) | vs final duck | vs all learned ducks (mean) |
|---|---|---|---|---|---|---|---|---|---|
| s | 0.376 | 0.504 | 0.566 | 0.752 | 0.712 | 0.594 | 0.478 | 0.414 | 0.470 |
| 1 | 0.378 | 0.508 | 0.548 | 0.748 | 0.692 | 0.670 | 0.516 | 0.530 | 0.571 |
| 2 | 0.356 | 0.522 | 0.506 | 0.758 | 0.679 | 0.734 | 0.490 | 0.632 | 0.654 |
| 3 | 0.326 | 0.494 | 0.534 | 0.766 | 0.690 | 0.762 | 0.506 | 0.658 | 0.702 |
| 4 | 0.372 | 0.482 | 0.532 | 0.766 | 0.673 | 0.726 | 0.584 | 0.688 | 0.721 |
| 5 | 0.370 | 0.476 | 0.528 | 0.728 | 0.656 | 0.744 | 0.610 | 0.674 | 0.715 |
| 6 | 0.360 | 0.472 | 0.528 | 0.736 | 0.641 | 0.724 | 0.578 | 0.686 | 0.725 |
| 7 | 0.356 | 0.474 | 0.502 | 0.704 | 0.640 | 0.718 | 0.528 | 0.658 | 0.712 |
| 8 | 0.358 | 0.498 | 0.508 | 0.700 | 0.627 | 0.748 | 0.614 | 0.700 | 0.740 |

- **Performance against archived opponents.**
  - Hunters: each new hunter is stronger against the whole learned-duck archive (mean hit
    0.47 → 0.74), most of it in generations 1–3.
  - Ducks: each new duck is harder to hit for the archived learned hunters (mean hit 0.71 → 0.63)
    and for the final hunter (0.75 → 0.70).
  - The latest agents beat the past on both sides. The final hunter's hit rate falls with the duck's generation
    (0.75–0.77 for stage 1 to generation 4, 0.70 for the final duck), and the final duck is hit
    less by older hunters.
- **Asymmetry.** The duck's progress is slow and its gain over the stage-1 duck against the final
  hunter is borderline (CI touches 0). The hunter gained much more. Plausible reasons, not
  isolated:
  - The duck's evasion is limited by physics: 0.25 s of warning at 2 g.
  - The duck must still finish the course on time.
  - The residual duck's exploration noise collapsed quickly (log-std ≈ −2.2 by generation 2).
- **Partial transfer.** Against the held-out `lead_ca` hunter, the learned ducks are no better
  than the racing line (0.50 vs 0.52 hit). Against the tuned lead-KF, which they also never saw,
  they cut hits from 0.62 to 0.51.
  - Tuning turned the lead hunter into a constant-acceleration filter as well (order 2), so the
    difference is not "constant-velocity vs constant-acceleration prediction".
  - `lead_ca` differs mainly in its fire rule: 0.2 m tolerance, needs 6 observations, and aims a
    full shot-delay ahead (the tuned hunter aims 0.67 of the delay ahead with 0.44 m tolerance).
  - Which feature the duck exploits is not isolated.

Figure: `media/r6h_matrix.png` (survival and hit-rate matrices, every duck × every hunter).

## Not done / caveats

- One seed for every learned agent (stage 1 and self-play). CIs cover course variation, not seed
  variation.
- The scripted evaders (jink, dodge) were hand-set; only the lead hunter got a tuning budget. The
  learned duck therefore has no tuned classical evader to beat. Its comparison is against the
  racing line (`straight`), which is the strongest scripted duck against every KF hunter here.
- The hunter's training shaping uses the duck's true position (training only; at play time it sees
  only noisy positions).
- Not built: pixel-based hunting with the tracker, and a hunter that knows the course (it would
  likely be much stronger, since the duck must pass known rings).

## Files and commands

- **Code:** `hunter.py` (env + scripted agents), `train_hunter.py` (stage 1 and resumable self-play),
  `tune_lead_r6h.py`, `eval_hunter.py`, `r6h_report.py`, `r6h_ci.py`, `make_hunter_demo.py` +
  `hunter_demo_template.html`, `r6h_chain.sh`.
- **Checkpoints:** `checkpoints/r6h/*.npz` (every archived agent), `checkpoints/r6h/selfplay_state.pkl`,
  `checkpoints/r6h_lead_tuned.json`.
- **Results:** `results/r6h_eval.json`, `results/r6h_tables.md`, `results/r6h_ci.json`,
  `results/r6h_stage1.json`, `results/r6h_selfplay_archive.json`, `results/r6h_train_*.jsonl`,
  `results/r6h_lead_tuning.json`.

```sh
sh r6h_chain.sh                                           # stage 1 -> self-play (resumable) -> eval
../bin/gpu_run ../.venv/bin/python tune_lead_r6h.py       # tuned lead-KF baseline
../bin/gpu_run ../.venv/bin/python eval_hunter.py && ../.venv/bin/python r6h_report.py && ../.venv/bin/python r6h_ci.py
../.venv/bin/python make_hunter_demo.py                   # media/hunter_demo.html
```
