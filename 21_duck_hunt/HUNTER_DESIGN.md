# Hunter game: design (R6, written before coding)

State-based arcade duel on top of the flight sim. No pixels, so it does not depend on the
tracker work. Everything lives in `hunter.py` (env + scripted agents), `train_hunter.py`
(PPO for either side, opponent pools, population self-play), `eval_hunter.py` (cross-play
tables) and `make_hunter_demo.py` (self-contained `media/hunter_demo.html`).

## Game

- **Duck**: the same point mass as `flight.py` (gravity, quadratic drag, 2 g thrust limit,
  0.1 s actuator lag, OU wind σ = 2 it does not observe), dt = 0.05 s.
- **Timed course**: 6 waypoints (radius 1 m, `flight.make_courses`), in order, within
  **T_LIM = 11 s** (the fastest flyers need about 6.4 s, so there is roughly 4.5 s of slack to
  spend on evasion). Timeout, touching the ground or leaving the arena = the duck loses. So
  hiding or leaving the screen cannot win.
- **Hunter**: a crosshair that moves at most **V_CROSS = 14 m/s** (the duck's top speed is
  about 13–15 m/s, so it cannot simply sit on the duck), **3 shells** per course, a
  **0.25 s shot delay** (the shot lands where the crosshair was when the trigger was pulled,
  5 steps later), a **0.4 s cooldown** between shots, hit radius **0.6 m**
  (duck within 0.6 m of the aim point at impact time).
- **Episode ends** when the duck is hit (hunter wins), completes the course (duck wins), or
  fails (crash / timeout: duck loses, counted as "not survived" and "not hit").

## Observations (both sides see the same world; no privileged course info for the hunter)

- Hunter: the duck's position with Gaussian noise σ = 0.3 m each step (no velocity), its own
  crosshair, shells left, cooldown, time. The learned hunter gets the last 6 noisy
  positions relative to the crosshair (a frame stack), not a filter: it has to learn the
  prediction.
- Duck: its 11 flight observations + crosshair offset and velocity relative to the duck, shells
  left, cooldown, and the pending shot (aim-point offset and time to impact, zero if none). The
  duck seeing the shot in flight is the "muzzle flash" rule: with 0.25 s of warning and 2 g it
  can move about 0.4–0.6 m, so dodging is possible but not free.

## Agents

Scripted evaders (baselines and held-out opponents):
- `straight`: PD+ racing line, ignores the hunter (fastest, most predictable).
- `jink`: PD+ plus a random-telegraph lateral thrust (switches side every 0.2–0.6 s).
- `dodge`: PD+ that, when a shot is pending, thrusts at full 2 g away from the aim point.
- held out (never trained against): `dodge_jink` (both of the above combined).

Scripted hunters:
- `naive`: crosshair chases the latest noisy position, fires when within 0.5 m.
- `lead`: constant-velocity Kalman filter on the noisy positions, aims at the predicted
  position at impact time (lead pursuit), fires when the crosshair is on the lead point and
  the KF position variance is small.
- held out (never trained against): `lead_ca`, a constant-acceleration Kalman filter with a
  different fire rule.

Learned (PPO in MLX, CPU, same machinery as `train_flight_r6.py`):
- Duck: zero-sum reward +1 survive / −1 hit / −1 fail, plus potential-based progress shaping
  (time-to-go, as in R6 flight) so it learns to fly first. Actions = thrust (2-D).
- Hunter: +1 hit / −1 when the duck survives, 0 on duck failure (the duck failed by itself);
  small −0.02 per shell fired to discourage spraying. Actions = crosshair velocity (2-D, scaled
  to V_CROSS) + trigger (fires when the 3rd output > 0).

## Training plan

1. Duck vs scripted hunters (mixture of `naive`, `lead`), hunter vs scripted evaders
   (`straight`, `jink`, `dodge`): learned vs scripted, both directions.
2. **Population self-play**: generations g = 1..G; in each, the duck trains vs opponents
   sampled from the hunter archive (scripted + every past learned hunter snapshot), then
   the hunter trains vs the duck archive. Each new snapshot joins the archive.

## Measures

For each (duck, hunter) pair on 500 held-out courses (new seed): hit rate, survival rate
(completed without being hit), course completion, mean completion time of survivors, shots
per course. Reported: learned vs scripted both ways, the final self-play agents vs every
archived opponent (does the latest beat the past?), and vs the held-out scripted opponents
(`dodge_jink`, `lead_ca`). Interesting failures get a clip in the demo.

## Deviations made while building (R6, after coding)

- The learned duck is a residual on PD+ (action = PD+ + 1.0 × policy, zero-init), so it starts by
  flying the racing line and learns evasion on top. Learning flight from scratch inside a duel was
  judged too slow for the time box.
- `jink` magnitude was cut from 0.7 to 0.25 × AMAX: at 0.7 the jinking duck could not finish
  within 11 s.
- Added a fairness baseline after the first evaluation: `lead_tuned`, the lead hunter with its 6
  parameters tuned (1001 configs) on training courses against straight/jink/dodge.
  `tune_lead_r6h.py` adds the lead-horizon multiplier `H_MULT` as one of those parameters. The
  untuned scripted hunters stay in the tables, and the pre-tuning evaluation is kept in
  `results/r6h_eval_pre_leadtune.json`.
- The hunter's training reward adds potential shaping on the true crosshair-to-duck distance
  (training only). The duck's adds shaping on the remaining path.
