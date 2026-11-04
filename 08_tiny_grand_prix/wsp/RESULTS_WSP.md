# Round 6: weight-space programming (WSP) on Tiny Grand Prix

**Question.** Can a frontier LLM read and write the weights of a tiny driving policy directly,
without gradient descent? We copied the setup of the MIT CSAIL "weight-space programming" post
(rikiyat.github.io, Nov 2026): the LLM sees weights as text plus the I/O layout and task, and
either answers from the weights alone or edits them with a small evaluation budget, keeping an edit
only if it scores better. **Astra** (OpenAI gpt-6-astra, Codex CLI, reasoning effort xhigh, Codex
sandbox, numpy only) wrote and read the weights. **Claude** set up the harness, ran the baselines
and scored everything on fresh held-out tracks. All numbers are from runs on 2026-11-04,
15:16–16:10 CST.

## TL;DR

* **Read (which policy wall-rides?): 3/3 correct, plus a correct "neither" on a catch trial.**
  Astra named the soft-wall policy at 95–97% confidence, from weights alone, with no rollouts and
  no code. It also said "neither" (88%) when both policies were legal drivers. **The units it named
  do not matter, though.** Clamping them changes wall contact by +0.2 to +0.4 steps per lap,
  inside the random-unit null. In a scan of all 64 units, no single unit matters either. The hack
  is not a circuit you can remove. It is a missing brake: the pedal logit is ≥ 3.6 on 99% of steps
  (median 4.5), so the car is at full throttle 100% of the time.
* **Edit (remove the wall-riding): it works, and generalises.** Both 25-call attempts reached 32/32
  legal laps on the training tracks. They used 12 and 10 calls (640–768 one-lap episodes,
  164–195k env steps). On **200 fresh held-out tracks** both finish **100%**, in **25.47 s / 25.77 s**
  per lap. For comparison, the tuned ray follower does 25.82 s and ES trained from scratch under
  real walls does 25.09 s. On **100 fresh hard (OOD) tracks** they finish **98% / 96%**, against
  74% for ES-from-scratch and 79% for the tuned ray follower, at about 1 s per lap slower.
  **Matched-budget baselines get nothing:** 25-call hill-climb (6 runs), ES fine-tune at 1× and
  10× the episode budget (3 seeds each) all finish 0%. ES needs **100× the episodes**
  (160k episodes, 56–79M steps, ~290–480× Astra's steps) to fix it (99–100%).
  **One-shot, with zero rollouts**, Astra's edits score 97.5% and 65.5% held-out.
* **But it is programming, not surgery.** Both attempts wrote a speed governor into spare units
  (target speed = f(forward clearance), pedal = tanh(gain·(target − speed))) and added a
  steering correction. They changed 14–18% of the weights, against 3.5% in the post's Kinetix
  edit. This is the shelf's standing lesson made literal: Astra compiled a hand-written controller
  into the net. The new pedal behaves more like ES-trained braking (r = 0.87–0.94) than like the
  ray follower (r = 0.54–0.64).
* **Grip (grip 0.5 with a policy trained at 1.0): an obstacle removed, not an ability added.**
  Both attempts found the same physics edit: scale the speed input's 32 weights by √2. Halving μ
  halves v² in a corner, so the car now drives as if it were 1.41× faster. That single row takes
  held-out grip-0.5 completion from **4.5% to 100%**. Fixing the grip-1.0 side effects with a few
  more edits gives **100% / 99.5% at 0.5 and 94% / 99.5% at 1.0** (4 and 8 calls). This beats
  matched ES (71–87% at 0.5, 3 seeds, with 3–6× more episodes) and grip-randomized PPO
  (92.5%). It costs **~4 s per lap at grip 1.0** (29.0 s vs 25.1 s): the net still cannot sense
  grip, so it now drives cautiously everywhere. ES at 10× budget matches it at 0.5 (95–99.5%) and
  is faster at 1.0 (27.5 s).
* **Verdict.** On this toy, WSP is a strong prior-driven programmer. Where the fix is "install a
  braking rule" or "rescale an input by known physics", it beats matched-budget search by orders of
  magnitude and its fixes generalise. Its *reading* is right at the policy level and wrong at the
  unit level. Here the bug is an absence, and ablation cannot find an absence.

## Setup

**Checkpoints** (both 13→32→32→2 tanh MLPs, 1,570 weights):
* Hacking policy: `checkpoints/es_soft_best.npz`, ES trained with soft walls (150 gens). It wall-rides:
  21.8 s laps under soft walls with ~20 wall-contact steps per lap, 0% under strict rules.
* Control and grip start: `checkpoints/es_best.npz`, ES trained with real walls (300 gens). It
  finishes 100% held-out at grip 1.0 but ~3–5% at grip 0.5.
* Catch-trial pair: `checkpoints/es_s2_best.npz` (ES seed 2, real walls) vs `es_best`.

**Harness (all in `wsp/`).**
* `astra_dir.sh`: same as `bin/astra_turn` (gpt-6-astra, xhigh, logs in
  `round5_logs/08_tiny_grand_prix_r6wsp_*`), but it runs Codex in an attempt subfolder, and it uses a
  read-only sandbox for the Read experiment. `bin/astra_turn` reuses its first argument for both
  the folder and the log name, so it cannot start Codex in a subfolder.
* `eval_train.py`: the **only** sanctioned simulator entry point for editors. It drives a candidate
  one lap from the start line on **train tracks 0–31**: strict and soft rules at grip 1.0 (hack
  task), or strict at grips 0.5 and 1.0 (grip task). It counts calls against the budget, archives
  every candidate (`candidates/callNN_<sha>.npz`), logs metrics, episodes and env steps
  (`eval_log.jsonl`), and copies a candidate to `best.npz` only if its score key beats the best so
  far. **The scored policy is the wrapper's `best.npz`, not the one Astra says is best.** Each call
  returns per-track results plus the last 15 observations and actions before up to 4 crashes.
  One call takes ~2.5 s in the Codex sandbox (turn 0 smoke).
* Score keys, fixed before any attempt. Hack: (strict completion, strict mean progress, −strict lap).
  Grip: (grip-1.0 completion ≥ start − 2/32, completion@0.5, progress@0.5, −lap@0.5).
* Rules given to Astra: write only in its attempt folder; no pip, no downloads, no git; no
  `racing/` import and no simulator re-implementation (reading `racing/*.py` is allowed); no other
  checkpoint, results or write-ups; no gradient descent, evolution or random search. Numpy on the
  weights themselves with hand-made inputs is allowed.
  **Audit:** we grepped every command in every jsonl. No reads of checkpoints, results or
  write-ups, no `racing` imports, and no evaluation calls outside the wrapper. The zero-budget
  threads ran no evaluations. The Read threads ran only `cat` on the three files in their folder.
  Both edit threads grepped `~/.codex/memories/MEMORY.md` for this project and found nothing about
  08.
* **Held-out scoring:** after the Astra turns, Claude picked **heldout seeds 1,090,000–1,090,199
  (`make_tracks("heldout", 200, start=90000)`) and hard seeds 2,090,000–2,090,099** (100 tracks).
  These ranges are used nowhere else in the folder (eval.py uses 0–99 / 0–49; R5 uses 30000–70063).
  No Astra thread saw them. `heldout_eval.py`, `ablate.py`, `unit_scan.py`, `mechanism_check.py` and
  `action_similarity.py` all use them. The Exp 1 ablations ran on them while the edit threads were
  still running; that is harmless, since those threads only had the wrapper's train tracks.

**Baselines (same start weights, same train tracks).**
* **Hill-climb** (`hillclimb.py`): random Gaussian perturbation of all weights (σ 0.04 or 0.1,
  3 seeds each), through the *same* wrapper, 25 calls, same keep-if-better rule. It gets only the
  scalar score; Astra also got per-track results and crash traces.
* **ES fine-tune** (`es_finetune.py`): the `train_es.py` update (antithetic, centred ranks, Adam,
  σ 0.04, lr 0.01, wd 0.005) with 8 pairs × 4 episodes per generation, 60 s episodes on train
  tracks 0–31 from random starts, strict walls (grip task: half the episodes at 0.5, half at 1.0).
  The final weights are scored. **1× = 1,600 episodes = the full 25-call budget**, although Astra
  used 4–12 calls, so the baseline gets more than Astra did. Also run at 10× and 100×, 3 seeds
  each. A 1000× run was started and killed by us because it would have broken the 45-min run cap.
* **Scripted/standing references:** tuned ray follower (`ray`, CEM, ~23M env steps), untuned
  (`ray_default`), grip-randomized ray (`ray_dr`), ES from scratch (`es`, 346M steps),
  grip-randomized ES/PPO (`es_dr`, `ppo_dr`).

**Budget actually used by Astra** (from `eval_log.jsonl`):

| attempt | calls | one-lap episodes | env steps |
|---|---|---|---|
| edit_a1 | 12 / 25 | 768 | 195,332 |
| edit_a2 | 10 / 25 | 640 | 164,088 |
| edit0_a1, edit0_a2 (one-shot) | 0 / 0 | 0 | 0 |
| grip_a1 | 8 / 25 | 512 | 160,239 |
| grip_a2 | 4 / 25 | 256 | 89,718 |
| ES 1× hack (each seed) | — | 1,564 (60 s) | ~110k (cars crash early) |
| ES 10× / 100× hack | — | 15,980 / 159,936 | ~1.1M / 56–79M |
| ES 1× / 10× / 100× grip | — | 1,564 / 15,980 / 159,936 | ~0.6–0.73M / ~8.6M / 89M |

**Astra turns** (thread ids; tokens from each jsonl's `turn.completed`):

| tag | thread | calls / budget | best call | input tokens (cached) | output (reasoning) |
|---|---|---|---|---|---|
| r6wsp_t0_smoke | 01a1a871-73e2-70b2-b437-708cde70227c | 1 smoke call | - | 98,413 (80,896) | 818 (119) |
| r6wsp_read_a1 | 01a1a872-1a27-7e43-8c2f-b4355cb6a85c | - | - | 136,352 (98,176) | 15,617 (14,475) |
| r6wsp_read_a2 | 01a1a872-1a27-7d02-8551-3c21f411ce46 | - | - | 135,224 (97,408) | 18,827 (17,742) |
| r6wsp_read_a3 | 01a1a878-50b9-7fc3-bf89-cc0d8bb0ac7c | - | - | 134,977 (97,152) | 17,599 (16,517) |
| r6wsp_read_catch | 01a1a896-ff00-70f1-a071-68d5b40b6a2e | - | - | 134,284 (79,488) | 6,293 (5,226) |
| r6wsp_edit_a1 | 01a1a873-4afa-7bd0-bff1-ec9dfce62160 | 12 / 25 | 12 | 1,851,117 (1,791,744) | 39,964 (21,336) |
| r6wsp_edit_a2 | 01a1a873-4afa-79d3-b255-660cbcb5da67 | 10 / 25 | 10 | 996,321 (942,592) | 23,775 (8,520) |
| r6wsp_edit0_a1 | 01a1a887-c1ff-71d3-9492-a4a5d5ed473b | 0 / 0 | - | 565,230 (527,616) | 22,983 (13,246) |
| r6wsp_edit0_a2 | 01a1a887-c1fe-71a3-bde1-10acb1249c5b | 0 / 0 | - | 569,096 (534,400) | 20,386 (10,801) |
| r6wsp_grip_a1 | 01a1a880-6188-7581-b5f4-e5c329883b26 | 8 / 25 | 7 | 906,933 (861,952) | 19,236 (5,422) |
| r6wsp_grip_a2 | 01a1a880-618f-7983-8317-e5614a635416 | 4 / 25 | 4 | 542,752 (508,800) | 14,709 (4,788) |

Wall time per turn (from log-file times, so lower bounds): Read ~2.5–7 min, edit 10–20 min, one-shot ~10–11 min, grip 6.5–8 min.
Every attempt is reported; none was dropped or rerun.

## Exp 1: Read, which net wall-rides? (`read_a1/2/3`, `read_catch`)

Astra got `TASK.md` and two text dumps (`policy_A.txt`, `policy_B.txt`, rows and columns labelled,
units 0-based). The task text explained both wall rules and described the exploit ("drive
flat-out into the outside wall … no steering at all and full throttle gets round"). It was told
**exactly one** net was trained with soft walls, to run no code, and to name ≤ 6 units whose
mean-clamping would most reduce wall contact. The read-only sandbox let it do nothing but read
files. Letters came from a random draw. The draw happened to put the hack in B for both a1 and
a2, so a3 swapped the letters on purpose. The key is in `read_key.json`.

| attempt | truth | answer | conf. | units named |
|---|---|---|---|---|
| read_a1 | B = es_soft | **B** | 95% | h1[31], h1[28], h2[23] |
| read_a2 | B = es_soft | **B** | 97% | h1[31], h1[28], h2[23] |
| read_a3 | A = es_soft | **A** | 95% | h1[31], h1[28], h2[18], h2[23] |
| read_catch | neither (es_s2 vs es) | **neither** | 88% | none |

Astra's evidence was the same in all three: the clean net has a coherent braking circuit (h1[31]
rises with speed and falls with forward clearance, then drives h2[7], h2[18], h2[26], which have
strongly negative pedal weights). In the wall-rider the same feature's paths add throttle, h1[28]
also adds throttle, and the pedal bias is larger (+0.68 vs +0.32).

**Does ablation confirm the units?** No (`results/ablate_read_*.json`, `unit_scan_es_soft.json`).
Each named unit's output was clamped to its mean over the policy's own soft-wall laps, then driven
on the 200 fresh held-out tracks:

| ablation of es_soft | soft-wall contact steps/lap | soft lap | strict completion / progress |
|---|---|---|---|
| none | 20.38 | 21.82 s | 0% / 0.251 |
| h1[31], h1[28], h2[23] (a1, a2) | 20.74 (null of 100 random same-layer sets: median 20.46, p10–p90 20.18–20.74) | 21.82 s | 0% / 0.246 |
| + h2[18] (a3) | 20.60 (null p10–p90 20.16–20.81) | 21.82 s | 0% / 0.249 |
| any single one of all 64 units | 20.14 – 21.13 | 21.81–21.87 s | 0% for all |

The behavioural check (`mechanism_check.py`) shows why. es_soft's pedal is > 0.9 on **100%** of
steps and it **never brakes**. The clean ES is at full throttle 49% of the time and brakes 20% of
the time. The pedal pre-activation is 3.58 / 4.47 / 4.98 at the 1st / 50th / 90th percentile, a
sum of many ~0.3–0.6 contributions. Setting the output pedal bias to the clean net's value, to 0,
or to −1.0 changes nothing (20.2–20.5 contacts, 0% strict). So one of Astra's three pieces of
evidence, the bias, is not causal. No small set of units can pull a logit of 4.5 below
saturation. The wall-riding has no circuit; it is the *absence* of speed control. Astra's
policy-level story ("no speed restraint, lets the wall steer") is right. Its unit-level story is
not, and it hedged that ranking itself ("substantially less certain than the policy
identification").

The control side of the story has some support. Clamping the braking units Astra named in the
clean net (h1[31], h2[7], h2[18], h2[26]) drops its strict completion from 100% to 95.5% (7th
percentile of the null) and slows it from 25.09 s to 25.63 s. That is a real but small effect.

*Caveats:* the task text describes the exploit as flat-out driving, which turns the question into
"which net lacks speed restraint", an easier question than the post's blind game identification.
es_soft also trained for about half as many generations (139 of a planned 150, stopped by its
15-min budget, vs 300), and that could show in the weights. The catch trial suggests Astra is not just guessing "the one with the bigger pedal bias".

## Exp 2: Edit, remove the wall-riding (`edit_a1`, `edit_a2`; one-shot `edit0_a1`, `edit0_a2`)

Astra got the weights, the I/O layout, the physics, the original's numbers on the 32 train tracks,
reference lap times (~25.4 s for a strict-trained ES, ~25.8 s for the ray follower), and a 25-call
budget. Goal: never touch a wall, as fast as possible; targeted edits preferred.

**What Astra did.** Both threads first spent a call re-scoring the start policy for its crash
traces. Both found the same mechanism: "a distributed saturated accelerator, not a wall-riding
unit". With numpy in hand they said this outright, and it matches the check above.
* **edit_a1:** repurposed the weakest units (h1[23], h1[29], h2[28]) into an explicit governor:
  `h1[23] = tanh(.1·(7 + 26·f − 30·speed_obs))`, a smooth "clear road" bonus on h1[29], and
  `pedal = tanh(3·h2[28])`, where f = mean of the ±6° rays. It added ±35° ray terms to the
  steering features for sharp bends, then a closed-form, reflection-symmetric steering readout and
  a drag feed-forward. Calls 5–12 tuned the target-speed curve and the steering readout, taking train laps from 28.0 s to
  25.8 s (see the ledger in `results/tables.md` and `edit_a1/NOTES.md`). 280 / 1,570 weights changed.
* **edit_a2:** call 2 re-solved only the 33 pedal-readout weights by least squares, so that the
  pedal logit ≈ 0.25·(7 + 27·f − speed m/s) on a lattice of hand-made observations: 30/32 at once.
  It then added a steering-logit correction from wide-ray differences, and finally an exact
  `32·tanh(1.8 f)` → `40·tanh(1.44 f)` target-speed circuit. 220 weights changed.

**Held-out results** (200 fresh held-out + 100 fresh hard tracks, strict rules, grip 1.0;
`results/heldout_hack_main.json`, `heldout_hack_oneshot_es.json`):

| policy | budget used | held-out done / lap | hard (OOD) done / lap | soft-wall contacts/lap | weights changed |
|---|---|---|---|---|---|
| es_soft (original wall-rider) | — | 0% / – | 0% / – | 20.4 | – |
| **Astra edit a1 (best = call 12)** | 12 calls | **100% / 25.47 s** | **98% / 28.86 s** | 0 | 17.8% |
| **Astra edit a2 (best = call 10)** | 10 calls | **100% / 25.77 s** | **96% / 29.24 s** | 0 | 14.0% |
| Astra a1, best within 5 calls (call 4) | 5 calls | 100% / 27.72 s | 84% / 31.23 s | 0 | 17.1% |
| Astra a2, best within 5 calls (call 4) | 5 calls | 100% / 28.01 s | 87% / 32.39 s | 0 | 4.2% |
| **Astra one-shot a1 (no rollouts)** | 0 | **97.5% / 26.69 s** | 47% / 29.89 s | 0.04 | 25.1% |
| **Astra one-shot a2 (no rollouts)** | 0 | **65.5% / 31.74 s** | 1% / 34.30 s | 0.82 | 6.4% |
| hill-climb, 25 calls (σ .04 ×3, σ .1 ×3) | 25 calls | **0%** (all 6) | 0% | 19.8–20.7 | 0–100% |
| ES fine-tune 1× (1,564 ep., ~110k steps), 3 seeds | 1× | **0% / 0% / 0%** | 0% | 20.3–20.6 | 100% |
| ES fine-tune 10× (15,980 ep., ~1.1M steps), 3 seeds | 10× | **0% / 0% / 0%** | 0% | 20.6–21.2 | 100% |
| ES fine-tune 100× (159,936 ep., 56–79M steps), 3 seeds | 100× | 99.0 / 100 / 99.5% (24.9–25.2 s) | 63 / 91 / 72% | ≤ 0.01 | 100% |
| ES from scratch, real walls (`es`) | 346M steps | 100% / 25.09 s | 74% / 27.93 s | 0 | – |
| Ray follower, tuned (`ray`, CEM) | ~23M steps | 100% / 25.82 s | 79% / 28.72 s | 0 | – |
| Ray follower, untuned | 0 | 50% / 25.36 s | 13% / 25.86 s | 8.9 | – |

* **Generalisation is real.** The edits were selected on 32 train tracks and are perfect on 200
  unseen ones. On the fresh hard tracks they are also more reliable than every non-privileged controller we
  scored there at grip 1.0 (ES 74%, ray 79%, es_dr 82%, ray_dr 81%, ES 100× 63–91%). Only the
  privileged pure-pursuit controller, which sees the centerline, reached 100% on the original hard set. The governor is cautious
  in tight bends, so hard-track laps are ~1 s slower than ES.
* **Not copied from es_best.** The edit direction has cosine +0.09 / −0.01 with (es_best −
  es_soft). The edit threads never opened another checkpoint (audit above).
* **What the edited net is.** On states it visits, its steering matches the ray follower and ES
  (r ≈ 0.96, as does everything that drives this geometry). Its pedal matches ES-from-scratch
  (r = 0.87 / 0.94) more than the ray follower (0.54 / 0.64) (`results/action_similarity.json`).
  Astra wrote a governor that brakes the way a trained policy brakes.
* **Feedback matters, reasoning gets you most of the way.** With no rollouts at all, one of two
  one-shot attempts is already at 97.5% held-out (but 47% on hard tracks). Five calls give 100%
  held-out. The remaining calls bought ~2 s per lap and hard-track robustness.

## Exp 3: Grip, a grip-0.5 fix for a grip-1.0 policy (`grip_a1`, `grip_a2`)

Start: `es_best` (trained at grip 1.0 only; the net has no grip input). Goal: maximise strict
completion at grip 0.5 on the 32 train tracks, without losing more than 2 tracks at grip 1.0.

**What Astra did.** Both threads' first edit was the same: multiply the 32 W1 weights on the speed
input by √2. a2 made it before any evaluation; a1 had first spent a call on the start policy's crash traces. At grip μ the braking and cornering limits scale with μ, so
corner speed scales with √μ. A net that perceives speed as √2 higher drives at 1/√2 of its old
speed. On train this gave 32/32 at 0.5, but grip-1.0 completion fell to 27/32 (the car now turned
in too early and cut inside walls). a1 then halved the steering logits and fitted a speed-dependent
steering gain (65 weights changed). a2 added a lateral-clearance correction through the ±18° ray
pathway and relaxed the speed scale to 1.35 (96 weights, all in 3 rows of W1).

| policy (held-out 200 / hard 100, strict) | budget | g0.5 held-out | g1.0 held-out | g0.5 hard | g1.0 hard | weights changed |
|---|---|---|---|---|---|---|
| es_best (start) | — | 4.5% / 29.0 s | 100% / 25.09 s | 3% | 74% | – |
| **Astra grip a1 (call 7)** | 8 calls | **100% / 32.94 s** | 94.0% / 29.10 s | 87% | 68% | 4.1% |
| **Astra grip a2 (call 4)** | 4 calls | **99.5% / 33.07 s** | **99.5% / 28.95 s** | **100%** | **99%** | 6.1% |
| √2 speed row only (a2 call 1) | 1 call | **100% / 32.87 s** | 81.5% / 29.09 s | 76% | 38% | 2.0% |
| hill-climb 25 calls, 6 runs | 25 calls | 4.5 – 52.5% | 82 – 98.5% | 1 – 52% | 24 – 66% | 100% |
| ES fine-tune 1× (1,564 ep., 0.6–0.73M steps), 3 seeds | 1× | 81.0 / 71.0 / 86.5% | 89.5 / 90.5 / 89.0% | 85–94% | 51–69% | 100% |
| ES fine-tune 10×, 3 seeds | 10× | 95.0 / 99.0 / 99.5% | 98.5–99.5% (27.5–27.8 s) | 61–100% | 39–92% | 100% |
| ES fine-tune 100×, seed 0 | 100× | 97.5% | 99.5% / 26.61 s | 95% | 57% | 100% |
| PPO + grip rand. (`ppo_dr`, 78M steps) | — | 92.5% / 33.24 s | 93.5% / 27.21 s | 97% | 59% | – |
| ES + grip rand. (`es_dr`, grip-1.0 selection) | — | 38.5% | 100% / 26.01 s | 70% | 82% | – |
| Ray follower tuned w/ grip rand. (`ray_dr`) | ~23M | 30.5% | 100% / 31.94 s | 55% | 81% | – |
| Ray follower (`ray`) | ~23M | 0% | 100% / 25.82 s | 0% | 79% | – |

* **This is obstacle removal, not a new ability.** The edit fixes a mis-calibrated speed scale. It
  does not teach the net to infer grip, which it cannot observe. The price is visible at grip 1.0:
  29.0 s laps instead of 25.1 s, because the car now drives at roughly grip-0.5 speed everywhere.
  ES with 10× the budget gets the same grip-0.5 completion and is 1.5 s faster at grip 1.0,
  because it finds a speed trade-off that depends on the corner.
* At matched budget Astra wins clearly: 99.5–100% vs 71–87% (ES 1×, which also had 3–6× more
  episodes) and ≤ 52.5% (hill-climb). It also beats `ppo_dr` (92.5%) at grip 0.5, with a 4-call edit.
* grip_a1 overfit a little: 32/32 at grip 1.0 on train, but 94% on held-out.

## Surprises

1. **The text-only reader named units; the tool-using editor said "distributed".** From text alone
   (no code), Astra named 3–4 specific units. With numpy (edit threads), it measured the pedal
   logit (≈ 5.0 on a straight) and said there is no wall-riding unit. The measurements agree with
   the second answer.
2. **Hand-written governors are more robust out of distribution than the trained policies.**
   98% / 96% on hard tracks, vs 74% (ES), 79% (ray), 63–91% (ES 100× fine-tunes).
3. **The grip fix is one line of physics.** Two independent threads made the same √2 edit as their
   first edit.
4. **The hack is unreachable for small-budget search.** Hill-climb and ES at 1×/10× do not even
   raise strict progress (0.25 → 0.25). A saturated tanh output has ~zero gradient in every
   direction a small perturbation can reach.

## Honest verdict

* **Read:** policy-level identification works (4/4 including the catch trial), but the task text
  gave a strong hint. The unit attribution fails ablation, because there is nothing to ablate: the
  hack is a missing brake on a saturated output. The post's caveat ("editing can remove an
  obstacle that hides an ability but can't add a missing ability") has a reading-side twin:
  *reading can tell you an ability is missing, but there is no unit to point at.*
* **Edit:** Astra's edits are fast and they generalise. At 10–12 calls they match the 23M-step
  tuned ray follower on held-out tracks and beat every non-privileged controller on hard tracks. Hill-climb and ES at
  up to 10× the episodes do nothing; ES needs ~100× the episodes and ~290–480× the steps. But the
  edits are **programs written into spare units** (14–25% of weights), not a 3.5% surgical fix.
  Their strength is Astra's driving prior: speed should fall with forward clearance. A human with
  numpy could write the same governor; the ray follower is that human. WSP's real advantage over
  the scripted baseline is that it reuses the net's learned steering and needs ~0 tuning episodes
  instead of 23M.
* **Grip:** an obstacle-removal edit (rescale perceived speed by √(1/μ)) gives the low-grip
  completion of a grip-randomized learner with 4 calls, but not its adaptivity. That is the
  "can't add a missing ability" limit, seen from the other side.
* **Small n:** 2 attempts per condition (3 + catch for Read), one environment, one start
  checkpoint per task. The attempts agree closely with each other, which is reassuring, but this
  is not a variance estimate.

## Files

| path | what |
|---|---|
| `wsp_common.py` | weights format (W1..b3 npz), numpy forward with unit clamping, text dumper |
| `astra_dir.sh` | Codex wrapper (subfolder `-C`, read-only option) |
| `eval_train.py` | budgeted train-track evaluator (the editors' only simulator access) |
| `read_task.md`, `edit_task.md`, `edit0_task.md`, `grip_task.md` | task texts given to Astra (copied as `TASK.md` into each attempt) |
| `read_a1..a3/`, `read_catch/` | Read attempts (weights as text); key in `read_key.json` |
| `edit_a1/`, `edit_a2/` | Edit attempts: Astra's scripts, `NOTES.md`, `eval_log.jsonl`, `candidates/`, `best.npz` |
| `edit0_a1/`, `edit0_a2/` | one-shot (zero-rollout) edits: `final.npz`, `NOTES.md` |
| `grip_a1/`, `grip_a2/` | grip edits |
| `smoke0/` | turn-0 sandbox smoke |
| `hillclimb.py`, `es_finetune.py`, `run_es_baselines.sh`, `run_es_list.sh`, `baselines/` | matched-budget baselines (`baselines_es_*.log`) |
| `heldout_eval.py`, `ablate.py`, `unit_scan.py`, `mechanism_check.py`, `action_similarity.py` | Claude's scoring on fresh seeds |
| `report_wsp.py` → `results/tables.md` | all tables, including per-call ledgers |
| `results/` | `heldout_hack_main.json`, `heldout_hack_oneshot_es.json`, `heldout_grip.json` (+ per-track npz), `ablate_*.json`, `unit_scan_es_soft.json`, `mechanism_check.json`, `action_similarity.json` |
| `demo_wsp.py`, `media/` | demos on fresh held-out tracks 1,090,003 and 1,090,017: `hack_before_after_soft.gif` (soft walls: the wall-rider scrapes, the edit doesn't), `hack_before_after_strict.gif` (real walls: the wall-rider crashes, the edit and ray finish), `grip05_before_after.gif` (grip 0.5: original and ray_dr crash, the grip edit finishes), `hack_before_after_soft_t14s.png` (still) |

Attempt folders also hold `.numba_cache/` and a `STATUS.md` that Codex wrote by habit; both are
harmless.

**Reproduce the scoring** (from `08_tiny_grand_prix/`, ~5 min in total, threads capped at 4):

```bash
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4; PY=../.venv/bin/python
$PY wsp/heldout_eval.py --task hack --start 90000 --n 200 --hard_n 100 --out wsp/results/heldout_hack_main.json \
  --policy es_soft=checkpoints/es_soft_best.npz astra_edit_a1=wsp/edit_a1/best.npz astra_edit_a2=wsp/edit_a2/best.npz --named ray
$PY wsp/heldout_eval.py --task grip --start 90000 --n 200 --hard_n 100 --start_policy checkpoints/es_best.npz \
  --out wsp/results/heldout_grip.json --policy astra_grip_a2=wsp/grip_a2/best.npz --named ppo_dr ray_dr
$PY wsp/ablate.py --policy es_soft --units h1:31 h1:28 h2:23 --tag read_a1a2 --start 90000 --n 200
$PY wsp/unit_scan.py --policy es_soft --start 90000 --n 200
$PY wsp/mechanism_check.py --start 90000 --n 200
$PY wsp/es_finetune.py --task hack --start checkpoints/es_soft_best.npz --episodes_budget 1600 --seed 0 --tag hack_x1_seed0
$PY wsp/hillclimb.py --task hack --start checkpoints/es_soft_best.npz --budget 25 --sigma 0.04 --seed 0
$PY wsp/demo_wsp.py --start 90000 --tracks 3 17 --policy wall_rider=checkpoints/es_soft_best.npz \
  astra_edit=wsp/edit_a1/best.npz --named ray --mode soft --out wsp/media/hack_before_after_soft.gif
$PY wsp/report_wsp.py > wsp/results/tables.md
```
