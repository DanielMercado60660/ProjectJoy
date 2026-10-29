# Round 5 — Revise the courier plan

**Question.** Does task SFT, pretraining, or success-filtered post-training help a tiny
policy recover from consequences and generalize to longer delivery compositions?
This is a controlled replanning experiment, not evidence of general reasoning.

**World and information.** Three rooms contain four parcels and three receptacles.
Requests name parcel → semantic label; commands name parcel → receptacle. Each episode
randomizes label bindings and locations. `INSPECT` discovers local contents/bindings;
movement alone reveals nothing. The target command never appears in the prompt.
Every policy gets exactly the same public notebook, including discoveries, the current
request and full queue, feedback and clock. External memory is supplied equally; recurrent memory is
not being tested. The 27-command grammar is fixed. All malformed/illegal actions and
repeated inspection consume one of 64 steps. No abstention or state-legality mask exists.

**Interventions.** In separate matched conditions, the first target receptacle is locked
(a failed delivery reveals this), a parcel moves when first collected (the failed action
invalidates its location survey), or the first request changes destination after pickup.
A fourth condition has no disruption. The planner repairs its plan from public feedback.
An agent can avoid exposure by guessing or never progressing; exposure is measured.

**Splits.** Train on 4,096 episodes of one/two deliveries in ascending parcel order,
excluding four parcel/label pairs, including after redirects. Validation uses separate
layouts. Test five strata: seen compositions, held-out pairs, reversed two-item order,
three deliveries and four deliveries with new orderings. Cross each with four disruption
conditions. Train/validation/rollout seed ranges start at 1M/5M/6M; planner tuning uses
4M. Fresh headline test seeds start at 81,241,000. A small secondary run uses the old
10M seed range **in the new environment**; original sandbox results remain historical,
not directly comparable. No test data select a model or enter reward replay.

**Methods and fairness.** Compare pilot d6 + SFT, identical d6 from scratch + SFT,
and a two-layer 256-wide MLP with verb/item/argument heads representing every command
and input slots for all four deliveries. All receive the same teacher records, including
20% of episodes with one initial random action for off-policy recovery coverage.
Each gets a seven-minute task-training allowance; MLP LR selection (3e-4/1e-3/3e-3,
20 seconds each, fixed validation NLL) is inside its allowance. LM AdamW LR is 3e-4.
Tune the symbolic planner over all six room survey orders × eager/full survey using
validation success, then length, then invalidity. LMs use token-greedy grammar decoding;
MLP maximizes factored command probability. Grammar restrictions never reveal legality.
Pretraining is explicitly extra data/compute, not a matched-total-compute comparison.

**Learning from reward.** From the pretrained SFT checkpoint, sample train-distribution
episodes at temperature 0.7 for 1.25 minutes; retain only terminal successes (reservoir
of at most 256 episodes). Fit for the remainder of 2.5 minutes with half successful
transitions and half original demonstrations, LR 1e-4. Compare against 2.5 minutes of
continued SFT from the exact same parent, resetting both optimizers. Zero successes
means an unchanged reward arm, explicitly reported. This isolates success filtering
from simply spending more time, but not extra rollout data from the reward objective.

**Budget and evidence.** Battery: smoke <2 min + suite ≤32.5 min, one slot, ≤4 threads.
Suite: preparation ≤2, learned arms 3×7, reward/control 2×2.5, evaluation ≤4 minutes.
Old d6 measurements are 51k contended / 203k sustained idle tokens/s (RESULTS §§1–2b):
B16×384 projects roughly 3.5k–13.9k updates per seven minutes before overhead, **not an
R5 measurement**. Remove epoch ceilings; log actual warmed and sustained rates and
sample continuously to the guard. AC extension repeats identical budgets with seeds
52/53. Turn 2 retains these budgets, validates all 597 rows, and uses fresh headline
seeds 91,024,000 (81,241,000 is secondary). Report per-cell success, invalidity, length, disruption exposure, conditional
success, recovery fraction and recovery steps with unresolved counts/capped times.
Keep paired scenarios, incomplete-run flags, checkpoint hashes, and first success/failure
transcripts in self-contained HTML. Last-at-budget selection; no test-selected snapshots.
Two rotating resume slots plus one final export hold weights, optimizer, RNG, step/config.
Only the selected final generations need preservation for distribution; intermediate
weights never accumulate. Turn-2 cached/full-forward parity remains pending Claude's smoke.
