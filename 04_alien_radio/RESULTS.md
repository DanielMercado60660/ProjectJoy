# Alien Radio: results

> **TL;DR.** Two agents usually invent a working code (stage 1: 6/8 Gumbel and 5/8 REINFORCE seeds perfect;
> stage 2: 0.96 to 0.995 on seen objects). On never-seen colour × shape × location combinations they score
> 0.86 to 0.89 in the 5-way game, **but this is mostly approximation, not compositional generalisation**:
> asked to pick the exact new object out of all 128, the receiver gets only 0.22 to 0.27 (the hand-designed
> compositional code gets 1.00), and 58 to 72% of new combinations simply reuse the message of a similar seen
> object. Independently trained partners cannot understand each other (0.19, chance 0.20).
> *(TL;DR added by the cross-reviewer.)*

All numbers come from the checkpoints in `checkpoints/`, evaluated by `analyze.py` (→ `results/metrics.json`).
Unless stated otherwise, accuracy means the receiver picks the target out of **K = 5** shuffled candidates
(chance = 0.20). It is measured on **20,000 fresh rounds** per split and reported as mean ± std over
**5 independently initialised seeds** (8 seeds in stage 1). Training budget: 15,000 Adam steps of batch 256
for every stage-2 method, including the receiver-only baselines.

## Headline

* **Stage 1 (first satisfying version, 8 colours, 1 symbol of 8):** a code appears within about 1,000 steps.
  **6/8 Gumbel seeds and 5/8 REINFORCE seeds reach 100%.** The rest get stuck in a "homonym trap" where two
  colours share one symbol (see the failure section). No-communication receivers score 0.20.
* **Stage 2 (8 colours × 4 shapes × 4 locations, 3 symbols of 8, 26/128 combinations held out):**

| method | seen combos | **held-out combos** | seen, hard rounds | held-out, hard rounds | shuffled-message control | pick exact object of 128 (seen objs) | of 128 (held-out objs) |
|---|---|---|---|---|---|---|---|
| Gumbel-ST (emergent) | 0.995 ± 0.002 | **0.860 ± 0.047** | 0.994 ± 0.003 | 0.820 ± 0.046 | 0.199 | 0.773 | 0.269 |
| REINFORCE (emergent) | 0.960 ± 0.007 | **0.888 ± 0.022** | 0.930 ± 0.008 | 0.831 ± 0.043 | 0.200 | 0.529 | 0.215 |
| REINFORCE, entropy 0.1→0.01 | 0.945 ± 0.013 | 0.871 ± 0.019 | 0.904 ± 0.017 | 0.818 ± 0.022 | 0.200 | 0.476 | 0.231 |
| *baseline:* no communication | 0.202 ± 0.003 | 0.198 ± 0.002 | 0.197 | 0.190 | 0.202 | 0.010 | 0.000 |
| *baseline:* random holistic code | 1.000 | 0.209 ± 0.011 | 1.000 | 0.194 | 0.201 | 0.967 | 0.015 |
| *reference:* hand-designed compositional code | 1.000 | **1.000** | 1.000 | 1.000 | 0.201 | 1.000 | 1.000 |

*Hard rounds* contain a distractor that differs from the target in exactly one attribute (53% of seen
rounds, 45% of held-out rounds). *Pick exact object of 128* gives the receiver every one of the 128 objects
as candidates. This is a much stricter test than the 5-way game.

**In one sentence:** the invented languages carry most of the meaning of never-seen combinations (0.86 to
0.89 in the 5-way game, where a structureless code that is perfect on seen objects gets 0.21). But they are
not cleanly compositional, and they do not let the receiver identify a new combination exactly (0.22 to 0.27
out of 128): 58 to 72% of new combinations are simply given the message of a similar seen object. A
hand-designed compositional code with the same receiver gets 1.00 on both. Swapped partners fall to chance
(0.19). Edits to single symbols show a few context-free "words" (for example REINFORCE seed 0's slot 2 is
mostly a location word) inside an otherwise entangled code.

Per-seed held-out accuracy: Gumbel 0.938 / 0.831 / 0.871 / 0.798 / 0.863. REINFORCE 0.915 / 0.879 / 0.911 / 0.856 / 0.881.

## Measure items

### Task success
See the table above and `media/training_curves.png`. Gumbel learns fastest: 0.90 on seen combinations
after 250 steps and 0.995 at the end. REINFORCE climbs more slowly (0.35 → 0.96). Held-out accuracy
plateaus at about 0.85 to 0.89 for both by roughly 5k steps. After that it fluctuates by ±0.05 between
evaluations while seen accuracy keeps creeping up.

### Message efficiency
The channel has 3 × log2(8) = 9 bits of capacity, and naming one of 128 objects needs 7 bits.

| method | distinct messages for 128 objects | distinct for the 26 held-out | message entropy (bits) | per-position entropy (bits, max 3) | topographic similarity | positional disentanglement |
|---|---|---|---|---|---|---|
| Gumbel | 104.4 | 24.2 | 6.62 | 2.92 / 2.93 / 2.94 | 0.373 ± 0.019 | 0.054 ± 0.020 |
| REINFORCE | 74.0 | 21.6 | 5.89 | 2.69 / 2.83 / 2.61 | 0.428 ± 0.028 | 0.146 ± 0.045 |
| REINFORCE, entropy decayed | 60.4 | 20.8 | 5.56 | 2.62 / 2.75 / 2.55 | 0.441 ± 0.028 | 0.158 ± 0.056 |
| hand-designed code | 128 | 26 | 7.00 | 3 / 2 / 2 | 1.000 | 1.000 |
| random holistic code | 128 | 26 | 7.00 | 2.97 / 2.95 / 2.97 | 0.002 | 0.060 |

Neither emergent language is a bijection. Gumbel uses 6.6 of the 7 needed bits, and REINFORCE uses only
5.9: it gives 128 objects 74 distinct messages. That is still enough for the 5-way game, because two objects
that share a message rarely appear in the same round. This is why "pick the exact object out of 128" is
much lower (0.53 for REINFORCE on seen objects). The agents learn exactly as much language as the game rewards.

### Unseen-combination accuracy
Held-out K=5 accuracy is 0.860 (Gumbel) and 0.888 (REINFORCE), against 0.209 for the holistic code and 0.198
for no communication. Held-out rounds are slightly easier than seen rounds (fewer one-attribute near misses:
45% vs 53%). Restricted to hard rounds, held-out accuracy is 0.820 / 0.831, against 0.994 / 0.930 on seen
hard rounds. So the generalisation gap is real, at about 0.10 to 0.17.

Note the inversion: **REINFORCE is worse on seen combinations but better on held-out ones**, and it has the
higher topographic similarity (0.43 vs 0.37) and positional disentanglement (0.15 vs 0.05). Its noisier,
lower-entropy code generalises a little better. This is consistent across seeds (REINFORCE's worst held-out
seed, 0.856, beats two of the five Gumbel seeds), but n = 5. *Reviewer note:* all methods share population seed 0, so this is a
paired comparison on 5 matched seeds: REINFORCE is ahead on 4/5 pairs, mean difference +0.028 (paired t ≈ 2.0,
p ≈ 0.12), smaller than Gumbel's seed std (0.047), and on hard rounds the gap is 0.831 vs 0.820. Treat the
inversion as suggestive, not established.

### Partner-transfer accuracy (partner swapping)
Sender from seed *a* + receiver from seed *b*, trained completely independently (`media/partner_swap.png`):

| pairing | own partner (diagonal) | swapped partner (off-diagonal mean, max) |
|---|---|---|
| Gumbel → Gumbel | 0.995 | 0.185 (max 0.221) |
| REINFORCE → REINFORCE | 0.960 | 0.191 (max 0.260) |
| Gumbel sender → REINFORCE receiver | n/a | 0.198 (max 0.310) |
| hand-designed code → emergent receivers | n/a | 0.21 to 0.22 |

**As expected, swapping partners fails completely: accuracy drops to chance.** Each pair invents its own
arbitrary code. No receiver understands another seed's sender, and none understands the human-designed code.

**Surprise: the language is largely fixed by the initial weights.** All runs use population seed 0, so
"REINFORCE seed *k*" and "REINFORCE-entropy-decayed seed *k*" start from **identical weights and see the same
data stream**. They differ only in the entropy-bonus schedule. Swapping between them gives **0.69 to 0.70**
(far above chance 0.20, far below the 0.95 own-partner score): mostly the same language, with dialect
drift. Gumbel seed *k* and REINFORCE seed *k* also share initial weights, but they end up unrelated
(0.226 vs 0.198 off-diagonal), so the learning rule matters more than the initialisation.

### Symbol interventions (the dictionary)
For every object, each message position is overwritten with each of the 8 symbols, and the receiver chooses
among all 128 objects (3,072 edited messages per agent). Two readouts (`media/dictionary_*.png`, and live in the demo):

1. **Which attribute does editing position p change?** For the hand-designed code it is perfectly clean:
   position 0 → colour 100%, position 1 → shape 86%, position 2 → location 86%, and nothing else. (86% = 6/7,
   because editing to one of the 4 unused symbols 4-7 sends the receiver to an arbitrary value.) For the
   emergent codes, editing *any* position changes the colour 62 to 84% of the time, and the shape or location
   typically 30 to 60% of the time (range 10 to 68%). Every slot carries some colour information.
2. **What does each symbol evoke, whatever the other two say?** Gumbel seed 0 has some fairly context-free
   words, such as *s0* at position 1 = **triangle** (80% of contexts, chance 25%), *s5* at position 1 =
   bottom-right (72%), *s3* at position 1 = star (63%), and *s1* at position 0 = red (50%, chance 12.5%).
   REINFORCE seed 0 has turned **position 2 into a mostly-location slot** (7 of 8 symbols evoke a location,
   45 to 68%) and has *s7* at position 1 = **circle (90%)**. The rest is entangled: a symbol's meaning depends
   on its neighbours, much like a holistic vocabulary with some compositional words mixed in.

## Baseline comparison (matched conditions)
Every row in the headline table uses the same receiver network, the same data stream, the same evaluation
rounds and 15,000 steps. Only the source of the message differs.
* **No communication** (message always blank) gives 0.20 everywhere. This is chance, and confirms there is
  no other route to the answer.
* **Hand-designed compositional code** gives 1.00 on held-out combinations. The receiver architecture can
  generalise perfectly when the language is compositional, so the emergent languages' 0.86 to 0.89 is a
  limit of the language, not of the listener.
* **Random holistic code** gives 1.00 on seen combinations and 0.21 on held-out ones. Memorisation alone
  does not transfer. The emergent codes sit between the two.

## Stage 1 details (colour only)
`checkpoints/color_*_s0`, 8 seeds each, 3,000 steps, V = 8 symbols for 8 colours (a zero-slack channel).

| method | per-seed accuracy (K=5, 20k rounds) | seeds at 100% |
|---|---|---|
| Gumbel (τ=2, entropy 0.1) | 1.000 0.927 1.000 1.000 0.931 1.000 1.000 1.000 | 6/8 |
| REINFORCE (entropy 0.3) | 0.929 1.000 1.000 1.000 0.926 0.853 1.000 1.000 | 5/8 |
| no communication | 0.192 to 0.201 | 0/8 |

Sensitivity sweep (`sweep_color.sh`, 8 seeds each; final accuracy per seed is in `metrics.json` under `color_stage._sweep`):

| setting | seeds at 100% |
|---|---|
| Gumbel τ=1, entropy 0 / 0.1 / 0.3 / 1.0 | 2 / 4 / 4 / 0 |
| Gumbel τ=2, entropy 0 / **0.1** / 0.3 / 1.0 | 2 / **6** / 0 / 0 |
| REINFORCE entropy 0.03 / 0.1 / **0.3** / 1.0 | 0 / 0 / **5** / 0 |
| REINFORCE entropy 0.3, lr 1e-3 | 0 |
| REINFORCE entropy 0.3, 9,000 steps (3× longer) | 5 (identical to 3k steps: stuck seeds never recover) |

The stage-1 defaults (τ=2, REINFORCE entropy 0.3) were chosen from this sweep. *Reviewer note:* the sweep used
the same 8 seeds (population seed 0) as the headline stage-1 runs, so "6/8" and "5/8" are the best of 14
configurations on those seeds and are optimistic; other settings gave 0 to 4/8. Stage 2 uses τ=1 and
entropy 0.1 for both methods, set before any stage-2 run. A 3-seed, 3k-step pilot hinted that τ=2 did not
help Gumbel but entropy 0.3 helped REINFORCE on held-out rounds (0.83 vs 0.79), so I ran REINFORCE with
entropy 0.3 as a full variant. At 15k steps it comes out *lower* (0.850 vs 0.888, variants table).

## The good kind of failure: the homonym trap (stage 1)
Seeds that miss 100% always converge to exactly **0.93 or 0.86**. That is exactly what you get when one or two
pairs of colours share a symbol. With 8 colours, K = 5 and one merged pair, the error rate is
P(target is in the pair) × P(partner colour among the 4 distractors) × ½ = 2/8 × 4/7 × ½ = 7.1%, giving 0.929.
Two merged pairs give 0.857. Measured: 0.927, 0.931, 0.929, 0.926 (one pair) and 0.853 (two pairs). For
example, Gumbel seed 1 says *s4* for both red and blue and never uses one symbol. The trap is stable: in
the 9k-step run every stuck seed stays stuck. To escape, the sender would have to move a colour to the unused
symbol, but the receiver has never seen that symbol, so it reads it as noise and the move loses reward. A
little entropy bonus keeps more seeds out of the trap, while too much (1.0) stops a code from forming at all.

## Side channels: what was checked
* The receiver's forward function takes only `(message one-hots, candidate features)` (unit test).
  Candidates are re-shuffled every batch, and the target is chosen uniformly among the candidates *after*
  the candidate set is built. A "pick the centre of the near-miss cluster" heuristic scores chance (unit test).
* **Shuffled-message control:** give each receiver the message from a different round, and accuracy is
  0.199 to 0.202 for every method (table above).
* **A side channel I found and fixed:** the first version of the held-out evaluation drew distractors from all
  objects, so the held-out target was usually the only *unfamiliar* object in the round. A no-message receiver
  scored up to 0.30 on held-out rounds by picking the odd one out. Held-out rounds now use only held-out
  objects, and no-communication scores 0.198 (asserted in the tests).
* Timing and simulator state: the game has no time steps and the receiver never sees RNG state. Senders and
  receivers in the population share nothing but the summed loss scalar, which gives no gradient between members.

## Channel-size, schedule and game-difficulty variants
Stage 2, 5 seeds, the same 15k steps, all evaluated on the same K=5 rounds plus the "exact object of 128" test.

| variant | L × V | trained with | seen | held-out (K=5) | topsim | distinct msgs / 128 | of 128: seen objs | of 128: held-out objs |
|---|---|---|---|---|---|---|---|---|
| Gumbel (main) | 3 × 8 | K=5 | 0.995 | 0.860 ± 0.047 | 0.373 | 104.4 | 0.773 | 0.269 |
| Gumbel | 3 × 16 | K=5 | 0.996 | 0.861 ± 0.024 | 0.321 | 120.4 | 0.775 | 0.346 |
| Gumbel | 2 × 16 | K=5 | 0.985 | 0.870 ± 0.020 | 0.291 | 87.6 | 0.600 | 0.277 |
| Gumbel | 4 × 8 | K=5 | 0.999 | 0.892 ± 0.017 | 0.392 | 122.4 | 0.878 | 0.354 |
| Gumbel | 3 × 8 | **K=20**, random distractors | 0.998 | 0.867 ± 0.033 | 0.410 | 105.4 | **0.892** | 0.223 |
| REINFORCE (main, entropy 0.1) | 3 × 8 | K=5 | 0.960 | 0.888 ± 0.022 | 0.428 | 74.0 | 0.529 | 0.215 |
| REINFORCE entropy 0.3 | 3 × 8 | K=5 | 0.955 | 0.850 ± 0.015 | 0.388 | 73.0 | 0.396 | 0.262 |
| REINFORCE entropy 0.1→0.01 | 3 × 8 | K=5 | 0.945 | 0.871 ± 0.019 | 0.441 | 60.4 | 0.476 | 0.231 |
| REINFORCE | 3 × 8 | **K=20**, random distractors | 0.931 | 0.863 ± 0.045 | 0.443 | 61.8 | 0.496 | 0.192 |

* A bigger vocabulary does not buy compositionality (3×16 topsim 0.32 < 0.37). A longer message helps a little
  (4×8: held-out 0.89, and 0.35 exact of 128). Differences in K=5 held-out accuracy are within about 1 std of seed noise.
* **A harder game (K=20) sharpens names for *seen* objects but not for new ones.** Gumbel's exact identification
  of seen objects rises from 0.77 to 0.89, while held-out identification stays at 0.22 (vs 0.27). During
  training, its K=20 held-out accuracy was only 0.65, against 0.995 on seen objects.

### How never-seen combinations get named
This explains the gap between "0.86 in the 5-way game" and "0.27 out of 128":

| | held-out objects whose message is **identical** to some seen object's message | attributes (of 3) the receiver's 128-way pick shares with a held-out target |
|---|---|---|
| Gumbel | 58% | 1.78 |
| REINFORCE | 72% | 1.95 |
| Gumbel K=20 | 74% | 1.72 |
| Gumbel 4×8 | 18% | 1.87 |
| hand-designed code | 0% | 3.00 |
| random holistic code | 0% | 0.60 (random guessing: 0.625) |

**The senders mostly describe a new combination with the word for a familiar, similar object.** The receiver
then reads it as an object that gets about two of the three attributes right. In a 5-way game that is usually
enough to rule out the other four candidates, so the K=5 accuracy looks like strong compositional generalisation.
It is really graceful *approximation*. Only the hand-designed code gives new combinations new, exactly
decodable names.

## What worked, what didn't
* Worked: a working code in both stages with both learning rules; clean controls (no-comm, shuffled message,
  holistic); exact checkpoint resume; partner-swap and intervention analyses; a demo where you can edit
  symbols and swap partners live.
* Didn't: compositional structure. The emergent codes are partly compositional at best (topsim ≤ 0.44,
  posdis ≤ 0.16), and exact identification of new combinations is poor (0.19 to 0.35 of 128). Neither entropy
  schedules, channel size nor a 4× harder game (K=20) changed that much: most new combinations are given the
  message of a similar seen object.
* Infrastructure note: the shared GPU queue was the bottleneck. Stage-2 training took about 1.5 min per
  method but waited about 1 h for a slot. Stage-1 runs (about 15 s each) were run directly.

## Next steps
* Pressure towards compositionality: iterated learning / receiver resets, training each sender with a
  population of receivers, or a generation-transfer bottleneck. Measure topsim and of-128 held-out
  identification as the main targets.
* An autoregressive (GRU) sender with variable-length messages and a length cost, to measure true minimal
  message length.
* Escape the stage-1 homonym trap with a small "unused-symbol" exploration bonus, and check whether that also
  helps stage 2.

## Reviewer notes / caveats
* "Seed k" = member k of a population run. Members are independent (own init, data, noise). Different
  *methods* share population seed 0 and therefore initial weights for member k (see the partner-swap surprise).
* Senders are evaluated greedily (argmax symbol per position). The demo shows these codebooks.
* **Showcased seed is the best one** (reviewer note). `media/demo.html` defaults to, and
  `media/example_rounds.png` and `media/dictionary_gumbel.png` show, Gumbel seed 0, which `analyze.py` picks as
  the seed with the highest held-out accuracy (0.938, against a 5-seed mean of 0.860). `dictionary_reinforce.png`
  is likewise REINFORCE's best held-out seed (seed 0, 0.915 vs mean 0.888). The "context-free words" quoted in the
  interventions section come from these best seeds. The example rounds themselves are fixed-seed draws, not
  filtered for correctness.
* Held-out accuracy fluctuates by about ±0.05 between evaluations late in training. Reported numbers are the
  final checkpoint, not the best one.
* The *hard rounds* metric conditions on a property of the target. For the no-message receiver it gives
  0.190 to 0.197 rather than exactly 0.20, so treat differences under about 0.01 as noise.
* Everything was run on the shared M5 Max under heavy load. Wall times in logs are not benchmarks.
