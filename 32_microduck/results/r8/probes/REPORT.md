# R8 Gate C (representation): interim handback, paused 17:12 CST 2026-11-05

> Saved by the session lead from the builder's final message (subagents can't write report files here).
> **Lead's checks:**
> - The probe numbers match `table.md`.
> - After the stop, no R8 process is running and `logs/r8_slots/` is empty.
> - Both run folders hold `ckpt/`.
>
> The round was paused at the owner's request ("wrap this up for now"), so the LeWM checkpoint-vs-scratch
> comparison is **unfinished, with no verdict**.

**Stopped cleanly.** Both LeWM trainers stopped right after their step-2000 checkpoint save. Only the builder's own
PIDs were stopped; nothing new was started.

## Probe bake-off (finished)

- **Setup:** held-out rooms 0270–0299. Labels are simulator truth (labels only). The primary probe is R6's linear
  probe on [CLS; mean patch].
- **CIs:** 95% cluster-bootstrap over the 30 held-out rooms (`ci.json`).

| arm | px | lamp acc [CI] | plate AUC | bearing MAE° (linear / attentive) |
|---|---|---|---|---|
| DINOv2-S | 224 | 0.980 [.960,.995] | 0.988 | 10.9 / 8.0 |
| V-JEPA 2.1-L | 224 | 0.962 [.938,.981] (attentive 0.992) | 0.989 | 10.6 / **5.6** |
| LeWM TwoRoom | 224 | 0.926 [.892,.956]; LeWM latent 0.887 | 0.913 | 12.0 / 11.6 |
| LeWM Reacher / PushT / Cube | 224 | 0.877 / 0.811 / 0.698; latent 0.747 / 0.774 / 0.611 | 0.95 / 0.92 / 0.90 | ~10–12 |
| random ViT-tiny (= scratch arm's initial encoder) | 224 | 0.622 (attentive 0.830) | 0.875 | 11.4 |
| P2-A′ s0 | 64 | 0.759 | 0.812 | 12.4 |
| R6 LeWM | 64 | 0.409 (AUC 0.35, below chance) | 0.880 | 11.9 |
| R6 random encoder | 64 | 0.572 | 0.840 | 13.0 |
| CNN, pre-registered seed 0 | 64 | 0.595, did not fit | 0.984 | **3.5** |

- **The cnn64 lamp row** is an optimisation failure, not missing signal:
  - The R6/P2 CNN recipe never trained at seed 0: train accuracy 0.517, and a constant output, so its accuracy
    is the majority class and its AUC 0.5.
  - Post hoc: seed 1 fits, at 0.929 (AUC 0.976); seed 2 fails like seed 0.
- **Dev-family rooms** (post hoc): lamp DINOv2 0.946, V-JEPA 0.913 (attentive 0.988), TwoRoom 0.859, P2-A′
  0.749. The ranking holds under the appearance shift.
- **Checkpoint pick** (pre-registered rule): `lewm-tworooms`, on the latent lamp probe (0.887 vs 0.774 next).
- **Parity:** 15/15 checks at 1e-3. Worst errors: DINOv2 1.2e-4, V-JEPA 1.9e-4, LeWM ≤ 1.3e-5. All LeWM
  state_dicts load into the torch reference with strict key matching.
- **Recorder check:** re-recording a stored M4 episode gives bit-identical 96 px frames and commands.

## LeWM checkpoint vs scratch (unfinished, no gate measured)

- Both arms reached **step 2000 of 12,000** on the same frozen data (540 episodes, 239,499 windows).
- Held-out R vs copy-last (256 windows):

| arm | step 1000 | step 2000 |
|---|---|---|
| checkpoint | 0.756 | 0.645 |
| scratch | 0.735 | 0.636 |

- **Not run:** the random-encoder control, the G1–G4 evaluations, the gate table and the curves.
- **Resume each arm** (bit-exact resume tested):

```
r8_rep/lane.sh lewm-ckpt runs/r8_lewm_ckpt
r8_rep/lane.sh lewm-scratch runs/r8_lewm_scratch
```

  Then `r8_rep/after_train.sh` runs the control, the evals (`rep_eval.py eval`), `rep_eval.py gates` and
  `rep_curves.py`. Long jobs go through `./r8_job.sh`, as before.

## Files

- **Code:** `r8_rep/` (recorder, ports, torch references, parity, probe chain, trainer, eval, lanes) and
  `tests/test_r8_rep.py` (8 pass). `smoke.sh`: 170 pass.
- **Results:**
  - `results/r8/probes/`: protocol, record_check, parity, probeset, probes.jsonl, table.json/.md, ci,
    dev_family, cnn_seed_diag.
  - `results/r8/lewm_ckpt/protocol.json`, with amendments 1–2.
- **Big files (gitignored), ~58 GB new:**
  - `episodes/r8_rep` 51 GB: 770 episodes at 224 px, uncompressed.
  - `data/r8_rep` 5.4 GB.
  - `runs/r8_lewm_*` 0.4 GB.
  - Downloads in `data/pretrained`.

## Deviations

- go_and_return was dropped from the training plan to save recording time.
- R8 probe labels are simulator truth; R6 used bot beliefs. So the recomputed R6 bars are on different labels and
  frames.
- The CNN seed and dev-family runs are post hoc.
- Two held-out episodes were rendered before the protocol, only to size the ≥ 16 px lamp threshold; no probe was
  computed on them.
- SIGReg follows the official weights, which is 2× R6's statistic.
- torch was used only to read `weights.pt` and in the parity tests.
