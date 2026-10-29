# Round 5 run smoke_20261024_105320_72158

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.0000 | 2.329 | 15.28 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| dagger | 0.0000 | 2.329 | 15.28 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| joint | 0.0000 | 2.329 | 15.27 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| oracle | 0.0000 | 2.329 | 15.28 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| cpg | 0.2500 | 1.667 | 24.24 | 1.0000 | 0.0833 | 0.0000 | 1.200 |
| zero | 0.0000 | 2.329 | 15.28 | 1.0000 | 0.0000 | 0.0000 | 1.200 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.0000 | 2.337 | 14.96 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| dagger | 0.0000 | 2.337 | 14.96 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| joint | 0.0000 | 2.337 | 14.96 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| oracle | 0.0000 | 2.336 | 14.96 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| cpg | 0.2176 | 1.906 | 20.75 | 1.0000 | 0.0000 | 0.0000 | 1.200 |
| zero | 0.0000 | 2.336 | 14.97 | 1.0000 | 0.0000 | 0.0000 | 1.200 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| adapter | fwd | 2 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| adapter | fwd | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| adapter | bwd | 2 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| adapter | bwd | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| adapter | slow | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| dagger | fwd | 1 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| dagger | fwd | 2 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| dagger | fwd | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| dagger | bwd | 2 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| dagger | bwd | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| dagger | slow | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| joint | fwd | 1 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| joint | fwd | 2 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| joint | fwd | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| joint | bwd | 2 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| joint | bwd | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |
| joint | slow | 3 | 0.0000 | 0.0000 | 0.0000 | unqualified (<5% at acquisition) |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 64 | 0 | 0 | 64 | 2 | 64 | 0.02 |
| adapter | 2 | 256 | 0 | 0 | 256 | 8 | 256 | 0.09 |
| adapter | 3 | 448 | 0 | 0 | 448 | 14 | 448 | 0.11 |
| dagger | 1 | 64 | 0 | 0 | 64 | 2 | 64 | 0.02 |
| dagger | 2 | 192 | 64 | 64 | 256 | 8 | 256 | 0.03 |
| dagger | 3 | 320 | 128 | 128 | 448 | 14 | 448 | 0.04 |
| joint | 1 | 64 | 0 | 0 | 64 | 2 | 64 | 0.02 |
| joint | 2 | 256 | 0 | 0 | 256 | 8 | 256 | 0.04 |
| joint | 3 | 448 | 0 | 0 | 448 | 14 | 448 | 0.06 |

CPG: 1920 search steps + 1344 validation steps; 0.11 seconds. Independently budgeted generous baseline.
Evaluation: 9024 completed environment steps; 0.64 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 710 | 2840 | 1341 | 20735 |
| adapter_s2 | 928 | 3712 | 1768 | 16396 |
| adapter_s3 | 1146 | 4584 | 2195 | 18897 |
| dagger_s1 | 710 | 2840 | 1341 | 20735 |
| dagger_s2 | 710 | 2840 | 1341 | 35267 |
| dagger_s3 | 710 | 2840 | 1341 | 35271 |
| joint_s1 | 710 | 2840 | 1341 | 20735 |
| joint_s2 | 710 | 2840 | 1341 | 20787 |
| joint_s3 | 710 | 2840 | 1341 | 20790 |
| oracle_s3 | 2130 | 8520 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
