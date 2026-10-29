# Round 5 run t2_s0

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

Primary reset bank: `test_t2`. Previous-test resets are diagnostic only.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9800 | 0.064 | 575.49 | 0.0625 | 0.0625 | 0.0000 | 1.897 |
| dagger | 0.9811 | 0.066 | 574.84 | 0.0208 | 0.0208 | 0.0000 | 1.877 |
| joint | 0.9626 | 0.186 | 534.26 | 0.0000 | 0.0000 | 0.0000 | 1.899 |
| oracle | 0.9812 | 0.066 | 574.83 | 0.0417 | 0.0417 | 0.0000 | 2.014 |
| cpg | 0.7161 | 0.564 | 449.85 | 0.5417 | 0.1458 | 0.0000 | 6.709 |
| zero | 0.0000 | 2.334 | 125.60 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## previous_test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9800 | 0.063 | 575.53 | 0.0000 | 0.0000 | 0.0000 | 1.902 |
| dagger | 0.9825 | 0.064 | 575.07 | 0.0000 | 0.0000 | 0.0000 | 1.975 |
| joint | 0.9628 | 0.187 | 534.07 | 0.0000 | 0.0000 | 0.0000 | 1.779 |
| oracle | 0.9822 | 0.064 | 575.12 | 0.0833 | 0.0833 | 0.0000 | 1.923 |
| cpg | 0.7017 | 0.598 | 442.87 | 0.5417 | 0.1250 | 0.0000 | 6.654 |
| zero | 0.0000 | 2.333 | 125.89 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9808 | 0.061 | 575.89 | 0.0833 | 0.0833 | 0.0000 | 2.229 |
| dagger | 0.9811 | 0.067 | 574.56 | 0.0000 | 0.0000 | 0.0000 | 1.838 |
| joint | 0.9581 | 0.187 | 534.01 | 0.0000 | 0.0000 | 0.0000 | 1.808 |
| oracle | 0.9811 | 0.065 | 574.89 | 0.0833 | 0.0833 | 0.0000 | 2.227 |
| cpg | 0.7219 | 0.516 | 460.03 | 0.3750 | 0.0000 | 0.0000 | 5.483 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.9712 | 0.9712 | 0.0000 | 1.0000 |
| adapter | fwd | 2 | 0.9712 | 0.9712 | 0.0000 | 1.0000 |
| adapter | fwd | 3 | 0.9712 | 0.9712 | 0.0000 | 1.0000 |
| adapter | bwd | 2 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| adapter | bwd | 3 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| adapter | slow | 3 | 0.9958 | 0.9958 | 0.0000 | 1.0000 |
| dagger | fwd | 1 | 0.9712 | 0.9712 | 0.0000 | 1.0000 |
| dagger | fwd | 2 | 0.9712 | 0.9725 | -0.0012 | 1.0013 |
| dagger | fwd | 3 | 0.9712 | 0.9717 | -0.0004 | 1.0004 |
| dagger | bwd | 2 | 0.9758 | 0.9758 | 0.0000 | 1.0000 |
| dagger | bwd | 3 | 0.9758 | 0.9742 | 0.0017 | 0.9983 |
| dagger | slow | 3 | 0.9975 | 0.9975 | 0.0000 | 1.0000 |
| joint | fwd | 1 | 0.9712 | 0.9712 | 0.0000 | 1.0000 |
| joint | fwd | 2 | 0.9712 | 0.9062 | 0.0650 | 0.9331 |
| joint | fwd | 3 | 0.9712 | 0.9500 | 0.0212 | 0.9781 |
| joint | bwd | 2 | 0.9592 | 0.9592 | 0.0000 | 1.0000 |
| joint | bwd | 3 | 0.9592 | 0.9521 | 0.0071 | 0.9926 |
| joint | slow | 3 | 0.9858 | 0.9858 | 0.0000 | 1.0000 |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Validation env steps | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 315.30 |
| adapter | 2 | 9437184 | 0 | 0 | 9437184 | 92160 | 94371840 | 32400 | 1015.63 |
| adapter | 3 | 15728640 | 0 | 0 | 15728640 | 153600 | 157286400 | 54000 | 1659.13 |
| dagger | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 315.30 |
| dagger | 2 | 7340032 | 2097152 | 2097152 | 9437184 | 71680 | 73400320 | 32400 | 715.19 |
| dagger | 3 | 11534336 | 4194304 | 4194304 | 15728640 | 112640 | 115343360 | 61200 | 1087.36 |
| joint | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 315.30 |
| joint | 2 | 9437184 | 0 | 0 | 9437184 | 92160 | 94371840 | 54000 | 859.25 |
| joint | 3 | 15728640 | 0 | 0 | 15728640 | 153600 | 157286400 | 118800 | 1310.91 |

Validation is separately counted; selecting an earlier checkpoint never refunds training credits.

CPG reused unchanged from `results/r5/main_s0/cpg.json`; no new search. The costs below are historical acquisition costs.

## Deployment checkpoint selection (validation only)

| Method | Stage | Selected step | Full charged steps | Worst-task validation tracking |
|---|---|---:|---:|---:|
| adapter | stage1 | 2097152 | 3145728 | 0.9717 |
| adapter | stage2 | 5767168 | 6291456 | 0.9717 |
| adapter | stage3 | 2621440 | 6291456 | 0.9989 |
| dagger | stage1 | 2097152 | 3145728 | 0.9717 |
| dagger | stage2 | 1048576 | 2097152 | 0.9717 |
| dagger | stage3 | 1048576 | 2097152 | 0.9722 |
| joint | stage1 | 2097152 | 3145728 | 0.9717 |
| joint | stage2 | 6291456 | 6291456 | 0.9094 |
| joint | stage3 | 6291456 | 6291456 | 0.9467 |

CPG: 1536000 search steps + 39600 validation steps; 34.49 seconds. Independently budgeted generous baseline.
Evaluation: 300800 completed environment steps; 20.56 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 19974 | 79896 | 39309 | 794500 |
| adapter_s2 | 24540 | 98160 | 48402 | 783437 |
| adapter_s3 | 29106 | 116424 | 57495 | 857904 |
| dagger_s1 | 19974 | 79896 | 39309 | 794500 |
| dagger_s2 | 19974 | 79896 | 39309 | 3124265 |
| dagger_s3 | 19974 | 79896 | 39309 | 3125063 |
| joint_s1 | 19974 | 79896 | 39309 | 794500 |
| joint_s2 | 19974 | 79896 | 39309 | 941858 |
| joint_s3 | 19974 | 79896 | 39309 | 942477 |
| oracle_s3 | 59922 | 239688 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
