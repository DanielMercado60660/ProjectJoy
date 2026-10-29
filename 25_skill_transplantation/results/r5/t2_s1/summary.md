# Round 5 run t2_s1

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

Primary reset bank: `test_t2`. Previous-test resets are diagnostic only.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9796 | 0.045 | 580.07 | 0.0625 | 0.0000 | 0.0625 | 2.202 |
| dagger | 0.9797 | 0.051 | 578.93 | 0.0000 | 0.0000 | 0.0000 | 1.695 |
| joint | 0.9687 | 0.177 | 539.91 | 0.0000 | 0.0000 | 0.0000 | 1.657 |
| oracle | 0.9807 | 0.049 | 579.14 | 0.0000 | 0.0000 | 0.0000 | 1.671 |
| cpg | 0.7350 | 0.477 | 461.96 | 0.5208 | 0.3958 | 0.0000 | 6.427 |
| zero | 0.0000 | 2.333 | 125.72 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## previous_test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9775 | 0.048 | 579.02 | 0.0833 | 0.0000 | 0.0833 | 2.252 |
| dagger | 0.9769 | 0.058 | 577.52 | 0.0000 | 0.0000 | 0.0000 | 1.627 |
| joint | 0.9669 | 0.177 | 540.14 | 0.0000 | 0.0000 | 0.0000 | 1.700 |
| oracle | 0.9769 | 0.056 | 577.41 | 0.0000 | 0.0000 | 0.0000 | 1.604 |
| cpg | 0.7106 | 0.523 | 448.60 | 0.6250 | 0.5000 | 0.0000 | 6.883 |
| zero | 0.0000 | 2.334 | 125.65 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9797 | 0.044 | 580.11 | 0.0000 | 0.0000 | 0.0000 | 1.819 |
| dagger | 0.9811 | 0.050 | 578.93 | 0.0000 | 0.0000 | 0.0000 | 1.606 |
| joint | 0.9653 | 0.175 | 540.41 | 0.0000 | 0.0000 | 0.0000 | 1.662 |
| oracle | 0.9803 | 0.049 | 579.12 | 0.0000 | 0.0000 | 0.0000 | 1.627 |
| cpg | 0.7169 | 0.494 | 458.69 | 0.5000 | 0.3333 | 0.0000 | 6.415 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| adapter | fwd | 2 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| adapter | fwd | 3 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| adapter | bwd | 2 | 0.9700 | 0.9700 | 0.0000 | 1.0000 |
| adapter | bwd | 3 | 0.9700 | 0.9700 | 0.0000 | 1.0000 |
| adapter | slow | 3 | 0.9958 | 0.9958 | 0.0000 | 1.0000 |
| dagger | fwd | 1 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| dagger | fwd | 2 | 0.9729 | 0.9721 | 0.0008 | 0.9991 |
| dagger | fwd | 3 | 0.9729 | 0.9713 | 0.0017 | 0.9983 |
| dagger | bwd | 2 | 0.9733 | 0.9733 | 0.0000 | 1.0000 |
| dagger | bwd | 3 | 0.9733 | 0.9729 | 0.0004 | 0.9996 |
| dagger | slow | 3 | 0.9950 | 0.9950 | 0.0000 | 1.0000 |
| joint | fwd | 1 | 0.9729 | 0.9729 | 0.0000 | 1.0000 |
| joint | fwd | 2 | 0.9729 | 0.9187 | 0.0542 | 0.9443 |
| joint | fwd | 3 | 0.9729 | 0.9704 | 0.0025 | 0.9974 |
| joint | bwd | 2 | 0.8088 | 0.8088 | 0.0000 | 1.0000 |
| joint | bwd | 3 | 0.8088 | 0.9467 | -0.1379 | 1.1705 |
| joint | slow | 3 | 0.9892 | 0.9892 | 0.0000 | 1.0000 |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Validation env steps | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 322.52 |
| adapter | 2 | 9437184 | 0 | 0 | 9437184 | 92160 | 94371840 | 32400 | 1029.50 |
| adapter | 3 | 15728640 | 0 | 0 | 15728640 | 153600 | 157286400 | 54000 | 1675.77 |
| dagger | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 322.52 |
| dagger | 2 | 7340032 | 2097152 | 2097152 | 9437184 | 71680 | 73400320 | 32400 | 723.85 |
| dagger | 3 | 11534336 | 4194304 | 4194304 | 15728640 | 112640 | 115343360 | 61200 | 1106.85 |
| joint | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 322.52 |
| joint | 2 | 9437184 | 0 | 0 | 9437184 | 92160 | 94371840 | 54000 | 878.10 |
| joint | 3 | 15728640 | 0 | 0 | 15728640 | 153600 | 157286400 | 118800 | 1328.13 |

Validation is separately counted; selecting an earlier checkpoint never refunds training credits.

CPG reused unchanged from `results/r5/main_s1/cpg.json`; no new search. The costs below are historical acquisition costs.

## Deployment checkpoint selection (validation only)

| Method | Stage | Selected step | Full charged steps | Worst-task validation tracking |
|---|---|---:|---:|---:|
| adapter | stage1 | 3145728 | 3145728 | 0.9711 |
| adapter | stage2 | 5242880 | 6291456 | 0.9756 |
| adapter | stage3 | 5767168 | 6291456 | 0.9944 |
| dagger | stage1 | 3145728 | 3145728 | 0.9711 |
| dagger | stage2 | 1572864 | 2097152 | 0.9711 |
| dagger | stage3 | 2097152 | 2097152 | 0.9711 |
| joint | stage1 | 3145728 | 3145728 | 0.9711 |
| joint | stage2 | 5767168 | 6291456 | 0.8533 |
| joint | stage3 | 6291456 | 6291456 | 0.9517 |

CPG: 1536000 search steps + 39600 validation steps; 34.61 seconds. Independently budgeted generous baseline.
Evaluation: 300800 completed environment steps; 15.51 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 19974 | 79896 | 39309 | 794495 |
| adapter_s2 | 24540 | 98160 | 48402 | 783440 |
| adapter_s3 | 29106 | 116424 | 57495 | 857906 |
| dagger_s1 | 19974 | 79896 | 39309 | 794495 |
| dagger_s2 | 19974 | 79896 | 39309 | 3124271 |
| dagger_s3 | 19974 | 79896 | 39309 | 3125062 |
| joint_s1 | 19974 | 79896 | 39309 | 794495 |
| joint_s2 | 19974 | 79896 | 39309 | 941861 |
| joint_s3 | 19974 | 79896 | 39309 | 942479 |
| oracle_s3 | 59922 | 239688 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
