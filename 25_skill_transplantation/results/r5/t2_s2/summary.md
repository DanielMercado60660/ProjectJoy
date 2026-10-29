# Round 5 run t2_s2

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

Primary reset bank: `test_t2`. Previous-test resets are diagnostic only.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9790 | 0.055 | 575.43 | 0.1667 | 0.0000 | 0.0000 | 3.216 |
| dagger | 0.9800 | 0.065 | 573.52 | 0.0000 | 0.0000 | 0.0000 | 1.945 |
| joint | 0.9529 | 0.205 | 526.94 | 0.0000 | 0.0000 | 0.0000 | 2.038 |
| oracle | 0.9792 | 0.066 | 573.35 | 0.0000 | 0.0000 | 0.0000 | 1.945 |
| cpg | 0.9383 | 0.242 | 526.54 | 0.0000 | 0.0000 | 0.0000 | 2.324 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## previous_test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9772 | 0.058 | 574.93 | 0.1667 | 0.0000 | 0.0000 | 3.154 |
| dagger | 0.9797 | 0.067 | 573.18 | 0.0000 | 0.0000 | 0.0000 | 1.931 |
| joint | 0.9533 | 0.208 | 526.33 | 0.0000 | 0.0000 | 0.0000 | 1.890 |
| oracle | 0.9783 | 0.067 | 573.15 | 0.0417 | 0.0417 | 0.0000 | 1.931 |
| cpg | 0.9536 | 0.227 | 530.61 | 0.0000 | 0.0000 | 0.0000 | 2.350 |
| zero | 0.0000 | 2.333 | 125.75 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.9800 | 0.053 | 575.91 | 0.1667 | 0.0000 | 0.0000 | 3.202 |
| dagger | 0.9800 | 0.065 | 573.47 | 0.0000 | 0.0000 | 0.0000 | 1.996 |
| joint | 0.9550 | 0.205 | 526.95 | 0.0000 | 0.0000 | 0.0000 | 1.946 |
| oracle | 0.9800 | 0.065 | 573.56 | 0.0417 | 0.0417 | 0.0000 | 1.915 |
| cpg | 0.9447 | 0.233 | 528.16 | 0.0000 | 0.0000 | 0.0000 | 2.340 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.9688 | 0.9688 | 0.0000 | 1.0000 |
| adapter | fwd | 2 | 0.9688 | 0.9688 | 0.0000 | 1.0000 |
| adapter | fwd | 3 | 0.9688 | 0.9688 | 0.0000 | 1.0000 |
| adapter | bwd | 2 | 0.9704 | 0.9704 | 0.0000 | 1.0000 |
| adapter | bwd | 3 | 0.9704 | 0.9704 | 0.0000 | 1.0000 |
| adapter | slow | 3 | 0.9979 | 0.9979 | 0.0000 | 1.0000 |
| dagger | fwd | 1 | 0.9688 | 0.9688 | 0.0000 | 1.0000 |
| dagger | fwd | 2 | 0.9688 | 0.9688 | 0.0000 | 1.0000 |
| dagger | fwd | 3 | 0.9688 | 0.9712 | -0.0025 | 1.0026 |
| dagger | bwd | 2 | 0.9717 | 0.9717 | 0.0000 | 1.0000 |
| dagger | bwd | 3 | 0.9717 | 0.9717 | 0.0000 | 1.0000 |
| dagger | slow | 3 | 0.9971 | 0.9971 | 0.0000 | 1.0000 |
| joint | fwd | 1 | 0.9688 | 0.9688 | 0.0000 | 1.0000 |
| joint | fwd | 2 | 0.9688 | 0.8950 | 0.0737 | 0.9239 |
| joint | fwd | 3 | 0.9688 | 0.9413 | 0.0275 | 0.9716 |
| joint | bwd | 2 | 0.9113 | 0.9113 | 0.0000 | 1.0000 |
| joint | bwd | 3 | 0.9113 | 0.9400 | -0.0287 | 1.0316 |
| joint | slow | 3 | 0.9775 | 0.9775 | 0.0000 | 1.0000 |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Validation env steps | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 316.07 |
| adapter | 2 | 9437184 | 0 | 0 | 9437184 | 92160 | 94371840 | 32400 | 1009.94 |
| adapter | 3 | 15728640 | 0 | 0 | 15728640 | 153600 | 157286400 | 54000 | 1656.30 |
| dagger | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 316.07 |
| dagger | 2 | 7340032 | 2097152 | 2097152 | 9437184 | 71680 | 73400320 | 32400 | 714.11 |
| dagger | 3 | 11534336 | 4194304 | 4194304 | 15728640 | 112640 | 115343360 | 61200 | 1081.62 |
| joint | 1 | 3145728 | 0 | 0 | 3145728 | 30720 | 31457280 | 10800 | 316.07 |
| joint | 2 | 9437184 | 0 | 0 | 9437184 | 92160 | 94371840 | 54000 | 858.27 |
| joint | 3 | 15728640 | 0 | 0 | 15728640 | 153600 | 157286400 | 118800 | 1308.22 |

Validation is separately counted; selecting an earlier checkpoint never refunds training credits.

CPG reused unchanged from `results/r5/main_s2/cpg.json`; no new search. The costs below are historical acquisition costs.

## Deployment checkpoint selection (validation only)

| Method | Stage | Selected step | Full charged steps | Worst-task validation tracking |
|---|---|---:|---:|---:|
| adapter | stage1 | 3145728 | 3145728 | 0.9689 |
| adapter | stage2 | 5242880 | 6291456 | 0.9728 |
| adapter | stage3 | 5242880 | 6291456 | 0.9994 |
| dagger | stage1 | 3145728 | 3145728 | 0.9689 |
| dagger | stage2 | 1048576 | 2097152 | 0.9683 |
| dagger | stage3 | 2097152 | 2097152 | 0.9689 |
| joint | stage1 | 3145728 | 3145728 | 0.9689 |
| joint | stage2 | 5767168 | 6291456 | 0.9022 |
| joint | stage3 | 4718592 | 6291456 | 0.9428 |

CPG: 1536000 search steps + 39600 validation steps; 34.98 seconds. Independently budgeted generous baseline.
Evaluation: 300800 completed environment steps; 22.72 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 19974 | 79896 | 39309 | 794489 |
| adapter_s2 | 24540 | 98160 | 48402 | 783443 |
| adapter_s3 | 29106 | 116424 | 57495 | 857909 |
| dagger_s1 | 19974 | 79896 | 39309 | 794489 |
| dagger_s2 | 19974 | 79896 | 39309 | 3124273 |
| dagger_s3 | 19974 | 79896 | 39309 | 3125066 |
| joint_s1 | 19974 | 79896 | 39309 | 794489 |
| joint_s2 | 19974 | 79896 | 39309 | 941864 |
| joint_s3 | 19974 | 79896 | 39309 | 942482 |
| oracle_s3 | 59922 | 239688 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
