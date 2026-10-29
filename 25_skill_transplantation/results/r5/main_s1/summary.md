# Round 5 run main_s1

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.4326 | 1.354 | 319.12 | 0.9167 | 0.7292 | 0.2500 | 9.380 |
| dagger | 0.4197 | 1.405 | 292.22 | 0.7500 | 0.3958 | 0.0833 | 7.869 |
| joint | 0.4093 | 0.974 | 320.97 | 0.9167 | 0.4167 | 0.0000 | 7.598 |
| oracle | 0.3967 | 1.686 | 264.56 | 0.9167 | 0.7083 | 0.2500 | 9.095 |
| cpg | 0.7357 | 0.460 | 464.49 | 0.5833 | 0.4375 | 0.0000 | 6.753 |
| zero | 0.0000 | 2.333 | 125.72 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.4433 | 1.182 | 338.70 | 0.8750 | 0.6667 | 0.2083 | 8.631 |
| dagger | 0.4375 | 1.420 | 297.03 | 0.7500 | 0.5000 | 0.0417 | 7.567 |
| joint | 0.3772 | 1.213 | 291.73 | 0.8750 | 0.4167 | 0.0000 | 8.356 |
| oracle | 0.4117 | 1.462 | 285.70 | 0.8750 | 0.5000 | 0.2083 | 8.954 |
| cpg | 0.7169 | 0.494 | 458.69 | 0.5000 | 0.3333 | 0.0000 | 6.415 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.0633 | 0.0633 | 0.0000 | 1.0000 |
| adapter | fwd | 2 | 0.0633 | 0.0633 | 0.0000 | 1.0000 |
| adapter | fwd | 3 | 0.0633 | 0.0633 | 0.0000 | 1.0000 |
| adapter | bwd | 2 | 0.3479 | 0.3479 | 0.0000 | 1.0000 |
| adapter | bwd | 3 | 0.3479 | 0.3479 | 0.0000 | 1.0000 |
| adapter | slow | 3 | 0.8867 | 0.8867 | 0.0000 | 1.0000 |
| dagger | fwd | 1 | 0.0633 | 0.0633 | 0.0000 | 1.0000 |
| dagger | fwd | 2 | 0.0633 | 0.1017 | -0.0383 | 1.6053 |
| dagger | fwd | 3 | 0.0633 | 0.1233 | -0.0600 | 1.9474 |
| dagger | bwd | 2 | 0.1583 | 0.1583 | 0.0000 | 1.0000 |
| dagger | bwd | 3 | 0.1583 | 0.1517 | 0.0067 | 0.9579 |
| dagger | slow | 3 | 0.9842 | 0.9842 | 0.0000 | 1.0000 |
| joint | fwd | 1 | 0.0633 | 0.0633 | 0.0000 | 1.0000 |
| joint | fwd | 2 | 0.0633 | 0.1558 | -0.0925 | 2.4605 |
| joint | fwd | 3 | 0.0633 | 0.2679 | -0.2046 | 4.2303 |
| joint | bwd | 2 | 0.1512 | 0.1512 | 0.0000 | 1.0000 |
| joint | bwd | 3 | 0.1512 | 0.1662 | -0.0150 | 1.0992 |
| joint | slow | 3 | 0.7938 | 0.7938 | 0.0000 | 1.0000 |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.31 |
| adapter | 2 | 786432 | 0 | 0 | 786432 | 6144 | 3145728 | 30.54 |
| adapter | 3 | 1310720 | 0 | 0 | 1310720 | 10240 | 5242880 | 55.73 |
| dagger | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.31 |
| dagger | 2 | 720896 | 393216 | 65536 | 786432 | 6400 | 3276800 | 25.78 |
| dagger | 3 | 1179648 | 786432 | 131072 | 1310720 | 10752 | 5505024 | 43.53 |
| joint | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.31 |
| joint | 2 | 786432 | 0 | 0 | 786432 | 6144 | 3145728 | 27.52 |
| joint | 3 | 1310720 | 0 | 0 | 1310720 | 10240 | 5242880 | 46.05 |

CPG: 1536000 search steps + 39600 validation steps; 34.61 seconds. Independently budgeted generous baseline.
Evaluation: 236000 completed environment steps; 10.43 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 19974 | 79896 | 39309 | 495031 |
| adapter_s2 | 24540 | 98160 | 48402 | 313427 |
| adapter_s3 | 29106 | 116424 | 57495 | 350593 |
| dagger_s1 | 19974 | 79896 | 39309 | 495031 |
| dagger_s2 | 19974 | 79896 | 39309 | 2886425 |
| dagger_s3 | 19974 | 79896 | 39309 | 2886487 |
| joint_s1 | 19974 | 79896 | 39309 | 495031 |
| joint_s2 | 19974 | 79896 | 39309 | 508336 |
| joint_s3 | 19974 | 79896 | 39309 | 508340 |
| oracle_s3 | 59922 | 239688 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
