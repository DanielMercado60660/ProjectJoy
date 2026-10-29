# Round 5 run main_s2

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.4782 | 1.231 | 329.45 | 0.9375 | 0.5000 | 0.2708 | 9.440 |
| dagger | 0.4237 | 1.468 | 287.06 | 0.9583 | 0.8333 | 0.1250 | 9.157 |
| joint | 0.3919 | 0.861 | 358.69 | 0.6875 | 0.0417 | 0.0208 | 7.862 |
| oracle | 0.4199 | 1.551 | 278.64 | 0.9583 | 0.8958 | 0.0208 | 9.633 |
| cpg | 0.9493 | 0.230 | 529.71 | 0.0000 | 0.0000 | 0.0000 | 2.368 |
| zero | 0.0000 | 2.334 | 125.68 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.4894 | 0.959 | 357.92 | 0.9583 | 0.2500 | 0.2500 | 9.265 |
| dagger | 0.4156 | 1.601 | 273.81 | 0.8750 | 0.7917 | 0.0833 | 8.923 |
| joint | 0.4036 | 0.797 | 370.21 | 0.6667 | 0.0000 | 0.0000 | 7.794 |
| oracle | 0.4372 | 1.298 | 302.03 | 0.9583 | 0.9167 | 0.0417 | 8.602 |
| cpg | 0.9447 | 0.233 | 528.16 | 0.0000 | 0.0000 | 0.0000 | 2.340 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.1017 | 0.1017 | 0.0000 | 1.0000 |
| adapter | fwd | 2 | 0.1017 | 0.1017 | 0.0000 | 1.0000 |
| adapter | fwd | 3 | 0.1017 | 0.1017 | 0.0000 | 1.0000 |
| adapter | bwd | 2 | 0.3575 | 0.3575 | 0.0000 | 1.0000 |
| adapter | bwd | 3 | 0.3575 | 0.3575 | 0.0000 | 1.0000 |
| adapter | slow | 3 | 0.9754 | 0.9754 | 0.0000 | 1.0000 |
| dagger | fwd | 1 | 0.1017 | 0.1017 | 0.0000 | 1.0000 |
| dagger | fwd | 2 | 0.1017 | 0.1029 | -0.0013 | 1.0123 |
| dagger | fwd | 3 | 0.1017 | 0.1333 | -0.0317 | 1.3115 |
| dagger | bwd | 2 | 0.1683 | 0.1683 | 0.0000 | 1.0000 |
| dagger | bwd | 3 | 0.1683 | 0.1542 | 0.0142 | 0.9158 |
| dagger | slow | 3 | 0.9838 | 0.9838 | 0.0000 | 1.0000 |
| joint | fwd | 1 | 0.1017 | 0.1017 | 0.0000 | 1.0000 |
| joint | fwd | 2 | 0.1017 | 0.2188 | -0.1171 | 2.1516 |
| joint | fwd | 3 | 0.1017 | 0.3087 | -0.2071 | 3.0369 |
| joint | bwd | 2 | 0.1946 | 0.1946 | 0.0000 | 1.0000 |
| joint | bwd | 3 | 0.1946 | 0.2658 | -0.0712 | 1.3662 |
| joint | slow | 3 | 0.6013 | 0.6013 | 0.0000 | 1.0000 |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.31 |
| adapter | 2 | 786432 | 0 | 0 | 786432 | 6144 | 3145728 | 30.48 |
| adapter | 3 | 1310720 | 0 | 0 | 1310720 | 10240 | 5242880 | 55.25 |
| dagger | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.31 |
| dagger | 2 | 720896 | 393216 | 65536 | 786432 | 6400 | 3276800 | 25.62 |
| dagger | 3 | 1179648 | 786432 | 131072 | 1310720 | 10752 | 5505024 | 43.02 |
| joint | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.31 |
| joint | 2 | 786432 | 0 | 0 | 786432 | 6144 | 3145728 | 27.45 |
| joint | 3 | 1310720 | 0 | 0 | 1310720 | 10240 | 5242880 | 45.72 |

CPG: 1536000 search steps + 39600 validation steps; 34.98 seconds. Independently budgeted generous baseline.
Evaluation: 236000 completed environment steps; 10.43 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 19974 | 79896 | 39309 | 495030 |
| adapter_s2 | 24540 | 98160 | 48402 | 313430 |
| adapter_s3 | 29106 | 116424 | 57495 | 350590 |
| dagger_s1 | 19974 | 79896 | 39309 | 495030 |
| dagger_s2 | 19974 | 79896 | 39309 | 2886423 |
| dagger_s3 | 19974 | 79896 | 39309 | 2886494 |
| joint_s1 | 19974 | 79896 | 39309 | 495030 |
| joint_s2 | 19974 | 79896 | 39309 | 508339 |
| joint_s3 | 19974 | 79896 | 39309 | 508337 |
| oracle_s3 | 59922 | 239688 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
