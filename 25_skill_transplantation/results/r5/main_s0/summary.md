# Round 5 run main_s0

One training seed per run. Fresh test resets are primary; legacy resets are secondary.
Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.
Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.

## test: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.3897 | 1.164 | 328.90 | 0.7708 | 0.4583 | 0.1042 | 8.037 |
| dagger | 0.3849 | 1.020 | 355.08 | 0.7083 | 0.3958 | 0.1042 | 7.716 |
| joint | 0.2193 | 1.382 | 230.10 | 1.0000 | 0.2292 | 0.0417 | 9.907 |
| oracle | 0.3715 | 1.149 | 336.80 | 0.6667 | 0.3125 | 0.0417 | 7.260 |
| cpg | 0.7119 | 0.587 | 445.84 | 0.5000 | 0.1042 | 0.0000 | 6.380 |
| zero | 0.0000 | 2.333 | 125.79 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## legacy: final three-skill repertoire

| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 0.3972 | 1.077 | 331.89 | 0.7500 | 0.2917 | 0.1250 | 8.108 |
| dagger | 0.3553 | 1.295 | 323.70 | 0.8333 | 0.7083 | 0.1250 | 8.360 |
| joint | 0.2317 | 1.379 | 235.64 | 1.0000 | 0.1667 | 0.0417 | 10.000 |
| oracle | 0.3664 | 1.073 | 336.42 | 0.7500 | 0.4167 | 0.1250 | 7.763 |
| cpg | 0.7219 | 0.516 | 460.03 | 0.3750 | 0.0000 | 0.0000 | 5.483 |
| zero | 0.0000 | 2.333 | 125.70 | 1.0000 | 0.0000 | 1.0000 | 10.000 |

## Retention (fresh tests; acquisition snapshot is the anchor)

| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |
|---|---|---:|---:|---:|---:|---:|
| adapter | fwd | 1 | 0.1675 | 0.1675 | 0.0000 | 1.0000 |
| adapter | fwd | 2 | 0.1675 | 0.1675 | 0.0000 | 1.0000 |
| adapter | fwd | 3 | 0.1675 | 0.1675 | 0.0000 | 1.0000 |
| adapter | bwd | 2 | 0.1112 | 0.1112 | 0.0000 | 1.0000 |
| adapter | bwd | 3 | 0.1112 | 0.1112 | 0.0000 | 1.0000 |
| adapter | slow | 3 | 0.8904 | 0.8904 | 0.0000 | 1.0000 |
| dagger | fwd | 1 | 0.1675 | 0.1675 | 0.0000 | 1.0000 |
| dagger | fwd | 2 | 0.1675 | 0.1554 | 0.0121 | 0.9279 |
| dagger | fwd | 3 | 0.1675 | 0.2071 | -0.0396 | 1.2363 |
| dagger | bwd | 2 | 0.0588 | 0.0588 | 0.0000 | 1.0000 |
| dagger | bwd | 3 | 0.0588 | 0.0671 | -0.0083 | 1.1418 |
| dagger | slow | 3 | 0.8804 | 0.8804 | 0.0000 | 1.0000 |
| joint | fwd | 1 | 0.1675 | 0.1675 | 0.0000 | 1.0000 |
| joint | fwd | 2 | 0.1675 | 0.1825 | -0.0150 | 1.0896 |
| joint | fwd | 3 | 0.1675 | 0.2450 | -0.0775 | 1.4627 |
| joint | bwd | 2 | 0.0863 | 0.0863 | 0.0000 | 1.0000 |
| joint | bwd | 3 | 0.0863 | 0.1779 | -0.0917 | 2.0628 |
| joint | slow | 3 | 0.2350 | 0.2350 | 0.0000 | 1.0000 |

## Completed training cost (common base charged to each arm)

Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.
Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.

| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.29 |
| adapter | 2 | 786432 | 0 | 0 | 786432 | 6144 | 3145728 | 30.39 |
| adapter | 3 | 1310720 | 0 | 0 | 1310720 | 10240 | 5242880 | 55.72 |
| dagger | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.29 |
| dagger | 2 | 720896 | 393216 | 65536 | 786432 | 6400 | 3276800 | 25.84 |
| dagger | 3 | 1179648 | 786432 | 131072 | 1310720 | 10752 | 5505024 | 43.45 |
| joint | 1 | 262144 | 0 | 0 | 262144 | 2048 | 1048576 | 9.29 |
| joint | 2 | 786432 | 0 | 0 | 786432 | 6144 | 3145728 | 27.39 |
| joint | 3 | 1310720 | 0 | 0 | 1310720 | 10240 | 5242880 | 45.38 |

CPG: 1536000 search steps + 39600 validation steps; 34.49 seconds. Independently budgeted generous baseline.
Evaluation: 236000 completed environment steps; 10.45 seconds.

## Memory

| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |
|---|---:|---:|---:|---:|
| adapter_s1 | 19974 | 79896 | 39309 | 495030 |
| adapter_s2 | 24540 | 98160 | 48402 | 313423 |
| adapter_s3 | 29106 | 116424 | 57495 | 350595 |
| dagger_s1 | 19974 | 79896 | 39309 | 495030 |
| dagger_s2 | 19974 | 79896 | 39309 | 2886424 |
| dagger_s3 | 19974 | 79896 | 39309 | 2886492 |
| joint_s1 | 19974 | 79896 | 39309 | 495030 |
| joint_s2 | 19974 | 79896 | 39309 | 508332 |
| joint_s3 | 19974 | 79896 | 39309 | 508342 |
| oracle_s3 | 59922 | 239688 | n/a | n/a |
| cpg_s3 | 57 | 456 | n/a | n/a |

Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.
Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.
Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.
The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).
