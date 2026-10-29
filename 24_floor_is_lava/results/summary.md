## Full gap range 0.05–0.20 m (500 held-out layouts)
| method | completion | lava | flip | progress | J/m (completed) | m/s (completed) | completion with 40 N push | fail ≤2 s after push |
|---|---|---|---|---|---|---|---|---|
| open-loop CPG, tuning round 1 | 2.4% [1.4, 4.1] | 94.0% | 3.2% | 0.36 | 1208 | 0.47 | 1.2% [0.6, 2.6] | 63.7% [56.5, 70.4] (n=182) |
| open-loop CPG, tuning round 2 | 6.8% [4.9, 9.4] | 92.2% | 0.6% | 0.40 | 1358 | 0.43 | 5.2% [3.6, 7.5] | 55.4% [48.6, 62.1] (n=202) |
| CPG + foothold planner, round 1 | 7.2% [5.2, 9.8] | 87.8% | 5.0% | 0.45 | 1051 | 0.47 | 4.6% [3.1, 6.8] | 54.9% [48.8, 60.8] (n=257) |
| CPG + foothold planner, round 2 | 31.0% [27.1, 35.2] | 67.4% | 1.6% | 0.67 | 903 | 0.52 | 33.6% [29.6, 37.9] | 32.0% [27.6, 36.7] (n=403) |
| CPG + foothold planner, round 3 | 62.6% [58.3, 66.7] | 34.2% | 3.2% | 0.84 | 843 | 0.44 | 57.6% [53.2, 61.9] | 14.7% [11.8, 18.3] (n=455) |
| CPG + foothold planner, round 4 | 69.4% [65.2, 73.3] | 28.8% | 1.8% | 0.85 | 971 | 0.44 | 64.6% [60.3, 68.7] | 12.3% [9.6, 15.6] (n=455) |
| residual PPO, blind (no stone obs) | 88.8% [85.7, 91.3] | 9.4% | 1.8% | 0.96 | 559 | 1.55 | 85.8% [82.5, 88.6] | 13.9% [11.1, 17.2] (n=498) |
| residual PPO (full obs) | 98.4% [96.9, 99.2] | 1.6% | 0.0% | 0.99 | 567 | 1.65 | 97.8% [96.1, 98.8] | 1.6% [0.8, 3.1] (n=497) |

## Easier: gaps 0.05–0.125 m (500 held-out layouts)
| method | completion | lava | flip | progress | J/m (completed) | m/s (completed) |
|---|---|---|---|---|---|---|
| open-loop CPG, tuning round 1 | 20.8% [17.5, 24.6] | 75.0% | 4.0% | 0.61 | 1210 | 0.47 |
| open-loop CPG, tuning round 2 | 46.6% [42.3, 51.0] | 51.8% | 1.6% | 0.74 | 1262 | 0.46 |
| CPG + foothold planner, round 1 | 45.0% [40.7, 49.4] | 46.6% | 8.0% | 0.76 | 980 | 0.50 |
| CPG + foothold planner, round 2 | 79.0% [75.2, 82.3] | 20.2% | 0.8% | 0.91 | 865 | 0.53 |
| CPG + foothold planner, round 3 | 87.8% [84.6, 90.4] | 9.0% | 3.0% | 0.94 | 794 | 0.46 |
| CPG + foothold planner, round 4 | 92.0% [89.3, 94.1] | 7.0% | 0.8% | 0.97 | 912 | 0.45 |
| residual PPO (full obs) | 99.6% [98.6, 99.9] | 0.4% | 0.0% | 1.00 | 575 | 1.59 |

## Push sweep on a flat walkway (200 held-out walkways, lateral force 0.1 s)
| method | 40 N | 80 N | 120 N |
|---|---|---|---|
| open-loop CPG r1 | 79.0% [72.8, 84.1] | 59.5% [52.6, 66.1] | 31.5% [25.5, 38.2] |
| open-loop CPG r2 | not run | 74.5% [68.0, 80.0] | not run |
| planner r1 | 100.0% [98.1, 100.0] | 90.5% [85.6, 93.8] | 25.5% [20.0, 32.0] |
| planner r4 | 100.0% [98.1, 100.0] | 83.0% [77.2, 87.6] | 2.5% [1.1, 5.7] |
| residual PPO | 100.0% [98.1, 100.0] | 98.5% [95.7, 99.5] | 42.0% [35.4, 48.9] |

## Paired comparisons (same layouts)
- `eval_run1_full` PPO − cpg_baseline: completion +0.960 [+0.942, +0.976], PPO-only 480, base-only 0; both completed n=12: J/m PPO 568 vs base 1208 (diff -640 [-700, -585]), speed 1.63 vs 0.47 m/s
- `eval_run1_full` PPO − cpg_planner_baseline: completion +0.912 [+0.886, +0.936], PPO-only 456, base-only 0; both completed n=36: J/m PPO 570 vs base 1051 (diff -481 [-522, -441]), speed 1.63 vs 0.47 m/s
- `eval_planner3_full` PPO − cpg_baseline: completion +0.960 [+0.942, +0.976], PPO-only 480, base-only 0; both completed n=12: J/m PPO 568 vs base 1208 (diff -640 [-700, -585]), speed 1.63 vs 0.47 m/s
- `eval_planner3_full` PPO − cpg_planner_baseline: completion +0.358 [+0.318, +0.400], PPO-only 183, base-only 4; both completed n=309: J/m PPO 567 vs base 843 (diff -275 [-281, -270]), speed 1.64 vs 0.44 m/s
- `eval_planner4_full` PPO − cpg_baseline: completion +0.960 [+0.942, +0.976], PPO-only 480, base-only 0; both completed n=12: J/m PPO 568 vs base 1208 (diff -640 [-700, -585]), speed 1.63 vs 0.47 m/s
- `eval_planner4_full` PPO − cpg_planner_baseline: completion +0.290 [+0.248, +0.332], PPO-only 150, base-only 5; both completed n=342: J/m PPO 567 vs base 970 (diff -403 [-410, -397]), speed 1.65 vs 0.44 m/s
- `eval_run1_gap0125` PPO − cpg_baseline: completion +0.788 [+0.750, +0.822], PPO-only 394, base-only 0; both completed n=104: J/m PPO 573 vs base 1210 (diff -636 [-666, -608]), speed 1.60 vs 0.47 m/s
- `eval_run1_gap0125` PPO − cpg_planner_baseline: completion +0.546 [+0.502, +0.590], PPO-only 274, base-only 1; both completed n=224: J/m PPO 573 vs base 981 (diff -408 [-427, -388]), speed 1.60 vs 0.50 m/s
- `eval_planner4_gap0125` PPO − cpg_baseline: completion +0.788 [+0.750, +0.822], PPO-only 394, base-only 0; both completed n=104: J/m PPO 573 vs base 1210 (diff -636 [-666, -608]), speed 1.60 vs 0.47 m/s
- `eval_planner4_gap0125` PPO − cpg_planner_baseline: completion +0.076 [+0.054, +0.102], PPO-only 39, base-only 1; both completed n=459: J/m PPO 574 vs base 912 (diff -338 [-342, -335]), speed 1.60 vs 0.45 m/s
