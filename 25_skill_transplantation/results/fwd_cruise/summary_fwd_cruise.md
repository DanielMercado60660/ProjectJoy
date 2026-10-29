# Auto-generated summary (seeds [0, 1, 2, 3], specialist branch = 2000000 env steps)

Returns are deterministic (mean-action) 1000-step episode returns on 10 held-out eval seeds, averaged per training seed; tables show mean ± std over training seeds. 'retained' = method return / oracle-switch (specialist) return on that task, computed per seed. 'gain retained' = (method - theta0) / (specialist - theta0), averaged over the two tasks: the share of what each branch learned AFTER the shared start that survives (in `init`, theta0 returns ~0 so it equals retained).


## Condition `pre`: shared pretraining (theta0 = 0.5M-step multitask base)

| method | fwd return | cruise return | fwd retained | cruise retained | mean retained | gain retained | lambda |
|---|---|---|---|---|---|---|---|
| oracle_switch | 6998 ± 891 | 5304 ± 1054 | 100% ± 0 | 100% ± 0 | 100% ± 0 | 100% ± 0 |  |
| theta0 | 3079 ± 649 | 1832 ± 359 | 44% ± 5 | 35% ± 6 | 39% ± 4 | 0% ± 0 |  |
| average | 5802 ± 683 | 3476 ± 490 | 84% ± 10 | 67% ± 11 | 76% ± 10 | 61% ± 16 |  |
| task_arith | 5920 ± 675 | 3514 ± 495 | 85% ± 7 | 68% ± 10 | 76% ± 8 | 62% ± 12 | 0.5, 0.5, 0.5, 0.5 |
| ties | 3827 ± 576 | 2205 ± 353 | 55% ± 4 | 43% ± 11 | 49% ± 7 | 16% ± 8 | 0.3, 0.5, 1.2, 0.5 |
| dare | 1474 ± 1102 | -340 ± 1347 | 21% ± 15 | -8% ± 23 | 7% ± 19 | -54% ± 34 | 0.3, 0.3, 0.3, 0.3 |
| zero_shot_fwd_spec | 6998 ± 891 | 2442 ± 443 | 100% ± 0 | 49% ± 16 | 74% ± 8 | 61% ± 10 |  |
| zero_shot_cruise_spec | 5091 ± 901 | 5304 ± 1054 | 72% ± 4 | 100% ± 0 | 86% ± 2 | 76% ± 2 |  |
| distill_theta0 | 7100 ± 975 | 5249 ± 1040 | 101% ± 2 | 99% ± 0 | 100% ± 1 | 101% ± 2 |  |
| distill_average | 7085 ± 963 | 5198 ± 1012 | 101% ± 1 | 98% ± 3 | 100% ± 1 | 100% ± 1 |  |
| ft_theta0 | 5968 ± 716 | 4048 ± 450 | 85% ± 2 | 78% ± 8 | 82% ± 4 | 70% ± 6 |  |
| ft_average | 6322 ± 1519 | 3933 ± 1292 | 89% ± 11 | 73% ± 14 | 81% ± 11 | 71% ± 15 |  |
| ft_task_arith | 5649 ± 1789 | 3588 ± 1553 | 79% ± 17 | 65% ± 19 | 72% ± 18 | 56% ± 27 |  |

**Interference (task arithmetic, same lambda):** return on each task with only its own task vector added vs with both added.

| task | theta0 + lam*tau_own | theta0 + lam*(tau_fwd+tau_cruise) | drop from adding the other vector |
|---|---|---|---|
| fwd | 4387 ± 1271 | 5920 ± 675 | -1533 ± 909 |
| cruise | 2932 ± 286 | 3514 ± 495 | -582 ± 387 |

**Branch length** (env steps each specialist trained after branching) vs merge quality, mean retained over both tasks:

| branch steps | specialist fwd / cruise return | average | task arith | TIES | DARE | TIES gain retained | cos(tau_fwd, tau_cruise) | |tau| / |theta0| |
|---|---|---|---|---|---|---|---|---|
| 125000 | 4441 ± 642 / 2816 ± 176 | 96% ± 2 | 99% ± 4 | 87% ± 7 | 75% ± 10 | 62% ± 16 | 0.38 ± 0.06 | 0.18 ± 0.01 |
| 250000 | 4881 ± 563 / 3110 ± 118 | 95% ± 3 | 96% ± 1 | 80% ± 8 | 64% ± 15 | 50% ± 15 | 0.45 ± 0.06 | 0.22 ± 0.02 |
| 500000 | 5661 ± 694 / 3775 ± 407 | 92% ± 1 | 91% ± 2 | 69% ± 6 | 41% ± 21 | 36% ± 12 | 0.51 ± 0.04 | 0.28 ± 0.02 |
| 1000000 | 6126 ± 486 / 4663 ± 823 | 85% ± 1 | 85% ± 2 | 56% ± 5 | 24% ± 13 | 22% ± 8 | 0.54 ± 0.03 | 0.34 ± 0.02 |
| 2000000 | 6998 ± 891 / 5304 ± 1054 | 76% ± 10 | 76% ± 8 | 49% ± 7 | 7% ± 19 | 16% ± 8 | 0.56 ± 0.03 | 0.38 ± 0.02 |

## Condition `init`: shared random init only (theta0 = init)

| method | fwd return | cruise return | fwd retained | cruise retained | mean retained | gain retained | lambda |
|---|---|---|---|---|---|---|---|
| oracle_switch | 4819 ± 1027 | 4180 ± 210 | 100% ± 0 | 100% ± 0 | 100% ± 0 | 100% ± 0 |  |
| theta0 | -0 ± 0 | -0 ± 0 | -0% ± 0 | -0% ± 0 | -0% ± 0 | 0% ± 0 |  |
| average | 2145 ± 260 | 155 ± 352 | 46% ± 8 | 4% ± 9 | 25% ± 6 | 25% ± 6 |  |
| task_arith | 2007 ± 166 | 336 ± 552 | 43% ± 6 | 8% ± 13 | 25% ± 6 | 25% ± 6 | 0.5, 0.5, 0.5, 0.3 |
| ties | -6 ± 12 | -73 ± 88 | -0% ± 0 | -2% ± 2 | -1% ± 1 | -1% ± 1 | 0.3, 0.3, 0.5, 0.3 |
| dare | 13 ± 277 | -1596 ± 721 | 0% ± 5 | -38% ± 16 | -19% ± 10 | -19% ± 10 | 0.3, 0.3, 0.3, 0.3 |
| zero_shot_fwd_spec | 4819 ± 1027 | 495 ± 324 | 100% ± 0 | 12% ± 7 | 56% ± 4 | 56% ± 4 |  |
| zero_shot_cruise_spec | 3669 ± 872 | 4180 ± 210 | 83% ± 35 | 100% ± 0 | 91% ± 17 | 91% ± 17 |  |
| distill_theta0 | 4775 ± 1014 | 4123 ± 231 | 99% ± 0 | 99% ± 1 | 99% ± 0 | 99% ± 0 |  |
| distill_average | 4692 ± 1098 | 4141 ± 211 | 97% ± 3 | 99% ± 0 | 98% ± 2 | 98% ± 2 |  |
| ft_theta0 | 3870 ± 1168 | 2503 ± 993 | 80% ± 13 | 59% ± 21 | 69% ± 17 | 69% ± 17 |  |
| ft_average | 4856 ± 746 | 3013 ± 608 | 102% ± 9 | 72% ± 14 | 87% ± 9 | 87% ± 9 |  |
| ft_task_arith | 4948 ± 788 | 3204 ± 590 | 104% ± 11 | 76% ± 13 | 90% ± 9 | 90% ± 9 |  |

**Interference (task arithmetic, same lambda):** return on each task with only its own task vector added vs with both added.

| task | theta0 + lam*tau_own | theta0 + lam*(tau_fwd+tau_cruise) | drop from adding the other vector |
|---|---|---|---|
| fwd | 1119 ± 794 | 2007 ± 166 | -888 ± 878 |
| cruise | 418 ± 332 | 336 ± 552 | 83 ± 814 |

**Branch length** (env steps each specialist trained after branching) vs merge quality, mean retained over both tasks:

| branch steps | specialist fwd / cruise return | average | task arith | TIES | DARE | TIES gain retained | cos(tau_fwd, tau_cruise) | |tau| / |theta0| |
|---|---|---|---|---|---|---|---|---|
| 125000 | 1038 ± 149 / 76 ± 100 | 845% ± 1372 | -3168% ± 5700 | -5461% ± 9808 | 639% ± 755 | -7194% ± 12811 | 0.18 ± 0.03 | 0.27 ± 0.01 |
| 250000 | 1861 ± 193 / 1346 ± 571 | 52% ± 8 | 37% ± 37 | 15% ± 32 | -4% ± 9 | 15% ± 32 | 0.19 ± 0.03 | 0.35 ± 0.02 |
| 500000 | 3017 ± 843 / 2741 ± 305 | 42% ± 16 | 48% ± 7 | 15% ± 9 | 0% ± 20 | 15% ± 9 | 0.19 ± 0.02 | 0.46 ± 0.02 |
| 1000000 | 4075 ± 1125 / 3581 ± 195 | 21% ± 11 | 32% ± 3 | 6% ± 5 | -16% ± 13 | 6% ± 5 | 0.22 ± 0.02 | 0.55 ± 0.02 |
| 2000000 | 4819 ± 1027 / 4180 ± 210 | 25% ± 6 | 25% ± 6 | -1% ± 1 | -19% ± 10 | -1% ± 1 | 0.25 ± 0.02 | 0.60 ± 0.02 |

## Adaptation cost: env steps of multitask PPO (8 fwd + 8 cruise envs) or DAgger distillation until BOTH tasks reach a threshold fraction of the specialists (first eval at or above; '>' = not reached)

| run | 50% | 80% | 90% | final fwd / cruise retained |
|---|---|---|---|---|
| pre_ft_theta0 | 203k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (0/4) | 85% ± 2 / 78% ± 8 |
| pre_ft_average | 0k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (median) (1/4 reached) | 89% ± 11 / 73% ± 14 |
| pre_ft_task_arith | 0k median (4/4 reached) | > budget (median) (1/4 reached) | > budget (0/4) | 79% ± 17 / 65% ± 19 |
| pre_distill_theta0 | 29k median (4/4 reached) | 29k median (4/4 reached) | 41k median (4/4 reached) | 101% ± 2 / 99% ± 0 |
| pre_distill_average | 0k median (4/4 reached) | 29k median (4/4 reached) | 41k median (4/4 reached) | 101% ± 1 / 98% ± 3 |
| pre_transfer_fwd2cruise | 127k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (median) (1/4 reached) | 40% ± 30 / 66% ± 33 |
| pre_transfer_cruise2fwd | 0k median (4/4 reached) | 502k median (3/4 reached) | > budget (median) (2/4 reached) | 76% ± 26 / 15% ± 39 |
| init_ft_theta0 | > budget (median) (2/4 reached) | > budget (median) (1/4 reached) | > budget (0/4) | 80% ± 13 / 59% ± 21 |
| init_ft_average | 352k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (0/4) | 102% ± 9 / 72% ± 14 |
| init_ft_task_arith | 252k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (0/4) | 104% ± 11 / 76% ± 13 |
| init_distill_theta0 | 53k median (4/4 reached) | 78k median (4/4 reached) | 102k median (4/4 reached) | 99% ± 0 / 99% ± 1 |
| init_distill_average | 29k median (4/4 reached) | 29k median (4/4 reached) | 29k median (4/4 reached) | 97% ± 3 / 99% ± 0 |
| multitask_scratch | 1128k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (median) (1/4 reached) | 86% ± 8 / 75% ± 12 |

## Sequential transfer (`pre`): PPO on a task starting from the OTHER specialist vs from theta0

| target task | start | return at 0 | at ~0.25M | at ~0.5M | at ~0.88M |
|---|---|---|---|---|---|
| cruise | other task's specialist | 2442 ± 443 | 2367 ± 1717 | 3066 ± 1838 | 3707 ± 2003 |
| cruise | theta0 (= the specialist's own training curve) | 1832 ± 359 | 3110 ± 118 | 3775 ± 407 | 4436 ± 724 |
| fwd | other task's specialist | 5091 ± 901 | 3925 ± 1595 | 4550 ± 1514 | 5124 ± 1548 |
| fwd | theta0 (= the specialist's own training curve) | 3079 ± 649 | 4881 ± 563 | 5661 ± 694 | 6337 ± 854 |
