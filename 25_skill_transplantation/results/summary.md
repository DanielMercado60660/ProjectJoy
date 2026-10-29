# Auto-generated summary (seeds [0, 1, 2, 3], specialist branch = 2000000 env steps)

Returns are deterministic (mean-action) 1000-step episode returns on 10 held-out eval seeds, averaged per training seed; tables show mean ± std over training seeds. 'retained' = method return / oracle-switch (specialist) return on that task, computed per seed. 'gain retained' = (method - theta0) / (specialist - theta0), averaged over the two tasks: the share of what each branch learned AFTER the shared start that survives (in `init`, theta0 returns ~0 so it equals retained).


## Condition `pre`: shared pretraining (theta0 = 0.5M-step multitask base)

| method | fwd return | bwd return | fwd retained | bwd retained | mean retained | gain retained | lambda |
|---|---|---|---|---|---|---|---|
| oracle_switch | 5320 ± 219 | 8026 ± 229 | 100% ± 0 | 100% ± 0 | 100% ± 0 | 100% ± 0 |  |
| theta0 | 1643 ± 208 | 4797 ± 151 | 31% ± 5 | 60% ± 2 | 45% ± 3 | 0% ± 0 |  |
| average | 429 ± 386 | 3358 ± 1823 | 8% ± 7 | 42% ± 23 | 25% ± 10 | -41% ± 30 |  |
| task_arith | 781 ± 782 | 4942 ± 245 | 15% ± 15 | 62% ± 4 | 38% ± 9 | -10% ± 14 | 0.3, 0.3, 0.3, 0.3 |
| ties | 766 ± 496 | 5087 ± 242 | 15% ± 9 | 63% ± 4 | 39% ± 6 | -7% ± 8 | 0.3, 0.3, 0.5, 0.5 |
| dare | 1211 ± 568 | 2822 ± 1709 | 23% ± 11 | 35% ± 21 | 29% ± 6 | -39% ± 25 | 0.3, 0.3, 0.3, 0.3 |
| zero_shot_fwd_spec | 5320 ± 219 | 897 ± 1111 | 100% ± 0 | 11% ± 14 | 56% ± 7 | -11% ± 20 |  |
| zero_shot_bwd_spec | -552 ± 292 | 8026 ± 229 | -10% ± 6 | 100% ± 0 | 45% ± 3 | 19% ± 8 |  |
| distill_theta0 | 5285 ± 197 | 7995 ± 239 | 99% ± 1 | 100% ± 0 | 99% ± 0 | 99% ± 0 |  |
| distill_average | 5003 ± 549 | 8039 ± 285 | 94% ± 10 | 100% ± 1 | 97% ± 5 | 96% ± 7 |  |
| ft_theta0 | 3068 ± 568 | 6266 ± 224 | 58% ± 12 | 78% ± 2 | 68% ± 5 | 43% ± 5 |  |
| ft_average | 3424 ± 327 | 7132 ± 194 | 65% ± 7 | 89% ± 2 | 77% ± 3 | 60% ± 2 |  |
| ft_task_arith | 3653 ± 910 | 6926 ± 112 | 69% ± 17 | 86% ± 2 | 78% ± 9 | 61% ± 12 |  |
| task_arith_fine | 1344 ± 368 | 4948 ± 284 | 25% ± 8 | 62% ± 4 | 44% ± 3 | -2% ± 3 | 0.0, 0.2, 0.0, 0.1 |
| ties_fine | 1593 ± 258 | 4961 ± 197 | 30% ± 5 | 62% ± 3 | 46% ± 4 | 2% ± 6 | 0.1, 0.1, 0.0, 0.2 |
| dare_fine | 1659 ± 282 | 4863 ± 272 | 31% ± 6 | 61% ± 4 | 46% ± 4 | 1% ± 4 | 0.0, 0.1, 0.2, 0.0 |
| scripted CPG gait (per-task, label-switched) | 3676 | 4372 | 69% | 54% | | |

**Interference (task arithmetic, same lambda):** return on each task with only its own task vector added vs with both added.

| task | theta0 + lam*tau_own | theta0 + lam*(tau_fwd+tau_bwd) | drop from adding the other vector |
|---|---|---|---|
| fwd | 1382 ± 888 | 781 ± 782 | 601 ± 1144 |
| bwd | 5745 ± 121 | 4942 ± 245 | 803 ± 255 |

**Branch length** (env steps each specialist trained after branching) vs merge quality, mean retained over both tasks:

| branch steps | specialist fwd / bwd return | average | task arith | TIES | DARE | TIES gain retained | cos(tau_fwd, tau_bwd) | |tau| / |theta0| |
|---|---|---|---|---|---|---|---|---|
| 125000 | 1800 ± 642 / 5605 ± 248 | 77% ± 11 | 85% ± 18 | 80% ± 12 | 78% ± 15 | 39% ± 37 | 0.07 ± 0.01 | 0.18 ± 0.00 |
| 250000 | 2481 ± 301 / 6126 ± 299 | 65% ± 11 | 67% ± 8 | 71% ± 7 | 70% ± 8 | -5% ± 28 | 0.09 ± 0.01 | 0.23 ± 0.01 |
| 500000 | 3155 ± 489 / 6724 ± 258 | 40% ± 13 | 60% ± 6 | 60% ± 8 | 50% ± 10 | -1% ± 18 | 0.13 ± 0.02 | 0.30 ± 0.01 |
| 1000000 | 4632 ± 194 / 7401 ± 196 | 31% ± 13 | 40% ± 9 | 43% ± 9 | 38% ± 7 | -10% ± 16 | 0.18 ± 0.01 | 0.37 ± 0.01 |
| 2000000 | 5320 ± 219 / 8026 ± 229 | 25% ± 10 | 38% ± 9 | 39% ± 6 | 29% ± 6 | -7% ± 8 | 0.21 ± 0.01 | 0.42 ± 0.01 |

## Condition `init`: shared random init only (theta0 = init)

| method | fwd return | bwd return | fwd retained | bwd retained | mean retained | gain retained | lambda |
|---|---|---|---|---|---|---|---|
| oracle_switch | 5464 ± 864 | 7969 ± 371 | 100% ± 0 | 100% ± 0 | 100% ± 0 | 100% ± 0 |  |
| theta0 | -0 ± 0 | 0 ± 0 | -0% ± 0 | 0% ± 0 | -0% ± 0 | 0% ± 0 |  |
| average | -2109 ± 800 | 2271 ± 345 | -38% ± 11 | 28% ± 3 | -5% ± 4 | -5% ± 4 |  |
| task_arith | -2111 ± 789 | 2218 ± 317 | -38% ± 11 | 28% ± 3 | -5% ± 4 | -5% ± 4 | 0.5, 1.0, 0.5, 0.5 |
| ties | -80 ± 393 | 680 ± 901 | -3% ± 9 | 9% ± 11 | 3% ± 2 | 3% ± 2 | 0.5, 0.7, 1.5, 1.2 |
| dare | -844 ± 526 | 285 ± 593 | -18% ± 15 | 4% ± 7 | -7% ± 4 | -7% ± 4 | 0.3, 0.3, 0.3, 0.3 |
| zero_shot_fwd_spec | 5464 ± 864 | -4914 ± 979 | 100% ± 0 | -62% ± 13 | 19% ± 6 | 19% ± 6 |  |
| zero_shot_bwd_spec | -5938 ± 696 | 7969 ± 371 | -110% ± 15 | 100% ± 0 | -5% ± 7 | -5% ± 7 |  |
| distill_theta0 | 5481 ± 911 | 7913 ± 401 | 100% ± 4 | 99% ± 2 | 100% ± 2 | 100% ± 2 |  |
| distill_average | 5284 ± 959 | 7942 ± 380 | 97% ± 9 | 100% ± 0 | 98% ± 5 | 98% ± 5 |  |
| ft_theta0 | 2006 ± 152 | 5529 ± 81 | 38% ± 10 | 70% ± 3 | 54% ± 5 | 54% ± 5 |  |
| ft_average | 4352 ± 902 | 7246 ± 164 | 79% ± 7 | 91% ± 2 | 85% ± 2 | 85% ± 2 |  |
| ft_task_arith | 2220 ± 2180 | 6215 ± 1645 | 43% ± 39 | 78% ± 21 | 61% ± 30 | 61% ± 30 |  |
| task_arith_fine | -2111 ± 789 | 2218 ± 317 | -38% ± 11 | 28% ± 3 | -5% ± 4 | -5% ± 4 | 0.5, 1.0, 0.5, 0.5 |
| ties_fine | -80 ± 393 | 680 ± 901 | -3% ± 9 | 9% ± 11 | 3% ± 2 | 3% ± 2 | 0.5, 0.7, 1.5, 1.2 |
| dare_fine | -0 ± 0 | 0 ± 0 | -0% ± 0 | 0% ± 0 | -0% ± 0 | -0% ± 0 | 0.0, 0.0, 0.0, 0.0 |
| scripted CPG gait (per-task, label-switched) | 3676 | 4372 | 67% | 55% | | |

**Interference (task arithmetic, same lambda):** return on each task with only its own task vector added vs with both added.

| task | theta0 + lam*tau_own | theta0 + lam*(tau_fwd+tau_bwd) | drop from adding the other vector |
|---|---|---|---|
| fwd | 3139 ± 1054 | -2111 ± 789 | 5249 ± 1798 |
| bwd | 5773 ± 1410 | 2218 ± 317 | 3555 ± 1287 |

**Branch length** (env steps each specialist trained after branching) vs merge quality, mean retained over both tasks:

| branch steps | specialist fwd / bwd return | average | task arith | TIES | DARE | TIES gain retained | cos(tau_fwd, tau_bwd) | |tau| / |theta0| |
|---|---|---|---|---|---|---|---|---|
| 125000 | 605 ± 349 / 3739 ± 492 | -699% ± 1159 | -1547% ± 2482 | -122% ± 210 | -444% ± 758 | -120% ± 205 | 0.09 ± 0.01 | 0.28 ± 0.01 |
| 250000 | 1825 ± 367 / 5033 ± 306 | -32% ± 12 | -27% ± 18 | -10% ± 11 | -9% ± 11 | -10% ± 11 | 0.10 ± 0.01 | 0.36 ± 0.01 |
| 500000 | 3271 ± 843 / 6063 ± 149 | -15% ± 5 | -13% ± 12 | 1% ± 3 | -5% ± 2 | 1% ± 3 | 0.12 ± 0.01 | 0.45 ± 0.02 |
| 1000000 | 4676 ± 1034 / 7104 ± 114 | -7% ± 3 | -6% ± 3 | 0% ± 2 | -8% ± 3 | 0% ± 2 | 0.15 ± 0.01 | 0.54 ± 0.02 |
| 2000000 | 5464 ± 864 / 7969 ± 371 | -5% ± 4 | -5% ± 4 | 3% ± 2 | -7% ± 4 | 3% ± 2 | 0.17 ± 0.01 | 0.60 ± 0.03 |

## Adaptation cost: env steps of multitask PPO (8 fwd + 8 bwd envs) or DAgger distillation until BOTH tasks reach a threshold fraction of the specialists (first eval at or above; '>' = not reached)

| run | 50% | 80% | 90% | final fwd / bwd retained |
|---|---|---|---|---|
| pre_ft_theta0 | 602k median (4/4 reached) | > budget (0/4) | > budget (0/4) | 58% ± 12 / 78% ± 2 |
| pre_ft_average | 651k median (4/4 reached) | > budget (0/4) | > budget (0/4) | 65% ± 7 / 89% ± 2 |
| pre_ft_task_arith | 303k median (3/4 reached) | > budget (median) (2/4 reached) | > budget (0/4) | 69% ± 17 / 86% ± 2 |
| pre_distill_theta0 | 29k median (4/4 reached) | 53k median (4/4 reached) | 53k median (4/4 reached) | 99% ± 1 / 100% ± 0 |
| pre_distill_average | 29k median (4/4 reached) | 29k median (4/4 reached) | 29k median (4/4 reached) | 94% ± 10 / 100% ± 1 |
| pre_transfer_fwd2bwd | 254k median (4/4 reached) | > budget (median) (1/4 reached) | > budget (0/4) | 20% ± 14 / 79% ± 4 |
| pre_transfer_bwd2fwd | > budget (0/4) | > budget (0/4) | > budget (0/4) | 19% ± 7 / 15% ± 20 |
| init_ft_theta0 | > budget (median) (1/4 reached) | > budget (0/4) | > budget (0/4) | 38% ± 10 / 70% ± 3 |
| init_ft_average | 303k median (4/4 reached) | > budget (median) (2/4 reached) | > budget (0/4) | 79% ± 7 / 91% ± 2 |
| init_ft_task_arith | > budget (median) (2/4 reached) | > budget (0/4) | > budget (0/4) | 43% ± 39 / 78% ± 21 |
| init_distill_theta0 | 29k median (4/4 reached) | 53k median (4/4 reached) | 78k median (4/4 reached) | 100% ± 4 / 99% ± 2 |
| init_distill_average | 29k median (4/4 reached) | 29k median (4/4 reached) | 29k median (3/4 reached) | 97% ± 9 / 100% ± 0 |
| multitask_scratch | 1128k median (4/4 reached) | 3002k median (3/4 reached) | > budget (median) (2/4 reached) | 85% ± 12 / 101% ± 4 |

## Sequential transfer (`pre`): PPO on a task starting from the OTHER specialist vs from theta0

| target task | start | return at 0 | at ~0.25M | at ~0.5M | at ~0.88M |
|---|---|---|---|---|---|
| bwd | other task's specialist | 897 ± 1111 | 4642 ± 415 | 5658 ± 404 | 6363 ± 230 |
| bwd | theta0 (= the specialist's own training curve) | 4797 ± 151 | 6126 ± 299 | 6724 ± 258 | 7307 ± 164 |
| fwd | other task's specialist | -552 ± 292 | 119 ± 145 | 519 ± 250 | 992 ± 397 |
| fwd | theta0 (= the specialist's own training curve) | 1643 ± 208 | 2481 ± 301 | 3155 ± 489 | 4287 ± 267 |
