# Grip-randomized runs: seeds x selection rule (strict rules, 1 lap)

Held-out: 100 unseen tracks; hard: 50 narrower/twistier OOD tracks. Cells: completion % (mean lap time of finished laps). Mean/range are over the seeds listed.


## PPO + grip rand., selected on grip-1.0 validation (original rule)

| seed | ckpt (env steps) | held-out 1.0 | held-out 0.8 | held-out 0.7 | held-out 0.6 | held-out 0.5 | hard 1.0 | hard 0.8 | hard 0.7 | hard 0.6 | hard 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (`ppo_dr`) | 4.6 M | 96% (27.3 s) | 100% (27.9 s) | 100% (28.9 s) | 100% (30.4 s) | 93% (33.4 s) | 60% | 66% | 90% | 98% | 98% |
| 1 (`ppo_dr_s1`) | 4.3 M | 92% (27.1 s) | 99% (27.9 s) | 100% (28.9 s) | 100% (30.4 s) | 96% (33.3 s) | 66% | 66% | 88% | 98% | 100% |
| 2 (`ppo_dr_s2`) | 14.1 M | 95% (27.0 s) | 99% (27.6 s) | 100% (28.6 s) | 100% (30.2 s) | 93% (33.3 s) | 70% | 76% | 78% | 90% | 98% |
| **mean** | | 94% (27.1 s) | 99% (27.8 s) | 100% (28.8 s) | 100% (30.3 s) | 94% (33.3 s) | 65% | 69% | 85% | 95% | 99% |
| **range** | | 92–96% | 99–100% | 100–100% | 100–100% | 93–96% | 60–70% | 66–76% | 78–90% | 90–98% | 98–100% |

## PPO + grip rand., selected on randomized-grip validation (μ ~ U(0.5, 1.1))

| seed | ckpt (env steps) | held-out 1.0 | held-out 0.8 | held-out 0.7 | held-out 0.6 | held-out 0.5 | hard 1.0 | hard 0.8 | hard 0.7 | hard 0.6 | hard 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (`ppo_dr_r0_rg`) | 4.6 M | 96% (27.3 s) | 100% (27.9 s) | 100% (28.9 s) | 100% (30.4 s) | 93% (33.4 s) | 60% | 66% | 90% | 98% | 98% |
| 1 (`ppo_dr_s1_rg`) | 9.2 M | 92% (27.4 s) | 99% (27.8 s) | 100% (28.7 s) | 100% (30.2 s) | 96% (33.1 s) | 66% | 72% | 90% | 98% | 100% |
| 2 (`ppo_dr_s2_rg`) | 7.2 M | 96% (27.1 s) | 100% (27.8 s) | 100% (28.7 s) | 100% (30.3 s) | 94% (33.3 s) | 76% | 62% | 84% | 98% | 100% |
| **mean** | | 95% (27.3 s) | 100% (27.8 s) | 100% (28.8 s) | 100% (30.3 s) | 94% (33.3 s) | 67% | 67% | 88% | 98% | 99% |
| **range** | | 92–96% | 99–100% | 100–100% | 100–100% | 93–96% | 60–76% | 62–72% | 84–90% | 98–98% | 98–100% |

## ES + grip rand., selected on grip-1.0 validation (original rule)

| seed | ckpt (env steps) | held-out 1.0 | held-out 0.8 | held-out 0.7 | held-out 0.6 | held-out 0.5 | hard 1.0 | hard 0.8 | hard 0.7 | hard 0.6 | hard 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (`es_dr`) | 304 M | 100% (26.0 s) | 100% (27.6 s) | 100% (28.9 s) | 95% (30.5 s) | 41% (32.7 s) | 90% | 100% | 100% | 100% | 66% |
| 1 (`es_dr_s1`) | 298.3 M | 100% (25.9 s) | 100% (27.6 s) | 100% (29.0 s) | 89% (30.8 s) | 21% (32.7 s) | 98% | 100% | 100% | 94% | 22% |
| 2 (`es_dr_s2`) | 309.5 M | 100% (26.2 s) | 100% (27.7 s) | 100% (28.9 s) | 93% (30.8 s) | 33% (32.2 s) | 98% | 100% | 100% | 100% | 50% |
| **mean** | | 100% (26.0 s) | 100% (27.6 s) | 100% (28.9 s) | 92% (30.7 s) | 32% (32.5 s) | 95% | 100% | 100% | 98% | 46% |
| **range** | | 100–100% | 100–100% | 100–100% | 89–95% | 21–41% | 90–98% | 100–100% | 100–100% | 94–100% | 22–66% |

## ES + grip rand., selected on randomized-grip validation (μ ~ U(0.5, 1.1))

| seed | ckpt (env steps) | held-out 1.0 | held-out 0.8 | held-out 0.7 | held-out 0.6 | held-out 0.5 | hard 1.0 | hard 0.8 | hard 0.7 | hard 0.6 | hard 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (`es_dr_r0_rg`) | 256.1 M | 98% (26.8 s) | 100% (27.9 s) | 100% (28.9 s) | 100% (30.6 s) | 90% (33.0 s) | 88% | 94% | 100% | 100% | 100% |
| 1 (`es_dr_s1_rg`) | 262.4 M | 99% (26.6 s) | 100% (27.8 s) | 100% (28.9 s) | 100% (30.6 s) | 84% (33.3 s) | 88% | 98% | 100% | 100% | 96% |
| 2 (`es_dr_s2_rg`) | 273.8 M | 99% (26.9 s) | 100% (28.0 s) | 100% (29.1 s) | 100% (30.8 s) | 76% (33.4 s) | 96% | 96% | 100% | 100% | 94% |
| **mean** | | 99% (26.8 s) | 100% (27.9 s) | 100% (29.0 s) | 100% (30.6 s) | 83% (33.2 s) | 91% | 96% | 100% | 100% | 97% |
| **range** | | 98–99% | 100–100% | 100–100% | 100–100% | 76–90% | 88–96% | 94–98% | 100–100% | 100–100% | 94–100% |

## Reference: ray follower tuned with grip rand. (deterministic, one tuning run)

| | held-out 1.0 | held-out 0.8 | held-out 0.7 | held-out 0.6 | held-out 0.5 | hard 1.0 | hard 0.8 | hard 0.7 | hard 0.6 | hard 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| ray_dr | 100% (31.8 s) | 100% (31.8 s) | 100% (31.9 s) | 100% (32.7 s) | 24% (35.8 s) | 90% | 88% | 92% | 96% | 38% |

## Checkpoint selection (validation tracks 5000–5127 only)

val_g1 = trainers' grip-1.0 validation score; val_rg = randomized-grip validation score (mean 60 s fitness, m). {"val_tracks": "train generator seeds 5000-5127", "starts": "rng(12345)", "grips": "U(0.5, 1.1), 4 draws per track, rng(777)", "episodes_val_rg": 512, "steps": 600}

| run | snapshots | grip-1.0 rule: step (env steps), val_g1 / val_rg | randomized rule: step (env steps), val_g1 / val_rg | g1 argmax = trainer's best file |
|---|---|---|---|---|
| ppo_dr_s1 | 238 | update 130 (4.3 M), 1246 / 1187 | update 280 (9.2 M), 1202 / 1198 | True |
| ppo_dr_s2 | 238 | update 430 (14.1 M), 1271 / 1184 | update 220 (7.2 M), 1212 / 1203 | True |
| ppo_dr_r0 | 238 | update 140 (4.6 M), 1238 / 1195 | update 140 (4.6 M), 1238 / 1195 | True |
| es_dr_s1 | 28 | gen 269 (298.3 M), 1354 / 1131 | gen 239 (262.4 M), 1316 / 1209 | True |
| es_dr_s2 | 28 | gen 269 (309.5 M), 1338 / 1163 | gen 239 (273.8 M), 1302 / 1198 | True |
| es_dr_r0 | 28 | gen 269 (304.0 M), 1345 / 1171 | gen 229 (256.1 M), 1291 / 1205 | True |
| ppo_dr (original file) | – | val_g1 1238 / val_rg 1195 | | |
| es_dr (original file) | – | val_g1 1345 / val_rg 1171 | | |
