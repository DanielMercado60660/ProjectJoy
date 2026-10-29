
### heldout tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| PPO + grip randomization | 1.0 | 96% | 4% | 0.04 | 27.3 s | 27.0 s (n=84) |
| ppo_dr_s1 | 1.0 | 92% | 8% | 0.08 | 27.1 s | 27.1 s (n=84) |
| ppo_dr_s2 | 1.0 | 95% | 5% | 0.05 | 27.0 s | 26.9 s (n=84) |
| ppo_dr_r0_rg | 1.0 | 96% | 4% | 0.04 | 27.3 s | 27.0 s (n=84) |
| ppo_dr_s1_rg | 1.0 | 92% | 8% | 0.08 | 27.4 s | 27.2 s (n=84) |
| ppo_dr_s2_rg | 1.0 | 96% | 4% | 0.04 | 27.1 s | 26.9 s (n=84) |
| ES + grip randomization | 1.0 | 100% | 0% | 0.00 | 26.0 s | 25.9 s (n=84) |
| es_dr_s1 | 1.0 | 100% | 0% | 0.00 | 25.9 s | 25.8 s (n=84) |
| es_dr_s2 | 1.0 | 100% | 0% | 0.00 | 26.2 s | 26.0 s (n=84) |
| es_dr_r0_rg | 1.0 | 98% | 2% | 0.02 | 26.8 s | 26.6 s (n=84) |
| es_dr_s1_rg | 1.0 | 99% | 1% | 0.01 | 26.6 s | 26.4 s (n=84) |
| es_dr_s2_rg | 1.0 | 99% | 1% | 0.01 | 26.9 s | 26.7 s (n=84) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 100% | 0% | 0.00 | 31.8 s | 31.6 s (n=84) |
| PPO + grip randomization | 0.8 | 100% | 0% | 0.00 | 27.9 s | 27.8 s (n=98) |
| ppo_dr_s1 | 0.8 | 99% | 1% | 0.01 | 27.9 s | 27.8 s (n=98) |
| ppo_dr_s2 | 0.8 | 99% | 1% | 0.01 | 27.6 s | 27.5 s (n=98) |
| ppo_dr_r0_rg | 0.8 | 100% | 0% | 0.00 | 27.9 s | 27.8 s (n=98) |
| ppo_dr_s1_rg | 0.8 | 99% | 1% | 0.01 | 27.8 s | 27.7 s (n=98) |
| ppo_dr_s2_rg | 0.8 | 100% | 0% | 0.00 | 27.8 s | 27.7 s (n=98) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 27.6 s | 27.5 s (n=98) |
| es_dr_s1 | 0.8 | 100% | 0% | 0.00 | 27.6 s | 27.4 s (n=98) |
| es_dr_s2 | 0.8 | 100% | 0% | 0.00 | 27.7 s | 27.5 s (n=98) |
| es_dr_r0_rg | 0.8 | 100% | 0% | 0.00 | 27.9 s | 27.7 s (n=98) |
| es_dr_s1_rg | 0.8 | 100% | 0% | 0.00 | 27.8 s | 27.6 s (n=98) |
| es_dr_s2_rg | 0.8 | 100% | 0% | 0.00 | 28.0 s | 27.9 s (n=98) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 100% | 0% | 0.00 | 31.8 s | 31.6 s (n=98) |
| PPO + grip randomization | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| ppo_dr_s1 | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| ppo_dr_s2 | 0.7 | 100% | 0% | 0.00 | 28.6 s | 28.6 s (n=100) |
| ppo_dr_r0_rg | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| ppo_dr_s1_rg | 0.7 | 100% | 0% | 0.00 | 28.7 s | 28.7 s (n=100) |
| ppo_dr_s2_rg | 0.7 | 100% | 0% | 0.00 | 28.7 s | 28.7 s (n=100) |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| es_dr_s1 | 0.7 | 100% | 0% | 0.00 | 29.0 s | 29.0 s (n=100) |
| es_dr_s2 | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| es_dr_r0_rg | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| es_dr_s1_rg | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| es_dr_s2_rg | 0.7 | 100% | 0% | 0.00 | 29.1 s | 29.1 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.7 | 100% | 0% | 0.00 | 31.9 s | 31.9 s (n=100) |
| PPO + grip randomization | 0.6 | 100% | 0% | 0.00 | 30.4 s | 30.2 s (n=84) |
| ppo_dr_s1 | 0.6 | 100% | 0% | 0.00 | 30.4 s | 30.2 s (n=84) |
| ppo_dr_s2 | 0.6 | 100% | 0% | 0.00 | 30.2 s | 29.9 s (n=84) |
| ppo_dr_r0_rg | 0.6 | 100% | 0% | 0.00 | 30.4 s | 30.2 s (n=84) |
| ppo_dr_s1_rg | 0.6 | 100% | 0% | 0.00 | 30.2 s | 30.0 s (n=84) |
| ppo_dr_s2_rg | 0.6 | 100% | 0% | 0.00 | 30.3 s | 30.1 s (n=84) |
| ES + grip randomization | 0.6 | 95% | 5% | 0.05 | 30.5 s | 30.5 s (n=84) |
| es_dr_s1 | 0.6 | 89% | 11% | 0.11 | 30.8 s | 30.7 s (n=84) |
| es_dr_s2 | 0.6 | 93% | 7% | 0.07 | 30.8 s | 30.7 s (n=84) |
| es_dr_r0_rg | 0.6 | 100% | 0% | 0.00 | 30.6 s | 30.4 s (n=84) |
| es_dr_s1_rg | 0.6 | 100% | 0% | 0.00 | 30.6 s | 30.3 s (n=84) |
| es_dr_s2_rg | 0.6 | 100% | 0% | 0.00 | 30.8 s | 30.5 s (n=84) |
| Ray follower (tuned w/ grip rand.) | 0.6 | 100% | 0% | 0.00 | 32.7 s | 32.5 s (n=84) |
| PPO + grip randomization | 0.5 | 93% | 7% | 0.07 | 33.4 s | 32.4 s (n=5) |
| ppo_dr_s1 | 0.5 | 96% | 4% | 0.04 | 33.3 s | 32.4 s (n=5) |
| ppo_dr_s2 | 0.5 | 93% | 7% | 0.07 | 33.3 s | 32.2 s (n=5) |
| ppo_dr_r0_rg | 0.5 | 93% | 7% | 0.07 | 33.4 s | 32.4 s (n=5) |
| ppo_dr_s1_rg | 0.5 | 96% | 4% | 0.04 | 33.1 s | 32.1 s (n=5) |
| ppo_dr_s2_rg | 0.5 | 94% | 6% | 0.06 | 33.3 s | 32.3 s (n=5) |
| ES + grip randomization | 0.5 | 41% | 59% | 0.59 | 32.7 s | 32.6 s (n=5) |
| es_dr_s1 | 0.5 | 21% | 79% | 0.79 | 32.7 s | 32.4 s (n=5) |
| es_dr_s2 | 0.5 | 33% | 67% | 0.67 | 32.2 s | 32.6 s (n=5) |
| es_dr_r0_rg | 0.5 | 90% | 10% | 0.10 | 33.0 s | 32.5 s (n=5) |
| es_dr_s1_rg | 0.5 | 84% | 16% | 0.16 | 33.3 s | 32.5 s (n=5) |
| es_dr_s2_rg | 0.5 | 76% | 24% | 0.24 | 33.4 s | 32.6 s (n=5) |
| Ray follower (tuned w/ grip rand.) | 0.5 | 24% | 76% | 0.76 | 35.8 s | 34.6 s (n=5) |

### hard tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| PPO + grip randomization | 1.0 | 60% | 40% | 0.40 | 32.6 s | 31.3 s (n=13) |
| ppo_dr_s1 | 1.0 | 66% | 34% | 0.34 | 31.4 s | 31.3 s (n=13) |
| ppo_dr_s2 | 1.0 | 70% | 30% | 0.30 | 32.1 s | 31.1 s (n=13) |
| ppo_dr_r0_rg | 1.0 | 60% | 40% | 0.40 | 32.6 s | 31.3 s (n=13) |
| ppo_dr_s1_rg | 1.0 | 66% | 34% | 0.34 | 32.3 s | 31.4 s (n=13) |
| ppo_dr_s2_rg | 1.0 | 76% | 24% | 0.24 | 31.5 s | 31.1 s (n=13) |
| ES + grip randomization | 1.0 | 90% | 10% | 0.10 | 29.5 s | 28.1 s (n=13) |
| es_dr_s1 | 1.0 | 98% | 2% | 0.02 | 28.9 s | 27.7 s (n=13) |
| es_dr_s2 | 1.0 | 98% | 2% | 0.02 | 29.8 s | 28.4 s (n=13) |
| es_dr_r0_rg | 1.0 | 88% | 12% | 0.12 | 31.1 s | 30.6 s (n=13) |
| es_dr_s1_rg | 1.0 | 88% | 12% | 0.12 | 30.9 s | 29.6 s (n=13) |
| es_dr_s2_rg | 1.0 | 96% | 4% | 0.04 | 31.0 s | 29.7 s (n=13) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 90% | 10% | 0.10 | 36.3 s | 34.5 s (n=13) |
| PPO + grip randomization | 0.8 | 66% | 34% | 0.34 | 32.9 s | 30.5 s (n=14) |
| ppo_dr_s1 | 0.8 | 66% | 34% | 0.34 | 31.1 s | 30.6 s (n=14) |
| ppo_dr_s2 | 0.8 | 76% | 24% | 0.24 | 31.7 s | 30.3 s (n=14) |
| ppo_dr_r0_rg | 0.8 | 66% | 34% | 0.34 | 32.9 s | 30.5 s (n=14) |
| ppo_dr_s1_rg | 0.8 | 72% | 28% | 0.28 | 32.1 s | 30.7 s (n=14) |
| ppo_dr_s2_rg | 0.8 | 62% | 38% | 0.38 | 31.6 s | 30.4 s (n=14) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 30.2 s | 28.7 s (n=14) |
| es_dr_s1 | 0.8 | 100% | 0% | 0.00 | 30.0 s | 28.5 s (n=14) |
| es_dr_s2 | 0.8 | 100% | 0% | 0.00 | 30.3 s | 28.7 s (n=14) |
| es_dr_r0_rg | 0.8 | 94% | 6% | 0.06 | 31.8 s | 30.0 s (n=14) |
| es_dr_s1_rg | 0.8 | 98% | 2% | 0.02 | 31.2 s | 29.4 s (n=14) |
| es_dr_s2_rg | 0.8 | 96% | 4% | 0.04 | 31.4 s | 29.6 s (n=14) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 88% | 12% | 0.12 | 35.4 s | 33.6 s (n=14) |
| PPO + grip randomization | 0.7 | 90% | 10% | 0.10 | 32.1 s | 31.8 s (n=28) |
| ppo_dr_s1 | 0.7 | 88% | 12% | 0.12 | 32.1 s | 31.8 s (n=28) |
| ppo_dr_s2 | 0.7 | 78% | 22% | 0.22 | 32.0 s | 31.6 s (n=28) |
| ppo_dr_r0_rg | 0.7 | 90% | 10% | 0.10 | 32.1 s | 31.8 s (n=28) |
| ppo_dr_s1_rg | 0.7 | 90% | 10% | 0.10 | 32.5 s | 31.8 s (n=28) |
| ppo_dr_s2_rg | 0.7 | 84% | 16% | 0.16 | 32.0 s | 31.6 s (n=28) |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 31.4 s | 30.9 s (n=28) |
| es_dr_s1 | 0.7 | 100% | 0% | 0.00 | 31.4 s | 30.8 s (n=28) |
| es_dr_s2 | 0.7 | 100% | 0% | 0.00 | 31.4 s | 30.9 s (n=28) |
| es_dr_r0_rg | 0.7 | 100% | 0% | 0.00 | 32.3 s | 31.6 s (n=28) |
| es_dr_s1_rg | 0.7 | 100% | 0% | 0.00 | 31.7 s | 31.1 s (n=28) |
| es_dr_s2_rg | 0.7 | 100% | 0% | 0.00 | 32.0 s | 31.4 s (n=28) |
| Ray follower (tuned w/ grip rand.) | 0.7 | 92% | 8% | 0.08 | 35.2 s | 34.5 s (n=28) |
| PPO + grip randomization | 0.6 | 98% | 2% | 0.02 | 33.5 s | 33.6 s (n=38) |
| ppo_dr_s1 | 0.6 | 98% | 2% | 0.02 | 33.6 s | 33.6 s (n=38) |
| ppo_dr_s2 | 0.6 | 90% | 10% | 0.10 | 33.5 s | 33.3 s (n=38) |
| ppo_dr_r0_rg | 0.6 | 98% | 2% | 0.02 | 33.5 s | 33.6 s (n=38) |
| ppo_dr_s1_rg | 0.6 | 98% | 2% | 0.02 | 33.2 s | 33.5 s (n=38) |
| ppo_dr_s2_rg | 0.6 | 98% | 2% | 0.02 | 33.1 s | 33.5 s (n=38) |
| ES + grip randomization | 0.6 | 100% | 0% | 0.00 | 33.3 s | 33.6 s (n=38) |
| es_dr_s1 | 0.6 | 94% | 6% | 0.06 | 33.6 s | 33.8 s (n=38) |
| es_dr_s2 | 0.6 | 100% | 0% | 0.00 | 33.4 s | 33.6 s (n=38) |
| es_dr_r0_rg | 0.6 | 100% | 0% | 0.00 | 33.5 s | 33.7 s (n=38) |
| es_dr_s1_rg | 0.6 | 100% | 0% | 0.00 | 33.2 s | 33.5 s (n=38) |
| es_dr_s2_rg | 0.6 | 100% | 0% | 0.00 | 33.5 s | 33.7 s (n=38) |
| Ray follower (tuned w/ grip rand.) | 0.6 | 96% | 4% | 0.04 | 35.7 s | 35.8 s (n=38) |
| PPO + grip randomization | 0.5 | 98% | 2% | 0.02 | 36.2 s | 35.8 s (n=5) |
| ppo_dr_s1 | 0.5 | 100% | 0% | 0.00 | 36.0 s | 35.8 s (n=5) |
| ppo_dr_s2 | 0.5 | 98% | 2% | 0.02 | 35.8 s | 35.3 s (n=5) |
| ppo_dr_r0_rg | 0.5 | 98% | 2% | 0.02 | 36.2 s | 35.8 s (n=5) |
| ppo_dr_s1_rg | 0.5 | 100% | 0% | 0.00 | 35.9 s | 35.4 s (n=5) |
| ppo_dr_s2_rg | 0.5 | 100% | 0% | 0.00 | 35.9 s | 35.6 s (n=5) |
| ES + grip randomization | 0.5 | 66% | 34% | 0.34 | 37.9 s | 36.1 s (n=5) |
| es_dr_s1 | 0.5 | 22% | 78% | 0.78 | 36.7 s | 36.5 s (n=5) |
| es_dr_s2 | 0.5 | 50% | 50% | 0.50 | 37.5 s | 36.4 s (n=5) |
| es_dr_r0_rg | 0.5 | 100% | 0% | 0.00 | 36.2 s | 35.6 s (n=5) |
| es_dr_s1_rg | 0.5 | 96% | 4% | 0.04 | 36.5 s | 35.8 s (n=5) |
| es_dr_s2_rg | 0.5 | 94% | 6% | 0.06 | 36.8 s | 36.1 s (n=5) |
| Ray follower (tuned w/ grip rand.) | 0.5 | 38% | 62% | 0.62 | 36.9 s | 37.3 s (n=5) |

### Driving style / wall-hugging check (strict rules)

|lat| = distance from the centerline as a fraction of the room to the wall (0 = centerline, 1 = touching). near-wall = share of time with |lat| > 0.8. jitter = mean |change of steering command| per step (range of steer is [-1, 1]).

| split | controller | grip | mean abs lat | p95 abs lat | time near wall | steer jitter | crashes into inside (apex) wall |
|---|---|---|---|---|---|---|---|
| heldout | PPO + grip randomization | 1.0 | 0.42 | 0.80 | 6.3% | 0.427 | 100% |
| heldout | ppo_dr_s1 | 1.0 | 0.41 | 0.79 | 5.6% | 0.341 | 100% |
| heldout | ppo_dr_s2 | 1.0 | 0.42 | 0.80 | 6.4% | 0.368 | 100% |
| heldout | ppo_dr_r0_rg | 1.0 | 0.42 | 0.80 | 6.3% | 0.427 | 100% |
| heldout | ppo_dr_s1_rg | 1.0 | 0.42 | 0.80 | 6.2% | 0.364 | 100% |
| heldout | ppo_dr_s2_rg | 1.0 | 0.40 | 0.79 | 5.2% | 0.405 | 100% |
| heldout | ES + grip randomization | 1.0 | 0.32 | 0.66 | 0.7% | 0.166 | no crashes |
| heldout | es_dr_s1 | 1.0 | 0.31 | 0.64 | 0.7% | 0.176 | no crashes |
| heldout | es_dr_s2 | 1.0 | 0.31 | 0.66 | 0.8% | 0.125 | no crashes |
| heldout | es_dr_r0_rg | 1.0 | 0.37 | 0.73 | 2.5% | 0.234 | 100% |
| heldout | es_dr_s1_rg | 1.0 | 0.37 | 0.73 | 2.3% | 0.207 | 100% |
| heldout | es_dr_s2_rg | 1.0 | 0.36 | 0.70 | 1.7% | 0.149 | 100% |
| heldout | Ray follower (tuned w/ grip rand.) | 1.0 | 0.44 | 0.72 | 1.2% | 0.217 | no crashes |
| heldout | PPO + grip randomization | 0.8 | 0.37 | 0.73 | 2.6% | 0.258 | no crashes |
| heldout | ppo_dr_s1 | 0.8 | 0.36 | 0.73 | 2.6% | 0.196 | 100% |
| heldout | ppo_dr_s2 | 0.8 | 0.37 | 0.74 | 3.0% | 0.204 | 100% |
| heldout | ppo_dr_r0_rg | 0.8 | 0.37 | 0.73 | 2.6% | 0.258 | no crashes |
| heldout | ppo_dr_s1_rg | 0.8 | 0.38 | 0.75 | 3.3% | 0.206 | 100% |
| heldout | ppo_dr_s2_rg | 0.8 | 0.35 | 0.72 | 2.5% | 0.233 | no crashes |
| heldout | ES + grip randomization | 0.8 | 0.24 | 0.54 | 0.1% | 0.109 | no crashes |
| heldout | es_dr_s1 | 0.8 | 0.24 | 0.54 | 0.0% | 0.118 | no crashes |
| heldout | es_dr_s2 | 0.8 | 0.22 | 0.55 | 0.1% | 0.080 | no crashes |
| heldout | es_dr_r0_rg | 0.8 | 0.30 | 0.64 | 0.6% | 0.135 | no crashes |
| heldout | es_dr_s1_rg | 0.8 | 0.29 | 0.63 | 0.4% | 0.121 | no crashes |
| heldout | es_dr_s2_rg | 0.8 | 0.28 | 0.60 | 0.2% | 0.091 | no crashes |
| heldout | Ray follower (tuned w/ grip rand.) | 0.8 | 0.40 | 0.70 | 0.9% | 0.147 | no crashes |
| heldout | PPO + grip randomization | 0.6 | 0.26 | 0.56 | 0.2% | 0.149 | no crashes |
| heldout | ppo_dr_s1 | 0.6 | 0.26 | 0.57 | 0.1% | 0.103 | no crashes |
| heldout | ppo_dr_s2 | 0.6 | 0.25 | 0.55 | 0.3% | 0.108 | no crashes |
| heldout | ppo_dr_r0_rg | 0.6 | 0.26 | 0.56 | 0.2% | 0.149 | no crashes |
| heldout | ppo_dr_s1_rg | 0.6 | 0.26 | 0.58 | 0.4% | 0.105 | no crashes |
| heldout | ppo_dr_s2_rg | 0.6 | 0.23 | 0.54 | 0.1% | 0.124 | no crashes |
| heldout | ES + grip randomization | 0.6 | 0.24 | 0.50 | 0.3% | 0.071 | 0% |
| heldout | es_dr_s1 | 0.6 | 0.30 | 0.63 | 1.0% | 0.085 | 0% |
| heldout | es_dr_s2 | 0.6 | 0.24 | 0.53 | 0.5% | 0.059 | 0% |
| heldout | es_dr_r0_rg | 0.6 | 0.22 | 0.50 | 0.0% | 0.080 | no crashes |
| heldout | es_dr_s1_rg | 0.6 | 0.24 | 0.50 | 0.0% | 0.073 | no crashes |
| heldout | es_dr_s2_rg | 0.6 | 0.20 | 0.47 | 0.0% | 0.059 | no crashes |
| heldout | Ray follower (tuned w/ grip rand.) | 0.6 | 0.35 | 0.64 | 0.2% | 0.080 | no crashes |
| hard | PPO + grip randomization | 1.0 | 0.45 | 0.85 | 12.1% | 0.558 | 100% |
| hard | ppo_dr_s1 | 1.0 | 0.46 | 0.85 | 11.0% | 0.448 | 100% |
| hard | ppo_dr_s2 | 1.0 | 0.44 | 0.85 | 11.5% | 0.495 | 100% |
| hard | ppo_dr_r0_rg | 1.0 | 0.45 | 0.85 | 12.1% | 0.558 | 100% |
| hard | ppo_dr_s1_rg | 1.0 | 0.46 | 0.85 | 12.1% | 0.495 | 100% |
| hard | ppo_dr_s2_rg | 1.0 | 0.44 | 0.85 | 12.6% | 0.548 | 100% |
| hard | ES + grip randomization | 1.0 | 0.39 | 0.78 | 5.0% | 0.215 | 100% |
| hard | es_dr_s1 | 1.0 | 0.40 | 0.75 | 3.6% | 0.195 | 100% |
| hard | es_dr_s2 | 1.0 | 0.38 | 0.73 | 2.7% | 0.171 | 100% |
| hard | es_dr_r0_rg | 1.0 | 0.41 | 0.80 | 5.6% | 0.321 | 100% |
| hard | es_dr_s1_rg | 1.0 | 0.43 | 0.79 | 5.9% | 0.262 | 100% |
| hard | es_dr_s2_rg | 1.0 | 0.41 | 0.76 | 3.7% | 0.198 | 100% |
| hard | Ray follower (tuned w/ grip rand.) | 1.0 | 0.49 | 0.79 | 5.4% | 0.293 | 100% |
| hard | PPO + grip randomization | 0.8 | 0.45 | 0.85 | 10.4% | 0.389 | 100% |
| hard | ppo_dr_s1 | 0.8 | 0.44 | 0.83 | 8.4% | 0.306 | 100% |
| hard | ppo_dr_s2 | 0.8 | 0.46 | 0.84 | 10.3% | 0.351 | 100% |
| hard | ppo_dr_r0_rg | 0.8 | 0.45 | 0.85 | 10.4% | 0.389 | 100% |
| hard | ppo_dr_s1_rg | 0.8 | 0.45 | 0.83 | 8.6% | 0.353 | 100% |
| hard | ppo_dr_s2_rg | 0.8 | 0.41 | 0.84 | 9.0% | 0.375 | 100% |
| hard | ES + grip randomization | 0.8 | 0.32 | 0.68 | 1.4% | 0.126 | no crashes |
| hard | es_dr_s1 | 0.8 | 0.32 | 0.68 | 0.9% | 0.118 | no crashes |
| hard | es_dr_s2 | 0.8 | 0.32 | 0.68 | 0.9% | 0.101 | no crashes |
| hard | es_dr_r0_rg | 0.8 | 0.40 | 0.77 | 3.9% | 0.209 | 100% |
| hard | es_dr_s1_rg | 0.8 | 0.39 | 0.77 | 4.1% | 0.158 | 100% |
| hard | es_dr_s2_rg | 0.8 | 0.37 | 0.74 | 3.0% | 0.130 | 100% |
| hard | Ray follower (tuned w/ grip rand.) | 0.8 | 0.47 | 0.78 | 4.4% | 0.227 | 100% |
| hard | PPO + grip randomization | 0.6 | 0.39 | 0.78 | 4.5% | 0.176 | 100% |
| hard | ppo_dr_s1 | 0.6 | 0.38 | 0.76 | 3.2% | 0.137 | 100% |
| hard | ppo_dr_s2 | 0.6 | 0.39 | 0.78 | 4.8% | 0.151 | 100% |
| hard | ppo_dr_r0_rg | 0.6 | 0.39 | 0.78 | 4.5% | 0.176 | 100% |
| hard | ppo_dr_s1_rg | 0.6 | 0.39 | 0.78 | 4.4% | 0.152 | 100% |
| hard | ppo_dr_s2_rg | 0.6 | 0.37 | 0.76 | 3.7% | 0.159 | 100% |
| hard | ES + grip randomization | 0.6 | 0.24 | 0.54 | 0.1% | 0.077 | no crashes |
| hard | es_dr_s1 | 0.6 | 0.30 | 0.63 | 0.6% | 0.081 | 0% |
| hard | es_dr_s2 | 0.6 | 0.22 | 0.54 | 0.1% | 0.065 | no crashes |
| hard | es_dr_r0_rg | 0.6 | 0.30 | 0.66 | 0.4% | 0.098 | no crashes |
| hard | es_dr_s1_rg | 0.6 | 0.29 | 0.63 | 0.4% | 0.084 | no crashes |
| hard | es_dr_s2_rg | 0.6 | 0.26 | 0.58 | 0.1% | 0.071 | no crashes |
| hard | Ray follower (tuned w/ grip rand.) | 0.6 | 0.39 | 0.73 | 2.5% | 0.124 | 100% |
