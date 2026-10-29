
### heldout tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 100% | 0% | 0.00 | 25.2 s | 25.1 s (n=98) |
| es_s2 | 1.0 | 100% | 0% | 0.00 | 25.3 s | 25.3 s (n=98) |
| PPO (MLX) | 1.0 | 99% | 1% | 0.01 | 25.2 s | 25.2 s (n=98) |
| ppo_s2 | 1.0 | 99% | 1% | 0.01 | 25.3 s | 25.3 s (n=98) |
| ES (neuroevolution) | 0.8 | 98% | 2% | 0.02 | 27.3 s | 27.3 s (n=98) |
| es_s2 | 0.8 | 100% | 0% | 0.00 | 27.5 s | 27.5 s (n=98) |
| PPO (MLX) | 0.8 | 100% | 0% | 0.00 | 27.3 s | 27.2 s (n=98) |
| ppo_s2 | 0.8 | 100% | 0% | 0.00 | 27.4 s | 27.3 s (n=98) |
| ES (neuroevolution) | 0.7 | 83% | 17% | 0.17 | 28.6 s | 28.2 s (n=68) |
| es_s2 | 0.7 | 75% | 25% | 0.25 | 28.7 s | 28.5 s (n=68) |
| PPO (MLX) | 0.7 | 90% | 10% | 0.10 | 28.7 s | 28.2 s (n=68) |
| ppo_s2 | 0.7 | 93% | 7% | 0.07 | 28.7 s | 28.2 s (n=68) |

### hard tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 80% | 20% | 0.20 | 26.9 s | 27.0 s (n=22) |
| es_s2 | 1.0 | 92% | 8% | 0.08 | 27.4 s | 27.4 s (n=22) |
| PPO (MLX) | 1.0 | 72% | 28% | 0.28 | 27.8 s | 27.2 s (n=22) |
| ppo_s2 | 1.0 | 76% | 24% | 0.24 | 27.3 s | 27.5 s (n=22) |
| ES (neuroevolution) | 0.8 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=47) |
| es_s2 | 0.8 | 100% | 0% | 0.00 | 29.4 s | 29.4 s (n=47) |
| PPO (MLX) | 0.8 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=47) |
| ppo_s2 | 0.8 | 94% | 6% | 0.06 | 29.4 s | 29.4 s (n=47) |
| ES (neuroevolution) | 0.7 | 94% | 6% | 0.06 | 30.9 s | 30.9 s (n=47) |
| es_s2 | 0.7 | 96% | 4% | 0.04 | 31.2 s | 31.2 s (n=47) |
| PPO (MLX) | 0.7 | 96% | 4% | 0.04 | 31.0 s | 31.1 s (n=47) |
| ppo_s2 | 0.7 | 100% | 0% | 0.00 | 31.0 s | 31.1 s (n=47) |

### Driving style / wall-hugging check (strict rules)

|lat| = distance from the centerline as a fraction of the room to the wall (0 = centerline, 1 = touching). near-wall = share of time with |lat| > 0.8. jitter = mean |change of steering command| per step (range of steer is [-1, 1]).

| split | controller | grip | mean abs lat | p95 abs lat | time near wall | steer jitter | crashes into inside (apex) wall |
|---|---|---|---|---|---|---|---|
| heldout | ES (neuroevolution) | 1.0 | 0.29 | 0.63 | 0.7% | 0.095 | no crashes |
| heldout | es_s2 | 1.0 | 0.25 | 0.59 | 0.5% | 0.117 | no crashes |
| heldout | PPO (MLX) | 1.0 | 0.33 | 0.67 | 1.6% | 0.163 | 100% |
| heldout | ppo_s2 | 1.0 | 0.30 | 0.66 | 1.6% | 0.238 | 100% |
| heldout | ES (neuroevolution) | 0.8 | 0.27 | 0.54 | 0.1% | 0.075 | 0% |
| heldout | es_s2 | 0.8 | 0.22 | 0.50 | 0.1% | 0.087 | no crashes |
| heldout | PPO (MLX) | 0.8 | 0.29 | 0.60 | 0.1% | 0.115 | no crashes |
| heldout | ppo_s2 | 0.8 | 0.28 | 0.57 | 0.2% | 0.164 | no crashes |
| hard | ES (neuroevolution) | 1.0 | 0.41 | 0.80 | 5.9% | 0.125 | 100% |
| hard | es_s2 | 1.0 | 0.37 | 0.78 | 4.4% | 0.142 | 100% |
| hard | PPO (MLX) | 1.0 | 0.42 | 0.81 | 6.4% | 0.224 | 100% |
| hard | ppo_s2 | 1.0 | 0.39 | 0.81 | 7.0% | 0.310 | 100% |
| hard | ES (neuroevolution) | 0.8 | 0.32 | 0.68 | 1.2% | 0.083 | no crashes |
| hard | es_s2 | 0.8 | 0.27 | 0.64 | 0.7% | 0.086 | no crashes |
| hard | PPO (MLX) | 0.8 | 0.33 | 0.72 | 1.8% | 0.128 | no crashes |
| hard | ppo_s2 | 0.8 | 0.33 | 0.71 | 2.5% | 0.175 | 100% |
