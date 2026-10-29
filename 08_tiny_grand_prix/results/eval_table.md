
### heldout tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 100% | 0% | 0.00 | 25.2 s | 25.2 s (n=50) |
| PPO (MLX) | 1.0 | 99% | 1% | 0.01 | 25.2 s | 25.2 s (n=50) |
| ES + grip randomization | 1.0 | 100% | 0% | 0.00 | 26.0 s | 25.9 s (n=50) |
| PPO + grip randomization | 1.0 | 96% | 4% | 0.04 | 27.3 s | 26.9 s (n=50) |
| Ray follower (tuned) | 1.0 | 100% | 0% | 0.00 | 25.8 s | 25.9 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 100% | 0% | 0.00 | 31.8 s | 31.5 s (n=50) |
| Ray follower (untuned) | 1.0 | 51% | 49% | 0.49 | 26.2 s | 26.4 s (n=50) |
| Pure pursuit (PRIVILEGED) | 1.0 | 100% | 0% | 0.00 | 29.1 s | 29.3 s (n=50) |
| ES (neuroevolution) | 0.9 | 100% | 0% | 0.00 | 26.1 s | 24.4 s (n=22) |
| PPO (MLX) | 0.9 | 100% | 0% | 0.00 | 26.1 s | 24.4 s (n=22) |
| ES + grip randomization | 0.9 | 100% | 0% | 0.00 | 26.7 s | 24.9 s (n=22) |
| PPO + grip randomization | 0.9 | 97% | 3% | 0.03 | 27.4 s | 25.3 s (n=22) |
| Ray follower (tuned) | 0.9 | 52% | 48% | 0.48 | 26.3 s | 25.1 s (n=22) |
| Ray follower (tuned w/ grip rand.) | 0.9 | 100% | 0% | 0.00 | 31.8 s | 29.0 s (n=22) |
| Ray follower (untuned) | 0.9 | 49% | 51% | 0.51 | 26.8 s | 25.2 s (n=22) |
| Pure pursuit (PRIVILEGED) | 0.9 | 100% | 0% | 0.00 | 29.1 s | 27.6 s (n=22) |
| ES (neuroevolution) | 0.8 | 98% | 2% | 0.02 | 27.3 s | 26.0 s (n=1) |
| PPO (MLX) | 0.8 | 100% | 0% | 0.00 | 27.3 s | 25.8 s (n=1) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 27.6 s | 26.6 s (n=1) |
| PPO + grip randomization | 0.8 | 100% | 0% | 0.00 | 27.9 s | 27.0 s (n=1) |
| Ray follower (tuned) | 0.8 | 3% | 97% | 0.97 | 27.1 s | 26.5 s (n=1) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 100% | 0% | 0.00 | 31.8 s | 30.9 s (n=1) |
| Ray follower (untuned) | 0.8 | 46% | 54% | 0.54 | 28.0 s | 27.4 s (n=1) |
| Pure pursuit (PRIVILEGED) | 0.8 | 94% | 6% | 0.06 | 29.4 s | 29.4 s (n=1) |
| ES (neuroevolution) | 0.7 | 83% | 17% | 0.17 | 28.6 s | n/a |
| PPO (MLX) | 0.7 | 90% | 10% | 0.10 | 28.7 s | n/a |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 28.9 s | n/a |
| PPO + grip randomization | 0.7 | 100% | 0% | 0.00 | 28.9 s | n/a |
| Ray follower (tuned) | 0.7 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.7 | 100% | 0% | 0.00 | 31.9 s | n/a |
| Ray follower (untuned) | 0.7 | 1% | 99% | 0.99 | 28.3 s | n/a |
| Pure pursuit (PRIVILEGED) | 0.7 | 5% | 95% | 0.95 | 28.8 s | n/a |
| ES (neuroevolution) | 0.6 | 28% | 72% | 0.72 | 29.6 s | n/a |
| PPO (MLX) | 0.6 | 43% | 57% | 0.57 | 29.9 s | n/a |
| ES + grip randomization | 0.6 | 95% | 5% | 0.05 | 30.5 s | n/a |
| PPO + grip randomization | 0.6 | 100% | 0% | 0.00 | 30.4 s | n/a |
| Ray follower (tuned) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.6 | 100% | 0% | 0.00 | 32.7 s | n/a |
| Ray follower (untuned) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| ES (neuroevolution) | 0.5 | 3% | 97% | 0.97 | 32.5 s | n/a |
| PPO (MLX) | 0.5 | 2% | 98% | 0.98 | 30.9 s | n/a |
| ES + grip randomization | 0.5 | 41% | 59% | 0.59 | 32.7 s | n/a |
| PPO + grip randomization | 0.5 | 93% | 7% | 0.07 | 33.4 s | n/a |
| Ray follower (tuned) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.5 | 24% | 76% | 0.76 | 35.8 s | n/a |
| Ray follower (untuned) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |

### heldout tracks, respawn rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 100% | 0% | 0.00 | 25.2 s | 25.2 s (n=100) |
| PPO (MLX) | 1.0 | 100% | 1% | 0.01 | 25.2 s | 25.2 s (n=100) |
| ES + grip randomization | 1.0 | 100% | 0% | 0.00 | 26.0 s | 26.0 s (n=100) |
| PPO + grip randomization | 1.0 | 100% | 4% | 0.04 | 27.3 s | 27.3 s (n=100) |
| Ray follower (tuned) | 1.0 | 100% | 0% | 0.00 | 25.8 s | 25.8 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 100% | 0% | 0.00 | 31.8 s | 31.8 s (n=100) |
| Ray follower (untuned) | 1.0 | 100% | 49% | 0.87 | 28.4 s | 28.4 s (n=100) |
| Pure pursuit (PRIVILEGED) | 1.0 | 100% | 0% | 0.00 | 29.1 s | 29.1 s (n=100) |
| ES (neuroevolution) | 0.9 | 100% | 0% | 0.00 | 26.1 s | 26.1 s (n=100) |
| PPO (MLX) | 0.9 | 100% | 0% | 0.00 | 26.1 s | 26.1 s (n=100) |
| ES + grip randomization | 0.9 | 100% | 0% | 0.00 | 26.7 s | 26.7 s (n=100) |
| PPO + grip randomization | 0.9 | 100% | 3% | 0.03 | 27.5 s | 27.5 s (n=100) |
| Ray follower (tuned) | 0.9 | 100% | 48% | 0.61 | 28.0 s | 28.0 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.9 | 100% | 0% | 0.00 | 31.8 s | 31.8 s (n=100) |
| Ray follower (untuned) | 0.9 | 100% | 51% | 0.85 | 28.9 s | 28.9 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.9 | 100% | 0% | 0.00 | 29.1 s | 29.1 s (n=100) |
| ES (neuroevolution) | 0.8 | 100% | 2% | 0.02 | 27.3 s | 27.3 s (n=100) |
| PPO (MLX) | 0.8 | 100% | 0% | 0.00 | 27.3 s | 27.3 s (n=100) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 27.6 s | 27.6 s (n=100) |
| PPO + grip randomization | 0.8 | 100% | 0% | 0.00 | 27.9 s | 27.9 s (n=100) |
| Ray follower (tuned) | 0.8 | 100% | 97% | 2.02 | 31.7 s | 31.7 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 100% | 0% | 0.00 | 31.8 s | 31.8 s (n=100) |
| Ray follower (untuned) | 0.8 | 100% | 54% | 0.89 | 30.0 s | 30.0 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.8 | 100% | 6% | 0.06 | 29.4 s | 29.4 s (n=100) |
| ES (neuroevolution) | 0.7 | 100% | 17% | 0.20 | 29.2 s | 29.2 s (n=100) |
| PPO (MLX) | 0.7 | 100% | 10% | 0.11 | 29.0 s | 29.0 s (n=100) |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| PPO + grip randomization | 0.7 | 100% | 0% | 0.00 | 28.9 s | 28.9 s (n=100) |
| Ray follower (tuned) | 0.7 | 100% | 100% | 3.17 | 34.6 s | 34.6 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.7 | 100% | 0% | 0.00 | 31.9 s | 31.9 s (n=100) |
| Ray follower (untuned) | 0.7 | 100% | 99% | 3.66 | 36.2 s | 36.2 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.7 | 100% | 95% | 1.92 | 33.2 s | 33.2 s (n=100) |
| ES (neuroevolution) | 0.6 | 100% | 72% | 1.23 | 31.9 s | 31.9 s (n=100) |
| PPO (MLX) | 0.6 | 100% | 57% | 0.81 | 31.6 s | 31.6 s (n=100) |
| ES + grip randomization | 0.6 | 100% | 5% | 0.05 | 30.8 s | 30.8 s (n=100) |
| PPO + grip randomization | 0.6 | 100% | 0% | 0.00 | 30.4 s | 30.4 s (n=100) |
| Ray follower (tuned) | 0.6 | 100% | 100% | 3.77 | 36.7 s | 36.7 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.6 | 100% | 0% | 0.00 | 32.7 s | 32.7 s (n=100) |
| Ray follower (untuned) | 0.6 | 100% | 100% | 4.85 | 39.2 s | 39.2 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.6 | 100% | 100% | 4.17 | 37.7 s | 37.7 s (n=100) |
| ES (neuroevolution) | 0.5 | 100% | 97% | 2.73 | 36.3 s | 36.3 s (n=100) |
| PPO (MLX) | 0.5 | 100% | 98% | 2.81 | 36.9 s | 36.9 s (n=100) |
| ES + grip randomization | 0.5 | 100% | 59% | 0.95 | 34.9 s | 34.9 s (n=100) |
| PPO + grip randomization | 0.5 | 100% | 7% | 0.07 | 33.5 s | 33.5 s (n=100) |
| Ray follower (tuned) | 0.5 | 100% | 100% | 4.15 | 39.6 s | 39.6 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.5 | 100% | 76% | 1.16 | 37.2 s | 37.2 s (n=100) |
| Ray follower (untuned) | 0.5 | 100% | 100% | 5.26 | 43.0 s | 43.0 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.5 | 100% | 100% | 4.88 | 41.4 s | 41.4 s (n=100) |

### hard tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 80% | 20% | 0.20 | 26.9 s | 27.6 s (n=2) |
| PPO (MLX) | 1.0 | 72% | 28% | 0.28 | 27.8 s | 28.2 s (n=2) |
| ES + grip randomization | 1.0 | 90% | 10% | 0.10 | 29.5 s | 29.3 s (n=2) |
| PPO + grip randomization | 1.0 | 60% | 40% | 0.40 | 32.6 s | 32.6 s (n=2) |
| Ray follower (tuned) | 1.0 | 82% | 18% | 0.18 | 27.8 s | 28.4 s (n=2) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 90% | 10% | 0.10 | 36.3 s | 35.8 s (n=2) |
| Ray follower (untuned) | 1.0 | 10% | 90% | 0.90 | 25.4 s | 29.4 s (n=2) |
| Pure pursuit (PRIVILEGED) | 1.0 | 100% | 0% | 0.00 | 32.1 s | 33.0 s (n=2) |
| ES (neuroevolution) | 0.9 | 92% | 8% | 0.08 | 27.7 s | 26.6 s (n=3) |
| PPO (MLX) | 0.9 | 88% | 12% | 0.12 | 28.0 s | 26.7 s (n=3) |
| ES + grip randomization | 0.9 | 96% | 4% | 0.04 | 29.5 s | 27.8 s (n=3) |
| PPO + grip randomization | 0.9 | 66% | 34% | 0.34 | 32.1 s | 30.6 s (n=3) |
| Ray follower (tuned) | 0.9 | 64% | 36% | 0.36 | 28.4 s | 27.6 s (n=3) |
| Ray follower (tuned w/ grip rand.) | 0.9 | 92% | 8% | 0.08 | 35.6 s | 33.2 s (n=3) |
| Ray follower (untuned) | 0.9 | 16% | 84% | 0.84 | 30.9 s | 28.0 s (n=3) |
| Pure pursuit (PRIVILEGED) | 0.9 | 100% | 0% | 0.00 | 32.1 s | 31.3 s (n=3) |
| ES (neuroevolution) | 0.8 | 100% | 0% | 0.00 | 29.2 s | n/a |
| PPO (MLX) | 0.8 | 100% | 0% | 0.00 | 29.2 s | n/a |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 30.2 s | n/a |
| PPO + grip randomization | 0.8 | 66% | 34% | 0.34 | 32.9 s | n/a |
| Ray follower (tuned) | 0.8 | 10% | 90% | 0.90 | 27.1 s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.8 | 88% | 12% | 0.12 | 35.4 s | n/a |
| Ray follower (untuned) | 0.8 | 14% | 86% | 0.86 | 31.4 s | n/a |
| Pure pursuit (PRIVILEGED) | 0.8 | 80% | 20% | 0.20 | 32.5 s | n/a |
| ES (neuroevolution) | 0.7 | 94% | 6% | 0.06 | 30.9 s | n/a |
| PPO (MLX) | 0.7 | 96% | 4% | 0.04 | 31.0 s | n/a |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 31.4 s | n/a |
| PPO + grip randomization | 0.7 | 90% | 10% | 0.10 | 32.1 s | n/a |
| Ray follower (tuned) | 0.7 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.7 | 92% | 8% | 0.08 | 35.2 s | n/a |
| Ray follower (untuned) | 0.7 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.7 | 0% | 100% | 1.00 | nan s | n/a |
| ES (neuroevolution) | 0.6 | 62% | 38% | 0.38 | 33.7 s | n/a |
| PPO (MLX) | 0.6 | 72% | 28% | 0.28 | 34.1 s | n/a |
| ES + grip randomization | 0.6 | 100% | 0% | 0.00 | 33.3 s | n/a |
| PPO + grip randomization | 0.6 | 98% | 2% | 0.02 | 33.5 s | n/a |
| Ray follower (tuned) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.6 | 96% | 4% | 0.04 | 35.7 s | n/a |
| Ray follower (untuned) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| ES (neuroevolution) | 0.5 | 6% | 94% | 0.94 | 43.4 s | n/a |
| PPO (MLX) | 0.5 | 6% | 94% | 0.94 | 40.1 s | n/a |
| ES + grip randomization | 0.5 | 66% | 34% | 0.34 | 37.9 s | n/a |
| PPO + grip randomization | 0.5 | 98% | 2% | 0.02 | 36.2 s | n/a |
| Ray follower (tuned) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.5 | 38% | 62% | 0.62 | 36.9 s | n/a |
| Ray follower (untuned) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |

### hard tracks, respawn rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 100% | 20% | 0.26 | 27.7 s | 27.7 s (n=50) |
| PPO (MLX) | 1.0 | 100% | 28% | 0.34 | 28.1 s | 28.1 s (n=50) |
| ES + grip randomization | 1.0 | 100% | 10% | 0.10 | 29.3 s | 29.3 s (n=50) |
| PPO + grip randomization | 1.0 | 100% | 40% | 0.50 | 33.2 s | 33.2 s (n=50) |
| Ray follower (tuned) | 1.0 | 100% | 18% | 0.18 | 28.3 s | 28.3 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 100% | 10% | 0.10 | 36.0 s | 36.0 s (n=50) |
| Ray follower (untuned) | 1.0 | 100% | 90% | 2.12 | 32.6 s | 32.6 s (n=50) |
| Pure pursuit (PRIVILEGED) | 1.0 | 100% | 0% | 0.00 | 32.1 s | 32.1 s (n=50) |
| ES (neuroevolution) | 0.9 | 100% | 8% | 0.10 | 28.1 s | 28.1 s (n=50) |
| PPO (MLX) | 0.9 | 100% | 12% | 0.16 | 28.3 s | 28.3 s (n=50) |
| ES + grip randomization | 0.9 | 100% | 4% | 0.04 | 29.6 s | 29.6 s (n=50) |
| PPO + grip randomization | 0.9 | 100% | 34% | 0.40 | 33.0 s | 33.0 s (n=50) |
| Ray follower (tuned) | 0.9 | 100% | 36% | 0.42 | 29.7 s | 29.7 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 0.9 | 100% | 8% | 0.08 | 35.8 s | 35.8 s (n=50) |
| Ray follower (untuned) | 0.9 | 100% | 84% | 2.02 | 32.8 s | 32.8 s (n=50) |
| Pure pursuit (PRIVILEGED) | 0.9 | 100% | 0% | 0.00 | 32.1 s | 32.1 s (n=50) |
| ES (neuroevolution) | 0.8 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=50) |
| PPO (MLX) | 0.8 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=50) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 30.2 s | 30.2 s (n=50) |
| PPO + grip randomization | 0.8 | 100% | 34% | 0.38 | 33.0 s | 33.0 s (n=50) |
| Ray follower (tuned) | 0.8 | 100% | 90% | 2.02 | 33.5 s | 33.5 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 100% | 12% | 0.12 | 35.6 s | 35.6 s (n=50) |
| Ray follower (untuned) | 0.8 | 100% | 86% | 1.96 | 33.8 s | 33.8 s (n=50) |
| Pure pursuit (PRIVILEGED) | 0.8 | 100% | 20% | 0.22 | 32.7 s | 32.7 s (n=50) |
| ES (neuroevolution) | 0.7 | 100% | 6% | 0.06 | 30.9 s | 30.9 s (n=50) |
| PPO (MLX) | 0.7 | 100% | 4% | 0.04 | 31.0 s | 31.0 s (n=50) |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 31.4 s | 31.4 s (n=50) |
| PPO + grip randomization | 0.7 | 100% | 10% | 0.12 | 32.7 s | 32.7 s (n=50) |
| Ray follower (tuned) | 0.7 | 100% | 100% | 3.66 | 37.2 s | 37.2 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 0.7 | 100% | 8% | 0.08 | 35.3 s | 35.3 s (n=50) |
| Ray follower (untuned) | 0.7 | 100% | 100% | 4.76 | 39.7 s | 39.7 s (n=50) |
| Pure pursuit (PRIVILEGED) | 0.7 | 100% | 100% | 2.54 | 36.6 s | 36.6 s (n=50) |
| ES (neuroevolution) | 0.6 | 100% | 38% | 0.42 | 33.6 s | 33.6 s (n=50) |
| PPO (MLX) | 0.6 | 100% | 28% | 0.36 | 33.7 s | 33.7 s (n=50) |
| ES + grip randomization | 0.6 | 100% | 0% | 0.00 | 33.3 s | 33.3 s (n=50) |
| PPO + grip randomization | 0.6 | 100% | 2% | 0.02 | 33.5 s | 33.5 s (n=50) |
| Ray follower (tuned) | 0.6 | 100% | 100% | 4.86 | 40.4 s | 40.4 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 0.6 | 100% | 4% | 0.04 | 35.6 s | 35.6 s (n=50) |
| Ray follower (untuned) | 0.6 | 100% | 100% | 6.92 | 44.3 s | 44.3 s (n=50) |
| Pure pursuit (PRIVILEGED) | 0.6 | 100% | 100% | 5.62 | 41.9 s | 41.9 s (n=50) |
| ES (neuroevolution) | 0.5 | 100% | 94% | 1.94 | 38.5 s | 38.5 s (n=50) |
| PPO (MLX) | 0.5 | 100% | 94% | 2.04 | 39.0 s | 39.0 s (n=50) |
| ES + grip randomization | 0.5 | 100% | 34% | 0.38 | 37.2 s | 37.2 s (n=50) |
| PPO + grip randomization | 0.5 | 100% | 2% | 0.02 | 36.2 s | 36.2 s (n=50) |
| Ray follower (tuned) | 0.5 | 100% | 100% | 4.88 | 43.9 s | 43.9 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 0.5 | 100% | 62% | 0.84 | 39.3 s | 39.3 s (n=50) |
| Ray follower (untuned) | 0.5 | 100% | 100% | 7.68 | 49.1 s | 49.1 s (n=50) |
| Pure pursuit (PRIVILEGED) | 0.5 | 100% | 100% | 6.76 | 46.1 s | 46.1 s (n=50) |

### train tracks, strict rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 100% | 0% | 0.00 | 25.6 s | 25.3 s (n=50) |
| PPO (MLX) | 1.0 | 98% | 2% | 0.02 | 25.5 s | 25.2 s (n=50) |
| ES + grip randomization | 1.0 | 100% | 0% | 0.00 | 26.5 s | 25.9 s (n=50) |
| PPO + grip randomization | 1.0 | 93% | 7% | 0.07 | 27.6 s | 26.9 s (n=50) |
| Ray follower (tuned) | 1.0 | 100% | 0% | 0.00 | 26.2 s | 25.8 s (n=50) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 100% | 0% | 0.00 | 32.5 s | 31.7 s (n=50) |
| Ray follower (untuned) | 1.0 | 54% | 46% | 0.46 | 26.6 s | 26.5 s (n=50) |
| Pure pursuit (PRIVILEGED) | 1.0 | 100% | 0% | 0.00 | 29.2 s | 28.9 s (n=50) |
| ES (neuroevolution) | 0.9 | 100% | 0% | 0.00 | 26.5 s | 23.4 s (n=23) |
| PPO (MLX) | 0.9 | 100% | 0% | 0.00 | 26.5 s | 23.4 s (n=23) |
| ES + grip randomization | 0.9 | 100% | 0% | 0.00 | 27.1 s | 23.8 s (n=23) |
| PPO + grip randomization | 0.9 | 99% | 1% | 0.01 | 27.8 s | 24.3 s (n=23) |
| Ray follower (tuned) | 0.9 | 47% | 53% | 0.53 | 24.9 s | 23.9 s (n=23) |
| Ray follower (tuned w/ grip rand.) | 0.9 | 100% | 0% | 0.00 | 32.4 s | 27.7 s (n=23) |
| Ray follower (untuned) | 0.9 | 50% | 50% | 0.50 | 27.0 s | 24.1 s (n=23) |
| Pure pursuit (PRIVILEGED) | 0.9 | 100% | 0% | 0.00 | 29.2 s | 26.1 s (n=23) |
| ES (neuroevolution) | 0.8 | 94% | 6% | 0.06 | 27.5 s | 23.1 s (n=2) |
| PPO (MLX) | 0.8 | 98% | 2% | 0.02 | 27.6 s | 23.1 s (n=2) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 28.0 s | 23.2 s (n=2) |
| PPO + grip randomization | 0.8 | 100% | 0% | 0.00 | 28.4 s | 23.1 s (n=2) |
| Ray follower (tuned) | 0.8 | 7% | 93% | 0.93 | 25.0 s | 22.9 s (n=2) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 100% | 0% | 0.00 | 32.4 s | 26.0 s (n=2) |
| Ray follower (untuned) | 0.8 | 47% | 53% | 0.53 | 28.3 s | 23.0 s (n=2) |
| Pure pursuit (PRIVILEGED) | 0.8 | 94% | 6% | 0.06 | 29.4 s | 24.2 s (n=2) |
| ES (neuroevolution) | 0.7 | 74% | 26% | 0.26 | 28.4 s | n/a |
| PPO (MLX) | 0.7 | 82% | 18% | 0.18 | 29.1 s | n/a |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 29.2 s | n/a |
| PPO + grip randomization | 0.7 | 100% | 0% | 0.00 | 29.3 s | n/a |
| Ray follower (tuned) | 0.7 | 1% | 99% | 0.99 | 26.5 s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.7 | 100% | 0% | 0.00 | 32.6 s | n/a |
| Ray follower (untuned) | 0.7 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.7 | 7% | 93% | 0.93 | 28.2 s | n/a |
| ES (neuroevolution) | 0.6 | 27% | 73% | 0.73 | 28.6 s | n/a |
| PPO (MLX) | 0.6 | 41% | 59% | 0.59 | 29.0 s | n/a |
| ES + grip randomization | 0.6 | 90% | 10% | 0.10 | 30.8 s | n/a |
| PPO + grip randomization | 0.6 | 100% | 0% | 0.00 | 30.8 s | n/a |
| Ray follower (tuned) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.6 | 99% | 1% | 0.01 | 33.3 s | n/a |
| Ray follower (untuned) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.6 | 0% | 100% | 1.00 | nan s | n/a |
| ES (neuroevolution) | 0.5 | 6% | 94% | 0.94 | 26.9 s | n/a |
| PPO (MLX) | 0.5 | 7% | 93% | 0.93 | 28.4 s | n/a |
| ES + grip randomization | 0.5 | 38% | 62% | 0.62 | 31.5 s | n/a |
| PPO + grip randomization | 0.5 | 81% | 19% | 0.19 | 33.5 s | n/a |
| Ray follower (tuned) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |
| Ray follower (tuned w/ grip rand.) | 0.5 | 29% | 71% | 0.71 | 34.1 s | n/a |
| Ray follower (untuned) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |
| Pure pursuit (PRIVILEGED) | 0.5 | 0% | 100% | 1.00 | nan s | n/a |

### train tracks, respawn rules

| controller | grip | completion | crash rate | collisions/lap | mean lap time (finished) | lap time on tracks every controller finished |
|---|---|---|---|---|---|---|
| ES (neuroevolution) | 1.0 | 100% | 0% | 0.00 | 25.6 s | 25.6 s (n=100) |
| PPO (MLX) | 1.0 | 100% | 2% | 0.02 | 25.6 s | 25.6 s (n=100) |
| ES + grip randomization | 1.0 | 100% | 0% | 0.00 | 26.5 s | 26.5 s (n=100) |
| PPO + grip randomization | 1.0 | 100% | 7% | 0.07 | 27.8 s | 27.8 s (n=100) |
| Ray follower (tuned) | 1.0 | 100% | 0% | 0.00 | 26.2 s | 26.2 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 1.0 | 100% | 0% | 0.00 | 32.5 s | 32.5 s (n=100) |
| Ray follower (untuned) | 1.0 | 100% | 46% | 0.77 | 28.6 s | 28.6 s (n=100) |
| Pure pursuit (PRIVILEGED) | 1.0 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=100) |
| ES (neuroevolution) | 0.9 | 100% | 0% | 0.00 | 26.5 s | 26.5 s (n=100) |
| PPO (MLX) | 0.9 | 100% | 0% | 0.00 | 26.5 s | 26.5 s (n=100) |
| ES + grip randomization | 0.9 | 100% | 0% | 0.00 | 27.1 s | 27.1 s (n=100) |
| PPO + grip randomization | 0.9 | 100% | 1% | 0.01 | 27.9 s | 27.9 s (n=100) |
| Ray follower (tuned) | 0.9 | 100% | 53% | 0.68 | 28.6 s | 28.6 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.9 | 100% | 0% | 0.00 | 32.4 s | 32.4 s (n=100) |
| Ray follower (untuned) | 0.9 | 100% | 50% | 0.84 | 29.3 s | 29.3 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.9 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=100) |
| ES (neuroevolution) | 0.8 | 100% | 6% | 0.06 | 27.8 s | 27.8 s (n=100) |
| PPO (MLX) | 0.8 | 100% | 2% | 0.02 | 27.7 s | 27.7 s (n=100) |
| ES + grip randomization | 0.8 | 100% | 0% | 0.00 | 28.0 s | 28.0 s (n=100) |
| PPO + grip randomization | 0.8 | 100% | 0% | 0.00 | 28.4 s | 28.4 s (n=100) |
| Ray follower (tuned) | 0.8 | 100% | 93% | 1.96 | 32.0 s | 32.0 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.8 | 100% | 0% | 0.00 | 32.4 s | 32.4 s (n=100) |
| Ray follower (untuned) | 0.8 | 100% | 53% | 0.96 | 30.5 s | 30.5 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.8 | 100% | 6% | 0.07 | 29.5 s | 29.5 s (n=100) |
| ES (neuroevolution) | 0.7 | 100% | 26% | 0.29 | 29.6 s | 29.6 s (n=100) |
| PPO (MLX) | 0.7 | 100% | 18% | 0.18 | 29.4 s | 29.4 s (n=100) |
| ES + grip randomization | 0.7 | 100% | 0% | 0.00 | 29.2 s | 29.2 s (n=100) |
| PPO + grip randomization | 0.7 | 100% | 0% | 0.00 | 29.3 s | 29.3 s (n=100) |
| Ray follower (tuned) | 0.7 | 100% | 99% | 3.01 | 34.8 s | 34.8 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.7 | 100% | 0% | 0.00 | 32.6 s | 32.6 s (n=100) |
| Ray follower (untuned) | 0.7 | 100% | 100% | 3.54 | 36.5 s | 36.5 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.7 | 100% | 93% | 1.81 | 33.2 s | 33.2 s (n=100) |
| ES (neuroevolution) | 0.6 | 100% | 73% | 1.26 | 32.3 s | 32.3 s (n=100) |
| PPO (MLX) | 0.6 | 100% | 59% | 0.80 | 31.9 s | 31.9 s (n=100) |
| ES + grip randomization | 0.6 | 100% | 10% | 0.10 | 31.2 s | 31.2 s (n=100) |
| PPO + grip randomization | 0.6 | 100% | 0% | 0.00 | 30.8 s | 30.8 s (n=100) |
| Ray follower (tuned) | 0.6 | 100% | 100% | 3.70 | 37.0 s | 37.0 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.6 | 100% | 1% | 0.01 | 33.3 s | 33.3 s (n=100) |
| Ray follower (untuned) | 0.6 | 100% | 100% | 4.93 | 39.8 s | 39.8 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.6 | 100% | 100% | 4.14 | 38.0 s | 38.0 s (n=100) |
| ES (neuroevolution) | 0.5 | 100% | 94% | 2.79 | 36.9 s | 36.9 s (n=100) |
| PPO (MLX) | 0.5 | 100% | 93% | 2.65 | 37.1 s | 37.1 s (n=100) |
| ES + grip randomization | 0.5 | 100% | 62% | 0.94 | 35.2 s | 35.2 s (n=100) |
| PPO + grip randomization | 0.5 | 100% | 19% | 0.20 | 34.0 s | 34.0 s (n=100) |
| Ray follower (tuned) | 0.5 | 100% | 100% | 3.98 | 39.7 s | 39.7 s (n=100) |
| Ray follower (tuned w/ grip rand.) | 0.5 | 100% | 71% | 1.03 | 37.5 s | 37.5 s (n=100) |
| Ray follower (untuned) | 0.5 | 100% | 100% | 5.24 | 43.5 s | 43.5 s (n=100) |
| Pure pursuit (PRIVILEGED) | 0.5 | 100% | 100% | 4.68 | 41.3 s | 41.3 s (n=100) |

### Driving style / wall-hugging check (strict rules)

|lat| = distance from the centerline as a fraction of the room to the wall (0 = centerline, 1 = touching). near-wall = share of time with |lat| > 0.8. jitter = mean |change of steering command| per step (range of steer is [-1, 1]).

| split | controller | grip | mean abs lat | p95 abs lat | time near wall | steer jitter | crashes into inside (apex) wall |
|---|---|---|---|---|---|---|---|
| heldout | ES (neuroevolution) | 1.0 | 0.29 | 0.63 | 0.7% | 0.095 | no crashes |
| heldout | PPO (MLX) | 1.0 | 0.33 | 0.67 | 1.6% | 0.163 | 100% |
| heldout | ES + grip randomization | 1.0 | 0.32 | 0.66 | 0.7% | 0.166 | no crashes |
| heldout | PPO + grip randomization | 1.0 | 0.42 | 0.80 | 6.3% | 0.427 | 100% |
| heldout | Ray follower (tuned) | 1.0 | 0.36 | 0.67 | 0.4% | 0.106 | no crashes |
| heldout | Ray follower (tuned w/ grip rand.) | 1.0 | 0.44 | 0.72 | 1.2% | 0.217 | no crashes |
| heldout | Ray follower (untuned) | 1.0 | 0.44 | 0.84 | 10.6% | 0.132 | 100% |
| heldout | Pure pursuit (PRIVILEGED) | 1.0 | 0.03 | 0.06 | 0.0% | 0.012 | no crashes |
| heldout | ES (neuroevolution) | 0.8 | 0.27 | 0.54 | 0.1% | 0.075 | 0% |
| heldout | PPO (MLX) | 0.8 | 0.29 | 0.60 | 0.1% | 0.115 | no crashes |
| heldout | ES + grip randomization | 0.8 | 0.24 | 0.54 | 0.1% | 0.109 | no crashes |
| heldout | PPO + grip randomization | 0.8 | 0.37 | 0.73 | 2.6% | 0.258 | no crashes |
| heldout | Ray follower (tuned) | 0.8 | 0.38 | 0.80 | 6.9% | 0.100 | 14% |
| heldout | Ray follower (tuned w/ grip rand.) | 0.8 | 0.40 | 0.70 | 0.9% | 0.147 | no crashes |
| heldout | Ray follower (untuned) | 0.8 | 0.48 | 0.89 | 15.0% | 0.097 | 98% |
| heldout | Pure pursuit (PRIVILEGED) | 0.8 | 0.08 | 0.33 | 0.6% | 0.017 | 0% |
| heldout | ES (neuroevolution) | 0.6 | 0.34 | 0.68 | 2.8% | 0.068 | 0% |
| heldout | PPO (MLX) | 0.6 | 0.36 | 0.73 | 4.5% | 0.093 | 7% |
| heldout | ES + grip randomization | 0.6 | 0.24 | 0.50 | 0.3% | 0.071 | 0% |
| heldout | PPO + grip randomization | 0.6 | 0.26 | 0.56 | 0.2% | 0.149 | no crashes |
| heldout | Ray follower (tuned) | 0.6 | 0.34 | 0.76 | 5.8% | 0.099 | 0% |
| heldout | Ray follower (tuned w/ grip rand.) | 0.6 | 0.35 | 0.64 | 0.2% | 0.080 | no crashes |
| heldout | Ray follower (untuned) | 0.6 | 0.40 | 0.85 | 11.9% | 0.096 | 16% |
| heldout | Pure pursuit (PRIVILEGED) | 0.6 | 0.12 | 0.66 | 3.5% | 0.023 | 1% |
| hard | ES (neuroevolution) | 1.0 | 0.41 | 0.80 | 5.9% | 0.125 | 100% |
| hard | PPO (MLX) | 1.0 | 0.42 | 0.81 | 6.4% | 0.224 | 100% |
| hard | ES + grip randomization | 1.0 | 0.39 | 0.78 | 5.0% | 0.215 | 100% |
| hard | PPO + grip randomization | 1.0 | 0.45 | 0.85 | 12.1% | 0.558 | 100% |
| hard | Ray follower (tuned) | 1.0 | 0.38 | 0.74 | 2.8% | 0.179 | 100% |
| hard | Ray follower (tuned w/ grip rand.) | 1.0 | 0.49 | 0.79 | 5.4% | 0.293 | 100% |
| hard | Ray follower (untuned) | 1.0 | 0.41 | 0.89 | 14.4% | 0.208 | 100% |
| hard | Pure pursuit (PRIVILEGED) | 1.0 | 0.04 | 0.08 | 0.0% | 0.020 | no crashes |
| hard | ES (neuroevolution) | 0.8 | 0.32 | 0.68 | 1.2% | 0.083 | no crashes |
| hard | PPO (MLX) | 0.8 | 0.33 | 0.72 | 1.8% | 0.128 | no crashes |
| hard | ES + grip randomization | 0.8 | 0.32 | 0.68 | 1.4% | 0.126 | no crashes |
| hard | PPO + grip randomization | 0.8 | 0.45 | 0.85 | 10.4% | 0.389 | 100% |
| hard | Ray follower (tuned) | 0.8 | 0.38 | 0.80 | 7.7% | 0.137 | 36% |
| hard | Ray follower (tuned w/ grip rand.) | 0.8 | 0.47 | 0.78 | 4.4% | 0.227 | 100% |
| hard | Ray follower (untuned) | 0.8 | 0.43 | 0.87 | 12.6% | 0.164 | 100% |
| hard | Pure pursuit (PRIVILEGED) | 0.8 | 0.11 | 0.45 | 1.3% | 0.028 | 0% |
| hard | ES (neuroevolution) | 0.6 | 0.31 | 0.62 | 1.1% | 0.070 | 5% |
| hard | PPO (MLX) | 0.6 | 0.34 | 0.70 | 2.0% | 0.100 | 14% |
| hard | ES + grip randomization | 0.6 | 0.24 | 0.54 | 0.1% | 0.077 | no crashes |
| hard | PPO + grip randomization | 0.6 | 0.39 | 0.78 | 4.5% | 0.176 | 100% |
| hard | Ray follower (tuned) | 0.6 | 0.35 | 0.76 | 6.6% | 0.136 | 6% |
| hard | Ray follower (tuned w/ grip rand.) | 0.6 | 0.39 | 0.73 | 2.5% | 0.124 | 100% |
| hard | Ray follower (untuned) | 0.6 | 0.37 | 0.84 | 10.8% | 0.153 | 32% |
| hard | Pure pursuit (PRIVILEGED) | 0.6 | 0.12 | 0.66 | 3.6% | 0.030 | 0% |
| train | ES (neuroevolution) | 1.0 | 0.28 | 0.61 | 0.6% | 0.095 | no crashes |
| train | PPO (MLX) | 1.0 | 0.31 | 0.66 | 1.2% | 0.167 | 100% |
| train | ES + grip randomization | 1.0 | 0.31 | 0.64 | 0.8% | 0.168 | no crashes |
| train | PPO + grip randomization | 1.0 | 0.40 | 0.79 | 6.0% | 0.429 | 100% |
| train | Ray follower (tuned) | 1.0 | 0.35 | 0.66 | 0.3% | 0.101 | no crashes |
| train | Ray follower (tuned w/ grip rand.) | 1.0 | 0.43 | 0.70 | 1.2% | 0.216 | no crashes |
| train | Ray follower (untuned) | 1.0 | 0.41 | 0.84 | 9.7% | 0.135 | 100% |
| train | Pure pursuit (PRIVILEGED) | 1.0 | 0.03 | 0.06 | 0.0% | 0.011 | no crashes |
| train | ES (neuroevolution) | 0.8 | 0.26 | 0.53 | 0.1% | 0.074 | 0% |
| train | PPO (MLX) | 0.8 | 0.27 | 0.58 | 0.1% | 0.120 | 0% |
| train | ES + grip randomization | 0.8 | 0.23 | 0.53 | 0.0% | 0.106 | no crashes |
| train | PPO + grip randomization | 0.8 | 0.36 | 0.71 | 2.3% | 0.268 | no crashes |
| train | Ray follower (tuned) | 0.8 | 0.38 | 0.80 | 7.9% | 0.096 | 5% |
| train | Ray follower (tuned w/ grip rand.) | 0.8 | 0.40 | 0.68 | 1.1% | 0.145 | no crashes |
| train | Ray follower (untuned) | 0.8 | 0.47 | 0.89 | 14.6% | 0.109 | 100% |
| train | Pure pursuit (PRIVILEGED) | 0.8 | 0.09 | 0.35 | 1.0% | 0.017 | 0% |
| train | ES (neuroevolution) | 0.6 | 0.32 | 0.67 | 2.5% | 0.071 | 1% |
| train | PPO (MLX) | 0.6 | 0.34 | 0.72 | 4.5% | 0.096 | 2% |
| train | ES + grip randomization | 0.6 | 0.23 | 0.49 | 0.4% | 0.069 | 0% |
| train | PPO + grip randomization | 0.6 | 0.25 | 0.54 | 0.1% | 0.155 | no crashes |
| train | Ray follower (tuned) | 0.6 | 0.33 | 0.77 | 5.5% | 0.096 | 0% |
| train | Ray follower (tuned w/ grip rand.) | 0.6 | 0.34 | 0.62 | 0.1% | 0.078 | 100% |
| train | Ray follower (untuned) | 0.6 | 0.39 | 0.85 | 11.5% | 0.109 | 20% |
| train | Pure pursuit (PRIVILEGED) | 0.6 | 0.12 | 0.67 | 3.5% | 0.022 | 0% |
