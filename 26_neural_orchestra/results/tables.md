Data: 7033 train / 879 val / 880 test distinct patterns. Each generator: 250 patterns per style x 8 styles, fixed seeds. Style classifier accuracy on real test patterns: 99.2%.

### Held-out likelihood of the hit grid (nats per 2-bar pattern, lower is better)

| model | val NLL | test NLL |
|---|---|---|
| bernoulli (exact) | 47.66 | 47.52 |
| markov (exact) | 37.44 | 37.69 |
| markov_bar (exact) | 24.90 | 25.07 |
| vae_cond_b1.0 (IWAE bound, K=200) | 19.31 | 19.54 |
| vae_cond_b0.5 (IWAE bound, K=200) | 20.69 | 20.78 |
| vae_cond_b0.25 (IWAE bound, K=200) | 23.19 | 23.06 |
| vae_uncond_b0.5 (IWAE bound, K=200) | 23.47 | 23.61 |
| arvae_cond_fb10 (IWAE bound, K=200) | 17.19 | 17.34 |
| vae_quick_b1.0 (IWAE bound, K=200) | 19.43 | 19.48 |

### Generation: novelty, realism, control, pulse (averaged over styles)

| set | C2ST acc ↓ (0.5 = indistinguishable; mean of 3 seeds) | style acc ↑ | NN Hamming to train | exact copies % | within-style pairwise Hamming | unique % | bar-1 vs bar-2 Hamming | beat/off-beat density | kick on beat | cell-freq L1 vs test ↓ | sparse % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real_test | 0.490 | 0.992 | 4.65 | 0.0 | 29.7 | 100.0 | 3.90 | 2.40 | 0.54 | 0.0000 | 0.0 |
| real_val | 0.511 | 0.998 | 4.56 | 0.0 | 29.4 | 100.0 | 3.88 | 2.37 | 0.54 | 0.0161 | 0.0 |
| bernoulli | 0.988 | 0.989 | 18.87 | 0.0 | 30.7 | 100.0 | 15.81 | 2.30 | 0.54 | 0.0127 | 0.0 |
| markov | 0.983 | 0.986 | 17.27 | 0.0 | 30.7 | 100.0 | 15.75 | 2.29 | 0.54 | 0.0133 | 0.0 |
| markov_bar | 0.969 | 0.976 | 14.35 | 0.0 | 30.2 | 100.0 | 4.89 | 2.31 | 0.54 | 0.0130 | 0.0 |
| copy_train | 0.474 | 1.000 | 0.00 | 100.0 | 30.4 | 86.9 | 3.84 | 2.27 | 0.54 | 0.0137 | 0.0 |
| vae_cond_b1.0:threshold | 0.661 | 0.990 | 4.48 | 8.8 | 26.4 | 97.5 | 3.71 | 2.57 | 0.57 | 0.0184 | 0.0 |
| vae_cond_b1.0:sample | 0.610 | 0.990 | 6.81 | 1.2 | 28.8 | 99.9 | 6.43 | 2.45 | 0.55 | 0.0169 | 0.0 |
| vae_cond_b1.0:sample_T0.7 | 0.618 | 0.991 | 5.83 | 2.7 | 27.6 | 99.8 | 5.40 | 2.50 | 0.56 | 0.0172 | 0.0 |
| vae_cond_b1.0:sample_T0.5 | 0.638 | 0.988 | 5.27 | 4.6 | 27.0 | 99.3 | 4.78 | 2.54 | 0.56 | 0.0178 | 0.0 |
| vae_cond_b0.5:threshold | 0.636 | 0.982 | 5.70 | 4.5 | 26.8 | 99.4 | 4.29 | 2.55 | 0.56 | 0.0205 | 0.0 |
| vae_cond_b0.5:sample | 0.640 | 0.986 | 7.40 | 1.0 | 28.5 | 100.0 | 6.37 | 2.46 | 0.54 | 0.0202 | 0.0 |
| vae_cond_b0.5:sample_T0.7 | 0.632 | 0.986 | 6.68 | 1.8 | 27.6 | 99.9 | 5.56 | 2.50 | 0.55 | 0.0201 | 0.0 |
| vae_cond_b0.5:sample_T0.5 | 0.631 | 0.985 | 6.25 | 2.3 | 27.3 | 99.8 | 5.10 | 2.51 | 0.55 | 0.0203 | 0.0 |
| vae_cond_b0.25:threshold | 0.644 | 0.974 | 6.36 | 3.2 | 27.7 | 99.6 | 4.36 | 2.54 | 0.56 | 0.0221 | 0.0 |
| vae_cond_b0.25:sample | 0.666 | 0.979 | 7.46 | 1.0 | 28.6 | 100.0 | 5.80 | 2.50 | 0.55 | 0.0220 | 0.0 |
| vae_cond_b0.25:sample_T0.7 | 0.659 | 0.979 | 6.98 | 1.7 | 28.2 | 99.8 | 5.26 | 2.52 | 0.55 | 0.0220 | 0.0 |
| vae_cond_b0.25:sample_T0.5 | 0.654 | 0.977 | 6.71 | 2.0 | 27.9 | 99.8 | 4.93 | 2.53 | 0.55 | 0.0218 | 0.0 |
| vae_uncond_b0.5:threshold | 0.676 | — | 7.21 | 2.2 | 31.0 | 99.8 | 4.34 | 2.56 | 0.57 | 0.0291 | 0.0 |
| vae_uncond_b0.5:sample | 0.696 | — | 8.96 | 0.6 | 32.7 | 100.0 | 6.51 | 2.46 | 0.55 | 0.0303 | 0.0 |
| vae_uncond_b0.5:sample_T0.7 | 0.698 | — | 8.24 | 1.0 | 31.9 | 100.0 | 5.64 | 2.50 | 0.56 | 0.0300 | 0.0 |
| vae_uncond_b0.5:sample_T0.5 | 0.697 | — | 7.82 | 1.6 | 31.5 | 100.0 | 5.14 | 2.52 | 0.56 | 0.0298 | 0.0 |
| arvae_cond_fb10:threshold | 0.645 | 0.990 | 3.88 | 16.3 | 28.0 | 94.7 | 4.23 | 2.33 | 0.55 | 0.0162 | 0.0 |
| arvae_cond_fb10:sample | 0.601 | 0.986 | 5.44 | 6.0 | 30.1 | 99.6 | 5.92 | 2.27 | 0.55 | 0.0159 | 0.0 |
| arvae_cond_fb10:sample_T0.7 | 0.589 | 0.989 | 4.65 | 10.2 | 29.0 | 98.5 | 5.11 | 2.30 | 0.55 | 0.0156 | 0.0 |
| arvae_cond_fb10:sample_T0.5 | 0.603 | 0.990 | 4.26 | 13.4 | 28.5 | 96.7 | 4.67 | 2.32 | 0.55 | 0.0156 | 0.0 |
| vae_quick_b1.0:threshold | 0.633 | 0.988 | 4.27 | 10.3 | 26.9 | 96.5 | 3.69 | 2.47 | 0.56 | 0.0163 | 0.0 |
| vae_quick_b1.0:sample | 0.587 | 0.990 | 6.14 | 2.0 | 28.6 | 99.8 | 5.82 | 2.40 | 0.55 | 0.0150 | 0.0 |
| vae_quick_b1.0:sample_T0.7 | 0.581 | 0.990 | 5.35 | 4.3 | 27.8 | 99.6 | 4.98 | 2.43 | 0.55 | 0.0153 | 0.0 |
| vae_quick_b1.0:sample_T0.5 | 0.610 | 0.988 | 4.90 | 6.0 | 27.3 | 98.7 | 4.52 | 2.45 | 0.56 | 0.0157 | 0.0 |

### Per-style pulse check: beat/off-beat onset density ratio and swing (odd minus even 16th offset), real test vs vae_cond_b1.0:sample

| style | real_test ratio | vae_cond_b1.0:sample ratio | markov_bar ratio | bernoulli ratio | real_test swing | vae_cond_b1.0:sample swing | real_test style acc | vae_cond_b1.0:sample style acc | markov_bar style acc | bernoulli style acc |
|---|---|---|---|---|---|---|---|---|---|---|
| rock | 3.98 | 3.74 | 3.53 | 3.54 | 0.003 | -0.001 | 1.00 | 1.00 | 1.00 | 1.00 |
| funk | 1.44 | 1.56 | 1.43 | 1.44 | 0.075 | 0.076 | 0.96 | 0.94 | 0.86 | 0.91 |
| house | 3.23 | 3.28 | 3.16 | 3.13 | 0.041 | 0.049 | 1.00 | 1.00 | 1.00 | 1.00 |
| breakbeat | 2.08 | 1.96 | 2.00 | 1.96 | 0.064 | 0.058 | 0.98 | 0.99 | 0.96 | 1.00 |
| hiphop | 2.67 | 2.73 | 2.54 | 2.55 | 0.259 | 0.246 | 1.00 | 0.99 | 0.98 | 1.00 |
| bossa | 2.01 | 2.16 | 2.03 | 1.99 | 0.000 | 0.003 | 1.00 | 1.00 | 1.00 | 1.00 |
| halftime | 2.05 | 2.28 | 2.06 | 2.09 | 0.035 | 0.042 | 1.00 | 1.00 | 1.00 | 1.00 |
| reggaeton | 1.76 | 1.90 | 1.73 | 1.73 | 0.040 | 0.045 | 1.00 | 1.00 | 1.00 | 1.00 |

### Interpolation (64 test pairs of different styles, 9 latent points each, threshold decode; crossfade = data-space baseline)

| method | mean step (cells) | max/mean step | midpoints NN Hamming to train ↓ | midpoints bar-repeat Hamming ↓ | P(target style) non-decreasing % | endpoint error (cells) | sparse points % |
|---|---|---|---|---|---|---|---|
| vae_cond_b1.0 | 6.15 | 2.87 | 4.82 | 3.96 | 99.8 | 2.49 | 0.0 |
| vae_cond_b0.5 | 6.22 | 2.93 | 5.39 | 4.58 | 99.8 | 1.87 | 0.0 |
| vae_cond_b0.25 | 6.29 | 2.82 | 5.78 | 4.54 | 99.4 | 1.50 | 0.0 |
| arvae_cond_fb10 | 11.24 | 2.42 | 5.65 | 5.46 | 98.6 | 4.98 | 0.0 |
| vae_quick_b1.0 | 6.51 | 2.93 | 4.87 | 4.07 | 99.6 | 2.37 | 0.0 |
| crossfade (data space) | 5.92 | 1.68 | 14.58 | 11.58 | 99.8 | 0.00 | 0.0 |

### Style swap (encode a real test groove, decode it with each other style label)

| model | target-style classifier acc | cells changed | per target |
|---|---|---|---|
| vae_cond_b1.0 | 99.3% | 29.5 | rock 100%, funk 96%, house 100%, breakbeat 99%, hiphop 99%, bossa 100%, halftime 100%, reggaeton 100% |
| vae_cond_b0.5 | 99.0% | 27.7 | rock 100%, funk 93%, house 100%, breakbeat 99%, hiphop 100%, bossa 100%, halftime 100%, reggaeton 100% |
| vae_cond_b0.25 | 97.9% | 26.2 | rock 100%, funk 90%, house 100%, breakbeat 97%, hiphop 97%, bossa 99%, halftime 99%, reggaeton 100% |
| arvae_cond_fb10 | 98.6% | 34.0 | rock 100%, funk 90%, house 100%, breakbeat 99%, hiphop 100%, bossa 100%, halftime 100%, reggaeton 100% |
| vae_quick_b1.0 | 99.5% | 30.6 | rock 100%, funk 96%, house 100%, breakbeat 100%, hiphop 100%, bossa 100%, halftime 100%, reggaeton 100% |

Selected by validation NLL: `checkpoints/arvae_cond_fb10/best`. Demo model: `checkpoints/vae_cond_b1.0/best`, decode mode `sample`.

- vae_cond_b1.0: config {'kind': 'mlp', 'latent': 16, 'hidden': 512, 'cond': True, 'trained_epochs': 100, 'val_neg_elbo': 22.478558953711605}; test reconstruction hit-F1 0.964, exact reconstructions 25.8%

- vae_cond_b0.5: config {'kind': 'mlp', 'latent': 16, 'hidden': 512, 'cond': True, 'trained_epochs': 70, 'val_neg_elbo': 24.583387216474687}; test reconstruction hit-F1 0.973, exact reconstructions 37.2%

- vae_cond_b0.25: config {'kind': 'mlp', 'latent': 16, 'hidden': 512, 'cond': True, 'trained_epochs': 90, 'val_neg_elbo': 28.351038666986632}; test reconstruction hit-F1 0.978, exact reconstructions 50.0%

- vae_uncond_b0.5: config {'kind': 'mlp', 'latent': 16, 'hidden': 512, 'cond': False, 'trained_epochs': 80, 'val_neg_elbo': 27.979851593607794}; test reconstruction hit-F1 0.975, exact reconstructions 42.4%

- arvae_cond_fb10: config {'kind': 'ar', 'latent': 16, 'hidden': 512, 'cond': True, 'rnn': 256, 'trained_epochs': 120, 'val_neg_elbo': 19.67614010061149}; test reconstruction hit-F1 0.924, exact reconstructions 16.8%

- vae_quick_b1.0: config {'kind': 'mlp', 'latent': 16, 'hidden': 512, 'cond': True, 'trained_epochs': 200}; test reconstruction hit-F1 0.964, exact reconstructions 26.8%
