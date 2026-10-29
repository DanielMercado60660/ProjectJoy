Capture rate (pursuer row vs evader column), 500 held-out episodes each, seed 12345; ± is one binomial standard error

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 0.328 ± 0.021 | 0.188 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| greedy | 0.994 ± 0.003 | 0.998 ± 0.002 | 0.034 ± 0.008 | 0.014 ± 0.005 | 0.008 ± 0.004 |
| lead | 0.998 ± 0.002 | 1.000 ± 0.000 | 0.176 ± 0.017 | 0.030 ± 0.008 | 0.024 ± 0.007 |
| learned | 1.000 ± 0.000 | 0.846 ± 0.016 | 0.940 ± 0.011 | 0.074 ± 0.012 | 0.040 ± 0.009 |
| learned-argmax | 0.998 ± 0.002 | 0.788 ± 0.018 | 0.972 ± 0.007 | 0.074 ± 0.012 | 0.058 ± 0.010 |

Mean time to capture (s, captured episodes only; episode limit 15 s)

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 7.54 | 9.09 | - | - | - |
| greedy | 4.16 | 4.11 | 7.33 | 8.66 | 7.74 |
| lead | 3.38 | 4.11 | 8.34 | 4.19 | 3.09 |
| learned | 3.51 | 5.64 | 6.82 | 6.79 | 3.64 |
| learned-argmax | 3.45 | 6.24 | 6.48 | 6.30 | 4.30 |

Energy proxy: pursuer / evader effort per second (thrust-seconds + sprint-seconds, per second; max 2.0)

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 1.32 / 1.30 | 1.32 / 1.03 | 1.30 / 1.00 | 1.30 / 1.00 | 1.30 / 0.96 |
| greedy | 1.41 / 1.35 | 1.26 / 1.19 | 1.49 / 1.42 | 1.22 / 1.35 | 1.16 / 1.24 |
| lead | 1.38 / 1.39 | 1.26 / 1.19 | 1.50 / 1.46 | 1.12 / 1.11 | 1.06 / 0.52 |
| learned | 1.45 / 1.39 | 1.40 / 1.37 | 1.49 / 1.43 | 1.22 / 1.13 | 1.18 / 0.72 |
| learned-argmax | 1.41 / 1.34 | 1.35 / 1.37 | 1.45 / 1.41 | 1.12 / 1.10 | 1.09 / 0.64 |

Pursuer energy per capture (effort units spent across all episodes / captures)

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 50.07 | 97.10 | - | - | - |
| greedy | 6.01 | 5.22 | 645.41 | 1301.44 | 2159.83 |
| lead | 4.69 | 5.18 | 117.79 | 546.46 | 644.00 |
| learned | 5.03 | 11.77 | 11.57 | 234.39 | 424.74 |
| learned-argmax | 4.85 | 13.98 | 10.06 | 214.57 | 266.99 |

Cross-play capture rate, archived pursuer snapshot (row, iteration) vs archived evader snapshot (column), 200 episodes per cell

| P iter \ E iter | 0 | 100 | 200 | 300 | 410 | 510 | 610 | 710 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 0.99 | 0.97 | 0.64 | 0.32 | 0.18 | 0.04 | 0.01 | 0.01 |
| 200 | 0.99 | 0.99 | 0.96 | 0.53 | 0.26 | 0.11 | 0.01 | 0.01 |
| 300 | 1.00 | 0.99 | 0.96 | 0.64 | 0.47 | 0.12 | 0.04 | 0.04 |
| 410 | 1.00 | 1.00 | 0.96 | 0.82 | 0.64 | 0.15 | 0.03 | 0.01 |
| 510 | 1.00 | 1.00 | 0.94 | 0.84 | 0.56 | 0.22 | 0.10 | 0.07 |
| 610 | 1.00 | 1.00 | 0.99 | 0.95 | 0.77 | 0.66 | 0.14 | 0.04 |
| 710 | 1.00 | 1.00 | 0.99 | 0.95 | 0.86 | 0.68 | 0.20 | 0.07 |

Each snapshot vs the scripted suite (200 episodes each): pursuer capture rate, evader escape rate

| iter | P_vs_random | P_vs_flee | P_vs_smartflee | E_escape_vs_random | E_escape_vs_greedy | E_escape_vs_lead |
|---|---|---|---|---|---|---|
| 0 | 0.22 | 0.03 | 0.00 | 0.78 | 0.00 | 0.00 |
| 100 | 0.91 | 0.11 | 0.91 | 0.99 | 0.53 | 0.27 |
| 200 | 0.95 | 0.40 | 0.84 | 0.99 | 0.89 | 0.57 |
| 300 | 1.00 | 0.85 | 0.89 | 1.00 | 0.86 | 0.67 |
| 410 | 0.99 | 0.80 | 0.94 | 0.99 | 0.91 | 0.75 |
| 510 | 0.99 | 0.73 | 0.92 | 1.00 | 0.96 | 0.92 |
| 610 | 1.00 | 0.77 | 0.94 | 1.00 | 0.97 | 0.97 |
| 710 | 1.00 | 0.86 | 0.96 | 1.00 | 0.98 | 0.96 |

Occupancy (fraction of ticks)

```
{
 "learned_selfplay": {
  "E_near_wall": 0.561847677191909,
  "E_in_corner": 0.015638363708426917,
  "P_near_wall": 0.004624060411784748
 },
 "lead_vs_smartflee": {
  "E_near_wall": 0.16721104433411724,
  "E_in_corner": 0.0020250849229161223
 },
 "uniform_area_reference": {
  "near_wall": 0.3193750000000001,
  "in_corner": 0.0625
 }
}
```
