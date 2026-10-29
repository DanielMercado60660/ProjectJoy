Capture rate (pursuer row vs evader column), 500 held-out episodes each, seed 12345; ± is one binomial standard error

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 0.328 ± 0.021 | 0.188 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| greedy | 0.994 ± 0.003 | 0.998 ± 0.002 | 0.034 ± 0.008 | 0.008 ± 0.004 | 0.006 ± 0.003 |
| lead | 0.998 ± 0.002 | 1.000 ± 0.000 | 0.176 ± 0.017 | 0.036 ± 0.008 | 0.020 ± 0.006 |
| learned | 1.000 ± 0.000 | 0.718 ± 0.020 | 0.966 ± 0.008 | 0.090 ± 0.013 | 0.110 ± 0.014 |
| learned-argmax | 1.000 ± 0.000 | 0.550 ± 0.022 | 0.982 ± 0.006 | 0.088 ± 0.013 | 0.044 ± 0.009 |

Mean time to capture (s, captured episodes only; episode limit 15 s)

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 7.54 | 9.09 | - | - | - |
| greedy | 4.16 | 4.11 | 7.33 | 9.65 | 6.02 |
| lead | 3.38 | 4.11 | 8.34 | 7.51 | 7.00 |
| learned | 3.65 | 4.69 | 6.80 | 7.91 | 10.45 |
| learned-argmax | 3.76 | 4.54 | 6.21 | 9.23 | 6.38 |

Energy proxy: pursuer / evader effort per second (thrust-seconds + sprint-seconds, per second; max 2.0)

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 1.32 / 1.30 | 1.32 / 1.03 | 1.30 / 1.00 | 1.30 / 1.06 | 1.30 / 1.03 |
| greedy | 1.41 / 1.35 | 1.26 / 1.19 | 1.49 / 1.42 | 1.21 / 1.36 | 1.15 / 1.28 |
| lead | 1.38 / 1.39 | 1.26 / 1.19 | 1.50 / 1.46 | 1.16 / 1.20 | 1.11 / 0.99 |
| learned | 1.32 / 1.39 | 1.31 / 1.33 | 1.39 / 1.42 | 1.19 / 1.14 | 1.14 / 0.81 |
| learned-argmax | 1.31 / 1.37 | 1.32 / 1.38 | 1.37 / 1.39 | 1.11 / 1.10 | 1.06 / 0.70 |

Pursuer energy per capture (effort units spent across all episodes / captures)

| pursuer \ evader | random | flee | smartflee | learned | learned-argmax |
|---|---|---|---|---|---|
| random | 50.07 | 97.10 | 9769.70 | 9727.50 | 9736.70 |
| greedy | 6.01 | 5.22 | 645.41 | 2253.76 | 2864.87 |
| lead | 4.69 | 5.18 | 117.79 | 475.26 | 823.17 |
| learned | 4.80 | 14.13 | 10.23 | 188.99 | 150.46 |
| learned-argmax | 4.88 | 22.63 | 8.90 | 181.03 | 349.25 |

Cross-play capture rate, archived pursuer snapshot (row, iteration) vs archived evader snapshot (column), 200 episodes per cell

| P iter \ E iter | 0 | 100 | 200 | 300 | 410 | 510 | 610 | 710 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 1.00 | 0.98 | 0.67 | 0.28 | 0.16 | 0.04 | 0.05 | 0.03 |
| 200 | 1.00 | 0.98 | 0.93 | 0.51 | 0.26 | 0.09 | 0.04 | 0.04 |
| 300 | 1.00 | 0.97 | 0.97 | 0.61 | 0.33 | 0.14 | 0.10 | 0.05 |
| 410 | 1.00 | 0.99 | 0.99 | 0.92 | 0.57 | 0.12 | 0.04 | 0.01 |
| 510 | 1.00 | 0.99 | 0.98 | 0.87 | 0.68 | 0.18 | 0.08 | 0.04 |
| 610 | 1.00 | 0.98 | 0.99 | 0.93 | 0.86 | 0.42 | 0.13 | 0.07 |
| 710 | 1.00 | 0.99 | 0.99 | 0.98 | 0.95 | 0.53 | 0.23 | 0.10 |

Each snapshot vs the scripted suite (200 episodes each): pursuer capture rate, evader escape rate

| iter | P_vs_random | P_vs_flee | P_vs_smartflee | E_escape_vs_random | E_escape_vs_greedy | E_escape_vs_lead |
|---|---|---|---|---|---|---|
| 0 | 0.26 | 0.02 | 0.00 | 0.76 | 0.00 | 0.00 |
| 100 | 0.98 | 0.42 | 0.92 | 0.98 | 0.77 | 0.31 |
| 200 | 0.99 | 0.59 | 0.95 | 0.98 | 0.70 | 0.39 |
| 300 | 0.99 | 0.78 | 0.89 | 0.99 | 0.76 | 0.57 |
| 410 | 1.00 | 0.89 | 0.90 | 1.00 | 0.95 | 0.80 |
| 510 | 1.00 | 1.00 | 0.94 | 1.00 | 0.97 | 0.93 |
| 610 | 1.00 | 0.93 | 0.96 | 1.00 | 0.99 | 0.92 |
| 710 | 0.99 | 0.73 | 0.98 | 1.00 | 0.97 | 0.99 |

Occupancy (fraction of ticks)

```
{
 "learned_selfplay": {
  "E_near_wall": 0.6307448384729357,
  "E_in_corner": 0.03244388440394921,
  "P_near_wall": 0.005725798634670507
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
