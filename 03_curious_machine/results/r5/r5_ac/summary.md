# Round 5 measured summary

Training seeds: [510, 511, 512]; budgets (pretrain, explore, task, expansions): (3145728, 512, 128, 4096).
Mean ± population SD over training seeds. One seed is descriptive only. Classical policies have no fitted training seed; repeated rows are not independent replicas.

| world | method | rooms | mechanisms | TV steps | task success | capped task steps | plan coverage | empty replay success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| plain | lp | 1.333 ± 0.471 | 0.333 ± 0.471 | 0.000 ± 0.000 | 0.083 ± 0.042 | 119.101 ± 6.044 | 0.048 ± 0.067 | 0.054 ± 0.000 |
| plain | rnd_ep | 6.354 ± 1.670 | 3.312 ± 1.692 | 0.000 ± 0.000 | 0.220 ± 0.120 | 117.723 ± 5.607 | 0.167 ± 0.120 | 0.054 ± 0.000 |
| plain | disagree | 4.021 ± 0.059 | 2.646 ± 0.290 | 0.000 ± 0.000 | 0.134 ± 0.015 | 116.607 ± 1.255 | 0.086 ± 0.008 | 0.054 ± 0.000 |
| plain | ngu | 3.938 ± 0.088 | 0.979 ± 0.029 | 0.000 ± 0.000 | 0.054 ± 0.000 | 123.375 ± 0.000 | 0.000 ± 0.000 | 0.054 ± 0.000 |
| plain | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 0.000 ± 0.000 | 0.286 ± 0.000 | 100.759 ± 0.000 | 0.286 ± 0.000 | 0.054 ± 0.000 |
| plain | random | 2.312 ± 0.000 | 1.375 ± 0.000 | 0.000 ± 0.000 | 0.152 ± 0.000 | 117.580 ± 0.000 | 0.134 ± 0.000 | 0.054 ± 0.000 |
| tv | lp | 1.667 ± 0.471 | 0.229 ± 0.324 | 288.625 ± 209.674 | 0.071 ± 0.025 | 120.717 ± 3.759 | 0.033 ± 0.046 | 0.054 ± 0.000 |
| tv | rnd_ep | 2.979 ± 0.029 | 1.896 ± 0.078 | 488.333 ± 4.993 | 0.182 ± 0.004 | 110.366 ± 0.452 | 0.128 ± 0.004 | 0.054 ± 0.000 |
| tv | disagree | 3.354 ± 0.511 | 2.458 ± 0.118 | 58.021 ± 14.195 | 0.149 ± 0.015 | 114.440 ± 2.517 | 0.107 ± 0.032 | 0.054 ± 0.000 |
| tv | ngu | 4.000 ± 0.000 | 1.000 ± 0.000 | 147.521 ± 24.681 | 0.054 ± 0.000 | 123.375 ± 0.000 | 0.000 ± 0.000 | 0.054 ± 0.000 |
| tv | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 65.188 ± 0.000 | 0.286 ± 0.000 | 100.857 ± 0.000 | 0.286 ± 0.000 | 0.054 ± 0.000 |
| tv | random | 2.375 ± 0.000 | 1.375 ± 0.000 | 170.188 ± 0.000 | 0.152 ± 0.000 | 117.536 ± 0.000 | 0.134 ± 0.000 | 0.054 ± 0.000 |

Sources:

- `results/r5/r5_ac/lp_plain_s510.json`
- `results/r5/r5_ac/lp_plain_s511.json`
- `results/r5/r5_ac/lp_plain_s512.json`
- `results/r5/r5_ac/rnd_ep_plain_s510.json`
- `results/r5/r5_ac/rnd_ep_plain_s511.json`
- `results/r5/r5_ac/rnd_ep_plain_s512.json`
- `results/r5/r5_ac/disagree_plain_s510.json`
- `results/r5/r5_ac/disagree_plain_s511.json`
- `results/r5/r5_ac/disagree_plain_s512.json`
- `results/r5/r5_ac/ngu_plain_s510.json`
- `results/r5/r5_ac/ngu_plain_s511.json`
- `results/r5/r5_ac/ngu_plain_s512.json`
- `results/r5/r5_ac/frontier_plain_s510.json`
- `results/r5/r5_ac/frontier_plain_s511.json`
- `results/r5/r5_ac/frontier_plain_s512.json`
- `results/r5/r5_ac/random_plain_s510.json`
- `results/r5/r5_ac/random_plain_s511.json`
- `results/r5/r5_ac/random_plain_s512.json`
- `results/r5/r5_ac/lp_tv_s510.json`
- `results/r5/r5_ac/lp_tv_s511.json`
- `results/r5/r5_ac/lp_tv_s512.json`
- `results/r5/r5_ac/rnd_ep_tv_s510.json`
- `results/r5/r5_ac/rnd_ep_tv_s511.json`
- `results/r5/r5_ac/rnd_ep_tv_s512.json`
- `results/r5/r5_ac/disagree_tv_s510.json`
- `results/r5/r5_ac/disagree_tv_s511.json`
- `results/r5/r5_ac/disagree_tv_s512.json`
- `results/r5/r5_ac/ngu_tv_s510.json`
- `results/r5/r5_ac/ngu_tv_s511.json`
- `results/r5/r5_ac/ngu_tv_s512.json`
- `results/r5/r5_ac/frontier_tv_s510.json`
- `results/r5/r5_ac/frontier_tv_s511.json`
- `results/r5/r5_ac/frontier_tv_s512.json`
- `results/r5/r5_ac/random_tv_s510.json`
- `results/r5/r5_ac/random_tv_s511.json`
- `results/r5/r5_ac/random_tv_s512.json`
