# Round 5 measured summary

Training seeds: [610, 611, 612]; budgets (pretrain, explore, task, expansions): (5996544, 512, 128, 100000).
Mean ± population SD over training seeds. One seed is descriptive only. Classical policies have no fitted training seed; repeated rows are not independent replicas.

| world | method | rooms | mechanisms | TV steps | task success | capped task steps | plan coverage | empty replay success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| plain | lpm | 1.667 ± 0.471 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 | 122.661 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 |
| plain | rnd_ep | 7.312 ± 0.368 | 4.979 ± 0.029 | 0.000 ± 0.000 | 0.640 ± 0.004 | 68.705 ± 0.467 | 0.568 ± 0.004 | 0.071 ± 0.000 |
| plain | disagree | 3.292 ± 0.230 | 2.104 ± 0.106 | 0.000 ± 0.000 | 0.202 ± 0.017 | 108.821 ± 1.858 | 0.131 ± 0.017 | 0.071 ± 0.000 |
| plain | ngu | 3.938 ± 0.088 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 | 122.661 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 |
| plain | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 0.000 ± 0.000 | 0.982 ± 0.000 | 46.366 ± 0.000 | 1.000 ± 0.000 | 0.071 ± 0.000 |
| plain | random | 2.312 ± 0.000 | 1.562 ± 0.000 | 0.000 ± 0.000 | 0.161 ± 0.000 | 110.259 ± 0.000 | 0.125 ± 0.000 | 0.071 ± 0.000 |
| tv | lpm | 1.771 ± 0.281 | 0.000 ± 0.000 | 4.292 ± 3.312 | 0.071 ± 0.000 | 122.661 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 |
| tv | rnd_ep | 2.667 ± 0.471 | 1.292 ± 0.913 | 483.521 ± 20.740 | 0.158 ± 0.061 | 113.080 ± 6.786 | 0.086 ± 0.061 | 0.071 ± 0.000 |
| tv | disagree | 3.229 ± 1.064 | 1.750 ± 0.184 | 30.229 ± 22.684 | 0.241 ± 0.098 | 104.423 ± 7.255 | 0.193 ± 0.078 | 0.071 ± 0.000 |
| tv | ngu | 3.979 ± 0.029 | 1.000 ± 0.000 | 148.812 ± 26.363 | 0.071 ± 0.000 | 122.661 ± 0.000 | 0.000 ± 0.000 | 0.071 ± 0.000 |
| tv | frontier | 9.000 ± 0.000 | 8.000 ± 0.000 | 66.812 ± 0.000 | 1.000 ± 0.000 | 45.714 ± 0.000 | 1.000 ± 0.000 | 0.071 ± 0.000 |
| tv | random | 2.250 ± 0.000 | 1.562 ± 0.000 | 115.312 ± 0.000 | 0.161 ± 0.000 | 110.259 ± 0.000 | 0.125 ± 0.000 | 0.071 ± 0.000 |

Sources:

- `results/r5/t2_ac/lpm_plain_s610.json`
- `results/r5/t2_ac/lpm_plain_s611.json`
- `results/r5/t2_ac/lpm_plain_s612.json`
- `results/r5/t2_ac/rnd_ep_plain_s610.json`
- `results/r5/t2_ac/rnd_ep_plain_s611.json`
- `results/r5/t2_ac/rnd_ep_plain_s612.json`
- `results/r5/t2_ac/disagree_plain_s610.json`
- `results/r5/t2_ac/disagree_plain_s611.json`
- `results/r5/t2_ac/disagree_plain_s612.json`
- `results/r5/t2_ac/ngu_plain_s610.json`
- `results/r5/t2_ac/ngu_plain_s611.json`
- `results/r5/t2_ac/ngu_plain_s612.json`
- `results/r5/t2_ac/frontier_plain_s610.json`
- `results/r5/t2_ac/frontier_plain_s611.json`
- `results/r5/t2_ac/frontier_plain_s612.json`
- `results/r5/t2_ac/random_plain_s610.json`
- `results/r5/t2_ac/random_plain_s611.json`
- `results/r5/t2_ac/random_plain_s612.json`
- `results/r5/t2_ac/lpm_tv_s610.json`
- `results/r5/t2_ac/lpm_tv_s611.json`
- `results/r5/t2_ac/lpm_tv_s612.json`
- `results/r5/t2_ac/rnd_ep_tv_s610.json`
- `results/r5/t2_ac/rnd_ep_tv_s611.json`
- `results/r5/t2_ac/rnd_ep_tv_s612.json`
- `results/r5/t2_ac/disagree_tv_s610.json`
- `results/r5/t2_ac/disagree_tv_s611.json`
- `results/r5/t2_ac/disagree_tv_s612.json`
- `results/r5/t2_ac/ngu_tv_s610.json`
- `results/r5/t2_ac/ngu_tv_s611.json`
- `results/r5/t2_ac/ngu_tv_s612.json`
- `results/r5/t2_ac/frontier_tv_s610.json`
- `results/r5/t2_ac/frontier_tv_s611.json`
- `results/r5/t2_ac/frontier_tv_s612.json`
- `results/r5/t2_ac/random_tv_s610.json`
- `results/r5/t2_ac/random_tv_s611.json`
- `results/r5/t2_ac/random_tv_s612.json`
