protocol: {'grow_steps': 200, 'regen_steps': 400, 'tau': 'pre + 0.2*(post_damage - pre), per trial', 'trials': 32, 'seed': 123, 'train_damage': ['circle'], 'heldout_damage': ['half', 'stripes', 'blobs'], 'repeated': 'after 200 growth steps, 10 rounds of (random held-out damage, 150 steps)'} 

| damage | model | pre-damage MSE | right after damage | +50 | +100 | +400 | excess error left at +400 | recovered (80%) | recovered & stable | median steps to recover |
|---|---|---|---|---|---|---|---|---|---|---|
| circle (train) | lizard_persist | 0.0020 | 0.0072 | 0.0069 | 0.0081 | 0.0085 ± 0.0075 | 125% | 16% | 12% | 38 |
| circle (train) | lizard_regen | 0.0023 | 0.0074 | 0.0033 | 0.0033 | 0.0033 ± 0.0012 | 20% | 62% | 47% | 36 |
| circle (train) | lizard_persist_s1 | 0.0190 | 0.0203 | 0.0204 | 0.0204 | 0.0204 ± 0.0095 | 107% | 72% | 66% | 0 |
| circle (train) | lizard_regen_s1 | 0.0015 | 0.0067 | 0.0024 | 0.0022 | 0.0023 ± 0.0011 | 15% | 75% | 59% | 32 |
| half (held-out) | lizard_persist | 0.0018 | 0.0140 | 0.0086 | 0.0086 | 0.0094 ± 0.0043 | 62% | 19% | 16% | 52 |
| half (held-out) | lizard_regen | 0.0024 | 0.0143 | 0.0058 | 0.0053 | 0.0053 ± 0.0019 | 25% | 44% | 41% | 50 |
| half (held-out) | lizard_persist_s1 | 0.0216 | 0.0243 | 0.0229 | 0.0230 | 0.0230 ± 0.0073 | 53% | 81% | 81% | 0 |
| half (held-out) | lizard_regen_s1 | 0.0016 | 0.0139 | 0.0041 | 0.0033 | 0.0034 ± 0.0011 | 15% | 75% | 66% | 48 |
| stripes (held-out) | lizard_persist | 0.0017 | 0.0110 | 0.0070 | 0.0084 | 0.0089 ± 0.0041 | 77% | 0% | 0% | n/a |
| stripes (held-out) | lizard_regen | 0.0023 | 0.0114 | 0.0041 | 0.0042 | 0.0042 ± 0.0014 | 21% | 75% | 69% | 27 |
| stripes (held-out) | lizard_persist_s1 | 0.0197 | 0.0223 | 0.0225 | 0.0226 | 0.0227 ± 0.0064 | 114% | 66% | 66% | 0 |
| stripes (held-out) | lizard_regen_s1 | 0.0017 | 0.0110 | 0.0030 | 0.0031 | 0.0032 ± 0.0016 | 16% | 100% | 91% | 22 |
| blobs (held-out) | lizard_persist | 0.0021 | 0.0113 | 0.0087 | 0.0104 | 0.0114 ± 0.0068 | 101% | 16% | 12% | 40 |
| blobs (held-out) | lizard_regen | 0.0023 | 0.0115 | 0.0044 | 0.0042 | 0.0043 ± 0.0014 | 21% | 72% | 69% | 42 |
| blobs (held-out) | lizard_persist_s1 | 0.0191 | 0.0217 | 0.0217 | 0.0222 | 0.0224 ± 0.0077 | 124% | 62% | 62% | 0 |
| blobs (held-out) | lizard_regen_s1 | 0.0015 | 0.0110 | 0.0030 | 0.0027 | 0.0028 ± 0.0009 | 14% | 88% | 81% | 32 |

Only trials where the seed actually grew (pre-damage MSE < half of an empty canvas):

| damage | model | grown | pre | right after damage | +400 | excess error left | recovered (80%) | recovered & stable |
|---|---|---|---|---|---|---|---|---|
| circle | lizard_persist | 100% | 0.0020 | 0.0072 | 0.0085 | 125% | 16% | 12% |
| circle | lizard_regen | 100% | 0.0023 | 0.0074 | 0.0033 | 20% | 62% | 47% |
| circle | lizard_persist_s1 | 38% | 0.0056 | 0.0091 | 0.0093 | 107% | 25% | 8% |
| circle | lizard_regen_s1 | 100% | 0.0015 | 0.0067 | 0.0023 | 15% | 75% | 59% |
| half | lizard_persist | 100% | 0.0018 | 0.0140 | 0.0094 | 62% | 19% | 16% |
| half | lizard_regen | 100% | 0.0024 | 0.0143 | 0.0053 | 25% | 44% | 41% |
| half | lizard_persist_s1 | 25% | 0.0051 | 0.0158 | 0.0108 | 53% | 25% | 25% |
| half | lizard_regen_s1 | 100% | 0.0016 | 0.0139 | 0.0034 | 15% | 75% | 66% |
| stripes | lizard_persist | 100% | 0.0017 | 0.0110 | 0.0089 | 77% | 0% | 0% |
| stripes | lizard_regen | 100% | 0.0023 | 0.0114 | 0.0042 | 21% | 75% | 69% |
| stripes | lizard_persist_s1 | 34% | 0.0057 | 0.0133 | 0.0144 | 114% | 0% | 0% |
| stripes | lizard_regen_s1 | 100% | 0.0017 | 0.0110 | 0.0032 | 16% | 100% | 91% |
| blobs | lizard_persist | 100% | 0.0021 | 0.0113 | 0.0114 | 101% | 16% | 12% |
| blobs | lizard_regen | 100% | 0.0023 | 0.0115 | 0.0043 | 21% | 72% | 69% |
| blobs | lizard_persist_s1 | 38% | 0.0057 | 0.0128 | 0.0145 | 124% | 0% | 0% |
| blobs | lizard_regen_s1 | 100% | 0.0015 | 0.0110 | 0.0028 | 14% | 88% | 81% |

| model | seeds that grew (by step 200) | MSE@100 | @200 | @500 | @1000 | @2000 | worst trial @2000 |
|---|---|---|---|---|---|---|---|
| lizard_persist | 100% | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0034 |
| lizard_regen | 100% | 0.0025 | 0.0023 | 0.0024 | 0.0024 | 0.0024 | 0.0033 |
| lizard_persist_s1 | 28% | 0.0212 | 0.0211 | 0.0211 | 0.0211 | 0.0211 | 0.0271 |
| lizard_regen_s1 | 100% | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0019 | 0.0028 |

| model | MSE after wound 1 | 3 | 5 | 10 | worst trial after 10 | trials < 0.01 after 10 |
|---|---|---|---|---|---|---|
| lizard_persist | 0.0084 | 0.0253 | 0.0598 | 0.6087 | 4.8838 | 0% |
| lizard_regen | 0.0044 | 0.0078 | 0.0104 | 0.0152 | 0.0229 | 6% |
| lizard_persist_s1 | 0.0252 | 0.0706 | 0.1313 | 10.7344 | 310.2599 | 0% |
| lizard_regen_s1 | 0.0028 | 0.0057 | 0.0082 | 0.0133 | 0.0241 | 31% |
