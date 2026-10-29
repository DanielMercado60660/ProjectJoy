protocol: {'grow_steps': 200, 'regen_steps': 400, 'tau': 'pre + 0.2*(post_damage - pre), per trial', 'trials': 16, 'seed': 123, 'train_damage': ['circle'], 'heldout_damage': ['half', 'stripes', 'blobs'], 'repeated': 'after 200 growth steps, 10 rounds of (random held-out damage, 150 steps)'} 

| damage | model | pre-damage MSE | right after damage | +50 | +100 | +400 | excess error left at +400 | recovered (80%) | recovered & stable | median steps to recover |
|---|---|---|---|---|---|---|---|---|---|---|
| circle (train) | octopus_regen | 0.0029 | 0.0172 | 0.0041 | 0.0037 | 0.0039 ± 0.0011 | 7% | 100% | 100% | 28 |
| half (held-out) | octopus_regen | 0.0028 | 0.0421 | 0.0074 | 0.0059 | 0.0062 ± 0.0023 | 9% | 94% | 94% | 42 |
| stripes (held-out) | octopus_regen | 0.0028 | 0.0327 | 0.0064 | 0.0055 | 0.0066 ± 0.0028 | 12% | 100% | 94% | 34 |
| blobs (held-out) | octopus_regen | 0.0031 | 0.0333 | 0.0053 | 0.0048 | 0.0050 ± 0.0015 | 6% | 100% | 100% | 29 |

Only trials where the seed actually grew (pre-damage MSE < half of an empty canvas):

| damage | model | grown | pre | right after damage | +400 | excess error left | recovered (80%) | recovered & stable |
|---|---|---|---|---|---|---|---|---|
| circle | octopus_regen | 100% | 0.0029 | 0.0172 | 0.0039 | 7% | 100% | 100% |
| half | octopus_regen | 100% | 0.0028 | 0.0421 | 0.0062 | 9% | 94% | 94% |
| stripes | octopus_regen | 100% | 0.0028 | 0.0327 | 0.0066 | 12% | 100% | 94% |
| blobs | octopus_regen | 100% | 0.0031 | 0.0333 | 0.0050 | 6% | 100% | 100% |

| model | seeds that grew (by step 200) | MSE@100 | @200 | @500 | @1000 | @2000 | worst trial @2000 |
|---|---|---|---|---|---|---|---|
| octopus_regen | 100% | 0.0032 | 0.0029 | 0.0032 | 0.0033 | 0.0034 | 0.0046 |

| model | MSE after wound 1 | 3 | 5 | 10 | worst trial after 10 | trials < 0.01 after 10 |
|---|---|---|---|---|---|---|
| octopus_regen | 0.0067 | 0.0117 | 0.0168 | 0.0281 | 0.0598 | 6% |
