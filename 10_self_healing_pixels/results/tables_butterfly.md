protocol: {'grow_steps': 200, 'regen_steps': 400, 'tau': 'pre + 0.2*(post_damage - pre), per trial', 'trials': 16, 'seed': 123, 'train_damage': ['circle'], 'heldout_damage': ['half', 'stripes', 'blobs'], 'repeated': 'after 200 growth steps, 10 rounds of (random held-out damage, 150 steps)'} 

| damage | model | pre-damage MSE | right after damage | +50 | +100 | +400 | excess error left at +400 | recovered (80%) | recovered & stable | median steps to recover |
|---|---|---|---|---|---|---|---|---|---|---|
| circle (train) | butterfly_regen | 0.0271 | 0.0381 | 0.0281 | 0.0277 | 0.0277 ± 0.0329 | 5% | 94% | 94% | 16 |
| half (held-out) | butterfly_regen | 0.0268 | 0.0515 | 0.0333 | 0.0301 | 0.0301 ± 0.0315 | 13% | 88% | 88% | 42 |
| stripes (held-out) | butterfly_regen | 0.0317 | 0.0481 | 0.0336 | 0.0337 | 0.0337 ± 0.0334 | 12% | 100% | 94% | 16 |
| blobs (held-out) | butterfly_regen | 0.0177 | 0.0381 | 0.0195 | 0.0190 | 0.0190 ± 0.0276 | 6% | 100% | 100% | 19 |

Only trials where the seed actually grew (pre-damage MSE < half of an empty canvas):

| damage | model | grown | pre | right after damage | +400 | excess error left | recovered (80%) | recovered & stable |
|---|---|---|---|---|---|---|---|---|
| circle | butterfly_regen | 69% | 0.0047 | 0.0207 | 0.0056 | 5% | 91% | 91% |
| half | butterfly_regen | 69% | 0.0042 | 0.0402 | 0.0090 | 13% | 82% | 82% |
| stripes | butterfly_regen | 62% | 0.0049 | 0.0312 | 0.0080 | 12% | 100% | 90% |
| blobs | butterfly_regen | 81% | 0.0041 | 0.0293 | 0.0057 | 6% | 100% | 100% |

| model | seeds that grew (by step 200) | MSE@100 | @200 | @500 | @1000 | @2000 | worst trial @2000 |
|---|---|---|---|---|---|---|---|
| butterfly_regen | 100% | 0.0046 | 0.0046 | 0.0047 | 0.0047 | 0.0047 | 0.0074 |

| model | MSE after wound 1 | 3 | 5 | 10 | worst trial after 10 | trials < 0.01 after 10 |
|---|---|---|---|---|---|---|
| butterfly_regen | 0.0241 | 0.0288 | 0.0300 | 0.0381 | 0.0765 | 0% |
