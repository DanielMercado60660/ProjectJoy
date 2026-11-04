Held-out courses: 500 (seed 5777000), play seed 7. Held-out scripted opponents: duck `dodge_jink`, hunter `lead_ca` (never trained against).

### Ducks vs scripted hunters (survival = completed and not hit; hit rate in brackets)

| duck | naive | lead (KF) | lead_ca (held out) | lead_tuned | course time of survivors vs lead |
|---|---|---|---|---|---|
| straight | 0.806 (0.194) | 0.488 (0.512) | 0.476 (0.524) | 0.380 (0.620) | 6.47 s |
| jink | 0.746 (0.246) | 0.456 (0.540) | 0.468 (0.530) | 0.310 (0.688) | 7.35 s |
| dodge | 0.694 (0.124) | 0.430 (0.568) | 0.598 (0.400) | 0.388 (0.594) | 6.94 s |
| dodge_jink | 0.606 (0.136) | 0.414 (0.572) | 0.626 (0.352) | 0.350 (0.626) | 7.83 s |
| duck_s | 0.840 (0.156) | 0.624 (0.376) | 0.496 (0.504) | 0.434 (0.566) | 6.75 s |
| duck_g8 | 0.800 (0.194) | 0.640 (0.358) | 0.502 (0.498) | 0.492 (0.508) | 7.20 s |

### Hunters vs scripted ducks (hit rate; shots per course in brackets)

| hunter | straight | jink | dodge | dodge_jink (held out) |
|---|---|---|---|---|
| naive | 0.194 (2.81) | 0.246 (2.77) | 0.124 (2.85) | 0.136 (2.86) |
| lead | 0.512 (2.43) | 0.540 (2.35) | 0.568 (2.36) | 0.572 (2.31) |
| lead_ca | 0.524 (2.44) | 0.530 (2.42) | 0.400 (2.56) | 0.352 (2.58) |
| lead_tuned | 0.620 (2.31) | 0.688 (2.21) | 0.594 (2.31) | 0.626 (2.27) |
| hunter_s | 0.594 (2.37) | 0.636 (2.30) | 0.548 (2.44) | 0.478 (2.47) |
| hunter_g8 | 0.748 (2.06) | 0.826 (1.95) | 0.686 (2.19) | 0.614 (2.30) |

### Self-play generations (hit rate of the hunter column against the duck row)

Each duck_g is tested against: the scripted lead and lead_ca hunters, the final hunter, and the mean over every learned hunter in the archive. Each hunter_g likewise against scripted and learned ducks.

| gen | duck_g vs lead | vs lead_ca (HO) | vs lead_tuned | vs final hunter | vs all learned hunters (mean) | hunter_g vs straight | vs dodge_jink (HO) | vs final duck | vs all learned ducks (mean) |
|---|---|---|---|---|---|---|---|---|---|
| s | 0.376 | 0.504 | 0.566 | 0.752 | 0.712 | 0.594 | 0.478 | 0.414 | 0.470 |
| g1 | 0.378 | 0.508 | 0.548 | 0.748 | 0.692 | 0.670 | 0.516 | 0.530 | 0.571 |
| g2 | 0.356 | 0.522 | 0.506 | 0.758 | 0.679 | 0.734 | 0.490 | 0.632 | 0.654 |
| g3 | 0.326 | 0.494 | 0.534 | 0.766 | 0.690 | 0.762 | 0.506 | 0.658 | 0.702 |
| g4 | 0.372 | 0.482 | 0.532 | 0.766 | 0.673 | 0.726 | 0.584 | 0.688 | 0.721 |
| g5 | 0.370 | 0.476 | 0.528 | 0.728 | 0.656 | 0.744 | 0.610 | 0.674 | 0.715 |
| g6 | 0.360 | 0.472 | 0.528 | 0.736 | 0.641 | 0.724 | 0.578 | 0.686 | 0.725 |
| g7 | 0.356 | 0.474 | 0.502 | 0.704 | 0.640 | 0.718 | 0.528 | 0.658 | 0.712 |
| g8 | 0.358 | 0.498 | 0.508 | 0.700 | 0.627 | 0.748 | 0.614 | 0.700 | 0.740 |

### Final self-play agents vs every archived opponent

`duck_g8` vs hunters (hit rate): naive 0.194, lead 0.358, lead_ca 0.498, lead_tuned 0.508, hunter_s 0.414, hunter_g1 0.530, hunter_g2 0.632, hunter_g3 0.658, hunter_g4 0.688, hunter_g5 0.674, hunter_g6 0.686, hunter_g7 0.658, hunter_g8 0.700

`hunter_g8` vs ducks (hit rate): straight 0.748, jink 0.826, dodge 0.686, dodge_jink 0.614, duck_s 0.752, duck_g1 0.748, duck_g2 0.758, duck_g3 0.766, duck_g4 0.766, duck_g5 0.728, duck_g6 0.736, duck_g7 0.704, duck_g8 0.700

Worst-case crash+timeout rate per duck over all hunters: straight 0.000, jink 0.008, dodge 0.182, dodge_jink 0.258, duck_s 0.004, duck_g1 0.008, duck_g2 0.020, duck_g3 0.006, duck_g4 0.010, duck_g5 0.016, duck_g6 0.010, duck_g7 0.012, duck_g8 0.006
