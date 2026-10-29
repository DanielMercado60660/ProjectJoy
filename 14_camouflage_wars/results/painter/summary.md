# Painter vs detectors (N=400 scenes, K=64 skins each, families=['marble', 'ripples', 'streaks'])

20:10  up 3 days,  1:35, 2 users, load averages: 14.33 7.14 4.58

Rows: evaluated detector. Columns: mean IoU on random skins (mean over K draws) and on the skin picked by each painter (the detector it minimised). Det = fraction with IoU >= 0.5.


## all

| detector | random | painted by unet0 | by unet1 | by base | by base_er | by stat-dist |
|---|---|---|---|---|---|---|
| unet0 | 0.875 | 0.603 | 0.686 | 0.822 | 0.828 | 0.842 |
| unet1 | 0.886 | 0.692 | 0.625 | 0.841 | 0.843 | 0.857 |
| base | 0.401 | 0.346 | 0.351 | 0.145 | 0.173 | 0.247 |
| base_er | 0.575 | 0.507 | 0.500 | 0.283 | 0.242 | 0.416 |
| stat_dist (skin vs bg) | 1.061 | 0.976 | 1.018 | 0.737 | 0.765 | (min) |

det rate (IoU>=0.5): unet0: random 0.97 / by unet0 0.73, unet1: random 0.97 / by unet0 0.83, base: random 0.42 / by unet0 0.30, base_er: random 0.64 / by unet0 0.50

## delta=0.25

| detector | random | painted by unet0 | by unet1 | by base | by base_er | by stat-dist |
|---|---|---|---|---|---|---|
| unet0 | 0.837 | 0.446 | 0.542 | 0.763 | 0.778 | 0.800 |
| unet1 | 0.847 | 0.507 | 0.437 | 0.778 | 0.792 | 0.822 |
| base | 0.256 | 0.183 | 0.192 | 0.062 | 0.080 | 0.137 |
| base_er | 0.413 | 0.311 | 0.305 | 0.124 | 0.086 | 0.217 |
| stat_dist (skin vs bg) | 0.598 | 0.542 | 0.535 | 0.403 | 0.417 | (min) |

det rate (IoU>=0.5): unet0: random 0.97 / by unet0 0.49, unet1: random 0.97 / by unet0 0.60, base: random 0.16 / by unet0 0.05, base_er: random 0.36 / by unet0 0.19

## delta=0.5

| detector | random | painted by unet0 | by unet1 | by base | by base_er | by stat-dist |
|---|---|---|---|---|---|---|
| unet0 | 0.869 | 0.600 | 0.699 | 0.821 | 0.827 | 0.837 |
| unet1 | 0.881 | 0.709 | 0.642 | 0.845 | 0.841 | 0.856 |
| base | 0.390 | 0.325 | 0.336 | 0.115 | 0.137 | 0.219 |
| base_er | 0.564 | 0.478 | 0.469 | 0.226 | 0.191 | 0.363 |
| stat_dist (skin vs bg) | 0.965 | 0.868 | 0.902 | 0.653 | 0.624 | (min) |

det rate (IoU>=0.5): unet0: random 0.96 / by unet0 0.75, unet1: random 0.97 / by unet0 0.90, base: random 0.39 / by unet0 0.24, base_er: random 0.60 / by unet0 0.44

## delta=0.75

| detector | random | painted by unet0 | by unet1 | by base | by base_er | by stat-dist |
|---|---|---|---|---|---|---|
| unet0 | 0.897 | 0.683 | 0.750 | 0.855 | 0.849 | 0.877 |
| unet1 | 0.909 | 0.767 | 0.708 | 0.883 | 0.875 | 0.884 |
| base | 0.436 | 0.389 | 0.403 | 0.165 | 0.192 | 0.275 |
| base_er | 0.635 | 0.587 | 0.593 | 0.341 | 0.293 | 0.503 |
| stat_dist (skin vs bg) | 1.218 | 1.130 | 1.175 | 0.874 | 0.917 | (min) |

det rate (IoU>=0.5): unet0: random 0.99 / by unet0 0.84, unet1: random 0.99 / by unet0 0.93, base: random 0.46 / by unet0 0.34, base_er: random 0.74 / by unet0 0.63

## delta=1.0

| detector | random | painted by unet0 | by unet1 | by base | by base_er | by stat-dist |
|---|---|---|---|---|---|---|
| unet0 | 0.897 | 0.683 | 0.752 | 0.849 | 0.856 | 0.855 |
| unet1 | 0.907 | 0.784 | 0.714 | 0.857 | 0.862 | 0.867 |
| base | 0.524 | 0.488 | 0.474 | 0.238 | 0.283 | 0.356 |
| base_er | 0.688 | 0.651 | 0.634 | 0.441 | 0.397 | 0.580 |
| stat_dist (skin vs bg) | 1.462 | 1.362 | 1.461 | 1.019 | 1.103 | (min) |

det rate (IoU>=0.5): unet0: random 0.97 / by unet0 0.85, unet1: random 0.97 / by unet0 0.91, base: random 0.66 / by unet0 0.56, base_er: random 0.87 / by unet0 0.75
