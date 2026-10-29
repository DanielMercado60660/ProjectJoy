# Painter vs detectors (N=1000 scenes, K=32 skins each, families=['stripes', 'checker', 'spots', 'fbm', 'cells', 'plaid'])

20:18  up 3 days,  1:42, 2 users, load averages: 17.14 15.30 9.52

Rows: evaluated detector. Columns: mean IoU on random skins (mean over K draws) and on the skin picked by each painter (the detector it minimised). Det = fraction with IoU >= 0.5.


## all

| detector | random | painted by unet0 | painted by unet1 | by stat-dist |
|---|---|---|---|---|
| unet0 | 0.939 | 0.774 | 0.805 | 0.898 |
| unet1 | 0.938 | 0.793 | 0.763 | 0.899 |
| stat_dist (skin vs bg) | 0.845 | 0.768 | 0.786 | (min) |

det rate (IoU>=0.5): unet0: random 1.00 / by unet0 0.93, unet1: random 1.00 / by unet0 0.93

## delta=0.25

| detector | random | painted by unet0 | painted by unet1 | by stat-dist |
|---|---|---|---|---|
| unet0 | 0.900 | 0.668 | 0.712 | 0.862 |
| unet1 | 0.898 | 0.681 | 0.642 | 0.859 |
| stat_dist (skin vs bg) | 0.458 | 0.434 | 0.434 | (min) |

det rate (IoU>=0.5): unet0: random 0.99 / by unet0 0.85, unet1: random 0.99 / by unet0 0.84

## delta=0.5

| detector | random | painted by unet0 | painted by unet1 | by stat-dist |
|---|---|---|---|---|
| unet0 | 0.937 | 0.785 | 0.811 | 0.901 |
| unet1 | 0.934 | 0.806 | 0.774 | 0.908 |
| stat_dist (skin vs bg) | 0.780 | 0.732 | 0.726 | (min) |

det rate (IoU>=0.5): unet0: random 1.00 / by unet0 0.95, unet1: random 0.99 / by unet0 0.96

## delta=0.75

| detector | random | painted by unet0 | painted by unet1 | by stat-dist |
|---|---|---|---|---|
| unet0 | 0.956 | 0.804 | 0.832 | 0.906 |
| unet1 | 0.956 | 0.824 | 0.800 | 0.907 |
| stat_dist (skin vs bg) | 0.979 | 0.838 | 0.873 | (min) |

det rate (IoU>=0.5): unet0: random 1.00 / by unet0 0.94, unet1: random 1.00 / by unet0 0.94

## delta=1.0

| detector | random | painted by unet0 | painted by unet1 | by stat-dist |
|---|---|---|---|---|
| unet0 | 0.965 | 0.840 | 0.866 | 0.923 |
| unet1 | 0.964 | 0.862 | 0.833 | 0.922 |
| stat_dist (skin vs bg) | 1.164 | 1.066 | 1.111 | (min) |

det rate (IoU>=0.5): unet0: random 1.00 / by unet0 0.98, unet1: random 1.00 / by unet0 0.98
