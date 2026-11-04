### Astra turns

| tag | thread | calls used / budget | best call | input tok (cached) | output tok (reasoning) |
|---|---|---|---|---|---|
| r6wsp_t0_smoke | 01a1a871-73e2-70b2-b437-708cde70227c | - | - | 98,413 (80,896) | 818 (119) |
| r6wsp_read_a1 | 01a1a872-1a27-7e43-8c2f-b4355cb6a85c | - | - | 136,352 (98,176) | 15,617 (14,475) |
| r6wsp_read_a2 | 01a1a872-1a27-7d02-8551-3c21f411ce46 | - | - | 135,224 (97,408) | 18,827 (17,742) |
| r6wsp_read_a3 | 01a1a878-50b9-7fc3-bf89-cc0d8bb0ac7c | - | - | 134,977 (97,152) | 17,599 (16,517) |
| r6wsp_read_catch | 01a1a896-ff00-70f1-a071-68d5b40b6a2e | - | - | 134,284 (79,488) | 6,293 (5,226) |
| r6wsp_edit_a1 | 01a1a873-4afa-7bd0-bff1-ec9dfce62160 | 12 / 25 | 12 | 1,851,117 (1,791,744) | 39,964 (21,336) |
| r6wsp_edit_a2 | 01a1a873-4afa-79d3-b255-660cbcb5da67 | 10 / 25 | 10 | 996,321 (942,592) | 23,775 (8,520) |
| r6wsp_edit0_a1 | 01a1a887-c1ff-71d3-9492-a4a5d5ed473b | 0 / 0 | - | 565,230 (527,616) | 22,983 (13,246) |
| r6wsp_edit0_a2 | 01a1a887-c1fe-71a3-bde1-10acb1249c5b | 0 / 0 | - | 569,096 (534,400) | 20,386 (10,801) |
| r6wsp_grip_a1 | 01a1a880-6188-7581-b5f4-e5c329883b26 | 8 / 25 | 7 | 906,933 (861,952) | 19,236 (5,422) |
| r6wsp_grip_a2 | 01a1a880-618f-7983-8317-e5614a635416 | 4 / 25 | 4 | 542,752 (508,800) | 14,709 (4,788) |

#### edit_a1

| call | key | improved | strict done | strict lap | soft wall steps/lap | note |
|---|---|---|---|---|---|---|
| 1 | [0.0, 0.232, -999.0] | False | 0.000 | nan | 19.34 | Call 1: unchanged start policy; diagnose units and crash observations |
| 2 | [0.812, 0.9, -27.719] | True | 0.812 | 27.72 | 0.34 | Call 2: preserve steering, reuse weak h1[23]/h2[28] for target 11+20*mean(front rays) m/s; constant-fold previous pedal at 1 |
| 3 | [0.844, 0.918, -27.685] | True | 0.844 | 27.69 | 0.25 | Call 3: lower short-range speed envelope to 7+26*mean(front rays); increase speed-error gain from .15 to .20 based on sharp-turn traces |
| 4 | [1.0, 1.0, -28.003] | True | 1.000 | 28.00 | 0.00 | Call 4: remaining low-speed crashes show inadequate sharp-turn steering; add half-strength +/-35-degree rays to existing +/-18-degree featur |
| 5 | [0.938, 0.973, -26.9] | False | 0.938 | 26.90 | 0.06 | Call 5: corrected steering now follows sharp turns; raise programmed speed target to 9+27*mean(front rays) m/s |
| 6 | [1.0, 1.0, -26.581] | True | 1.000 | 26.58 | 0.00 | Call 6: call5 showed grip-limited sharp-turn crashes; restore call4 short-range targets, add h1[29] clear-road bonus only as forward range g |
| 7 | [1.0, 1.0, -26.322] | True | 1.000 | 26.32 | 0.00 | Call 7: unchanged speed target; pedal output gain 2 to 3 brings max throttle .964 to .995 and reduces speed tracking lag; local brake-loop g |
| 8 | [1.0, 1.0, -26.141] | True | 1.000 | 26.14 | 0.00 | Call 8: forward range 24m implies broad enough bends for higher target; move smooth clear-road bonus midpoint from 30m to 24m, retaining nea |
| 9 | [1.0, 1.0, -26.147] | False | 1.000 | 26.15 | 0.00 | Call 9: closed-form steering-readout correction toward reflection symmetry on handmade vectors; speed circuit unchanged; held-out handmade R |
| 10 | [1.0, 1.0, -25.897] | True | 1.000 | 25.90 | 0.00 | Call 10: symmetry correction halves near-wall time; increase clear-road bonus .45 to .60, mainly adding speed on straights/broad bends while |
| 11 | [1.0, 1.0, -25.803] | True | 1.000 | 25.80 | 0.00 | Call 11: add existing-feature readout for explicit drag-logit surrogate .8*speed_obs-.25, closed-form handmade fit; held-out logit RMSE .023 |
| 12 | [1.0, 1.0, -25.775] | True | 1.000 | 25.78 | 0.00 | Call 12: restore 60 original W1 ray weights; compile call11 steering function and same drag surrogate into existing readout using handmade i |

#### edit_a2

| call | key | improved | strict done | strict lap | soft wall steps/lap | note |
|---|---|---|---|---|---|---|
| 1 | [0.0, 0.232, -999.0] | False | 0.000 | nan | 19.34 | Call 1: unchanged start baseline; inspect crash traces and throttle mechanism |
| 2 | [0.938, 0.968, -28.103] | True | 0.938 | 28.10 | 0.06 | Call 2: change only 33 pedal-readout weights via least squares on handmade sensor lattice; desired pedal logit=.25*(7+27*mean(front +/-6 ray |
| 3 | [0.844, 0.919, -27.219] | False | 0.844 | 27.22 | 0.19 | Call 3: same analytic clearance/speed pedal readout; pin prev_pedal input internally to +1 via W1 row12=0,b1+=old row12 so changing braking  |
| 4 | [1.0, 1.0, -28.203] | True | 1.000 | 28.20 | 0.00 | Call 4: from call2 add steer-logit correction .8*(ray+35-ray-35)+.4*(ray+60-ray-60)+(ray+90-ray-90), fitted by linear algebra on handmade in |
| 5 | [0.938, 0.983, -26.91] | False | 0.938 | 26.91 | 0.06 | Call 5: call4 steering retained; prescribe pedal target speed 9+27*front mean, +2m/s versus conservative call4. For a centered R~50m corner, |
| 6 | [1.0, 1.0, -27.131] | True | 1.000 | 27.13 | 0.00 | Call 6: retain prescribed speed9+27*frontmean and broad steering; fix readout extrapolation by deterministic handmade peripheral openings at |
| 7 | [0.906, 0.967, -27.172] | False | 0.906 | 27.17 | 0.09 | Call 7: exact pedal circuit h1[14]=tanh(.1*(9+27*frontmean-speed)), h2[16]=tanh(h1[14]), pedal=tanh(3*h2[16]); compensate removed h1 mean an |
| 8 | [1.0, 1.0, -26.716] | True | 1.000 | 26.72 | 0.00 | Call 8: exact saturating target speed=32*tanh(1.8*frontmean), pedal=tanh(3*tanh(.15*(target-vhat))); h1[14] clearance, h1[28] near-linear sp |
| 9 | [1.0, 1.0, -26.284] | True | 1.000 | 26.28 | 0.00 | Call 9: preserve low-clearance speed slope 32*1.8=36*1.6=57.6 and identical steering; raise only open-road target via 36*tanh(1.6*f). Call8  |
| 10 | [1.0, 1.0, -26.113] | True | 1.000 | 26.11 | 0.00 | Call 10: final physical-ceiling check, target=40*tanh(1.44*f), keeping low-clearance slope 57.6 and unchanged steering. 40m/s is drag-limite |

#### grip_a1

| call | key | improved | strict done | strict lap | soft wall steps/lap | note |
|---|---|---|---|---|---|---|
| 1 | [1, 0.031, 0.267, -28.3] | False | g0.5 0.031 / g1.0 1.000 | g0.5 28.30 / g1.0 25.39 | - | Call 1: unchanged start policy; baseline and crash observations |
| 2 | [0, 1.0, 1.0, -32.913] | False | g0.5 1.000 / g1.0 0.844 | g0.5 32.91 / g1.0 29.90 | - | Call 2: W1 speed row times sqrt(2), matching v squared to half grip; 32 weights changed |
| 3 | [1, 1.0, 1.0, -33.053] | True | g0.5 1.000 / g1.0 0.969 | g0.5 33.05 / g1.0 29.57 | - | Call 3: sqrt(2) speed row, halve steering logits to reduce dry-grip over-turning; 65 scalars |
| 4 | [0, 0.688, 0.866, -33.214] | False | g0.5 0.688 / g1.0 0.625 | g0.5 33.21 / g1.0 30.22 | - | Call 4: sqrt(2) speed, quarter steering logits; call 3 dry failure commands 0.32-0.67 versus half-grip threshold 0.12-0.17, test reducing ex |
| 5 | [1, 0.938, 0.989, -33.013] | False | g0.5 0.938 / g1.0 0.969 | g0.5 33.01 / g1.0 28.68 | - | Call 5: restore half steering after quarter gain loses low-speed turning; speed scale 1.35 tests smaller slowdown because quadratic drag pre |
| 6 | [1, 0.969, 0.996, -33.406] | False | g0.5 0.969 / g1.0 0.969 | g0.5 33.41 / g1.0 30.54 | - | Call 6: from start, sqrt(2) speed and half steering; inject 0.5*(left90-right90) into left18 and its negative into right18 via two W1 rows,  |
| 7 | [1, 1.0, 1.0, -33.013] | True | g0.5 1.000 / g1.0 1.000 | g0.5 33.01 / g1.0 29.56 | - | Call 7: sqrt(2) speed; closed-form steering readout fit on handmade inputs to gain(v)=0.5+0.5*clip(1-v/15,0,1), restoring low-speed corner a |
| 8 | [1, 1.0, 1.0, -33.047] | False | g0.5 1.000 / g1.0 0.938 | g0.5 33.05 / g1.0 28.32 | - | Call 8: combine 1.35 speed scale with explicit low-speed steering restoration from call 7; call 5 failed at low speed; same handmade closed- |

#### grip_a2

| call | key | improved | strict done | strict lap | soft wall steps/lap | note |
|---|---|---|---|---|---|---|
| 1 | [0, 1.0, 1.0, -32.913] | False | g0.5 1.000 / g1.0 0.844 | g0.5 32.91 / g1.0 29.90 | - | Physical speed rescaling sqrt(2): multiply the 32 W1 speed-input weights; all other parameters unchanged |
| 2 | [0, 0.688, 0.866, -33.214] | False | g0.5 0.688 / g1.0 0.625 | g0.5 33.21 / g1.0 30.22 | - | Retain sqrt(2) speed perception; scale steering logits by 0.25 to reduce high-grip inner-wall cutting at reduced speeds |
| 3 | [1, 1.0, 1.0, -33.209] | True | g0.5 1.000 / g1.0 1.000 | g0.5 33.21 / g1.0 29.98 | - | sqrt(2) speed plus lateral-clearance correction through +/-18-degree ray pathway with gain 0.75; restore original steering head |
| 4 | [1, 1.0, 1.0, -33.194] | True | g0.5 1.000 / g1.0 1.000 | g0.5 33.19 / g1.0 29.06 | - | Keep the wall-clearance pathway; modest relaxation to 1.35 perceived-speed gain after call3 virtually eliminated low-grip near-wall time |

### heldout_grip.json

seeds: {'heldout': [90000, 90199], 'hard': [90000, 90099]}

| policy | heldout_strict_g0.5 | heldout_strict_g1.0 | hard_strict_g0.5 | hard_strict_g1.0 | weights changed | rel. change | cos(change, es_best - start) |
|---|---|---|---|---|---|---|---|
| es_start | 4.5% / 29.00 s | 100.0% / 25.09 s | 3.0% / 30.00 s | 74.0% / 27.93 s | 0.0% | 0.00 | +0.00 |
| astra_grip_a1 | 100.0% / 32.94 s | 94.0% / 29.10 s | 87.0% / 38.05 s | 68.0% / 37.02 s | 4.1% | 0.16 | +0.00 |
| astra_grip_a2 | 99.5% / 33.07 s | 99.5% / 28.95 s | 100.0% / 37.87 s | 99.0% / 35.93 s | 6.1% | 0.36 | +0.00 |
| astra_grip_sqrt2_speed_only | 100.0% / 32.87 s | 81.5% / 29.09 s | 76.0% / 36.35 s | 38.0% / 34.20 s | 2.0% | 0.10 | +0.00 |
| hc_grip_s0.04_seed0_b25 | 15.0% / 29.26 s | 91.5% / 26.18 s | 5.0% / 33.12 s | 37.0% / 31.85 s | 100.0% | 0.35 | +0.00 |
| hc_grip_s0.04_seed1_b25 | 52.5% / 32.31 s | 88.5% / 27.31 s | 28.0% / 36.74 s | 43.0% / 32.58 s | 100.0% | 0.42 | +0.00 |
| hc_grip_s0.04_seed2_b25 | 17.5% / 30.65 s | 98.5% / 25.46 s | 52.0% / 37.22 s | 53.0% / 29.25 s | 100.0% | 0.29 | +0.00 |
| hc_grip_s0.1_seed0_b25 | 4.5% / 28.22 s | 98.0% / 25.71 s | 1.0% / 33.10 s | 54.0% / 29.61 s | 100.0% | 0.37 | +0.00 |
| hc_grip_s0.1_seed1_b25 | 11.5% / 29.50 s | 98.5% / 25.56 s | 12.0% / 36.66 s | 66.0% / 29.29 s | 100.0% | 0.52 | +0.00 |
| hc_grip_s0.1_seed2_b25 | 17.5% / 30.39 s | 82.0% / 28.58 s | 7.0% / 34.16 s | 24.0% / 36.81 s | 100.0% | 0.73 | +0.00 |
| es_ft_grip_x100_seed0 | 97.5% / 33.44 s | 99.5% / 26.61 s | 95.0% / 37.88 s | 57.0% / 31.92 s | 100.0% | 1.70 | +0.00 |
| es_ft_grip_x10_seed0 | 95.0% / 33.06 s | 99.5% / 27.76 s | 61.0% / 37.10 s | 39.0% / 31.97 s | 100.0% | 0.57 | +0.00 |
| es_ft_grip_x10_seed1 | 99.0% / 33.54 s | 98.5% / 27.52 s | 100.0% / 37.70 s | 84.0% / 34.26 s | 100.0% | 0.55 | +0.00 |
| es_ft_grip_x10_seed2 | 99.5% / 33.23 s | 99.0% / 27.53 s | 100.0% / 37.35 s | 92.0% / 33.32 s | 100.0% | 0.54 | +0.00 |
| es_ft_grip_x1_seed0 | 81.0% / 32.70 s | 89.5% / 26.38 s | 93.0% / 37.32 s | 69.0% / 33.83 s | 100.0% | 0.20 | +0.00 |
| es_ft_grip_x1_seed1 | 71.0% / 32.60 s | 90.5% / 25.88 s | 85.0% / 36.93 s | 51.0% / 31.76 s | 100.0% | 0.20 | +0.00 |
| es_ft_grip_x1_seed2 | 86.5% / 32.63 s | 89.0% / 27.00 s | 94.0% / 36.73 s | 57.0% / 33.74 s | 100.0% | 0.20 | +0.00 |
| es_dr | 38.5% / 32.04 s | 100.0% / 26.01 s | 70.0% / 37.55 s | 82.0% / 30.19 s | - | - | - |
| ppo_dr | 92.5% / 33.24 s | 93.5% / 27.21 s | 97.0% / 37.22 s | 59.0% / 32.85 s | - | - | - |
| ray | 0.0% / - | 100.0% / 25.82 s | 0.0% / - | 79.0% / 28.72 s | - | - | - |
| ray_dr | 30.5% / 34.50 s | 100.0% / 31.94 s | 55.0% / 37.65 s | 81.0% / 37.14 s | - | - | - |

### heldout_hack_main.json

seeds: {'heldout': [90000, 90199], 'hard': [90000, 90099]}

| policy | heldout_strict_g1.0 | hard_strict_g1.0 | heldout_soft_g1.0 | weights changed | rel. change | cos(change, es_best - start) |
|---|---|---|---|---|---|---|
| es_soft | 0.0% / - | 0.0% / - | 100.0% / 21.82 s | 0.0% | 0.00 | +0.00 |
| astra_edit_a1 | 100.0% / 25.47 s | 98.0% / 28.86 s | 100.0% / 25.47 s | 17.8% | 1.01 | +0.09 |
| astra_edit_a2 | 100.0% / 25.77 s | 96.0% / 29.24 s | 100.0% / 25.77 s | 14.0% | 4.78 | -0.01 |
| astra_edit_a1_at5calls | 100.0% / 27.72 s | 84.0% / 31.23 s | 100.0% / 27.72 s | 17.1% | 0.89 | +0.09 |
| astra_edit_a2_at5calls | 100.0% / 28.01 s | 87.0% / 32.39 s | 100.0% / 28.01 s | 4.2% | 2.89 | -0.01 |
| es_strict_full | 100.0% / 25.09 s | 74.0% / 27.93 s | 100.0% / 25.09 s | 100.0% | 0.91 | +1.00 |
| hc_hack_s0.04_seed0_b25 | 0.0% / - | 0.0% / - | 100.0% / 21.96 s | 100.0% | 0.33 | +0.03 |
| hc_hack_s0.04_seed1_b25 | 0.0% / - | 0.0% / - | 100.0% / 21.86 s | 100.0% | 0.29 | +0.01 |
| hc_hack_s0.04_seed2_b25 | 0.0% / - | 0.0% / - | 100.0% / 21.82 s | 100.0% | 0.23 | -0.00 |
| hc_hack_s0.1_seed0_b25 | 0.0% / - | 0.0% / - | 100.0% / 21.82 s | 0.0% | 0.00 | +0.00 |
| hc_hack_s0.1_seed1_b25 | 0.0% / - | 0.0% / - | 100.0% / 21.84 s | 100.0% | 0.41 | +0.02 |
| hc_hack_s0.1_seed2_b25 | 0.0% / - | 0.0% / - | 100.0% / 21.82 s | 0.0% | 0.00 | +0.00 |
| ray | 100.0% / 25.82 s | 79.0% / 28.72 s | 100.0% / 25.82 s | - | - | - |
| ray_default | 50.0% / 25.36 s | 13.0% / 25.86 s | 100.0% / 26.77 s | - | - | - |

### heldout_hack_oneshot_es.json

seeds: {'heldout': [90000, 90199], 'hard': [90000, 90099]}

| policy | heldout_strict_g1.0 | hard_strict_g1.0 | heldout_soft_g1.0 | weights changed | rel. change | cos(change, es_best - start) |
|---|---|---|---|---|---|---|
| astra_oneshot_a1 | 97.5% / 26.69 s | 47.0% / 29.89 s | 99.5% / 26.76 s | 25.1% | 4.49 | +0.05 |
| astra_oneshot_a2 | 65.5% / 31.74 s | 1.0% / 34.30 s | 100.0% / 32.07 s | 6.4% | 1.55 | +0.01 |
| es_ft_hack_x100_seed0 | 99.0% / 25.03 s | 63.0% / 27.52 s | 100.0% / 25.00 s | 100.0% | 1.92 | +0.05 |
| es_ft_hack_x100_seed1 | 100.0% / 25.17 s | 91.0% / 27.68 s | 100.0% / 25.17 s | 100.0% | 1.94 | +0.07 |
| es_ft_hack_x100_seed2 | 99.5% / 24.92 s | 72.0% / 27.48 s | 100.0% / 24.91 s | 100.0% | 1.98 | +0.02 |
| es_ft_hack_x10_seed0 | 0.0% / - | 0.0% / - | 100.0% / 21.94 s | 100.0% | 0.65 | +0.01 |
| es_ft_hack_x10_seed1 | 0.0% / - | 0.0% / - | 100.0% / 21.88 s | 100.0% | 0.66 | +0.01 |
| es_ft_hack_x10_seed2 | 0.0% / - | 0.0% / - | 100.0% / 21.89 s | 100.0% | 0.68 | +0.03 |
| es_ft_hack_x1_seed0 | 0.0% / - | 0.0% / - | 100.0% / 21.84 s | 100.0% | 0.23 | -0.01 |
| es_ft_hack_x1_seed1 | 0.0% / - | 0.0% / - | 100.0% / 21.84 s | 100.0% | 0.23 | +0.00 |
| es_ft_hack_x1_seed2 | 0.0% / - | 0.0% / - | 100.0% / 21.82 s | 100.0% | 0.23 | +0.01 |

