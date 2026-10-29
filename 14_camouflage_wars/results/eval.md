Baseline = `outlier` (sigma 2.0), chosen on val. Load: 19:05  up 3 days, 29 mins, 2 users, load averages: 11.64 13.25 10.82

| split | method | scene IoU | pixel IoU | det@IoU0.5 | inst IoU | det@0.5 isolated only | coverage≥0.5 | FP comps/scene | ms/scene (under load) |
|---|---|---|---|---|---|---|---|---|---|
| test_id | unet | 0.896 | 0.892 | 0.738 | 0.725 | 0.934 | 0.951 | 0.03 | 0.5 |
| test_id | baseline | 0.485 | 0.412 | 0.456 | 0.419 | 0.584 | 0.745 | 3.26 | 3.4 |
| test_ood | unet | 0.832 | 0.827 | 0.717 | 0.698 | 0.898 | 0.923 | 0.30 | 0.5 |
| test_ood | baseline | 0.433 | 0.375 | 0.421 | 0.404 | 0.532 | 0.756 | 3.77 | 3.4 |

**By camouflage knob δ (generation parameter).** det(iso) = instance IoU ≥ 0.5 among creatures not touching another creature; cov = ≥ 50% of the creature's pixels predicted (all creatures)

| split | bin | n (isolated) | U-Net det(iso) | baseline det(iso) | U-Net cov | baseline cov | U-Net inst IoU | baseline inst IoU |
|---|---|---|---|---|---|---|---|---|
| test_id | other family | 505 (338) | 0.982 | 0.790 | 0.998 | 0.950 | 0.807 | 0.524 |
| test_id | same δ<0.25 | 347 (209) | 0.737 | 0.177 | 0.764 | 0.271 | 0.523 | 0.176 |
| test_id | same 0.25–0.5 | 399 (259) | 0.954 | 0.506 | 0.972 | 0.657 | 0.728 | 0.387 |
| test_id | same 0.5–0.75 | 370 (240) | 0.983 | 0.629 | 0.989 | 0.832 | 0.770 | 0.460 |
| test_id | same 0.75–1 | 379 (226) | 0.969 | 0.695 | 1.000 | 0.910 | 0.752 | 0.495 |
| test_ood | other family | 525 (333) | 0.979 | 0.715 | 0.998 | 0.941 | 0.766 | 0.502 |
| test_ood | same δ<0.25 | 361 (234) | 0.620 | 0.128 | 0.637 | 0.321 | 0.437 | 0.178 |
| test_ood | same 0.25–0.5 | 372 (253) | 0.937 | 0.387 | 0.978 | 0.661 | 0.736 | 0.343 |
| test_ood | same 0.5–0.75 | 384 (249) | 0.948 | 0.647 | 0.987 | 0.854 | 0.739 | 0.456 |
| test_ood | same 0.75–1 | 373 (251) | 0.964 | 0.697 | 0.973 | 0.912 | 0.772 | 0.492 |

**By measured texture-statistics distance (quartiles from train).** det(iso) = instance IoU ≥ 0.5 among creatures not touching another creature; cov = ≥ 50% of the creature's pixels predicted (all creatures)

| split | bin | n (isolated) | U-Net det(iso) | baseline det(iso) | U-Net cov | baseline cov | U-Net inst IoU | baseline inst IoU |
|---|---|---|---|---|---|---|---|---|
| test_id | Q1 (closest match) | 462 (280) | 0.796 | 0.221 | 0.814 | 0.325 | 0.573 | 0.198 |
| test_id | Q2 | 519 (337) | 0.961 | 0.540 | 0.983 | 0.765 | 0.754 | 0.421 |
| test_id | Q3 | 512 (329) | 0.979 | 0.714 | 0.996 | 0.896 | 0.781 | 0.497 |
| test_id | Q4 (least match) | 507 (326) | 0.979 | 0.810 | 0.998 | 0.953 | 0.776 | 0.540 |
| test_ood | Q1 (closest match) | 324 (217) | 0.673 | 0.134 | 0.670 | 0.302 | 0.481 | 0.180 |
| test_ood | Q2 | 417 (279) | 0.900 | 0.455 | 0.930 | 0.700 | 0.704 | 0.369 |
| test_ood | Q3 | 488 (318) | 0.965 | 0.623 | 0.982 | 0.857 | 0.753 | 0.454 |
| test_ood | Q4 (least match) | 786 (506) | 0.953 | 0.688 | 0.987 | 0.911 | 0.749 | 0.483 |

**By creature skin family.** det(iso) = instance IoU ≥ 0.5 among creatures not touching another creature; cov = ≥ 50% of the creature's pixels predicted (all creatures)

| split | bin | n (isolated) | U-Net det(iso) | baseline det(iso) | U-Net cov | baseline cov | U-Net inst IoU | baseline inst IoU |
|---|---|---|---|---|---|---|---|---|
| test_id | cells | 322 (201) | 0.915 | 0.507 | 0.932 | 0.686 | 0.719 | 0.367 |
| test_id | checker | 318 (198) | 0.949 | 0.753 | 0.959 | 0.843 | 0.723 | 0.489 |
| test_id | fbm | 366 (240) | 0.929 | 0.371 | 0.934 | 0.593 | 0.740 | 0.335 |
| test_id | plaid | 376 (238) | 0.903 | 0.584 | 0.963 | 0.785 | 0.701 | 0.441 |
| test_id | spots | 277 (187) | 0.963 | 0.695 | 0.953 | 0.798 | 0.749 | 0.461 |
| test_id | stripes | 341 (208) | 0.952 | 0.644 | 0.965 | 0.783 | 0.722 | 0.435 |
| test_ood | marble | 657 (411) | 0.898 | 0.494 | 0.928 | 0.756 | 0.684 | 0.390 |
| test_ood | ripples | 689 (450) | 0.889 | 0.547 | 0.935 | 0.779 | 0.692 | 0.416 |
| test_ood | streaks | 669 (459) | 0.908 | 0.551 | 0.906 | 0.732 | 0.717 | 0.405 |
