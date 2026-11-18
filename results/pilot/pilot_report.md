# DCFG pilot report (VALIDATION split only, PILOT - not paper numbers)

700 val videos, 22354 frames per condition; 1000 video-level bootstrap replicates (95% percentile CI). Models: Xception detector; base = frozen detector, fixed/dcfg = front-end + frozen detector, augft = fine-tuned detector; 5 epochs of the real 30-epoch schedule. Not used for selection of anything on the test split.

## (a) Frame AUC [95% CI]

| model | clean | noise_s10 | blur_s3 | webp_q20 | mixed_heavy | c40 | mean(degraded) |
|---|---|---|---|---|---|---|---|
| base | 0.9764 [0.9684, 0.9832] | 0.7192 [0.6850, 0.7552] | 0.7827 [0.7474, 0.8147] | 0.8418 [0.8182, 0.8640] | 0.6932 [0.6635, 0.7196] | 0.7966 [0.7668, 0.8268] | 0.7592 [0.7387, 0.7787] |
| fixed | 0.9669 [0.9569, 0.9753] | 0.8292 [0.8006, 0.8532] | 0.8084 [0.7752, 0.8373] | 0.8438 [0.8191, 0.8654] | 0.7144 [0.6867, 0.7394] | 0.8000 [0.7706, 0.8283] | 0.7989 [0.7765, 0.8180] |
| augft | 0.9754 [0.9661, 0.9831] | 0.9372 [0.9205, 0.9512] | 0.9086 [0.8864, 0.9279] | 0.8921 [0.8708, 0.9121] | 0.8576 [0.8335, 0.8792] | 0.8650 [0.8419, 0.8881] | 0.8989 [0.8793, 0.9171] |
| dcfg | 0.9762 [0.9676, 0.9831] | 0.8737 [0.8487, 0.8955] | 0.8097 [0.7763, 0.8413] | 0.8408 [0.8161, 0.8640] | 0.7355 [0.7076, 0.7612] | 0.7921 [0.7616, 0.8222] | 0.8149 [0.7951, 0.8341] |

Degraded mean = mean over noise_s10, blur_s3, webp_q20, mixed_heavy (synthetic degradations of the c23 frames; c40 is not part of it); its CI resamples videos jointly across conditions. c40 (val, real H.264 compression, no synthetic op) has its own video set and resample.

## (b) Paired bootstrap delta-AUC (dcfg minus baseline); `*` = CI excludes 0

| comparison | noise_s10 | blur_s3 | webp_q20 | mixed_heavy | mean(degraded) | c40 (val) |
|---|---|---|---|---|---|---|
| dcfg - augft | -0.0634 [-0.0822, -0.0472] * | -0.0989 [-0.1270, -0.0713] * | -0.0513 [-0.0676, -0.0354] * | -0.1221 [-0.1470, -0.0989] * | -0.0839 [-0.0984, -0.0684] * | -0.0728 [-0.0915, -0.0557] * |
| dcfg - fixed | +0.0446 [+0.0234, +0.0688] * | +0.0014 [-0.0190, +0.0231] | -0.0030 [-0.0096, +0.0040] | +0.0211 [+0.0036, +0.0406] * | +0.0160 [+0.0063, +0.0280] * | -0.0079 [-0.0163, +0.0003] |

**dcfg beats augft and fixed beyond the CIs on degraded conditions: MIXED**

- Paired-CI basis: dcfg-augft: degraded-mean -0.0839 [-0.0984, -0.0684]; per-condition CI>0 in 0, CI<0 in 4 of 4; dcfg-fixed: degraded-mean +0.0160 [+0.0063, +0.0280]; per-condition CI>0 in 2, CI<0 in 0 of 4
- Plain CI-overlap view: dcfg vs augft: degraded-mean CIs do NOT overlap (0.8149 [0.7951, 0.8341] vs 0.8989 [0.8793, 0.9171]; corrected 2026-11-17, generator bug fixed); dcfg vs fixed: degraded-mean CIs overlap
- Rule: YES = degraded-mean delta CI > 0 against both and no degraded condition with delta CI < 0; NO = degraded-mean delta CI not > 0 against either; MIXED = otherwise.

## (c) GPU / CPU summary (training phase only; validation excluded)

| run | GPU util mean/median % (train) | data-wait share | CPU util mean % | peak GPU mem MiB | max temp C | wall train/val (h) | CPU-bound? |
|---|---|---|---|---|---|---|---|
| pilot_fixed_xception | 98.6 / 99.0 | 0.1% | 42.5 | 5689 | 82 | 0.87 / 0.29 | no |
| pilot_augft_xception | 98.2 / 99.0 | 0.1% | 36.1 | 5360 | 81 | 1.22 / 0.26 | no |
| pilot_dcfg_xception | 98.7 / 99.0 | 0.1% | 36.0 | 6018 | 81 | 1.20 / 0.37 | no |

Flag rule: CPU-bound if mean GPU util < 85% or data-wait share > 15%. No run flagged.

## (d) Pilot wall time

runner log span (first pilot job start to pilot_report start): 4.97 h; sum of training total_time_s (done.json): 4.21 h

## (e) c40

c40 VAL frames present: included as its own column (699 videos, real compression, no synthetic op).

## Addendum: GATE 8 ETA recomputed from pilot timings (added 2026-11-17)

Measured per epoch (Xception, from resources_summary.json, wall incl. val): dcfg 18.8 min (train 14.4 + val 4.4), fixed 14.0 min (train 10.5 + val 3.5), augft 17.8 min (train 14.6 + val 3.2).
Phase-8 estimate per epoch: dcfg 19.6 min, fixed 15.1 min, augft 19.6 min. The phase-8 estimate was slightly pessimistic; the earlier '~23 min/epoch' quoted in chat was a mis-read of in-progress logs.
Recomputation: each run's per-epoch time scaled by the measured/estimated ratio of its Xception variant class (dcfg-class incl. nolive/frozensensor; fixed-class incl. nosensor; augft); jointft and non-Xception detectors use the same class ratio (ASSUMPTION: only Xception was measured; jointft unscaled). Test-eval time unchanged.

| group | runs | worst 30 ep (GPU-days) | likely 15 ep (GPU-days, assumption) | cumulative worst | cumulative likely |
|---|---|---|---|---|---|
| 1 | 12 | 6.24 | 3.26 | 6.24 | 3.26 |
| 2 | 4 | 1.66 | 0.86 | 7.89 | 4.13 |
| 3 | 2 | 0.82 | 0.43 | 8.71 | 4.55 |
| 4 | 2 | 0.62 | 0.33 | 9.33 | 4.88 |
