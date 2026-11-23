# LUMEN2 evaluation: LUMEN2 minus GenD-aug (pre-declared metrics)

All numbers: frame AUC unless stated, FF++ c23 test + c40, GenD crops, degradation code/seeds/frame keys identical to GenD-aug. Paired differences LUMEN2 - GenD-aug, 95% video-level bootstrap CI (1000 resamples, ONE shared resample matrix, aggregates bootstrapped jointly). Win = CI entirely on the favourable side of 0 (ECE: lower is better); safety checks pass unless the CI is entirely below 0. Metric definitions: docs/LUMEN2_BRIEF.md (heavy-damage list amended 2026-11-21). The LUMEN-uncond control is NOT run yet, so this is not yet the full publishing rule.

## Pre-declared metrics

| metric | role | GenD-aug | LUMEN2 | LUMEN2 - GenD-aug | outcome |
|---|---|---|---|---|---|
| Synthetic-degraded mean AUC (17) | primary | 0.9170 [0.9033, 0.9320] | 0.9275 [0.9136, 0.9413] | +0.0105 [+0.0049, +0.0158] | win |
| c40 AUC | secondary | 0.8570 [0.8324, 0.8801] | 0.8517 [0.8282, 0.8740] | -0.0053 [-0.0159, +0.0045] | tie |
| Held-out mean AUC (WebP q50/q20, motion k7/k11) | secondary | 0.9159 [0.9024, 0.9314] | 0.9101 [0.8945, 0.9250] | -0.0059 [-0.0133, +0.0009] | tie |
| ECE (15 bins, all degraded frames) | secondary | 0.1045 [0.0885, 0.1209] | 0.1097 [0.0939, 0.1264] | +0.0052 [+0.0003, +0.0098] | loss |
| Selective AUC, 80% coverage | secondary | 0.9457 [0.9341, 0.9572] | 0.9495 [0.9376, 0.9599] | +0.0038 [-0.0007, +0.0084] | tie |
| Selective AUC, 90% coverage | secondary | 0.9312 [0.9186, 0.9445] | 0.9360 [0.9232, 0.9483] | +0.0048 [-0.0002, +0.0099] | tie |
| Worst-case AUC (min over 22 degraded conditions) | secondary | 0.8403 [0.8169, 0.8613] | 0.8517 [0.8282, 0.8740] | +0.0114 [-0.0064, +0.0305] | tie |
| Heavy-damage mean AUC (jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy) | secondary | 0.8775 [0.8604, 0.8961] | 0.8954 [0.8785, 0.9126] | +0.0179 [+0.0104, +0.0255] | win |
| Video AUC, synthetic mean (17) | secondary | 0.9438 [0.9306, 0.9577] | 0.9526 [0.9393, 0.9646] | +0.0088 [+0.0037, +0.0141] | win |
| Video AUC, c40 | secondary | 0.9084 [0.8834, 0.9316] | 0.9123 [0.8885, 0.9343] | +0.0039 [-0.0072, +0.0152] | tie |
| DFD under damage (frame AUC, mixed_heavy) | secondary | 0.6719 [0.6535, 0.6914] | 0.6651 [0.6449, 0.6858] | -0.0068 [-0.0175, +0.0031] | tie |
| Clean c23 AUC | safety | 0.9730 [0.9646, 0.9807] | 0.9759 [0.9674, 0.9831] | +0.0029 [-0.0008, +0.0063] | ok |
| Clean DFD AUC | safety | 0.8741 [0.8620, 0.8858] | 0.8921 [0.8803, 0.9028] | +0.0180 [+0.0129, +0.0231] | ok |

Selective-AUC rule (win if either coverage wins): **tie**. ECE is computed over 492488 frames (22 degraded conditions concatenated: 17 synthetic + 4 held-out + c40).

## Verdict on the pre-declared rule

- Primary (synthetic-degraded mean): **win**
- Secondary wins: synth_mean, heavy_mean, video_synth_mean
- Safety: clean c23 ok, clean DFD ok
- "Worth publishing" rule (wins primary or >= 1 secondary, passes both safety checks): **PASSES** - still requires beating LUMEN-uncond on the won metric (not run).

## Exploratory (never used as a claim)

Per-condition frame AUC, paired difference LUMEN2 - GenD-aug [95% CI], video-AUC difference, and AP / EER (point estimates, no CI):

| condition | GenD-aug AUC | LUMEN2 AUC | frame diff [CI] | video diff [CI] | AP (GenD-aug / LUMEN2) | EER (GenD-aug / LUMEN2) |
|---|---|---|---|---|---|---|
| clean | 0.9730 | 0.9759 | +0.0029 [-0.0008, +0.0063] | +0.0021 [-0.0005, +0.0053] | 0.9933 / 0.9939 | 0.0738 / 0.0641 |
| jpeg_q90 | 0.9654 | 0.9655 | +0.0000 [-0.0046, +0.0044] | -0.0010 [-0.0047, +0.0027] | 0.9914 / 0.9911 | 0.0890 / 0.0854 |
| jpeg_q70 | 0.9509 | 0.9514 | +0.0005 [-0.0049, +0.0053] | -0.0006 [-0.0051, +0.0038] | 0.9879 / 0.9877 | 0.1139 / 0.1105 |
| jpeg_q50 | 0.9334 | 0.9375 | +0.0041 [-0.0021, +0.0100] | +0.0027 [-0.0032, +0.0085] | 0.9832 / 0.9842 | 0.1391 / 0.1310 |
| jpeg_q30 | 0.9159 | 0.9231 | +0.0072 [-0.0000, +0.0148] | +0.0062 [-0.0012, +0.0135] | 0.9784 / 0.9802 | 0.1625 / 0.1504 |
| jpeg_q10 | 0.8403 | 0.8664 | +0.0261 [+0.0130, +0.0389] | +0.0243 [+0.0099, +0.0386] | 0.9560 / 0.9642 | 0.2415 / 0.2164 |
| blur_s1 | 0.9576 | 0.9641 | +0.0064 [+0.0013, +0.0116] | +0.0061 [+0.0017, +0.0110] | 0.9890 / 0.9909 | 0.1036 / 0.0866 |
| blur_s2 | 0.9286 | 0.9458 | +0.0172 [+0.0097, +0.0254] | +0.0174 [+0.0092, +0.0255] | 0.9812 / 0.9862 | 0.1442 / 0.1204 |
| blur_s3 | 0.8999 | 0.9208 | +0.0209 [+0.0102, +0.0307] | +0.0204 [+0.0093, +0.0318] | 0.9734 / 0.9795 | 0.1787 / 0.1543 |
| blur_s4 | 0.8738 | 0.8932 | +0.0194 [+0.0070, +0.0324] | +0.0168 [+0.0021, +0.0310] | 0.9660 / 0.9720 | 0.2074 / 0.1884 |
| noise_s5 | 0.9485 | 0.9500 | +0.0015 [-0.0034, +0.0065] | -0.0001 [-0.0054, +0.0053] | 0.9871 / 0.9875 | 0.1185 / 0.1130 |
| noise_s10 | 0.9197 | 0.9279 | +0.0082 [+0.0019, +0.0149] | +0.0072 [+0.0010, +0.0142] | 0.9795 / 0.9818 | 0.1587 / 0.1442 |
| noise_s15 | 0.9023 | 0.9139 | +0.0117 [+0.0037, +0.0194] | +0.0092 [+0.0014, +0.0173] | 0.9748 / 0.9781 | 0.1779 / 0.1650 |
| noise_s20 | 0.8884 | 0.9009 | +0.0125 [+0.0038, +0.0207] | +0.0094 [+0.0002, +0.0184] | 0.9708 / 0.9744 | 0.1952 / 0.1818 |
| resize_0.5 | 0.9663 | 0.9706 | +0.0044 [+0.0003, +0.0083] | +0.0036 [+0.0005, +0.0071] | 0.9916 / 0.9927 | 0.0865 / 0.0768 |
| resize_0.25 | 0.9360 | 0.9448 | +0.0088 [+0.0028, +0.0152] | +0.0102 [+0.0041, +0.0165] | 0.9836 / 0.9861 | 0.1349 / 0.1210 |
| mixed_light | 0.9130 | 0.9197 | +0.0066 [-0.0009, +0.0140] | +0.0020 [-0.0058, +0.0095] | 0.9774 / 0.9794 | 0.1605 / 0.1576 |
| mixed_heavy | 0.8492 | 0.8717 | +0.0225 [+0.0102, +0.0349] | +0.0167 [+0.0024, +0.0297] | 0.9588 / 0.9658 | 0.2351 / 0.2104 |
| webp_q50 | 0.9124 | 0.9016 | -0.0107 [-0.0200, -0.0021] | -0.0051 [-0.0151, +0.0038] | 0.9781 / 0.9745 | 0.1641 / 0.1708 |
| webp_q20 | 0.8881 | 0.8719 | -0.0161 [-0.0271, -0.0062] | -0.0103 [-0.0228, +0.0014] | 0.9712 / 0.9662 | 0.1911 / 0.2047 |
| motion_k7 | 0.9448 | 0.9441 | -0.0007 [-0.0078, +0.0070] | +0.0014 [-0.0049, +0.0078] | 0.9854 / 0.9853 | 0.1169 / 0.1192 |
| motion_k11 | 0.9185 | 0.9226 | +0.0041 [-0.0062, +0.0134] | +0.0079 [-0.0014, +0.0168] | 0.9779 / 0.9791 | 0.1522 / 0.1474 |
| c40 | 0.8570 | 0.8517 | -0.0053 [-0.0159, +0.0045] | +0.0039 [-0.0072, +0.0152] | 0.9566 / 0.9553 | 0.2184 / 0.2245 |

## DFD (DeepFakeDetection, never used in training)

|  | GenD-aug | LUMEN2 | difference |
|---|---|---|---|
| frame AUC, clean | 0.8741 [0.8620, 0.8858] | 0.8921 [0.8803, 0.9028] | +0.0180 [+0.0129, +0.0231] |
| frame AUC, mixed_heavy | 0.6719 [0.6535, 0.6914] | 0.6651 [0.6449, 0.6858] | -0.0068 [-0.0175, +0.0031] |
| video AUC, clean | 0.9175 [0.9050, 0.9302] | 0.9369 [0.9256, 0.9479] | +0.0194 [+0.0137, +0.0252] |
| video AUC, mixed_heavy | 0.7188 [0.6942, 0.7433] | 0.7093 [0.6850, 0.7351] | -0.0094 [-0.0228, +0.0036] |

3431 videos; mixed_heavy uses the same seeding rule as the FF++ conditions.

## Damage sensor inside LUMEN2

Presence threshold 0.5; labelled conditions pooled (clean + 17 synthetic), strength metrics only where the op is applied (strength scale 0..1 over the training range). The sensor here is trained jointly with LUMEN2 (loss weight 0.1).

| op | presence acc | presence F1 | strength MAE | Spearman | standalone F1 | standalone MAE |
|---|---|---|---|---|---|---|
| jpeg | 0.989 | 0.985 | 0.045 | 0.969 | 0.992 | 0.020 |
| blur | 0.875 | 0.830 | 0.086 | 0.939 | 0.929 | 0.049 |
| noise | 0.977 | 0.966 | 0.095 | 0.898 | 0.978 | 0.038 |
| resize | 0.897 | 0.535 | 0.348 | -0.851 | 0.831 | 0.056 |

Per labelled condition (true ops, exact op-set accuracy, mean P(op) in order jpeg / blur / noise / resize):

| condition | true ops | exact set acc | mean P(jpeg/blur/noise/resize) |
|---|---|---|---|
| clean | none | 0.984 | 0.02 / 0.03 / 0.00 / 0.03 |
| jpeg_q90 | jpeg | 0.869 | 0.96 / 0.10 / 0.02 / 0.20 |
| jpeg_q70 | jpeg | 0.822 | 1.00 / 0.16 / 0.05 / 0.31 |
| jpeg_q50 | jpeg | 0.898 | 1.00 / 0.13 / 0.06 / 0.27 |
| jpeg_q30 | jpeg | 0.872 | 1.00 / 0.14 / 0.10 / 0.32 |
| jpeg_q10 | jpeg | 0.607 | 1.00 / 0.25 / 0.24 / 0.40 |
| blur_s1 | blur | 0.707 | 0.09 / 0.92 / 0.06 / 0.40 |
| blur_s2 | blur | 0.943 | 0.09 / 1.00 / 0.15 / 0.28 |
| blur_s3 | blur | 0.910 | 0.12 / 1.00 / 0.22 / 0.30 |
| blur_s4 | blur | 0.824 | 0.18 / 1.00 / 0.32 / 0.31 |
| noise_s5 | noise | 0.906 | 0.13 / 0.16 / 1.00 / 0.20 |
| noise_s10 | noise | 0.898 | 0.22 / 0.18 / 1.00 / 0.23 |
| noise_s15 | noise | 0.856 | 0.27 / 0.21 / 1.00 / 0.26 |
| noise_s20 | noise | 0.799 | 0.29 / 0.26 / 1.00 / 0.26 |
| resize_0.5 | resize | 0.958 | 0.09 / 0.01 / 0.00 / 0.91 |
| resize_0.25 | resize | 0.040 | 0.05 / 0.74 / 0.05 / 0.23 |
| mixed_light | jpeg+blur+noise | 0.502 | 1.00 / 0.51 / 0.93 / 0.33 |
| mixed_heavy | jpeg+blur+noise | 0.970 | 1.00 / 0.82 / 0.95 / 0.22 |

Unseen conditions (no ground truth: real compression c40, WebP, motion blur): fraction of frames with any op detected and mean P(op) per op.

| condition | any op present | P(jpeg) | P(blur) | P(noise) | P(resize) |
|---|---|---|---|---|---|
| webp_q50 | 0.69 | 0.64 | 0.01 | 0.00 | 0.00 |
| webp_q20 | 0.45 | 0.46 | 0.01 | 0.00 | 0.00 |
| motion_k7 | 0.81 | 0.23 | 0.74 | 0.01 | 0.03 |
| motion_k11 | 0.98 | 0.56 | 0.93 | 0.01 | 0.06 |
| c40 | 0.03 | 0.01 | 0.05 | 0.01 | 0.05 |

## Cost

- LUMEN parameters: 1,741,424 (gend_trainable (LayerNorms + head, = GenD-aug): 104,450, sensor: 772,264, gate (MLP + 2 LN): 132,160, ln_generator (MLP + U,V): 837,000)
- trainable-only file: LUMEN2 7.105 MB vs GenD-aug 0.432 MB
- inference (batch 64, NVIDIA GeForce RTX 3060, CLIP fp16 autocast, LUMEN parts fp32): LUMEN2 11.71 ms/image vs GenD-aug 10.57 ms/image (x1.11)
- training: stop reason **patience**, best epoch 6, stop epoch 12 (1-based; cap 18, patience 6); best val mean AUC 0.9305; train 16.17 h + val 1.20 h = 17.37 h; 17220 optimiser steps
- val curve: 1: 0.9076, 2: 0.9190, 3: 0.9232, 4: 0.9244, 5: 0.9276, 6: 0.9305, 7: 0.9269, 8: 0.9256, 9: 0.9261, 10: 0.9274, 11: 0.9239, 12: 0.9249
- parameters per part: gend_trainable (LayerNorms + head, = GenD-aug): 104,450, sensor: 772,264, gate (MLP + 2 LN): 132,160, ln_generator (MLP + U,V): 837,000, LUMEN parts total: 1,741,424

<!-- DFD begin -->
## DFD (DeepFakeDetection, FF++ release; never used in training)

All DFD videos (real actors + DFD fakes), 32 frames per video, video-level bootstrap 95% CI (1000 resamples). `mixed_heavy` uses the same degradation and seeding rule as on FF++ (per-frame key). GenD models use GenD RetinaFace crops; Xception models use DeepfakeBench crops.

| model | frame AUC clean | video AUC clean | frame AUC mixed_heavy | video AUC mixed_heavy |
|---|---|---|---|---|
| GenD | 0.8924 [0.8831, 0.9016] | 0.9349 [0.9252, 0.9438] | 0.6557 [0.6349, 0.6753] | 0.6938 [0.6672, 0.7185] |
| GenD-aug | 0.8741 [0.8630, 0.8847] | 0.9175 [0.9059, 0.9287] | 0.6719 [0.6521, 0.6904] | 0.7188 [0.6937, 0.7426] |
| LUMEN2 | 0.8921 [0.8819, 0.9019] | 0.9369 [0.9263, 0.9462] | 0.6651 [0.6459, 0.6839] | 0.7093 [0.6851, 0.7337] |
<!-- DFD end -->
