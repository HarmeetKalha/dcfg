# GenD-aug (LUMEN2 baseline): GenD fine-tuned with its own recipe on our degraded stream

> **FLAG: the run hit the 18-epoch cap while validation was still improving** (best epoch 17 of 18). GenD-aug may be under-trained relative to a longer schedule.

Frame/video AUC on the FF++ c23 test split (+ c40), 95% video-level bootstrap CI (1000 resamples, one shared resample matrix: aggregates are joint and every difference is paired). Same GenD crops, degradation code, seeds and frame keys as `results/lumen2/gend_public`. Sources: results/lumen2/gend_public, results/lumen2/gend_aug, results/raw (Xception).

## Frame AUC [95% CI]

| model | clean | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | 0.9601 [0.9496, 0.9702] | 0.8680 [0.8503, 0.8864] | 0.8695 [0.8516, 0.8879] | 0.8010 [0.7727, 0.8280] | 0.8395 [0.8117, 0.8663] | 0.8211 [0.7925, 0.8485] | 0.8049 [0.7731, 0.8329] | 0.7604 [0.7261, 0.7927] | 0.7882 [0.7593, 0.8154] | 0.7845 [0.7555, 0.8130] |
| GenD-aug | 0.9730 [0.9644, 0.9804] | 0.9170 [0.9033, 0.9320] | 0.9159 [0.9024, 0.9314] | 0.8570 [0.8303, 0.8817] | 0.9023 [0.8837, 0.9182] | 0.8884 [0.8684, 0.9066] | 0.8999 [0.8810, 0.9179] | 0.8738 [0.8519, 0.8945] | 0.8403 [0.8165, 0.8623] | 0.8492 [0.8261, 0.8715] |
| base_xception | 0.9761 [0.9684, 0.9826] | 0.8175 [0.8017, 0.8317] | 0.8771 [0.8576, 0.8956] | 0.8046 [0.7756, 0.8332] | 0.5640 [0.5238, 0.6077] | 0.5405 [0.5023, 0.5814] | 0.7682 [0.7342, 0.8003] | 0.6100 [0.5653, 0.6533] | 0.7086 [0.6764, 0.7399] | 0.6877 [0.6603, 0.7146] |
| augft_xception | 0.9743 [0.9654, 0.9819] | 0.9286 [0.9159, 0.9410] | 0.9174 [0.9009, 0.9320] | 0.8722 [0.8496, 0.8921] | 0.9187 [0.9016, 0.9334] | 0.9034 [0.8852, 0.9188] | 0.9115 [0.8914, 0.9288] | 0.8774 [0.8543, 0.8996] | 0.8368 [0.8131, 0.8586] | 0.8548 [0.8329, 0.8754] |
| lumen_xception | 0.9742 [0.9660, 0.9815] | 0.8995 [0.8833, 0.9151] | 0.8871 [0.8687, 0.9036] | 0.8265 [0.7994, 0.8523] | 0.8775 [0.8551, 0.8975] | 0.8540 [0.8305, 0.8752] | 0.8671 [0.8417, 0.8889] | 0.8160 [0.7853, 0.8436] | 0.7722 [0.7427, 0.7995] | 0.7910 [0.7626, 0.8185] |

## Video AUC [95% CI]

| model | clean | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | 0.9778 [0.9685, 0.9862] | 0.8993 [0.8819, 0.9177] | 0.9079 [0.8902, 0.9265] | 0.8690 [0.8390, 0.8976] | 0.8747 [0.8450, 0.9032] | 0.8598 [0.8300, 0.8899] | 0.8369 [0.8030, 0.8657] | 0.7910 [0.7537, 0.8251] | 0.8253 [0.7935, 0.8569] | 0.8267 [0.7943, 0.8586] |
| GenD-aug | 0.9861 [0.9789, 0.9921] | 0.9438 [0.9306, 0.9577] | 0.9440 [0.9313, 0.9582] | 0.9084 [0.8821, 0.9320] | 0.9325 [0.9142, 0.9488] | 0.9216 [0.9012, 0.9401] | 0.9287 [0.9096, 0.9460] | 0.9054 [0.8830, 0.9262] | 0.8820 [0.8554, 0.9053] | 0.8916 [0.8674, 0.9150] |
| base_xception | 0.9895 [0.9830, 0.9946] | 0.8555 [0.8390, 0.8699] | 0.9223 [0.9029, 0.9399] | 0.8650 [0.8333, 0.8938] | 0.5846 [0.5331, 0.6402] | 0.5442 [0.4921, 0.5965] | 0.8095 [0.7725, 0.8448] | 0.6283 [0.5776, 0.6768] | 0.7680 [0.7291, 0.8050] | 0.7868 [0.7456, 0.8260] |
| augft_xception | 0.9875 [0.9802, 0.9933] | 0.9598 [0.9489, 0.9703] | 0.9518 [0.9375, 0.9651] | 0.9185 [0.8950, 0.9381] | 0.9587 [0.9435, 0.9724] | 0.9522 [0.9353, 0.9667] | 0.9410 [0.9218, 0.9579] | 0.9095 [0.8860, 0.9308] | 0.8981 [0.8729, 0.9208] | 0.9216 [0.8984, 0.9431] |
| lumen_xception | 0.9888 [0.9826, 0.9942] | 0.9353 [0.9196, 0.9498] | 0.9293 [0.9123, 0.9454] | 0.8811 [0.8521, 0.9075] | 0.9255 [0.9030, 0.9460] | 0.9079 [0.8838, 0.9310] | 0.9032 [0.8749, 0.9266] | 0.8524 [0.8180, 0.8817] | 0.8367 [0.8028, 0.8680] | 0.8517 [0.8183, 0.8825] |

## Paired difference GenD-aug minus GenD (public) [95% CI]

| aggregate | frame AUC | video AUC |
|---|---|---|
| clean | +0.0129 [+0.0069, +0.0198] | +0.0083 [+0.0040, +0.0132] |
| synth-degraded mean (17) | +0.0490 [+0.0391, +0.0592] | +0.0444 [+0.0336, +0.0554] |
| held-out mean (4) | +0.0464 [+0.0368, +0.0559] | +0.0361 [+0.0262, +0.0462] |
| c40 | +0.0560 [+0.0410, +0.0705] | +0.0393 [+0.0231, +0.0552] |

Clean frame AUC change: +0.0129 [+0.0069, +0.0198] -> **clean AUC did NOT drop** (CI not entirely below 0).

## Drop from own clean score (frame AUC, paired bootstrap) [95% CI]

| model | synth-degraded mean (17) | held-out mean (4) | c40 |
|---|---|---|---|
| GenD (public) | -0.0921 [-0.1048, -0.0796] | -0.0906 [-0.1030, -0.0778] | -0.1591 [-0.1837, -0.1345] |
| GenD-aug | -0.0560 [-0.0656, -0.0465] | -0.0570 [-0.0664, -0.0470] | -0.1160 [-0.1375, -0.0946] |
| augft_xception | -0.0457 [-0.0532, -0.0385] | -0.0569 [-0.0677, -0.0473] | -0.1021 [-0.1215, -0.0825] |

## Calibration and selective prediction over all degraded test frames

All non-clean conditions concatenated (21 synthetic + c40 = 22 conditions, 492488 frames). ECE: 15 equal-width bins, predicted P(fake) vs label (lower is better). Selective AUC: keep the most confident 80%/90% by |p-0.5|. CIs: video-level bootstrap; difference = GenD-aug minus GenD, paired.

| metric | GenD (public) | GenD-aug | difference |
|---|---|---|---|
| ECE | 0.0867 [0.0718, 0.1079] | 0.1045 [0.0892, 0.1217] | +0.0178 [+0.0061, +0.0256] |
| selective AUC 80% | 0.8850 [0.8680, 0.8996] | 0.9457 [0.9333, 0.9558] | +0.0607 [+0.0520, +0.0697] |
| selective AUC 90% | 0.8694 [0.8524, 0.8847] | 0.9312 [0.9177, 0.9427] | +0.0618 [+0.0533, +0.0704] |
| AUC (all frames) | 0.8521 [0.8356, 0.8677] | 0.9146 [0.9005, 0.9274] | +0.0625 [+0.0539, +0.0713] |

## Training

- stop reason: **epoch_cap**; best epoch 17, stop epoch 18 (1-based; cap 18, patience 6); best val mean AUC 0.9199 (degraded val set, 8 of 32 frames per video, 6 conditions: clean, jpeg_q50, jpeg_q30, blur_s2, noise_s10, mixed_heavy)
- wall time: train 21.19 h (0.88 d), val 1.63 h (0.07 d), total 22.82 h (0.95 d) (excludes crop preparation); 25830 optimiser steps (1435/epoch, effective batch 128), non-finite skipped steps: 0
- trainable parameters: 104,450; trainable-only file 0.432 MB vs full model file 1156.7 MB (fp32)
- val curve (mean frame AUC over the val conditions): 1: 0.8937, 2: 0.9023, 3: 0.9080, 4: 0.9121, 5: 0.9146, 6: 0.9164, 7: 0.9168, 8: 0.9175, 9: 0.9181, 10: 0.9188, 11: 0.9187, 12: 0.9197, 13: 0.9197, 14: 0.9199, 15: 0.9197, 16: 0.9199, 17: 0.9199, 18: 0.9198
