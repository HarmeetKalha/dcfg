# GenD room-to-improve (LUMEN2 step 1)

Public GenD CLIP-L/14 evaluated on our 23 FF++ test conditions (frame AUC, 95% video-level bootstrap CI, 1000 resamples, one shared resample matrix so aggregates are joint and all differences paired). Source: results/lumen2/gend_public/*.json + scores, results/raw (Xception).

## Frame AUC [95% CI]

| model | clean | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | 0.9601 [0.9496, 0.9702] | 0.8680 [0.8503, 0.8864] | 0.8695 [0.8516, 0.8879] | 0.8010 [0.7727, 0.8280] | 0.8395 [0.8117, 0.8663] | 0.8211 [0.7925, 0.8485] | 0.8049 [0.7731, 0.8329] | 0.7604 [0.7261, 0.7927] | 0.7882 [0.7593, 0.8154] | 0.7845 [0.7555, 0.8130] |
| base_xception | 0.9761 [0.9684, 0.9826] | 0.8175 [0.8017, 0.8317] | 0.8771 [0.8576, 0.8956] | 0.8046 [0.7756, 0.8332] | 0.5640 [0.5238, 0.6077] | 0.5405 [0.5023, 0.5814] | 0.7682 [0.7342, 0.8003] | 0.6100 [0.5653, 0.6533] | 0.7086 [0.6764, 0.7399] | 0.6877 [0.6603, 0.7146] |
| augft_xception | 0.9743 [0.9654, 0.9819] | 0.9286 [0.9159, 0.9410] | 0.9174 [0.9009, 0.9320] | 0.8722 [0.8496, 0.8921] | 0.9187 [0.9016, 0.9334] | 0.9034 [0.8852, 0.9188] | 0.9115 [0.8914, 0.9288] | 0.8774 [0.8543, 0.8996] | 0.8368 [0.8131, 0.8586] | 0.8548 [0.8329, 0.8754] |
| lumen_xception | 0.9742 [0.9660, 0.9815] | 0.8995 [0.8833, 0.9151] | 0.8871 [0.8687, 0.9036] | 0.8265 [0.7994, 0.8523] | 0.8775 [0.8551, 0.8975] | 0.8540 [0.8305, 0.8752] | 0.8671 [0.8417, 0.8889] | 0.8160 [0.7853, 0.8436] | 0.7722 [0.7427, 0.7995] | 0.7910 [0.7626, 0.8185] |

## Video AUC [95% CI]

| model | clean | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | 0.9778 [0.9685, 0.9862] | 0.8993 [0.8819, 0.9177] | 0.9079 [0.8902, 0.9265] | 0.8690 [0.8390, 0.8976] | 0.8747 [0.8450, 0.9032] | 0.8598 [0.8300, 0.8899] | 0.8369 [0.8030, 0.8657] | 0.7910 [0.7537, 0.8251] | 0.8253 [0.7935, 0.8569] | 0.8267 [0.7943, 0.8586] |
| base_xception | 0.9895 [0.9830, 0.9946] | 0.8555 [0.8390, 0.8699] | 0.9223 [0.9029, 0.9399] | 0.8650 [0.8333, 0.8938] | 0.5846 [0.5331, 0.6402] | 0.5442 [0.4921, 0.5965] | 0.8095 [0.7725, 0.8448] | 0.6283 [0.5776, 0.6768] | 0.7680 [0.7291, 0.8050] | 0.7868 [0.7456, 0.8260] |
| augft_xception | 0.9875 [0.9802, 0.9933] | 0.9598 [0.9489, 0.9703] | 0.9518 [0.9375, 0.9651] | 0.9185 [0.8950, 0.9381] | 0.9587 [0.9435, 0.9724] | 0.9522 [0.9353, 0.9667] | 0.9410 [0.9218, 0.9579] | 0.9095 [0.8860, 0.9308] | 0.8981 [0.8729, 0.9208] | 0.9216 [0.8984, 0.9431] |
| lumen_xception | 0.9888 [0.9826, 0.9942] | 0.9353 [0.9196, 0.9498] | 0.9293 [0.9123, 0.9454] | 0.8811 [0.8521, 0.9075] | 0.9255 [0.9030, 0.9460] | 0.9079 [0.8838, 0.9310] | 0.9032 [0.8749, 0.9266] | 0.8524 [0.8180, 0.8817] | 0.8367 [0.8028, 0.8680] | 0.8517 [0.8183, 0.8825] |

## Drop from own clean score, paired bootstrap delta [95% CI] (frame AUC)

| model | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | -0.0921 [-0.1048, -0.0796] | -0.0906 [-0.1030, -0.0778] | -0.1591 [-0.1837, -0.1345] | -0.1206 [-0.1462, -0.0985] | -0.1389 [-0.1655, -0.1159] | -0.1552 [-0.1811, -0.1295] | -0.1996 [-0.2309, -0.1690] | -0.1719 [-0.1963, -0.1477] | -0.1756 [-0.1999, -0.1515] |
| base_xception | -0.1585 [-0.1709, -0.1473] | -0.0990 [-0.1140, -0.0856] | -0.1715 [-0.1975, -0.1441] | -0.4121 [-0.4557, -0.3695] | -0.4356 [-0.4773, -0.3940] | -0.2079 [-0.2416, -0.1733] | -0.3661 [-0.4092, -0.3211] | -0.2675 [-0.3019, -0.2374] | -0.2884 [-0.3158, -0.2604] |
| augft_xception | -0.0457 [-0.0532, -0.0385] | -0.0569 [-0.0677, -0.0473] | -0.1021 [-0.1215, -0.0825] | -0.0556 [-0.0682, -0.0439] | -0.0709 [-0.0854, -0.0575] | -0.0628 [-0.0780, -0.0478] | -0.0969 [-0.1173, -0.0782] | -0.1375 [-0.1589, -0.1179] | -0.1195 [-0.1372, -0.1024] |
| lumen_xception | -0.0747 [-0.0862, -0.0640] | -0.0871 [-0.0993, -0.0751] | -0.1477 [-0.1707, -0.1242] | -0.0967 [-0.1151, -0.0790] | -0.1202 [-0.1401, -0.0997] | -0.1070 [-0.1271, -0.0875] | -0.1582 [-0.1868, -0.1320] | -0.2020 [-0.2302, -0.1754] | -0.1832 [-0.2068, -0.1594] |

## Gate (step 4 of the brief)

- clean -> synthetic-degraded mean: -0.0921 (threshold -0.03); clean -> c40: -0.1591 (threshold -0.03).
- **Gate TRIGGERED (drop >= 0.03): DFD download + evaluation required.**

## Calibration / selective prediction over all degraded test frames (reference)

All non-clean conditions concatenated (21 synthetic + c40 = 22 conditions), ECE 15 equal-width bins, selective AUC keeps the most confident 80%/90% by |p-0.5|. GenD with video-level bootstrap CI; Xception rows point estimates.

| model | frames | ECE | selective AUC 80% | selective AUC 90% | AUC (all frames) |
|---|---|---|---|---|---|
| GenD (public) | 492488 | 0.0867 [0.0718, 0.1079] | 0.8850 [0.8680, 0.8996] | 0.8694 [0.8524, 0.8847] | 0.8521 |
| base_xception | 492488 | 0.1129 | 0.8532 | 0.8298 | 0.8061 |
| augft_xception | 492488 | 0.1062 | 0.9507 | 0.9369 | 0.9208 |
| lumen_xception | 492488 | 0.0987 | 0.9189 | 0.9022 | 0.8840 |

## Cost on the RTX 3060 (12 GB)

- gpu: NVIDIA GeForce RTX 3060
- weights_fp32_gb: 1.13
- inference_fp16: [{'batch': 32, 'img_per_s': 104.1, 'peak_alloc_gb': 2.05, 'peak_reserved_gb': 2.15}, {'batch': 64, 'img_per_s': 106.5, 'peak_alloc_gb': 2.39, 'peak_reserved_gb': 2.53}]
- ln_only_train_bf16: [{'batch': 32, 'trainable_params': 104450, 'img_per_s': 43.8, 'peak_alloc_gb': 7.99}, {'batch': 16, 'trainable_params': 104450, 'img_per_s': 42.9, 'peak_alloc_gb': 4.84}]
- epoch_estimate_h: {'frames': 184000, 'batch': 32, 'hours': 1.17, 'note': 'GPU fwd+bwd only; excludes data loading/degradation and validation'}

## Checks

Frame-set identity with base_xception per condition: clean: same frame keys as base_xception; jpeg_q90: same frame keys as base_xception; jpeg_q70: same frame keys as base_xception ... (full list in results/lumen2/gend_room_checks.txt)
