# Design selection study (5 epochs, validation)

**PILOT / appendix material. VALIDATION split only; not paper main numbers; no design is selected here (the choice is the user's).** Xception detector, seed 42, batch 32, same TrainDegradation, 5 epochs of the real 30-epoch LR schedule. Models: base = frozen detector; fixed / dcfg = front-end + frozen detector; augft = detector fine-tuned; jointft = dcfg module + detector fine-tuned (lr 1e-3 / 1e-4 / 2e-5); dcfgcons = dcfg + consistency loss (frozen detector); dcfgfilm = dcfgcons + FiLM at entry/block7/block11/exit; dcfgfilmlite = FiLM at entry/exit only (small MLP).

700 val videos, 22354 frames per condition; 1000 video-level bootstrap replicates (95% percentile CI, one resample shared by all models/conditions, so deltas are paired).

## 1. Cost: trainable parameters and inference overhead

| model | trainable params added | % of Xception | ms/img b32 (total / added) | ms/img b1 (total / added) | inference graph measured as |
|---|---|---|---|---|---|
| base | 0 | 0.00% | 2.65 / +0.00 | 3.78 / +0.00 | base |
| fixed | 192 | 0.00% | 2.75 / +0.10 | 4.29 / +0.51 | fixed |
| augft | 21,861,162 (whole detector counted as trainable) | 100.00% | 2.65 / +0.00 | 3.78 / +0.00 | base |
| dcfg | 394,369 | 1.80% | 3.11 / +0.46 | 5.14 / +1.35 | dcfg |
| jointft | 22,255,531 (whole detector counted as trainable) | 101.80% | 3.11 / +0.46 | 5.14 / +1.35 | dcfg |
| dcfgcons | 394,369 | 1.80% | 3.11 / +0.46 | 5.14 / +1.35 | dcfg |
| dcfgfilm | 2,602,641 | 11.91% | 3.16 / +0.51 | 5.29 / +1.50 | film |
| dcfgfilmlite | 763,505 | 3.49% | 3.14 / +0.49 | 5.27 / +1.48 | filmlite |

Trainable params ADDED to the frozen Xception (sensor counted as trainable: it is fine-tuned at lr 1e-4); for augft/jointft the whole detector counts as trainable. Overhead: inference, fp32 detector, no_grad, CUDA events, median; 'added' = pipeline minus the plain detector (front-end, FiLM MLP and hooks included). Training cost of dcfgcons/dcfgfilm/dcfgfilmlite additionally includes one extra no-grad detector forward per step (the teacher).

## 2. Frame AUC [95% CI] (validation)

| model | clean | d clean vs base | noise_s10 | blur_s3 | webp_q20 | mixed_heavy | mean(degraded) | c40 |
|---|---|---|---|---|---|---|---|---|
| base | 0.9764 [0.9684, 0.9832] | - | 0.7192 [0.6850, 0.7552] | 0.7827 [0.7474, 0.8147] | 0.8418 [0.8182, 0.8640] | 0.6932 [0.6635, 0.7196] | 0.7592 [0.7387, 0.7787] | 0.7966 [0.7668, 0.8268] |
| fixed | 0.9669 [0.9569, 0.9753] | -0.0095 [-0.0139, -0.0050] * | 0.8292 [0.8006, 0.8532] | 0.8084 [0.7752, 0.8373] | 0.8438 [0.8191, 0.8654] | 0.7144 [0.6867, 0.7394] | 0.7989 [0.7765, 0.8180] | 0.8000 [0.7706, 0.8283] |
| augft | 0.9754 [0.9661, 0.9831] | -0.0010 [-0.0058, +0.0037] | 0.9372 [0.9205, 0.9512] | 0.9086 [0.8864, 0.9279] | 0.8921 [0.8708, 0.9121] | 0.8576 [0.8335, 0.8792] | 0.8989 [0.8793, 0.9171] | 0.8650 [0.8419, 0.8881] |
| dcfg | 0.9762 [0.9676, 0.9831] | -0.0002 [-0.0022, +0.0017] | 0.8737 [0.8487, 0.8955] | 0.8097 [0.7763, 0.8413] | 0.8408 [0.8161, 0.8640] | 0.7355 [0.7076, 0.7612] | 0.8149 [0.7951, 0.8341] | 0.7921 [0.7616, 0.8222] |
| jointft | 0.9741 [0.9643, 0.9819] | -0.0023 [-0.0075, +0.0026] | 0.9382 [0.9219, 0.9521] | 0.9125 [0.8916, 0.9315] | 0.8911 [0.8697, 0.9112] | 0.8635 [0.8393, 0.8851] | 0.9013 [0.8818, 0.9186] | 0.8624 [0.8384, 0.8859] |
| dcfgcons | 0.9765 [0.9684, 0.9833] | +0.0001 [-0.0008, +0.0011] | 0.8763 [0.8503, 0.8983] | 0.8097 [0.7781, 0.8414] | 0.8429 [0.8191, 0.8644] | 0.7340 [0.7065, 0.7603] | 0.8157 [0.7969, 0.8346] | 0.7962 [0.7671, 0.8260] |
| dcfgfilm | 0.9760 [0.9681, 0.9830] | -0.0004 [-0.0029, +0.0021] | 0.9108 [0.8924, 0.9270] | 0.8734 [0.8480, 0.8967] | 0.8506 [0.8278, 0.8722] | 0.8009 [0.7759, 0.8241] | 0.8589 [0.8398, 0.8764] | 0.8213 [0.7931, 0.8489] |
| dcfgfilmlite | 0.9763 [0.9679, 0.9831] | -0.0001 [-0.0020, +0.0020] | 0.9105 [0.8912, 0.9272] | 0.8785 [0.8558, 0.8994] | 0.8503 [0.8255, 0.8722] | 0.8004 [0.7761, 0.8236] | 0.8599 [0.8409, 0.8773] | 0.8166 [0.7881, 0.8453] |

Degraded mean = mean over noise_s10, blur_s3, webp_q20, mixed_heavy; c40 = real H.264 compression (own video set), not in the mean. 'd clean vs base' is a paired delta; * = CI excludes 0.

## 3. Paired bootstrap delta-AUC of each new variant vs dcfg, augft, jointft; `*` = CI excludes 0

| new variant vs | noise_s10 | blur_s3 | webp_q20 | mixed_heavy | mean(degraded) | c40 |
|---|---|---|---|---|---|---|
| jointft - dcfg | +0.0645 [+0.0474, +0.0832] * | +0.1028 [+0.0754, +0.1302] * | +0.0503 [+0.0353, +0.0667] * | +0.1280 [+0.1055, +0.1530] * | +0.0864 [+0.0710, +0.1017] * | +0.0702 [+0.0533, +0.0891] * |
| jointft - augft | +0.0010 [-0.0025, +0.0043] | +0.0039 [-0.0006, +0.0083] | -0.0010 [-0.0046, +0.0025] | +0.0059 [+0.0012, +0.0107] * | +0.0025 [-0.0005, +0.0054] | -0.0026 [-0.0060, +0.0009] |
| dcfgcons - dcfg | +0.0025 [-0.0027, +0.0084] | -0.0001 [-0.0076, +0.0066] | +0.0021 [-0.0026, +0.0070] | -0.0014 [-0.0104, +0.0071] | +0.0008 [-0.0031, +0.0046] | +0.0040 [+0.0009, +0.0071] * |
| dcfgcons - augft | -0.0609 [-0.0793, -0.0447] * | -0.0990 [-0.1258, -0.0720] * | -0.0492 [-0.0646, -0.0346] * | -0.1235 [-0.1500, -0.0995] * | -0.0832 [-0.0980, -0.0679] * | -0.0688 [-0.0867, -0.0520] * |
| dcfgcons - jointft | -0.0619 [-0.0811, -0.0453] * | -0.1029 [-0.1301, -0.0763] * | -0.0483 [-0.0633, -0.0333] * | -0.1295 [-0.1554, -0.1041] * | -0.0856 [-0.1012, -0.0702] * | -0.0662 [-0.0845, -0.0499] * |
| dcfgfilm - dcfg | +0.0371 [+0.0224, +0.0540] * | +0.0636 [+0.0390, +0.0890] * | +0.0098 [+0.0016, +0.0181] * | +0.0654 [+0.0462, +0.0843] * | +0.0440 [+0.0331, +0.0549] * | +0.0292 [+0.0201, +0.0380] * |
| dcfgfilm - augft | -0.0264 [-0.0372, -0.0160] * | -0.0352 [-0.0534, -0.0187] * | -0.0415 [-0.0568, -0.0265] * | -0.0567 [-0.0754, -0.0389] * | -0.0400 [-0.0520, -0.0277] * | -0.0436 [-0.0583, -0.0297] * |
| dcfgfilm - jointft | -0.0274 [-0.0380, -0.0164] * | -0.0391 [-0.0559, -0.0238] * | -0.0405 [-0.0552, -0.0258] * | -0.0626 [-0.0810, -0.0452] * | -0.0424 [-0.0544, -0.0307] * | -0.0410 [-0.0557, -0.0274] * |
| dcfgfilmlite - dcfg | +0.0367 [+0.0245, +0.0506] * | +0.0687 [+0.0428, +0.0950] * | +0.0095 [+0.0017, +0.0171] * | +0.0650 [+0.0452, +0.0851] * | +0.0450 [+0.0344, +0.0549] * | +0.0245 [+0.0163, +0.0334] * |
| dcfgfilmlite - augft | -0.0267 [-0.0384, -0.0164] * | -0.0302 [-0.0470, -0.0142] * | -0.0418 [-0.0570, -0.0265] * | -0.0571 [-0.0756, -0.0395] * | -0.0390 [-0.0509, -0.0271] * | -0.0484 [-0.0638, -0.0345] * |
| dcfgfilmlite - jointft | -0.0278 [-0.0393, -0.0170] * | -0.0341 [-0.0503, -0.0185] * | -0.0408 [-0.0555, -0.0255] * | -0.0631 [-0.0824, -0.0464] * | -0.0414 [-0.0535, -0.0294] * | -0.0458 [-0.0617, -0.0321] * |

## 4. Highlighted: blur_s3, webp_q20, c40

| model | blur_s3 | webp_q20 | c40 |
|---|---|---|---|
| **base** | 0.7827 [0.7474, 0.8147] | 0.8418 [0.8182, 0.8640] | 0.7966 [0.7668, 0.8268] |
| **fixed** | 0.8084 [0.7752, 0.8373] | 0.8438 [0.8191, 0.8654] | 0.8000 [0.7706, 0.8283] |
| **augft** | 0.9086 [0.8864, 0.9279] | 0.8921 [0.8708, 0.9121] | 0.8650 [0.8419, 0.8881] |
| **dcfg** | 0.8097 [0.7763, 0.8413] | 0.8408 [0.8161, 0.8640] | 0.7921 [0.7616, 0.8222] |
| **jointft** | 0.9125 [0.8916, 0.9315] | 0.8911 [0.8697, 0.9112] | 0.8624 [0.8384, 0.8859] |
| **dcfgcons** | 0.8097 [0.7781, 0.8414] | 0.8429 [0.8191, 0.8644] | 0.7962 [0.7671, 0.8260] |
| **dcfgfilm** | 0.8734 [0.8480, 0.8967] | 0.8506 [0.8278, 0.8722] | 0.8213 [0.7931, 0.8489] |
| **dcfgfilmlite** | 0.8785 [0.8558, 0.8994] | 0.8503 [0.8255, 0.8722] | 0.8166 [0.7881, 0.8453] |

| new variant vs | blur_s3 | webp_q20 | c40 |
|---|---|---|---|
| jointft - dcfg | +0.1028 [+0.0754, +0.1302] * | +0.0503 [+0.0353, +0.0667] * | +0.0702 [+0.0533, +0.0891] * |
| jointft - augft | +0.0039 [-0.0006, +0.0083] | -0.0010 [-0.0046, +0.0025] | -0.0026 [-0.0060, +0.0009] |
| dcfgcons - dcfg | -0.0001 [-0.0076, +0.0066] | +0.0021 [-0.0026, +0.0070] | +0.0040 [+0.0009, +0.0071] * |
| dcfgcons - augft | -0.0990 [-0.1258, -0.0720] * | -0.0492 [-0.0646, -0.0346] * | -0.0688 [-0.0867, -0.0520] * |
| dcfgcons - jointft | -0.1029 [-0.1301, -0.0763] * | -0.0483 [-0.0633, -0.0333] * | -0.0662 [-0.0845, -0.0499] * |
| dcfgfilm - dcfg | +0.0636 [+0.0390, +0.0890] * | +0.0098 [+0.0016, +0.0181] * | +0.0292 [+0.0201, +0.0380] * |
| dcfgfilm - augft | -0.0352 [-0.0534, -0.0187] * | -0.0415 [-0.0568, -0.0265] * | -0.0436 [-0.0583, -0.0297] * |
| dcfgfilm - jointft | -0.0391 [-0.0559, -0.0238] * | -0.0405 [-0.0552, -0.0258] * | -0.0410 [-0.0557, -0.0274] * |
| dcfgfilmlite - dcfg | +0.0687 [+0.0428, +0.0950] * | +0.0095 [+0.0017, +0.0171] * | +0.0245 [+0.0163, +0.0334] * |
| dcfgfilmlite - augft | -0.0302 [-0.0470, -0.0142] * | -0.0418 [-0.0570, -0.0265] * | -0.0484 [-0.0638, -0.0345] * |
| dcfgfilmlite - jointft | -0.0341 [-0.0503, -0.0185] * | -0.0408 [-0.0555, -0.0255] * | -0.0458 [-0.0617, -0.0321] * |

## 5. Resource summary per run (training phase only)

| run | GPU util mean/median % (train) | data-wait share | CPU util mean % | peak GPU mem MiB | max temp C | wall train/val (h) | CPU-bound? |
|---|---|---|---|---|---|---|---|
| pilot_fixed_xception | 98.6 / 99.0 | 0.1% | 42.5 | 5689 | 82 | 0.87 / 0.29 | no |
| pilot_augft_xception | 98.2 / 99.0 | 0.1% | 36.1 | 5360 | 81 | 1.22 / 0.26 | no |
| pilot_dcfg_xception | 98.7 / 99.0 | 0.1% | 36.0 | 6018 | 81 | 1.20 / 0.37 | no |
| pilot_jointft_xception | 98.9 / 99.0 | 0.1% | 31.8 | 6378 | 81 | 1.56 / 0.35 | no |
| pilot_dcfgcons_xception | 98.9 / 99.0 | 0.1% | 33.7 | 6438 | 82 | 1.49 / 0.35 | no |
| pilot_dcfgfilm_xception | 98.9 / 99.0 | 0.1% | 33.4 | 6531 | 82 | 1.51 / 0.35 | no |
| pilot_dcfgfilmlite_xception | 98.9 / 99.0 | 0.1% | 33.7 | 6512 | 81 | 1.50 / 0.35 | no |

Flag rule: CPU-bound if mean GPU util < 85% or data-wait share > 15%. No run flagged.

## 6. Measured epoch times (for the matrix ETA)

| run | train min/epoch | val min/epoch | total min/epoch | epochs |
|---|---|---|---|---|
| pilot_fixed_xception | 10.5 | 3.5 | 14.0 | 5 |
| pilot_augft_xception | 14.6 | 3.2 | 17.8 | 5 |
| pilot_dcfg_xception | 14.4 | 4.4 | 18.8 | 5 |
| pilot_jointft_xception | 18.8 | 4.2 | 22.9 | 5 |
| pilot_dcfgcons_xception | 17.9 | 4.2 | 22.1 | 5 |
| pilot_dcfgfilm_xception | 18.1 | 4.2 | 22.3 | 5 |
| pilot_dcfgfilmlite_xception | 18.0 | 4.2 | 22.2 | 5 |

From resources_summary.json (wall_train_s / wall_val_s) divided by epochs_run.

## 7. Total wall time

runner log span (first design-selection job start to report start): 8.00 h; sum of the training total_time_s of the new runs (done.json): 7.46 h

