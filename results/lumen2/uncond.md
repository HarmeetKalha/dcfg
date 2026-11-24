# LUMEN-uncond control: LUMEN2 minus LUMEN-uncond (pre-declared metrics)

LUMEN-uncond = LUMEN2 with the per-image conditioning input of the DCT gate and the LayerNorm generator replaced by ONE learned constant vector (definition: docs/DECISIONS.md 2026-11-23); the sensor still trains with its damage loss. Same code, data, degradation stream, seed and schedule. All numbers frame AUC unless stated, FF++ c23 test + c40, GenD crops, identical frames/degradations for all three models. Paired differences, 95% video-level bootstrap CI (1000 resamples, one shared resample matrix for the three models). Win = CI entirely on the favourable side of 0 (ECE: lower is better).

Training: stop reason **patience**, best epoch 6 of 12 (val mean AUC 0.9268), parameters 1741744 LUMEN parts.

## The control question

Does LUMEN2 beat LUMEN-uncond on the metrics it won against GenD-aug?

- Synthetic-degraded mean AUC (17): LUMEN2 - uncond +0.0056 [+0.0011, +0.0097] -> **yes** (win)
- Heavy-damage mean AUC (jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy): LUMEN2 - uncond +0.0097 [+0.0037, +0.0159] -> **yes** (win)
- Video AUC, synthetic mean (17): LUMEN2 - uncond +0.0050 [+0.0006, +0.0089] -> **yes** (win)

**Verdict: LUMEN2 beats uncond on ALL its won metrics: the gain needs per-image conditioning.**

## Every pre-declared metric

| metric | role | GenD-aug | LUMEN-uncond | LUMEN2 | LUMEN2 - LUMEN-uncond | outcome | LUMEN-uncond - GenD-aug (context) |
|---|---|---|---|---|---|---|---|
| Synthetic-degraded mean AUC (17) | primary | 0.9170 [0.9033, 0.9320] | 0.9218 [0.9075, 0.9361] | 0.9275 [0.9136, 0.9413] | +0.0056 [+0.0011, +0.0097] | win | +0.0048 [-0.0005, +0.0099] (tie) |
| c40 AUC | secondary | 0.8570 [0.8324, 0.8801] | 0.8636 [0.8400, 0.8864] | 0.8517 [0.8282, 0.8740] | -0.0119 [-0.0224, -0.0015] | loss | +0.0066 [-0.0016, +0.0146] (tie) |
| Held-out mean AUC (WebP q50/q20, motion k7/k11) | secondary | 0.9159 [0.9024, 0.9314] | 0.9208 [0.9069, 0.9351] | 0.9101 [0.8945, 0.9250] | -0.0107 [-0.0184, -0.0032] | loss | +0.0048 [-0.0011, +0.0108] (tie) |
| ECE (15 bins, all degraded frames) | secondary | 0.1045 [0.0885, 0.1209] | 0.1025 [0.0860, 0.1198] | 0.1097 [0.0939, 0.1264] | +0.0073 [+0.0031, +0.0116] | loss | -0.0021 [-0.0065, +0.0024] (tie) |
| Selective AUC, 80% coverage | secondary | 0.9457 [0.9341, 0.9572] | 0.9509 [0.9390, 0.9620] | 0.9495 [0.9376, 0.9599] | -0.0014 [-0.0055, +0.0026] | tie | +0.0052 [+0.0009, +0.0096] (win) |
| Selective AUC, 90% coverage | secondary | 0.9312 [0.9186, 0.9445] | 0.9370 [0.9240, 0.9498] | 0.9360 [0.9232, 0.9483] | -0.0011 [-0.0056, +0.0033] | tie | +0.0059 [+0.0013, +0.0106] (win) |
| Worst-case AUC (min over 22 degraded conditions) | secondary | 0.8403 [0.8169, 0.8613] | 0.8573 [0.8329, 0.8756] | 0.8517 [0.8282, 0.8740] | -0.0056 [-0.0186, +0.0134] | tie | +0.0170 [+0.0034, +0.0266] (win) |
| Heavy-damage mean AUC (jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy) | secondary | 0.8775 [0.8604, 0.8961] | 0.8857 [0.8678, 0.9042] | 0.8954 [0.8785, 0.9126] | +0.0097 [+0.0037, +0.0159] | win | +0.0082 [+0.0010, +0.0153] (win) |
| Video AUC, synthetic mean (17) | secondary | 0.9438 [0.9306, 0.9577] | 0.9476 [0.9336, 0.9611] | 0.9526 [0.9393, 0.9646] | +0.0050 [+0.0006, +0.0089] | win | +0.0039 [-0.0016, +0.0090] (tie) |
| Video AUC, c40 | secondary | 0.9084 [0.8834, 0.9316] | 0.9132 [0.8896, 0.9351] | 0.9123 [0.8885, 0.9343] | -0.0009 [-0.0121, +0.0100] | tie | +0.0048 [-0.0043, +0.0143] (tie) |
| DFD under damage (frame AUC, mixed_heavy) | secondary | 0.6719 [0.6535, 0.6914] | 0.6605 [0.6402, 0.6817] | 0.6651 [0.6449, 0.6858] | +0.0046 [-0.0015, +0.0107] | tie | -0.0115 [-0.0206, -0.0024] (loss) |
| Clean c23 AUC | safety | 0.9730 [0.9646, 0.9807] | 0.9722 [0.9635, 0.9803] | 0.9759 [0.9674, 0.9831] | +0.0037 [-0.0004, +0.0078] | ok | -0.0008 [-0.0046, +0.0027] (ok) |
| Clean DFD AUC | safety | 0.8741 [0.8620, 0.8858] | 0.8722 [0.8597, 0.8840] | 0.8921 [0.8803, 0.9028] | +0.0199 [+0.0157, +0.0241] | ok | -0.0019 [-0.0066, +0.0024] (ok) |

Outcome for safety rows: ok unless the CI is entirely below 0. Selective AUC: win if either coverage wins.

## Validation curve (val mean AUC per epoch, 8 of 32 frames/video)

| epoch | LUMEN-uncond | LUMEN2 | GenD-aug |
|---|---|---|---|
| 1 | 0.9030 | 0.9076 | 0.8937 |
| 2 | 0.9163 | 0.9190 | 0.9023 |
| 3 | 0.9216 | 0.9232 | 0.9080 |
| 4 | 0.9256 | 0.9244 | 0.9121 |
| 5 | 0.9237 | 0.9276 | 0.9146 |
| 6 | 0.9268 | 0.9305 | 0.9164 |
| 7 | 0.9195 | 0.9269 | 0.9168 |
| 8 | 0.9233 | 0.9256 | 0.9175 |
| 9 | 0.9206 | 0.9261 | 0.9181 |
| 10 | 0.9212 | 0.9274 | 0.9188 |
| 11 | 0.9170 | 0.9238 | 0.9187 |
| 12 | 0.9185 | 0.9249 | 0.9197 |
| 13 | - | - | 0.9196 |
| 14 | - | - | 0.9199 |
| 15 | - | - | 0.9197 |
| 16 | - | - | 0.9199 |
| 17 | - | - | 0.9199 |
| 18 | - | - | 0.9198 |
