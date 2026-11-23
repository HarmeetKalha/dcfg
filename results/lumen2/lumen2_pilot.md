# LUMEN2 pilot report

LUMEN2 = GenD-aug (GenD LayerNorm tuning on our degraded stream, the fair baseline) + the LUMEN parts (jointly trained damage sensor, DCT gate, degradation-conditioned LayerNorm). Everything else (data, degradation stream, schedule, seed) is identical. Fixed pre-declared metrics: docs/LUMEN2_BRIEF.md. Full detail: `lumen2_eval.md`, `gend_aug.md`.

## Main table (frame AUC, point estimates)

| model | clean | synthetic mean (17) | held-out (4) | c40 | heavy-damage (5) | worst-case (22) | DFD clean | DFD mixed_heavy |
|---|---|---|---|---|---|---|---|---|
| LUMEN2 | 0.9759 | 0.9275 | 0.9101 | 0.8517 | 0.8954 | 0.8517 | 0.8921 | 0.6651 |
| GenD-aug | 0.9730 | 0.9170 | 0.9159 | 0.8570 | 0.8775 | 0.8403 | 0.8741 | 0.6719 |
| GenD (public) | 0.9601 | 0.8680 | 0.8695 | 0.8010 | 0.8086 | 0.7604 | 0.8924 | 0.6557 |

## Pre-declared metrics: LUMEN2 minus GenD-aug (paired, 1000 video-level bootstrap resamples, 95% CI)

Outcome: win = CI entirely favourable; loss = entirely unfavourable; tie otherwise. Safety: ok unless CI entirely below 0.

| metric | role | LUMEN2 - GenD-aug [95% CI] | outcome |
|---|---|---|---|
| Synthetic-degraded mean AUC (17) | primary | +0.0105 [+0.0049, +0.0158] | win |
| c40 AUC | secondary | -0.0053 [-0.0159, +0.0045] | tie |
| Held-out mean AUC (WebP q50/q20, motion k7/k11) | secondary | -0.0059 [-0.0133, +0.0009] | tie |
| ECE (15 bins, all degraded frames) | secondary | +0.0052 [+0.0003, +0.0098] | loss |
| Selective AUC, 80% coverage | secondary | +0.0038 [-0.0007, +0.0084] | tie |
| Selective AUC, 90% coverage | secondary | +0.0048 [-0.0002, +0.0099] | tie |
| Worst-case AUC (min over 22 degraded conditions) | secondary | +0.0114 [-0.0064, +0.0305] | tie |
| Heavy-damage mean AUC (jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy) | secondary | +0.0179 [+0.0104, +0.0255] | win |
| Video AUC, synthetic mean (17) | secondary | +0.0088 [+0.0037, +0.0141] | win |
| Video AUC, c40 | secondary | +0.0039 [-0.0072, +0.0152] | tie |
| DFD under damage (frame AUC, mixed_heavy) | secondary | -0.0068 [-0.0175, +0.0031] | tie |
| Clean c23 AUC | safety | +0.0029 [-0.0008, +0.0063] | ok |
| Clean DFD AUC | safety | +0.0180 [+0.0129, +0.0231] | ok |

Secondary + primary tally (selective counted per coverage): 3 win, 1 loss, 7 tie.

## Exploratory (not claims)

AP, EER and every single condition are in `lumen2_eval.md` (point estimates / paired per-condition CIs).

## Context: Xception results (DIFFERENT BACKBONE, not comparable as a head-to-head)

Frame AUC point estimates from results/raw (FF++ c23 test + c40, DFB crops) and results/dfd.

| model | clean | synthetic mean | held-out | c40 | heavy-damage | worst-case | DFD clean | DFD mixed_heavy |
|---|---|---|---|---|---|---|---|---|
| base (different backbone: Xception) | 0.9761 | 0.8175 | 0.8771 | 0.8046 | 0.6914 | 0.5405 | 0.8471 | 0.5277 |
| augft (full fine-tune) (different backbone: Xception) | 0.9743 | 0.9286 | 0.9174 | 0.8722 | 0.8820 | 0.8368 | 0.7733 | 0.5490 |
| LUMEN1 (different backbone: Xception) | 0.9742 | 0.8995 | 0.8871 | 0.8265 | 0.8299 | 0.7722 | 0.8111 | 0.5536 |
| fixed gate (different backbone: Xception) | 0.9651 | 0.8414 | 0.8745 | 0.8108 | 0.7235 | 0.6476 | 0.8159 | 0.5243 |
| LUMEN1 without gate (different backbone: Xception) | 0.9747 | 0.8891 | 0.8852 | 0.8335 | 0.8137 | 0.7642 | 0.8259 | 0.5420 |

### DFD (DeepFakeDetection, FF++ release; never used in training)

All DFD videos (real actors + DFD fakes), 32 frames per video, video-level bootstrap 95% CI (1000 resamples). `mixed_heavy` uses the same degradation and seeding rule as on FF++ (per-frame key). GenD models use GenD RetinaFace crops; Xception models use DeepfakeBench crops.

| model | frame AUC clean | video AUC clean | frame AUC mixed_heavy | video AUC mixed_heavy |
|---|---|---|---|---|
| LUMEN2 | 0.8921 [0.8819, 0.9019] | 0.9369 [0.9263, 0.9462] | 0.6651 [0.6459, 0.6839] | 0.7093 [0.6851, 0.7337] |
| GenD-aug | 0.8741 [0.8630, 0.8847] | 0.9175 [0.9059, 0.9287] | 0.6719 [0.6521, 0.6904] | 0.7188 [0.6937, 0.7426] |
| GenD (public) | 0.8924 [0.8831, 0.9016] | 0.9349 [0.9252, 0.9438] | 0.6557 [0.6349, 0.6753] | 0.6938 [0.6672, 0.7185] |
| base (Xception) | 0.8471 [0.8340, 0.8605] | 0.8964 [0.8808, 0.9113] | 0.5277 [0.5117, 0.5456] | 0.5694 [0.5413, 0.6013] |
| augft (full fine-tune) (Xception) | 0.7733 [0.7552, 0.7918] | 0.8194 [0.7976, 0.8402] | 0.5490 [0.5266, 0.5725] | 0.5652 [0.5337, 0.5951] |
| LUMEN1 (Xception) | 0.8111 [0.7950, 0.8262] | 0.8581 [0.8382, 0.8761] | 0.5536 [0.5312, 0.5747] | 0.5696 [0.5367, 0.5996] |
| fixed gate (Xception) | 0.8159 [0.8020, 0.8309] | 0.8726 [0.8562, 0.8893] | 0.5243 [0.5077, 0.5413] | 0.5521 [0.5224, 0.5839] |
| LUMEN1 without gate (Xception) | 0.8259 [0.8093, 0.8406] | 0.8722 [0.8520, 0.8903] | 0.5420 [0.5193, 0.5639] | 0.5516 [0.5198, 0.5821] |

## Damage sensor

Accuracy per op, per condition and on c40 / WebP / motion blur: `lumen2_eval.md` (section Damage sensor inside LUMEN2), next to the standalone sensor (`results/sensor/standalone.md`).

## Cost

- LUMEN parameters 1,741,424; trainable file 7.11 MB (GenD-aug 0.43 MB)
- inference 11.71 ms/image vs 10.57 (x1.11)
- training lumen2: best epoch 6, stop epoch 12 (patience), 17.4 h
- training gend_aug: best epoch 17, stop epoch 18 (epoch_cap), 22.8 h

## Verdict

- Primary (synthetic-degraded mean): win; wins: synth_mean, heavy_mean, video_synth_mean
- Safety: clean c23 ok, clean DFD ok
- 'Worth publishing' rule (wins primary or >= 1 secondary AND passes both safety checks): **PASSES**.
- The LUMEN-uncond control (same architecture, constant conditioning input) is NOT run: the rule additionally requires LUMEN2 to beat it on the won metric. Running it is justified only if the rule passes.
