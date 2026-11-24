# STATUS FOR MEETING (auto-generated)

Generated Tue 24 Nov 10:47 IST by `ops/jobs/meeting_status.py`. Meeting: Tue 24 Nov 08:30 IST. All times IST.

Last commit: 25c272f after dfd_lumen2_uncond: DFD lumen2_uncond: frame AUC clean 0.8722, mixed_heavy 0.6605

## Queue and ETA

### Runner: LUMEN-uncond (alive: yes)
- `lumen2u_smoke`: done
- `lumen2_uncond_train`: done
- `uncond_eval`: done
- `dfd_lumen2_uncond`: done
- `uncond_report`: RUNNING: running; ETA Tue 24 Nov 10:57 IST

### Runner: meeting (alive: yes)
- `gate_heatmaps_lumen1`: done
- `sensor_train`: done
- `sensor_eval`: done
- `lumen_nogate_xception`: done
- `eval_lumen_nogate_xception`: done
- `gate_ablation`: done
- `lumen2_smoke`: done
- `lumen2_train`: done
- `lumen2_eval`: done
- `lumen2_bench`: done
- `lumen2_report`: done
- `gate_heatmaps_lumen2_collect`: done
- `gate_heatmaps_lumen2`: done
- `dfd_gend_public`: done
- `dfd_gend_aug`: done
- `dfd_lumen2`: done
- `dfd_base_xception`: done
- `dfd_augft_xception`: done
- `dfd_lumen_xception`: done
- `dfd_fixed_xception`: done
- `dfd_lumen_nogate_xception`: done
- `dfd_reports`: done
- `lumen2_final`: done

### Runner: background (DFD) (alive: NO - restart it if jobs are pending)
- `bg_dl_dfd_original`: done
- `bg_dl_dfd_fakes`: done
- `bg_recrop_dfd`: done
- `bg_dfb_pre_dfd`: done

LUMEN-uncond queue estimated finish (assuming the earliest patience stop): Tue 24 Nov 10:57 IST

## Finished results

### Step 1: LUMEN1 gate heatmaps (Xception) - done
Per-damage change of the gate weights vs clean (`results/gate/lumen1/`):
- jpeg_q10: attenuates (w down) relative to clean, mean change -0.299; strongest in the mid band (-0.379); luma (Y) -0.281 vs chroma (Cb, Cr) -0.309.
- blur_s2: opens up (w up) relative to clean, mean change +0.117; strongest in the low band (+0.133); luma (Y) +0.093 vs chroma (Cb, Cr) +0.128.
- blur_s4: opens up (w up) relative to clean, mean change +0.105; strongest in the low band (+0.149); luma (Y) +0.076 vs chroma (Cb, Cr) +0.120.
- noise_s10: attenuates (w down) relative to clean, mean change -0.385; strongest in the mid band (-0.481); luma (Y) -0.362 vs chroma (Cb, Cr) -0.397.
- noise_s20: attenuates (w down) relative to clean, mean change -0.488; strongest in the mid band (-0.573); luma (Y) -0.457 vs chroma (Cb, Cr) -0.503.
- resize_0.5: opens up (w up) relative to clean, mean change +0.063; strongest in the mid band (+0.083); luma (Y) +0.060 vs chroma (Cb, Cr) +0.065.
- resize_0.25: opens up (w up) relative to clean, mean change +0.065; strongest in the mid band (+0.080); luma (Y) +0.053 vs chroma (Cb, Cr) +0.070.
- mixed_heavy: attenuates (w down) relative to clean, mean change -0.491; strongest in the mid band (-0.579); luma (Y) -0.457 vs chroma (Cb, Cr) -0.508.
- c40: leaves unchanged relative to clean, mean change +0.004; strongest in the mid band (+0.007); luma (Y) +0.004 vs chroma (Cb, Cr) +0.005.
- webp_q20: opens up (w up) relative to clean, mean change +0.058; strongest in the mid band (+0.069); luma (Y) +0.060 vs chroma (Cb, Cr) +0.058.
- motion_k11: opens up (w up) relative to clean, mean change +0.076; strongest in the mid band (+0.092); luma (Y) +0.063 vs chroma (Cb, Cr) +0.082.

### Step 2: standalone damage sensor - done
772264 params. Presence F1 / strength Spearman per op: jpeg 0.99 / 0.97; blur 0.93 / 0.95; noise 0.98 / 0.96; resize 0.83 / 0.87
Full report: `results/sensor/standalone.md`.

### Earlier: LIVE-pretrained sensor (quality regression)
{"job_id": "sensor_live", "n_train": 930, "n_val": 116, "n_test": 116, "best_epoch": 66, "best_val_srcc": 0.6867950640064583, "epochs_run": 86}

### Step 3: gate ablation (Xception) - done
## Frame AUC (95% CI)
| model | clean | synthetic mean (17) | held-out mean (4) | c40 | heavy-damage mean (5) | worst case (min of 22 degraded) |
|---|---|---|---|---|---|---|
| Xception base | 0.9761 [0.9685, 0.9828] | 0.8175 [0.8017, 0.8317] | 0.8771 [0.8576, 0.8956] | 0.8046 [0.7770, 0.8324] | 0.6914 [0.6696, 0.7113] | 0.5405 [0.4986, 0.5801] |
| Fixed gate | 0.9651 [0.9547, 0.9742] | 0.8414 [0.8235, 0.8574] | 0.8745 [0.8538, 0.8946] | 0.8108 [0.7820, 0.8376] | 0.7235 [0.7024, 0.7438] | 0.6476 [0.6029, 0.6787] |
| LUMEN1 without gate | 0.9747 [0.9657, 0.9823] | 0.8891 [0.8726, 0.9045] | 0.8852 [0.8673, 0.9031] | 0.8335 [0.8086, 0.8587] | 0.8137 [0.7931, 0.8345] | 0.7642 [0.7352, 0.7881] |
| LUMEN1 | 0.9742 [0.9647, 0.9816] | 0.8995 [0.8833, 0.9151] | 0.8871 [0.8687, 0.9036] | 0.8265 [0.8003, 0.8516] | 0.8299 [0.8085, 0.8515] | 0.7722 [0.7405, 0.7998] |
| augft (full fine-tune) | 0.9743 [0.9656, 0.9820] | 0.9286 [0.9159, 0.9410] | 0.9174 [0.9009, 0.9320] | 0.8722 [0.8511, 0.8932] | 0.8820 [0.8650, 0.8986] | 0.8368 [0.8122, 0.8595] |
## Video AUC (95% CI)
| model | clean | synthetic mean (17) | held-out mean (4) | c40 | heavy-damage mean (5) | worst case (min of 22 degraded) |
|---|---|---|---|---|---|---|
| Xception base | 0.9895 [0.9832, 0.9946] | 0.8555 [0.8390, 0.8699] | 0.9223 [0.9029, 0.9399] | 0.8650 [0.8349, 0.8933] | 0.7357 [0.7100, 0.7597] | 0.5442 [0.4896, 0.5969] |
Full table + paired differences: `results/gate/ablation_xception.md`.

### Xception frame AUC, point estimates
| run | clean | synthetic mean | held-out | c40 | heavy (5) |
|---|---|---|---|---|---|
| base_xception | 0.976 | 0.818 | 0.877 | 0.805 | 0.691 |
| fixed_xception | 0.965 | 0.841 | 0.874 | 0.811 | 0.724 |
| lumen_nogate_xception | 0.975 | 0.889 | 0.885 | 0.833 | 0.814 |
| lumen_xception | 0.974 | 0.899 | 0.887 | 0.826 | 0.830 |
| augft_xception | 0.974 | 0.929 | 0.917 | 0.872 | 0.882 |

### LUMEN-uncond control (final)
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

## LUMEN-uncond at Tue 24 Nov 07:30 IST (snapshot, training NOT stopped)

- Current: epoch 12 in progress (step 16289, 11 epochs completed)
- Best epoch so far: 6 (val mean AUC 0.9268)
- LUMEN2 best: epoch 6 (0.9305); GenD-aug best: epoch 16 (0.9199)

Validation mean AUC (8 of 32 frames/video, 6 val conditions) at the same epoch:

| epoch | LUMEN-uncond | LUMEN2 | GenD-aug | uncond - LUMEN2 |
|---|---|---|---|---|
| 1 | 0.9030 | 0.9076 | 0.8937 | -0.0046 |
| 2 | 0.9163 | 0.9190 | 0.9023 | -0.0026 |
| 3 | 0.9216 | 0.9232 | 0.9080 | -0.0016 |
| 4 | 0.9256 | 0.9244 | 0.9121 | +0.0012 |
| 5 | 0.9237 | 0.9276 | 0.9146 | -0.0039 |
| 6 | 0.9268 | 0.9305 | 0.9164 | -0.0037 |
| 7 | 0.9195 | 0.9269 | 0.9168 | -0.0074 |
| 8 | 0.9233 | 0.9256 | 0.9175 | -0.0024 |
| 9 | 0.9206 | 0.9261 | 0.9181 | -0.0055 |
| 10 | 0.9212 | 0.9274 | 0.9188 | -0.0062 |
| 11 | 0.9170 | 0.9238 | 0.9187 | -0.0068 |
| 12 | - | 0.9249 | 0.9197 | - |
| 13 | - | - | 0.9196 | - |
| 14 | - | - | 0.9199 | - |
| 15 | - | - | 0.9197 | - |
| 16 | - | - | 0.9199 | - |
| 17 | - | - | 0.9199 | - |
| 18 | - | - | 0.9198 | - |

### LUMEN-uncond smoke test
# LUMEN2 smoke test
Verdict: **PASS**
| check | result |
|---|---|
| step-0 LUMEN2 == GenD | max |logit diff| 6.03e-04, max |embedding diff| 1.15e-04, w all 1: True -> ok |
| damaged-image stream == GenD-aug | 224 view and 256 view hashes equal in 2 slices; 25830 record-order hashes vs runs/gendaug_train/batches.log, 0 mismatches -> ok |
| 300 optimiser steps (1200 micro-steps) | main loss -1.3391 -> -1.4349, sensor loss 0.7629 -> 0.5125 (first -> last quarter), NaN steps 0, U/V abs sums 533/552 -> ok |
| resume | resumed True, batch hashes equal True, max |param diff| 4.93e-04, BN buffers (max relative diff) 4.08e-03 -> ok |
| time | 36.3 img/s -> epoch train 84.3 min + val 6.3 min; 18 epochs = **27.2 h** (limit 30 h); peak GPU 8486 MB |
## Parameters per part
| part | params |
|---|---|

### LUMEN2 track: GenD (public) / GenD-aug
# GenD-aug (LUMEN2 baseline): GenD fine-tuned with its own recipe on our degraded stream
> **FLAG: the run hit the 18-epoch cap while validation was still improving** (best epoch 17 of 18). GenD-aug may be under-trained relative to a longer schedule.
Frame/video AUC on the FF++ c23 test split (+ c40), 95% video-level bootstrap CI (1000 resamples, one shared resample matrix: aggregates are joint and every difference is paired). Same GenD crops, degradation code, seeds and frame keys as `results/lumen2/gend_public`. Sources: results/lumen2/gend_public, results/lumen2/gend_aug, results/raw (Xception).
## Frame AUC [95% CI]
| model | clean | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | 0.9601 [0.9496, 0.9702] | 0.8680 [0.8503, 0.8864] | 0.8695 [0.8516, 0.8879] | 0.8010 [0.7727, 0.8280] | 0.8395 [0.8117, 0.8663] | 0.8211 [0.7925, 0.8485] | 0.8049 [0.7731, 0.8329] | 0.7604 [0.7261, 0.7927] | 0.7882 [0.7593, 0.8154] | 0.7845 [0.7555, 0.8130] |
| GenD-aug | 0.9730 [0.9644, 0.9804] | 0.9170 [0.9033, 0.9320] | 0.9159 [0.9024, 0.9314] | 0.8570 [0.8303, 0.8817] | 0.9023 [0.8837, 0.9182] | 0.8884 [0.8684, 0.9066] | 0.8999 [0.8810, 0.9179] | 0.8738 [0.8519, 0.8945] | 0.8403 [0.8165, 0.8623] | 0.8492 [0.8261, 0.8715] |

### LUMEN2
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

### LUMEN2 smoke test
# LUMEN2 smoke test
Verdict: **NO PASS (training not started)**
| check | result |
|---|---|
| step-0 LUMEN2 == GenD | max |logit diff| 6.03e-04, max |embedding diff| 1.15e-04, w all 1: True -> ok |
| damaged-image stream == GenD-aug | 224 view and 256 view hashes equal in 2 slices; 25830 record-order hashes vs runs/gendaug_train/batches.log, 0 mismatches -> ok |
| 300 optimiser steps (1200 micro-steps) | main loss -1.3488 -> -1.4377, sensor loss 0.7782 -> 0.7268 (first -> last quarter), NaN steps 0, U/V abs sums 407/413 -> ok |
| resume | resumed True, batch hashes equal True, max |param diff| 3.62e-04, BN buffers (max relative diff) 7.36e-01 -> FAIL |
| time | 37.6 img/s -> epoch train 81.4 min + val 6.4 min; 18 epochs = **26.3 h** (limit 30 h); peak GPU 8486 MB |
## Parameters per part
| part | params |
|---|---|

### DFD (GenD public, clean)
frame AUC 0.8924, video AUC 0.9349

