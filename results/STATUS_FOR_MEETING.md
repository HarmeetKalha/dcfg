# STATUS FOR MEETING (auto-generated)

Generated Sun 22 Nov 08:59 IST by `ops/jobs/meeting_status.py`. Meeting: Tue 24 Nov 08:30 IST. All times IST.

Last commit: 9b4de81 after lumen2_report: LUMEN2 eval report written

## Queue and ETA

### Runner: meeting (alive: yes)
- `gate_heatmaps_lumen1`: done
- `sensor_train`: done
- `sensor_eval`: done
- `lumen_nogate_xception`: done
- `eval_lumen_nogate_xception`: done
- `gate_ablation`: done
- `lumen2_smoke`: done
- `lumen2_train`: FAILED
- `lumen2_eval`: FAILED
- `lumen2_bench`: FAILED
- `lumen2_report`: FAILED
- `gate_heatmaps_lumen2_collect`: FAILED
- `gate_heatmaps_lumen2`: FAILED
- `dfd_gend_public`: done
- `dfd_gend_aug`: pending, ETA Sun 22 Nov 09:59 IST
- `dfd_lumen2`: pending, ETA Sun 22 Nov 10:59 IST
- `dfd_base_xception`: pending, ETA Sun 22 Nov 11:59 IST
- `dfd_augft_xception`: pending, ETA Sun 22 Nov 12:59 IST
- `dfd_lumen_xception`: pending, ETA Sun 22 Nov 13:59 IST
- `dfd_fixed_xception`: pending, ETA Sun 22 Nov 14:59 IST
- `dfd_lumen_nogate_xception`: pending, ETA Sun 22 Nov 15:59 IST
- `dfd_reports`: pending, ETA Sun 22 Nov 16:59 IST
- `lumen2_final`: pending, ETA Sun 22 Nov 17:59 IST

### Runner: background (DFD) (alive: yes)
- `bg_dl_dfd_original`: done
- `bg_dl_dfd_fakes`: done
- `bg_recrop_dfd`: done
- `bg_dfb_pre_dfd`: RUNNING: running; ETA Sun 22 Nov 09:29 IST

Meeting-queue estimated finish (upper bound for LUMEN1-no-gate: all 30 epochs, early stopping may end sooner): Sun 22 Nov 17:59 IST

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

### LUMEN2 track: GenD (public) / GenD-aug
# GenD-aug (LUMEN2 baseline): GenD fine-tuned with its own recipe on our degraded stream
> **FLAG: the run hit the 18-epoch cap while validation was still improving** (best epoch 17 of 18). GenD-aug may be under-trained relative to a longer schedule.
Frame/video AUC on the FF++ c23 test split (+ c40), 95% video-level bootstrap CI (1000 resamples, one shared resample matrix: aggregates are joint and every difference is paired). Same GenD crops, degradation code, seeds and frame keys as `results/lumen2/gend_public`. Sources: results/lumen2/gend_public, results/lumen2/gend_aug, results/raw (Xception).
## Frame AUC [95% CI]
| model | clean | synth-degraded mean (17) | held-out mean (4) | c40 | noise_s15 | noise_s20 | blur_s3 | blur_s4 | jpeg_q10 | mixed_heavy |
|---|---|---|---|---|---|---|---|---|---|---|
| GenD (public) | 0.9601 [0.9496, 0.9702] | 0.8680 [0.8503, 0.8864] | 0.8695 [0.8516, 0.8879] | 0.8010 [0.7727, 0.8280] | 0.8395 [0.8117, 0.8663] | 0.8211 [0.7925, 0.8485] | 0.8049 [0.7731, 0.8329] | 0.7604 [0.7261, 0.7927] | 0.7882 [0.7593, 0.8154] | 0.7845 [0.7555, 0.8130] |
| GenD-aug | 0.9730 [0.9644, 0.9804] | 0.9170 [0.9033, 0.9320] | 0.9159 [0.9024, 0.9314] | 0.8570 [0.8303, 0.8817] | 0.9023 [0.8837, 0.9182] | 0.8884 [0.8684, 0.9066] | 0.8999 [0.8810, 0.9179] | 0.8738 [0.8519, 0.8945] | 0.8403 [0.8165, 0.8623] | 0.8492 [0.8261, 0.8715] |

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

