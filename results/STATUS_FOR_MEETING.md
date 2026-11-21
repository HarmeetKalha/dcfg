# STATUS FOR MEETING (auto-generated)

Generated Sat 21 Nov 19:15 IST by `ops/jobs/meeting_status.py`. Meeting: Tue 24 Nov 08:30 IST. All times IST.

Last commit: 3898f06 LUMEN2 metrics amendment (before any results): heavy-damage mean = synthetic types only; Mon

## Queue and ETA

### Runner: meeting (alive: yes)
- `gate_heatmaps_lumen1`: done
- `sensor_train`: pending, ETA Sat 21 Nov 21:20 IST
- `sensor_eval`: pending, ETA Sat 21 Nov 22:15 IST
- `lumen_nogate_xception`: pending, ETA Sun 22 Nov 09:45 IST
- `eval_lumen_nogate_xception`: pending, ETA Sun 22 Nov 10:50 IST
- `gate_ablation`: pending, ETA Sun 22 Nov 11:25 IST

### Runner: background (DFD) (alive: yes)
- `bg_dl_dfd_original`: done
- `bg_dl_dfd_fakes`: RUNNING: running; ETA Sat 21 Nov 19:45 IST
- `bg_recrop_dfd`: pending, ETA Sat 21 Nov 20:15 IST

Meeting-queue estimated finish (upper bound for LUMEN1-no-gate: all 30 epochs, early stopping may end sooner): Sun 22 Nov 11:55 IST

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

### Earlier: LIVE-pretrained sensor (quality regression)
{"job_id": "sensor_live", "n_train": 930, "n_val": 116, "n_test": 116, "best_epoch": 66, "best_val_srcc": 0.6867950640064583, "epochs_run": 86}

### Xception frame AUC, point estimates
| run | clean | synthetic mean | held-out | c40 | heavy (5) |
|---|---|---|---|---|---|
| base_xception | 0.976 | 0.818 | 0.877 | 0.805 | 0.691 |
| fixed_xception | 0.965 | 0.841 | 0.874 | 0.811 | 0.724 |
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

### DFD prep (background, CPU)
bg_dl_dfd_original: done
bg_dl_dfd_fakes: pending/running
bg_recrop_dfd: pending/running
bg_dfb_pre_dfd: pending/running

