# LUMEN2 brief (change of direction, 2026-11-19)

## Why the CNN version of LUMEN will not be published
Xception, FF++ c23 test, frame AUC (results/raw, group 1 runs):

| | synthetic-degraded mean | held-out | c40 |
|---|---|---|---|
| augft | 0.929 | 0.917 | 0.872 |
| LUMEN | 0.899 | 0.887 | 0.826 |

LUMEN beats base and fixed beyond the CIs and keeps clean AUC, but full fine-tuning (augft) beats LUMEN on every degraded
aggregate. augft also trained faster (best at epoch 3, 3.0 h, vs LUMEN epoch 18, 9.3 h), and LUMEN needs the detector's full
weights anyway (white-box gradients). On small CNN detectors LUMEN has no strong advantage; the CNN version will not be
published. The CNN matrix is PAUSED (not deleted); see PROGRESS.md.

## New direction: LUMEN2
Modern deepfake detectors build on large pretrained foundation models (CLIP ViT-L/14). There, full fine-tuning generalizes
badly: GenD (WACV 2026, github.com/yermandy/GenD) reports ~57.9 mean cross-dataset AUROC for full fine-tuning vs ~91 for
LayerNorm-only tuning, so LayerNorm tuning is the standard. These detectors still lose accuracy under degradation.

**LUMEN2** = CLIP ViT-L/14 with GenD-style LayerNorm tuning, where a degradation signal adjusts the LayerNorm scale and
shift per image (FiLM on LayerNorm, zero-initialized, so it starts as plain GenD), plus the optional DCT gate at the input.
Its opponent is GenD-style LayerNorm tuning trained on the same degraded data, NOT full fine-tuning.

## Decision rules (fixed 2026-11-19, before any LUMEN2 code or training)
Comparisons (all CLIP ViT-L/14, same data, same degradation stream, same schedule):
- **GenD**: public weights, clean-trained (reference only).
- **GenD-aug**: GenD retrained by us with our degradation stream. **The fair baseline.**
- **GenD-aug + LUMEN**: degradation-conditioned LayerNorm (+ optional DCT gate).
- **GenD-aug + LUMEN-uncond**: same architecture and parameters, but the conditioning network gets a constant input
  instead of the degradation signal.

Every comparison: paired bootstrap (1,000 video-level resamples, 95% CI) of LUMEN minus GenD-aug.

| Metric | Definition | Role | Win when |
|---|---|---|---|
| Synthetic-degraded mean AUC | mean frame AUC over the 17 synthetic conditions | **Primary** | CI of the difference > 0 |
| c40 AUC | frame AUC on FF++ c40 test videos | Secondary | CI > 0 |
| Held-out mean AUC | mean over WebP q50/q20, motion blur k7/k11 | Secondary | CI > 0 |
| ECE | expected calibration error over all degraded test frames, 15 equal-width bins, predicted fake-probability vs label | Secondary | ECE lower and CI of the difference < 0 |
| Selective-prediction AUC | sort frames by confidence abs(p - 0.5), keep the most confident 80% and 90%, AUC on kept frames, all degraded conditions combined | Secondary | CI > 0 at 80% or 90% coverage |
| Clean AUC | frame AUC on clean c23 test | **Safety check** | must not drop (CI not entirely < 0) |
| DFD AUC | frame and video AUC on DeepFakeDetection (FF++ release), never used in training | **Safety check** | must not drop (CI not entirely < 0) |

**"Worth publishing" rule:** GenD-aug + LUMEN wins the primary metric OR at least one secondary metric, passes BOTH safety
checks, AND beats GenD-aug + LUMEN-uncond on the same metric it wins. Every metric is reported for every model regardless of
outcome; no metric is dropped after seeing results.

Helpers: src/dcfg/eval/metrics.py (ece, selective_auc, paired_bootstrap_metric), tests/test_metrics_calib.py.

## First step (2026-11-19): measure the room to improve, no training
Evaluate public GenD (`yermandy/GenD_CLIP_L_14`) on our 23 conditions (same degradation code/params/seeds). Gate: only if
GenD loses >= 0.03 AUC from its clean score on the synthetic-degraded mean OR on c40, download + preprocess DFD and
evaluate GenD on it; otherwise skip DFD and report that LUMEN2 has little room to improve.

## Gate result and next step (2026-11-19)
Public GenD, c23 test frame AUC: clean 0.960, synthetic-degraded mean 0.868 (-0.092), held-out 0.869, c40 0.801 (-0.159).
Gate PASSED. DFD paused (see PROGRESS.md).

**GenD-aug decisions (user, 2026-11-19):**
- GenD-aug starts from GenD's public weights (`yermandy/GenD_CLIP_L_14`), not plain CLIP.
- Any later LUMEN2 model starts from the same weights, with the same data, degradation stream and schedule.
- Whether we continue to LUMEN2 is decided by the user after seeing GenD-aug results; there is no automatic rule.

## FIXED training schedule for GenD-aug and every later LUMEN2 run (user decision, 2026-11-20)
- Training frames: exactly the augft training frames (FF++ c23 train, all frames per video, same epoch definition: all fake frames, reals re-cycled to 1:1), GenD RetinaFace crops, our PLAN 7.2 degradation applied to the 256 px crop before GenD's resize + normalisation.
- Validation: the same degraded val set as augft (clean, jpeg_q50, jpeg_q30, blur_s2, noise_s10, mixed_heavy), subsampled to 8 of the ~32 frames per video with `np.linspace` over the key-sorted frames (deterministic, hash recorded); selection metric = mean frame AUC over these conditions (as augft); val only.
- AdamW, cosine over the 18-epoch cap, max 18 epochs, patience 6, seed 42, GenD's recipe (LayerNorm + L2-normalised head, CE + 0.5 uniformity + 0.1 alignment, lr 3e-4, weight decay 0, bf16), effective batch 128 (4 x 32), min lr 1e-5, 1-epoch warm-up. Details and deviations: DECISIONS.md rows of 2026-11-20.
- Every run logs `stop_reason` ("patience" or "epoch_cap"), best epoch, stop epoch and the val curve; a run that hits the cap while still improving is flagged.

## LUMEN2 pre-declared metrics (fixed 2026-11-21, before any LUMEN2 code or training)
All metrics: paired difference LUMEN2 minus GenD-aug, same frames, 1,000 video-level bootstrap resamples (one shared
resample matrix), 95% CI. Frame AUC unless stated.

| Metric | Definition | Role | Win when |
|---|---|---|---|
| Synthetic-degraded mean AUC | mean frame AUC over the 17 synthetic conditions | **Primary** | CI > 0 |
| c40 AUC | FF++ c40 test | Secondary | CI > 0 |
| Held-out mean AUC | mean over webp_q50, webp_q20, motion_k7, motion_k11 | Secondary | CI > 0 |
| ECE | 15 equal-width bins, P(fake) vs label, all degraded test frames | Secondary | CI < 0 |
| Selective AUC 80% / 90% | most confident 80% / 90% by abs(p-0.5), all degraded frames | Secondary | CI > 0 at either |
| Worst-case AUC | min frame AUC over all 22 degraded conditions (17 synthetic + 4 held-out + c40); per resample, min over conditions computed for each model, then differenced | Secondary | CI > 0 |
| Heavy-damage mean AUC | mean frame AUC over exactly: jpeg_q10, blur_s4, noise_s20, resize_0.25, webp_q20, motion_k11, mixed_heavy (strongest level of each degradation type in the condition table, plus mixed-heavy; c40 excluded, reported on its own) | Secondary | CI > 0 |
| Video AUC, synthetic mean | video AUC (mean frame prob per video), mean over the 17 synthetic conditions | Secondary | CI > 0 |
| Video AUC, c40 | video AUC on c40 | Secondary | CI > 0 |
| DFD under damage | frame AUC on DFD with mixed_heavy applied (same seeding rule) | Secondary | CI > 0 |
| Clean c23 AUC | frame AUC, clean c23 test | **Safety** | CI not entirely < 0 |
| Clean DFD AUC | frame AUC, clean DFD | **Safety** | CI not entirely < 0 |

Exploratory (reported, never used as a claim): AP, EER, every single condition, sensor accuracy.
"Worth publishing" rule unchanged (wins primary OR >= 1 secondary, passes BOTH safety checks, and a later LUMEN-uncond
control must lose to LUMEN2 on the same metric it wins). No metric is dropped after seeing results.
