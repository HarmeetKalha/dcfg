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
