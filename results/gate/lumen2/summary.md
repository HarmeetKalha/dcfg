# LUMEN2 (CLIP ViT-L/14) gate heatmaps: `lumen2`

Test split, all frames, same degradation code and per-frame seeds as `results/raw`. w = 2*sigmoid(z); w = 1 means the coefficient is passed unchanged, w < 1 attenuated, w > 1 amplified. The gate has no bands; low / mid / high below are an after-the-fact grouping of the 8x8 grid by u+v (low 0-3, mid 4-7, high 8-14; u = vertical, v = horizontal frequency index, DC at (0,0)).

Shared colour scale for all w plots: 1 +/- 0.998; difference plots: +/- 0.754.

Files: `gate_means.csv` (mean, std, per class real/fake, per coefficient), `w_<cond>.png`, `w_all_conditions.png`, `w_fixed.png`, `diff_<cond>.png` (condition minus clean), `diff_all_conditions.png`.

## Mean w per band (all test frames)

| condition | Y low | Y mid | Y high | Cb low | Cb mid | Cb high | Cr low | Cr mid | Cr high |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.997 | 1.001 | 1.004 | 0.995 | 0.995 | 0.996 | 1.003 | 0.990 | 0.999 |
| jpeg_q50 | 0.975 | 0.914 | 0.946 | 0.918 | 0.968 | 0.979 | 0.916 | 0.946 | 1.006 |
| jpeg_q10 | 0.952 | 0.817 | 0.874 | 0.837 | 0.943 | 0.964 | 0.821 | 0.899 | 1.014 |
| blur_s2 | 1.102 | 1.049 | 1.018 | 1.082 | 0.997 | 1.011 | 1.097 | 1.029 | 0.997 |
| blur_s4 | 1.139 | 1.053 | 1.016 | 1.106 | 0.991 | 1.014 | 1.120 | 1.035 | 0.996 |
| noise_s10 | 0.963 | 0.843 | 0.901 | 0.872 | 0.934 | 0.967 | 0.849 | 0.903 | 1.017 |
| noise_s20 | 0.925 | 0.713 | 0.808 | 0.782 | 0.894 | 0.951 | 0.721 | 0.840 | 1.039 |
| resize_0.5 | 0.999 | 1.008 | 1.009 | 1.001 | 0.997 | 0.998 | 1.009 | 0.993 | 0.998 |
| resize_0.25 | 1.038 | 1.022 | 1.010 | 1.030 | 0.997 | 1.002 | 1.041 | 1.007 | 0.998 |
| mixed_heavy | 0.937 | 0.766 | 0.835 | 0.794 | 0.930 | 0.957 | 0.767 | 0.875 | 1.021 |
| c40 | 1.000 | 1.004 | 1.005 | 0.998 | 0.996 | 0.997 | 1.006 | 0.991 | 0.998 |
| webp_q20 | 0.995 | 0.992 | 0.998 | 0.988 | 0.992 | 0.994 | 0.995 | 0.986 | 1.000 |
| motion_k11 | 1.068 | 1.033 | 1.013 | 1.054 | 0.997 | 1.006 | 1.065 | 1.018 | 0.998 |

## Change relative to clean (band mean of w(condition) - w(clean))

| condition | Y low | Y mid | Y high | Cb low | Cb mid | Cb high | Cr low | Cr mid | Cr high |
|---|---|---|---|---|---|---|---|---|---|
| jpeg_q50 | -0.023 | -0.087 | -0.057 | -0.076 | -0.027 | -0.017 | -0.087 | -0.044 | +0.007 |
| jpeg_q10 | -0.045 | -0.184 | -0.129 | -0.157 | -0.052 | -0.032 | -0.183 | -0.090 | +0.015 |
| blur_s2 | +0.104 | +0.048 | +0.014 | +0.088 | +0.002 | +0.015 | +0.094 | +0.039 | -0.002 |
| blur_s4 | +0.142 | +0.052 | +0.013 | +0.111 | -0.005 | +0.018 | +0.117 | +0.045 | -0.003 |
| noise_s10 | -0.035 | -0.158 | -0.103 | -0.123 | -0.061 | -0.029 | -0.154 | -0.087 | +0.018 |
| noise_s20 | -0.073 | -0.287 | -0.196 | -0.212 | -0.101 | -0.045 | -0.282 | -0.149 | +0.040 |
| resize_0.5 | +0.001 | +0.007 | +0.006 | +0.006 | +0.002 | +0.002 | +0.006 | +0.003 | -0.001 |
| resize_0.25 | +0.041 | +0.021 | +0.006 | +0.036 | +0.002 | +0.006 | +0.038 | +0.017 | -0.001 |
| mixed_heavy | -0.061 | -0.235 | -0.169 | -0.200 | -0.065 | -0.039 | -0.236 | -0.115 | +0.022 |
| c40 | +0.003 | +0.003 | +0.001 | +0.003 | +0.001 | +0.000 | +0.003 | +0.001 | -0.000 |
| webp_q20 | -0.002 | -0.008 | -0.005 | -0.007 | -0.003 | -0.002 | -0.008 | -0.004 | +0.001 |
| motion_k11 | +0.070 | +0.032 | +0.009 | +0.059 | +0.002 | +0.010 | +0.062 | +0.028 | -0.001 |

## Headline

- jpeg_q50: attenuates (w down) relative to clean, mean change -0.046; strongest in the low band (-0.062); luma (Y) -0.056 vs chroma (Cb, Cr) -0.041.
- jpeg_q10: attenuates (w down) relative to clean, mean change -0.095; strongest in the low band (-0.128); luma (Y) -0.119 vs chroma (Cb, Cr) -0.083.
- blur_s2: opens up (w up) relative to clean, mean change +0.045; strongest in the low band (+0.095); luma (Y) +0.055 vs chroma (Cb, Cr) +0.039.
- blur_s4: opens up (w up) relative to clean, mean change +0.054; strongest in the low band (+0.123); luma (Y) +0.069 vs chroma (Cb, Cr) +0.047.
- noise_s10: attenuates (w down) relative to clean, mean change -0.081; strongest in the low band (-0.104); luma (Y) -0.099 vs chroma (Cb, Cr) -0.073.
- noise_s20: attenuates (w down) relative to clean, mean change -0.145; strongest in the low band (-0.189); luma (Y) -0.185 vs chroma (Cb, Cr) -0.125.
- resize_0.5: leaves unchanged relative to clean, mean change +0.004; strongest in the low band (+0.005); luma (Y) +0.005 vs chroma (Cb, Cr) +0.003.
- resize_0.25: leaves unchanged relative to clean, mean change +0.019; strongest in the low band (+0.038); luma (Y) +0.023 vs chroma (Cb, Cr) +0.016.
- mixed_heavy: attenuates (w down) relative to clean, mean change -0.122; strongest in the low band (-0.166); luma (Y) -0.155 vs chroma (Cb, Cr) -0.105.
- c40: leaves unchanged relative to clean, mean change +0.002; strongest in the low band (+0.003); luma (Y) +0.002 vs chroma (Cb, Cr) +0.001.
- webp_q20: leaves unchanged relative to clean, mean change -0.004; strongest in the low band (-0.006); luma (Y) -0.005 vs chroma (Cb, Cr) -0.004.
- motion_k11: opens up (w up) relative to clean, mean change +0.030; strongest in the low band (+0.064); luma (Y) +0.037 vs chroma (Cb, Cr) +0.027.

## In plain language

- **jpeg_q50**: lowered vs clean: Y low (-0.023), Y mid (-0.087), Y high (-0.057), Cb low (-0.076), Cb mid (-0.027), Cr low (-0.087), Cr mid (-0.044); raised: none above 0.02. Largest single change: Y (u=0, v=7) -0.180.
- **jpeg_q10**: lowered vs clean: Y low (-0.045), Y mid (-0.184), Y high (-0.129), Cb low (-0.157), Cb mid (-0.052), Cb high (-0.032), Cr low (-0.183), Cr mid (-0.090); raised: none above 0.02. Largest single change: Y (u=0, v=7) -0.370.
- **blur_s2**: lowered vs clean: none above 0.02; raised: Y low (+0.104), Y mid (+0.048), Cb low (+0.088), Cr low (+0.094), Cr mid (+0.039). Largest single change: Y (u=2, v=1) +0.263.
- **blur_s4**: lowered vs clean: none above 0.02; raised: Y low (+0.142), Y mid (+0.052), Cb low (+0.111), Cr low (+0.117), Cr mid (+0.045). Largest single change: Y (u=2, v=1) +0.356.
- **noise_s10**: lowered vs clean: Y low (-0.035), Y mid (-0.158), Y high (-0.103), Cb low (-0.123), Cb mid (-0.061), Cb high (-0.029), Cr low (-0.154), Cr mid (-0.087); raised: none above 0.02. Largest single change: Y (u=0, v=7) -0.319.
- **noise_s20**: lowered vs clean: Y low (-0.073), Y mid (-0.287), Y high (-0.196), Cb low (-0.212), Cb mid (-0.101), Cb high (-0.045), Cr low (-0.282), Cr mid (-0.149); raised: Cr high (+0.040). Largest single change: Y (u=0, v=7) -0.510.
- **resize_0.5**: lowered vs clean: none above 0.02; raised: none above 0.02. Largest single change: Y (u=4, v=0) +0.023.
- **resize_0.25**: lowered vs clean: none above 0.02; raised: Y low (+0.041), Y mid (+0.021), Cb low (+0.036), Cr low (+0.038). Largest single change: Y (u=1, v=2) +0.104.
- **mixed_heavy**: lowered vs clean: Y low (-0.061), Y mid (-0.235), Y high (-0.169), Cb low (-0.200), Cb mid (-0.065), Cb high (-0.039), Cr low (-0.236), Cr mid (-0.115); raised: Cr high (+0.022). Largest single change: Y (u=0, v=7) -0.453.
- **c40**: lowered vs clean: none above 0.02; raised: none above 0.02. Largest single change: Y (u=2, v=2) +0.007.
- **webp_q20**: lowered vs clean: none above 0.02; raised: none above 0.02. Largest single change: Cb (u=2, v=6) -0.021.
- **motion_k11**: lowered vs clean: none above 0.02; raised: Y low (+0.070), Y mid (+0.032), Cb low (+0.059), Cr low (+0.062), Cr mid (+0.028). Largest single change: Y (u=1, v=2) +0.179.

Clean-image baseline of the gate (what it does with no damage): Y low 1.00, Y mid 1.00, Y high 1.00, Cb low 0.99, Cb mid 1.00, Cb high 1.00, Cr low 1.00, Cr mid 0.99, Cr high 1.00.

## Image dependence

| condition | mean per-coefficient std over frames | max |real - fake| mean-w gap |
|---|---|---|
| clean | 0.0067 | 0.0032 |
| jpeg_q50 | 0.0174 | 0.0020 |
| jpeg_q10 | 0.0168 | 0.0059 |
| blur_s2 | 0.0077 | 0.0054 |
| blur_s4 | 0.0054 | 0.0020 |
| noise_s10 | 0.0317 | 0.0108 |
| noise_s20 | 0.0369 | 0.0198 |
| resize_0.5 | 0.0064 | 0.0045 |
| resize_0.25 | 0.0096 | 0.0093 |
| mixed_heavy | 0.0134 | 0.0059 |
| c40 | 0.0079 | 0.0039 |
| webp_q20 | 0.0051 | 0.0021 |
| motion_k11 | 0.0152 | 0.0149 |

## LUMEN1 (Xception) vs LUMEN2 (CLIP)

LUMEN1: gate on the 256 px DFB crop (32x32 blocks), LIVE-pretrained sensor, frozen CNN. LUMEN2: gate on the 224 px GenD-resized crop (28x28 blocks), sensor trained jointly from scratch, LayerNorm-tuned CLIP. The (u, v) coefficient grid is the same 8x8 DCT grid, but the spatial scale of a coefficient differs because the image is resized, so compare patterns, not exact values.

Mean |w - 1| over the 192 coefficients (how far the gate departs from the identity) and mean w:

| condition | LUMEN1 mean abs(w-1) | LUMEN2 mean abs(w-1) | LUMEN1 mean w | LUMEN2 mean w |
|---|---|---|---|---|
| clean | 0.322 | 0.014 | 0.678 | 0.998 |
| jpeg_q50 | 0.438 | 0.065 | 0.562 | 0.957 |
| jpeg_q10 | 0.643 | 0.128 | 0.357 | 0.912 |
| blur_s2 | 0.226 | 0.058 | 0.790 | 1.028 |
| blur_s4 | 0.249 | 0.076 | 0.771 | 1.033 |
| noise_s10 | 0.735 | 0.118 | 0.265 | 0.923 |
| noise_s20 | 0.831 | 0.208 | 0.169 | 0.866 |
| resize_0.5 | 0.253 | 0.014 | 0.747 | 1.001 |
| resize_0.25 | 0.258 | 0.022 | 0.743 | 1.011 |
| mixed_heavy | 0.836 | 0.163 | 0.165 | 0.889 |
| c40 | 0.317 | 0.013 | 0.683 | 0.999 |
| webp_q20 | 0.260 | 0.018 | 0.740 | 0.994 |
| motion_k11 | 0.247 | 0.038 | 0.754 | 1.019 |

Change relative to clean, band means (average over Y, Cb, Cr); LUMEN1 -> LUMEN2:

| condition | low | mid | high |
|---|---|---|---|
| jpeg_q50 | -0.053 -> -0.062 | -0.148 -> -0.053 | -0.109 -> -0.023 |
| jpeg_q10 | -0.213 -> -0.128 | -0.379 -> -0.109 | -0.305 -> -0.049 |
| blur_s2 | +0.133 -> +0.095 | +0.131 -> +0.030 | +0.086 -> +0.009 |
| blur_s4 | +0.149 -> +0.123 | +0.104 -> +0.031 | +0.062 -> +0.009 |
| noise_s10 | -0.275 -> -0.104 | -0.481 -> -0.102 | -0.399 -> -0.038 |
| noise_s20 | -0.401 -> -0.189 | -0.573 -> -0.179 | -0.489 -> -0.067 |
| resize_0.5 | +0.042 -> +0.005 | +0.083 -> +0.004 | +0.065 -> +0.002 |
| resize_0.25 | +0.061 -> +0.038 | +0.080 -> +0.014 | +0.053 -> +0.004 |
| mixed_heavy | -0.401 -> -0.166 | -0.579 -> -0.138 | -0.494 -> -0.062 |
| c40 | +0.002 -> +0.003 | +0.007 -> +0.002 | +0.005 -> +0.000 |
| webp_q20 | +0.045 -> -0.006 | +0.069 -> -0.005 | +0.061 -> -0.002 |
| motion_k11 | +0.073 -> +0.064 | +0.092 -> +0.021 | +0.063 -> +0.006 |

Files: `side_by_side/w_<cond>.png` and `side_by_side/diff_<cond>.png` (LUMEN1 on top, LUMEN2 below, identical colour scales); `w_*.png` / `diff_*.png` in this directory are the LUMEN2 plots on that same scale.
