# LUMEN1 (Xception) gate heatmaps: `lumen_xception`

Test split, all frames, same degradation code and per-frame seeds as `results/raw`. w = 2*sigmoid(z); w = 1 means the coefficient is passed unchanged, w < 1 attenuated, w > 1 amplified. The gate has no bands; low / mid / high below are an after-the-fact grouping of the 8x8 grid by u+v (low 0-3, mid 4-7, high 8-14; u = vertical, v = horizontal frequency index, DC at (0,0)).

Shared colour scale for all w plots: 1 +/- 0.998; difference plots: +/- 0.754.

Files: `gate_means.csv` (mean, std, per class real/fake, per coefficient), `w_<cond>.png`, `w_all_conditions.png`, `w_fixed.png`, `diff_<cond>.png` (condition minus clean), `diff_all_conditions.png`.

## Mean w per band (all test frames)

| condition | Y low | Y mid | Y high | Cb low | Cb mid | Cb high | Cr low | Cr mid | Cr high |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.929 | 0.687 | 0.498 | 0.801 | 0.698 | 0.755 | 0.791 | 0.584 | 0.670 |
| jpeg_q50 | 0.910 | 0.533 | 0.395 | 0.748 | 0.565 | 0.659 | 0.704 | 0.426 | 0.541 |
| jpeg_q10 | 0.784 | 0.291 | 0.196 | 0.577 | 0.330 | 0.473 | 0.519 | 0.210 | 0.338 |
| blur_s2 | 1.047 | 0.792 | 0.554 | 0.951 | 0.833 | 0.849 | 0.922 | 0.737 | 0.776 |
| blur_s4 | 1.066 | 0.751 | 0.523 | 0.973 | 0.818 | 0.832 | 0.926 | 0.712 | 0.754 |
| noise_s10 | 0.738 | 0.174 | 0.114 | 0.520 | 0.226 | 0.376 | 0.437 | 0.124 | 0.236 |
| noise_s20 | 0.614 | 0.079 | 0.047 | 0.384 | 0.116 | 0.270 | 0.317 | 0.053 | 0.139 |
| resize_0.5 | 0.957 | 0.770 | 0.566 | 0.843 | 0.772 | 0.812 | 0.844 | 0.674 | 0.740 |
| resize_0.25 | 0.978 | 0.758 | 0.537 | 0.868 | 0.775 | 0.810 | 0.858 | 0.675 | 0.735 |
| mixed_heavy | 0.622 | 0.074 | 0.045 | 0.381 | 0.109 | 0.264 | 0.314 | 0.049 | 0.133 |
| c40 | 0.929 | 0.693 | 0.501 | 0.803 | 0.704 | 0.759 | 0.794 | 0.591 | 0.676 |
| webp_q20 | 0.967 | 0.758 | 0.569 | 0.848 | 0.763 | 0.807 | 0.840 | 0.656 | 0.728 |
| motion_k11 | 0.988 | 0.767 | 0.547 | 0.882 | 0.788 | 0.818 | 0.869 | 0.689 | 0.746 |
| fixed_gate | 0.879 | 0.479 | 0.218 | 0.905 | 0.623 | 0.566 | 0.592 | 0.450 | 0.462 |

## Change relative to clean (band mean of w(condition) - w(clean))

| condition | Y low | Y mid | Y high | Cb low | Cb mid | Cb high | Cr low | Cr mid | Cr high |
|---|---|---|---|---|---|---|---|---|---|
| jpeg_q50 | -0.018 | -0.154 | -0.102 | -0.053 | -0.133 | -0.096 | -0.087 | -0.158 | -0.129 |
| jpeg_q10 | -0.144 | -0.396 | -0.302 | -0.224 | -0.368 | -0.282 | -0.272 | -0.374 | -0.331 |
| blur_s2 | +0.118 | +0.105 | +0.057 | +0.150 | +0.135 | +0.094 | +0.131 | +0.153 | +0.106 |
| blur_s4 | +0.137 | +0.064 | +0.025 | +0.173 | +0.120 | +0.077 | +0.136 | +0.129 | +0.085 |
| noise_s10 | -0.190 | -0.512 | -0.384 | -0.281 | -0.472 | -0.379 | -0.354 | -0.460 | -0.434 |
| noise_s20 | -0.314 | -0.608 | -0.450 | -0.416 | -0.581 | -0.485 | -0.473 | -0.530 | -0.531 |
| resize_0.5 | +0.029 | +0.083 | +0.068 | +0.043 | +0.074 | +0.057 | +0.054 | +0.091 | +0.070 |
| resize_0.25 | +0.050 | +0.072 | +0.039 | +0.067 | +0.077 | +0.055 | +0.068 | +0.091 | +0.065 |
| mixed_heavy | -0.306 | -0.613 | -0.453 | -0.419 | -0.589 | -0.491 | -0.477 | -0.534 | -0.537 |
| c40 | +0.000 | +0.007 | +0.004 | +0.002 | +0.006 | +0.004 | +0.004 | +0.008 | +0.006 |
| webp_q20 | +0.038 | +0.071 | +0.071 | +0.047 | +0.065 | +0.053 | +0.049 | +0.072 | +0.059 |
| motion_k11 | +0.059 | +0.080 | +0.050 | +0.081 | +0.090 | +0.063 | +0.079 | +0.105 | +0.076 |

## Headline

- jpeg_q50: attenuates (w down) relative to clean, mean change -0.103; strongest in the mid band (-0.148); luma (Y) -0.091 vs chroma (Cb, Cr) -0.109.
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

## In plain language

- **jpeg_q50**: lowered vs clean: Y mid (-0.154), Y high (-0.102), Cb low (-0.053), Cb mid (-0.133), Cb high (-0.096), Cr low (-0.087), Cr mid (-0.158), Cr high (-0.129); raised: none above 0.02. Largest single change: Cr (u=0, v=5) -0.287.
- **jpeg_q10**: lowered vs clean: Y low (-0.144), Y mid (-0.396), Y high (-0.302), Cb low (-0.224), Cb mid (-0.368), Cb high (-0.282), Cr low (-0.272), Cr mid (-0.374), Cr high (-0.331); raised: none above 0.02. Largest single change: Y (u=3, v=4) -0.520.
- **blur_s2**: lowered vs clean: none above 0.02; raised: Y low (+0.118), Y mid (+0.105), Y high (+0.057), Cb low (+0.150), Cb mid (+0.135), Cb high (+0.094), Cr low (+0.131), Cr mid (+0.153), Cr high (+0.106). Largest single change: Cr (u=0, v=6) +0.299.
- **blur_s4**: lowered vs clean: none above 0.02; raised: Y low (+0.137), Y mid (+0.064), Y high (+0.025), Cb low (+0.173), Cb mid (+0.120), Cb high (+0.077), Cr low (+0.136), Cr mid (+0.129), Cr high (+0.085). Largest single change: Y (u=6, v=0) +0.288.
- **noise_s10**: lowered vs clean: Y low (-0.190), Y mid (-0.512), Y high (-0.384), Cb low (-0.281), Cb mid (-0.472), Cb high (-0.379), Cr low (-0.354), Cr mid (-0.460), Cr high (-0.434); raised: none above 0.02. Largest single change: Y (u=3, v=4) -0.642.
- **noise_s20**: lowered vs clean: Y low (-0.314), Y mid (-0.608), Y high (-0.450), Cb low (-0.416), Cb mid (-0.581), Cb high (-0.485), Cr low (-0.473), Cr mid (-0.530), Cr high (-0.531); raised: none above 0.02. Largest single change: Cb (u=1, v=4) -0.741.
- **resize_0.5**: lowered vs clean: none above 0.02; raised: Y low (+0.029), Y mid (+0.083), Y high (+0.068), Cb low (+0.043), Cb mid (+0.074), Cb high (+0.057), Cr low (+0.054), Cr mid (+0.091), Cr high (+0.070). Largest single change: Cr (u=0, v=6) +0.150.
- **resize_0.25**: lowered vs clean: none above 0.02; raised: Y low (+0.050), Y mid (+0.072), Y high (+0.039), Cb low (+0.067), Cb mid (+0.077), Cb high (+0.055), Cr low (+0.068), Cr mid (+0.091), Cr high (+0.065). Largest single change: Cr (u=0, v=5) +0.167.
- **mixed_heavy**: lowered vs clean: Y low (-0.306), Y mid (-0.613), Y high (-0.453), Cb low (-0.419), Cb mid (-0.589), Cb high (-0.491), Cr low (-0.477), Cr mid (-0.534), Cr high (-0.537); raised: none above 0.02. Largest single change: Cb (u=1, v=4) -0.754.
- **c40**: lowered vs clean: none above 0.02; raised: none above 0.02. Largest single change: Cr (u=0, v=5) +0.021.
- **webp_q20**: lowered vs clean: none above 0.02; raised: Y low (+0.038), Y mid (+0.071), Y high (+0.071), Cb low (+0.047), Cb mid (+0.065), Cb high (+0.053), Cr low (+0.049), Cr mid (+0.072), Cr high (+0.059). Largest single change: Y (u=6, v=0) +0.143.
- **motion_k11**: lowered vs clean: none above 0.02; raised: Y low (+0.059), Y mid (+0.080), Y high (+0.050), Cb low (+0.081), Cb mid (+0.090), Cb high (+0.063), Cr low (+0.079), Cr mid (+0.105), Cr high (+0.076). Largest single change: Cr (u=0, v=6) +0.196.

Clean-image baseline of the gate (what it does with no damage): Y low 0.93, Y mid 0.69, Y high 0.50, Cb low 0.80, Cb mid 0.70, Cb high 0.75, Cr low 0.79, Cr mid 0.58, Cr high 0.67.

## Image dependence

| condition | mean per-coefficient std over frames | max |real - fake| mean-w gap |
|---|---|---|
| clean | 0.0686 | 0.0039 |
| jpeg_q50 | 0.0703 | 0.0058 |
| jpeg_q10 | 0.0627 | 0.0119 |
| blur_s2 | 0.0385 | 0.0032 |
| blur_s4 | 0.0264 | 0.0007 |
| noise_s10 | 0.0544 | 0.0222 |
| noise_s20 | 0.0331 | 0.0161 |
| resize_0.5 | 0.0656 | 0.0044 |
| resize_0.25 | 0.0715 | 0.0042 |
| mixed_heavy | 0.0279 | 0.0086 |
| c40 | 0.0717 | 0.0179 |
| webp_q20 | 0.0466 | 0.0034 |
| motion_k11 | 0.0613 | 0.0062 |

## Fixed gate (static, `fixed_xception`)

Band means: Y low 0.879, Y mid 0.479, Y high 0.218, Cb low 0.905, Cb mid 0.623, Cb high 0.566, Cr low 0.592, Cr mid 0.450, Cr high 0.462.
