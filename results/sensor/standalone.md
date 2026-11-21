# Standalone damage sensor: test-split evaluation

Checkpoint `runs/damage_sensor/best.pt` (best val loss epoch 124, 772264 parameters; val loss 0.1347). 128x128 native centre crop + 192 DCT log-energy statistics. Test split, 23 conditions, same degradation code and per-frame seeds as `results/raw`. Test only: nothing was selected on it.

Labelled conditions pooled: 18 (clean + 17 synthetic), 402984 frames. Presence threshold 0.5. Strength is scaled 0..1 over the training range (jpeg q95->0, q20->1; blur 0.5->4; noise 2->20; resize 0.9->0.25), clipped; strength metrics only over frames where the op is truly applied.

## Per operation

| op | presence acc | presence F1 | strength MAE | Pearson | Spearman | n applied | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| jpeg | 0.994 | 0.992 | 0.020 | 0.997 | 0.972 | 156716 | 156714 | 2594 | 2 |
| blur | 0.950 | 0.929 | 0.049 | 0.975 | 0.952 | 134328 | 131114 | 16865 | 3214 |
| noise | 0.985 | 0.978 | 0.038 | 0.986 | 0.956 | 134328 | 134328 | 6171 | 0 |
| resize | 0.955 | 0.831 | 0.056 | 0.964 | 0.866 | 44776 | 44775 | 18206 | 1 |

## Per condition (labelled)

| condition | true ops | true strength | mean P(jpeg/blur/noise/resize) | mean strength where present | exact op-set acc |
|---|---|---|---|---|---|
| clean | none | - | 0.00 / 0.01 / 0.00 / 0.00 | blur 0.07, resize 0.23 | 0.994 |
| jpeg_q90 | jpeg | jpeg 0.07 | 1.00 / 0.03 / 0.00 / 0.12 | jpeg 0.07, blur 0.09, noise 0.05, resize 0.34 | 0.907 |
| jpeg_q70 | jpeg | jpeg 0.33 | 1.00 / 0.09 / 0.01 / 0.16 | jpeg 0.34, blur 0.13, noise 0.13, resize 0.43 | 0.851 |
| jpeg_q50 | jpeg | jpeg 0.60 | 1.00 / 0.10 / 0.02 / 0.17 | jpeg 0.61, blur 0.14, noise 0.13, resize 0.45 | 0.847 |
| jpeg_q30 | jpeg | jpeg 0.87 | 1.00 / 0.11 / 0.06 / 0.20 | jpeg 0.88, blur 0.18, noise 0.15, resize 0.47 | 0.834 |
| jpeg_q10 | jpeg | jpeg 1.00 | 1.00 / 0.09 / 0.28 / 0.11 | jpeg 0.99, blur 0.24, noise 0.26, resize 0.53 | 0.781 |
| blur_s1 | blur | blur 0.14 | 0.00 / 1.00 / 0.01 / 0.16 | jpeg 0.05, blur 0.15, noise 0.06, resize 0.46 | 0.929 |
| blur_s2 | blur | blur 0.43 | 0.01 / 1.00 / 0.03 / 0.20 | jpeg 0.08, blur 0.44, noise 0.10, resize 0.63 | 0.945 |
| blur_s3 | blur | blur 0.71 | 0.03 / 1.00 / 0.06 / 0.24 | jpeg 0.12, blur 0.75, noise 0.16, resize 0.68 | 0.963 |
| blur_s4 | blur | blur 1.00 | 0.07 / 1.00 / 0.07 / 0.25 | jpeg 0.19, blur 0.95, noise 0.27 | 0.966 |
| noise_s5 | noise | noise 0.17 | 0.01 / 0.07 / 1.00 / 0.11 | jpeg 0.14, blur 0.15, noise 0.18, resize 0.42 | 0.887 |
| noise_s10 | noise | noise 0.44 | 0.04 / 0.11 / 1.00 / 0.17 | jpeg 0.17, blur 0.19, noise 0.45, resize 0.46 | 0.836 |
| noise_s15 | noise | noise 0.72 | 0.07 / 0.12 / 1.00 / 0.20 | jpeg 0.27, blur 0.22, noise 0.74, resize 0.50 | 0.820 |
| noise_s20 | noise | noise 1.00 | 0.11 / 0.13 / 1.00 / 0.23 | jpeg 0.37, blur 0.23, noise 0.96, resize 0.53 | 0.812 |
| resize_0.5 | resize | resize 0.62 | 0.00 / 0.07 / 0.01 / 1.00 | jpeg 0.01, blur 0.08, noise 0.03, resize 0.65 | 0.948 |
| resize_0.25 | resize | resize 1.00 | 0.02 / 0.12 / 0.01 / 1.00 | jpeg 0.05, blur 0.19, noise 0.25, resize 0.93 | 0.925 |
| mixed_light | jpeg+blur+noise | jpeg 0.33, blur 0.14, noise 0.17 | 1.00 / 0.83 / 1.00 / 0.09 | jpeg 0.35, blur 0.16, noise 0.21, resize 0.55 | 0.861 |
| mixed_heavy | jpeg+blur+noise | jpeg 0.87, blur 0.43, noise 0.44 | 1.00 / 0.98 / 1.00 / 0.04 | jpeg 0.86, blur 0.42, noise 0.46, resize 0.80 | 0.990 |

## Confusion: true op set -> predicted op set (top 5, fraction of frames)

| condition | true | predicted |
|---|---|---|
| clean | none | none 0.99; blur 0.00; resize 0.00; blur+resize 0.00 |
| jpeg_q90 | jpeg | jpeg 0.91; jpeg+resize 0.07; jpeg+blur 0.02; jpeg+blur+resize 0.00; jpeg+noise 0.00 |
| jpeg_q70 | jpeg | jpeg 0.85; jpeg+resize 0.08; jpeg+blur 0.05; jpeg+blur+resize 0.01; jpeg+blur+noise 0.00 |
| jpeg_q50 | jpeg | jpeg 0.85; jpeg+resize 0.07; jpeg+blur 0.06; jpeg+blur+resize 0.01; jpeg+blur+noise 0.00 |
| jpeg_q30 | jpeg | jpeg 0.83; jpeg+blur 0.07; jpeg+resize 0.06; jpeg+blur+resize 0.01; jpeg+noise 0.01 |
| jpeg_q10 | jpeg | jpeg 0.78; jpeg+noise 0.15; jpeg+blur 0.04; jpeg+blur+noise 0.02; jpeg+resize 0.00 |
| blur_s1 | blur | blur 0.93; blur+resize 0.07; blur+noise 0.00; jpeg+blur 0.00; blur+noise+resize 0.00 |
| blur_s2 | blur | blur 0.94; blur+resize 0.04; blur+noise 0.02; jpeg+blur 0.00; blur+noise+resize 0.00 |
| blur_s3 | blur | blur 0.96; blur+noise 0.02; jpeg+blur 0.01; blur+resize 0.00; jpeg+blur+noise 0.00 |
| blur_s4 | blur | blur 0.97; jpeg+blur 0.02; blur+noise 0.01; jpeg+blur+noise 0.00 |
| noise_s5 | noise | noise 0.89; noise+resize 0.06; blur+noise 0.04; blur+noise+resize 0.01; jpeg+noise 0.00 |
| noise_s10 | noise | noise 0.84; noise+resize 0.07; blur+noise 0.07; blur+noise+resize 0.01; jpeg+noise 0.01 |
| noise_s15 | noise | noise 0.82; blur+noise 0.07; noise+resize 0.07; jpeg+noise 0.01; blur+noise+resize 0.01 |
| noise_s20 | noise | noise 0.81; blur+noise 0.07; noise+resize 0.07; jpeg+noise 0.02; blur+noise+resize 0.01 |
| resize_0.5 | resize | resize 0.95; blur+resize 0.05; noise+resize 0.00; blur+noise+resize 0.00; jpeg+resize 0.00 |
| resize_0.25 | resize | resize 0.93; blur+resize 0.07; jpeg+resize 0.00; noise+resize 0.00; jpeg+blur+resize 0.00 |
| mixed_light | jpeg+blur+noise | jpeg+blur+noise 0.86; jpeg+noise 0.11; jpeg+noise+resize 0.03; jpeg+blur+noise+resize 0.00 |
| mixed_heavy | jpeg+blur+noise | jpeg+blur+noise 0.99; jpeg+noise 0.01; jpeg+blur+noise+resize 0.00; jpeg+noise+resize 0.00 |

## Unseen conditions (no ground truth: c40 real compression, WebP, motion blur)

| condition | frac any op present | P(jpeg) | P(blur) | P(noise) | P(resize) | str jpeg | str blur | str noise | str resize |
|---|---|---|---|---|---|---|---|---|---|
| webp_q50 | 0.89 | 0.87 | 0.00 | 0.00 | 0.00 | 0.03 | 0.05 | 0.01 | 0.14 |
| webp_q20 | 0.88 | 0.86 | 0.01 | 0.00 | 0.00 | 0.03 | 0.05 | 0.01 | 0.19 |
| motion_k7 | 0.86 | 0.01 | 0.85 | 0.00 | 0.00 | 0.03 | 0.15 | 0.03 | 0.10 |
| motion_k11 | 0.87 | 0.03 | 0.85 | 0.00 | 0.00 | 0.03 | 0.22 | 0.02 | 0.13 |
| c40 | 0.04 | 0.00 | 0.02 | 0.03 | 0.00 | 0.03 | 0.02 | 0.06 | 0.12 |
