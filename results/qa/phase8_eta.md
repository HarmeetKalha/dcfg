# Phase 8 ETA (Tiers 1-2)

Worst case = 30 full epochs (no early stop). Likely case = early stop at epoch 15 (ASSUMPTION, not measured). Each run = epochs x (train + val over the 6 val conditions) + test evaluation over all conditions.

| run | variant | det | bs | img/s | steps/ep | train/ep (min) | val/ep (min) | epoch (min) | test eval (min) | 30 ep total (h) | 15 ep total (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dcfg_xception | dcfg | xception | 32 | 216 | 5743 | 14.2 | 5.5 | 19.6 | 24.9 (log (22/23 conds)) | 10.2 | 5.3 |
| fixed_xception | fixed | xception | 32 | 317 | 5743 | 9.7 | 5.5 | 15.1 | 24.9 (log (22/23 conds)) | 8.0 | 4.2 |
| augft_xception | augft | xception | 32 | 203 | 5743 | 15.1 | 4.5 | 19.6 | 24.9 (log (22/23 conds)) | 10.2 | 5.3 |
| dcfg_efficientnetb4 | dcfg | efficientnetb4 | 32 | 123 | 5743 | 25.0 | 7.2 | 32.2 | 36.8 (log (22/23 conds)) | 16.7 | 8.7 |
| fixed_efficientnetb4 | fixed | efficientnetb4 | 32 | 152 | 5743 | 20.2 | 7.2 | 27.4 | 36.8 (log (22/23 conds)) | 14.3 | 7.5 |
| augft_efficientnetb4 | augft | efficientnetb4 | 32 | 123 | 5743 | 24.9 | 6.2 | 31.2 | 36.8 (log (22/23 conds)) | 16.2 | 8.4 |
| dcfg_f3net | dcfg | f3net | 32 | 205 | 5743 | 14.9 | 5.7 | 20.6 | 26.8 (log (22/23 conds)) | 10.7 | 5.6 |
| fixed_f3net | fixed | f3net | 32 | 269 | 5743 | 11.4 | 5.7 | 17.1 | 26.8 (log (22/23 conds)) | 9.0 | 4.7 |
| augft_f3net | augft | f3net | 32 | 183 | 5743 | 16.7 | 4.8 | 21.6 | 26.8 (log (22/23 conds)) | 11.2 | 5.8 |
| dcfg_ucf | dcfg | ucf | 32 | 165 | 5743 | 18.6 | 8.9 | 27.5 | 49.0 (log (22/23 conds)) | 14.6 | 7.7 |
| fixed_ucf | fixed | ucf | 32 | 203 | 5743 | 15.1 | 8.9 | 24.0 | 49.0 (log (22/23 conds)) | 12.8 | 6.8 |
| augft_ucf | augft | ucf | 32 | 70 | 5743 | 43.9 | 7.6 | 51.5 | 49.0 (log (22/23 conds)) | 26.6 | 13.7 |
| gate_nosensor_xception | nosensor | xception | 32 | 317 | 5743 | 9.7 | 5.5 | 15.1 | 24.9 (log (22/23 conds)) | 8.0 | 4.2 |
| dcfg_frozensensor_xception | dcfg | xception | 32 | 216 | 5743 | 14.2 | 5.5 | 19.6 | 24.9 (log (22/23 conds)) | 10.2 | 5.3 |
| dcfg_nolive_xception | dcfg_nolive | xception | 32 | 216 | 5743 | 14.2 | 5.5 | 19.6 | 24.9 (log (22/23 conds)) | 10.2 | 5.3 |
| dcfg_jointft_xception | dcfg_jointft | xception | 32 | 158 | 5743 | 19.4 | 5.5 | 24.8 | 24.9 (log (22/23 conds)) | 12.8 | 6.6 |
| dcfg_xception_s43 | dcfg | xception | 32 | 216 | 5743 | 14.2 | 5.5 | 19.6 | 24.9 (log (22/23 conds)) | 10.2 | 5.3 |
| dcfg_xception_s44 | dcfg | xception | 32 | 216 | 5743 | 14.2 | 5.5 | 19.6 | 24.9 (log (22/23 conds)) | 10.2 | 5.3 |
| fixed_xception_s43 | fixed | xception | 32 | 317 | 5743 | 9.7 | 5.5 | 15.1 | 24.9 (log (22/23 conds)) | 8.0 | 4.2 |
| fixed_xception_s44 | fixed | xception | 32 | 317 | 5743 | 9.7 | 5.5 | 15.1 | 24.9 (log (22/23 conds)) | 8.0 | 4.2 |

## Per priority group (you can stop after any group)

| group | contents | runs | worst (GPU-days) | likely (GPU-days) | cumulative worst | cumulative likely |
|---|---|---|---|---|---|---|
| 1 | dcfg/fixed/augft x 4 detectors | 12 | 6.69 | 3.49 | 6.69 | 3.49 |
| 2 | Xception ablations (nosensor, frozensensor, nolive, jointft) | 4 | 1.72 | 0.89 | 8.41 | 4.38 |
| 3 | dcfg_xception extra seeds (43, 44) | 2 | 0.85 | 0.44 | 9.27 | 4.83 |
| 4 | fixed_xception extra seeds (43, 44) | 2 | 0.66 | 0.35 | 9.93 | 5.18 |

**Total (20 runs): worst case 9.93 GPU-days; likely case (15 epochs, assumption) 5.18 GPU-days.**

- Per-epoch val time = sum over the 6 val conditions at full val size, extrapolated linearly from a ~10% video sample (includes DataLoader start-up per condition: slightly conservative).
- Timing classes: dcfg (also nolive, frozensensor: frozen sensor is slightly cheaper), fixed (also nosensor), augft, jointft (Xception only). Val time of module variants = measured dcfg val time.
- Test eval time is measured without the front-end; the module adds a small cost per image.
