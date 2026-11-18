# LUMEN smoke tests

300-step smokes (lumen) and 100-step smokes (bnft, lumen_nocons), batch 32, real config except warmup_epochs=0, val on a 64-video stratified subset, W&B disabled, runs in /tmp/claude-1000/lumen_smoke. Loss = first vs last log window (lumen: BCE+KL+feat-MSE). img/s = mean of log windows after the warm-up (step >= 100 for 300-step runs, >= 50 for 100-step runs).

| run | det | variant | loss first -> last | ckpt | val AUC (64 vid) | img/s | peak VRAM MB | min scaler scale | skipped steps | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| lumen_xception | xception | lumen | 0.6930 -> 0.5705 | True | 0.9016 | 177.0 | 3997 | 65536 | 0 | True |
| lumen_efficientnetb4 | efficientnetb4 | lumen | 0.7598 -> 0.5801 | True | 0.8559 | 105.1 | 5787 | 65536 | 0 | True |
| lumen_f3net | f3net | lumen | 0.6777 -> 0.5460 | True | 0.9100 | 160.6 | 4039 | 32768 | 1 | True |
| bnft_xception | xception | bnft | 0.8583 -> 0.6033 | True | 0.9091 | 321.1 | 3076 | 65536 | 0 | True |
| lumen_nocons_xception | xception | lumen_nocons | 0.4628 -> 0.3503 | True | 0.9087 | 252.9 | 3979 | 65536 | 0 | True |

## Resume (lumen, xception)

- pass: True; killed at ~150 steps (ckpt every 100), resumed to 300; repeated hashes 0; missing steps n/a; hash conflicts []; reference mismatches []

## Notes

- UCF lumen not smoked: its FiLM sites are ambiguous (see docs/DECISIONS.md); stopped before implementing.
- bnft trainable params (Xception): 55,552
- total GPU wall time of this job: 8.8 min
- overall pass: True
