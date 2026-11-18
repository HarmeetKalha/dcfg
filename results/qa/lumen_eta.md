# LUMEN matrix ETA

Worst case = 30 full epochs (no early stop). Likely case = 15 epochs (ASSUMPTION, not measured). Per run: epochs x (train + val per epoch) + test evaluation over 23 conditions. Units: GPU-days (24 h).

## Per group (you can stop after any group)

| group | contents | runs | worst | likely | cumulative worst | cumulative likely |
|---|---|---|---|---|---|---|
| G1 | core: lumen/fixed/augft x 4 detectors | 12 | 6.92 | 3.60 | 6.92 | 3.60 |
| G2 | jointft_xception, bnft_xception | 2 | 0.75 | 0.39 | 7.66 | 3.99 |
| G3 | Xception ablation chain | 5 | 2.19 | 1.14 | 9.85 | 5.13 |
| G4 | seeds (lumen, augft Xception s43/s44) | 4 | 1.74 | 0.90 | 11.59 | 6.03 |

**Total (23 runs): worst 11.59 GPU-days; likely (15 ep, assumption) 6.03 GPU-days.**

## Per run

| run | group | epoch (min) | eval (min) | worst (GPU-days) | likely (GPU-days) | epoch time from |
|---|---|---|---|---|---|---|
| lumen_xception | G1 | 22.2 | 24.9 | 0.48 | 0.25 | pilot (measured) |
| fixed_xception | G1 | 14.0 | 24.9 | 0.31 | 0.16 | pilot (measured) |
| augft_xception | G1 | 17.8 | 24.9 | 0.39 | 0.20 | pilot (measured) |
| lumen_efficientnetb4 | G1 | 37.4 | 36.8 | 0.80 | 0.42 | pilot x img/s ratio |
| fixed_efficientnetb4 | G1 | 29.3 | 36.8 | 0.64 | 0.33 | pilot x img/s ratio |
| augft_efficientnetb4 | G1 | 29.4 | 36.8 | 0.64 | 0.33 | pilot x img/s ratio |
| lumen_f3net | G1 | 24.5 | 26.8 | 0.53 | 0.27 | pilot x img/s ratio |
| fixed_f3net | G1 | 16.5 | 26.8 | 0.36 | 0.19 | pilot x img/s ratio |
| augft_f3net | G1 | 19.7 | 26.8 | 0.43 | 0.22 | pilot x img/s ratio |
| lumen_ucf | G1 | 33.8 | 49.0 | 0.74 | 0.39 | pilot x img/s ratio |
| fixed_ucf | G1 | 21.9 | 49.0 | 0.49 | 0.26 | pilot x img/s ratio |
| augft_ucf | G1 | 51.7 | 49.0 | 1.11 | 0.57 | pilot x img/s ratio |
| jointft_xception | G2 | 22.9 | 24.9 | 0.49 | 0.26 | pilot (measured) |
| bnft_xception | G2 | 11.2 | 24.9 | 0.25 | 0.13 | pilot augft x smoke ratio bnft |
| dcfg_xception | G3 | 18.8 | 24.9 | 0.41 | 0.21 | pilot (measured) |
| dcfgcons_xception | G3 | 22.1 | 24.9 | 0.48 | 0.25 | pilot (measured) |
| lumen_nocons_xception | G3 | 15.5 | 24.9 | 0.34 | 0.18 | pilot lumen x smoke ratio nocons |
| lumen_nolive_xception | G3 | 22.2 | 24.9 | 0.48 | 0.25 | pilot (measured) |
| lumen_frozensensor_xception | G3 | 22.2 | 24.9 | 0.48 | 0.25 | pilot (measured) |
| lumen_xception_s43 | G4 | 22.2 | 24.9 | 0.48 | 0.25 | pilot (measured) |
| lumen_xception_s44 | G4 | 22.2 | 24.9 | 0.48 | 0.25 | pilot (measured) |
| augft_xception_s43 | G4 | 17.8 | 24.9 | 0.39 | 0.20 | pilot (measured) |
| augft_xception_s44 | G4 | 17.8 | 24.9 | 0.39 | 0.20 | pilot (measured) |

## Notes

- Epoch time includes val; scaling by steady img/s ratios applies to the whole epoch (val share is similar across detectors).
- Test eval time = base_<det> logs; module variants add a small per-image cost not included. xception 24.9 min (log (22/23 conds)); efficientnetb4 36.8 min (log (22/23 conds)); f3net 26.8 min (log (22/23 conds)); ucf 49.0 min (log (22/23 conds))
- lumen_frozensensor and lumen_nolive are timed as lumen (frozen sensor is slightly cheaper: conservative).
- lumen_ucf uses its measured smoke img/s when present in lumen_smoke.json (otherwise the phase-8 dcfg ratio, labelled ASSUMPTION). augft_ucf uses UCF's native loss (phase-8 img/s).
- img/s: {"lumen": {"xception": 177.0, "efficientnetb4": 105.1, "f3net": 160.6, "ucf": 116.1}, "lumen_nocons": {"xception": 252.9}, "bnft": {"xception": 321.1}, "fixed": {"f3net": 268.6, "xception": 317.4, "ucf": 203.0, "efficientnetb4": 151.9}, "augft": {"efficientnetb4": 122.8, "xception": 202.7, "ucf": 69.7, "f3net": 182.9}, "dcfg": {"ucf": 164.9, "xception": 216.0, "f3net": 205.0, "efficientnetb4": 122.6}}
