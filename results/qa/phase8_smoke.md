# Phase 8 smoke

## (a) Xception, 300 steps, val on 64 videos (stratified)

| variant | loss first | loss last | decreased | val mean AUC | ckpt | img/s | peak VRAM MB | pass |
|---|---|---|---|---|---|---|---|---|
| dcfg | 0.4541 | 0.3906 | True | 0.8877 | True | 219 | 3951 | True |
| augft | 0.5723 | 0.3260 | True | 0.9174 | True | 202 | 3323 | True |
| dcfg_jointft | 0.5683 | 0.3271 | True | 0.9158 | True | 150 | 4188 | True |
| dcfg_nolive | 0.4715 | 0.3982 | True | 0.9012 | True | 219 | 3951 | True |
| dcfg_frozensensor | 0.4592 | 0.4103 | True | 0.8990 | True | 284 | 3183 | True |
| fixed | 0.4921 | 0.4375 | True | 0.8705 | True | 294 | 3184 | True |
| nosensor | 0.4583 | 0.3902 | True | 0.8945 | True | 293 | 3184 | True |

## (b) other detectors (3 steps) and batch size

- xception: largest batch dcfg=32, augft=32 -> **32**
- f3net: largest batch dcfg=32, augft=32 -> **32**; fixed one-step ok=True
- efficientnetb4: largest batch dcfg=32, augft=32 -> **32**; fixed one-step ok=True
- ucf: largest batch dcfg=32, augft=32 -> **32**; fixed one-step ok=True; UCF native-loss augft: amp=true ok=True finite=True, amp=false ok=True finite=True

## (c) resume

- pass: True; killed at ~250 steps, resume markers [{'epoch': 0, 'step_in_epoch': 200, 'step': 200}]; steps logged 400/400; missing []; hash conflicts []; ref mismatches []; repeated hashes 0; re-executed steps (ckpt..kill) [200, 249]

## (d) timing (img/s, steady state)

- efficientnetb4_augft: bs 32, 123 img/s, peak 5079 MB, val (full, extrapolated) 6.2 min
- ucf_dcfg: bs 32, 165 img/s, peak 6990 MB, val (full, extrapolated) 8.9 min
- xception_dcfg: bs 32, 216 img/s, peak 3951 MB, val (full, extrapolated) 5.5 min
- f3net_fixed: bs 32, 269 img/s, peak 3223 MB
- f3net_dcfg: bs 32, 205 img/s, peak 3991 MB, val (full, extrapolated) 5.7 min
- xception_fixed: bs 32, 317 img/s, peak 3184 MB
- xception_augft: bs 32, 203 img/s, peak 3323 MB, val (full, extrapolated) 4.5 min
- efficientnetb4_dcfg: bs 32, 123 img/s, peak 5734 MB, val (full, extrapolated) 7.2 min
- ucf_augft: bs 32, 70 img/s, peak 7246 MB, val (full, extrapolated) 7.6 min
- ucf_fixed: bs 32, 203 img/s, peak 6225 MB
- efficientnetb4_fixed: bs 32, 152 img/s, peak 4970 MB
- xception_dcfg_jointft: bs 32, 158 img/s, peak 4188 MB
- f3net_augft: bs 32, 183 img/s, peak 3495 MB, val (full, extrapolated) 4.8 min

ETA table: phase8_eta.md
Overall pass (a+c): True
