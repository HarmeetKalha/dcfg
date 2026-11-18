# Design selection: recommendation (validation only; PILOT, 5 epochs) — proposal, pending user decision

Source: results/pilot/design_selection/report.md (val split, 700 videos, paired video bootstrap).

## Findings
- dcfgfilmlite (degraded mean 0.8599 [0.8409, 0.8773]) ≈ dcfgfilm (0.8589 [0.8398, 0.8764]); both beat dcfg by +0.045 (CI excludes 0) on every degraded condition and on c40. Clean AUC unchanged vs base (Δ -0.0001).
- filmlite is cheaper: 0.76M added params (3.5%) vs 2.60M (11.9%); +0.49 vs +0.51 ms/img (b32).
- Consistency alone adds nothing: dcfgcons - dcfg = +0.0008 [-0.0031, +0.0046] (only c40 +0.004*). FiLM variants include consistency, so FiLM's contribution without it is untested.
- No frozen design reaches fine-tuning: filmlite - augft = -0.039 [-0.051, -0.027]; filmlite recovers ~72% of the base->augft degraded-mean gap (0.759 -> 0.860 vs 0.899).
- jointft ≈ augft: +0.0025 [-0.0005, +0.0054] (only mixed_heavy +0.006*). DCFG adds ~nothing on top of fine-tuning at 5 epochs.
- Caveats: 5 of 30 epochs; models selected on the same val split they are scored on (equal bias across variants); Xception only.

## Recommended final method: dcfg_filmlite (frozen detector)
Chosen because it is within CI of dcfgfilm and cheaper; it is the best frozen-detector design.

## Proposed restructured matrix (test evaluation; full 30-epoch schedule, early stopping)
| group | runs | n | worst GPU-days (30 ep) | likely (15 ep, assumption) | cumulative worst / likely |
|---|---|---|---|---|---|
| 1 core | filmlite, fixed, augft x {xception, effb4, f3net, ucf} | 12 | 6.60 | 3.44 | 6.60 / 3.44 |
| 2 joint FT | jointft (plain DCFG gate + fine-tuned detector) x 4 detectors | 4 | 3.11 | 1.60 | 9.71 / 5.05 |
| 3 Xception ablation chain | dcfg (plain gate) -> dcfgcons (+consistency) -> filmlite (from G1); plus filmlite_nocons (FiLM without consistency), filmlite_nolive, filmlite_frozensensor | 5 | 2.26 | 1.17 | 11.97 / 6.22 |
| 4 seeds | filmlite_xception s43/s44, augft_xception s43/s44 | 4 | 1.74 | 0.90 | 13.71 / 7.12 |
base_<det> test evaluations already exist (no training).
ETA basis: measured Xception pilot epoch times (filmlite 22.2, fixed 14.0, augft 17.8, jointft 22.9, dcfg 18.8, dcfgcons 22.1 min/epoch incl. val), scaled to other detectors by the phase-8 per-detector ratios of the same variant class (ASSUMPTION; jointft scaled like augft, so jointft_ucf ~60 min/epoch is conservative); filmlite_nocons assumed = dcfg time; test eval per run from base_<det> logs.
Dropped vs old matrix: gate_nosensor (FiLM is conditioned on the sensor, so "no sensor" needs a new design), dcfgfilm 4-site (≈ filmlite; stays as appendix pilot), dcfg/fixed seed repeats replaced by filmlite/augft seeds.
