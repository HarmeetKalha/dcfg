# DCFG project — rules for Claude Code
Plan: docs/PLAN.md. Progress: docs/PROGRESS.md (update at end of every session). Decisions/deviations: docs/DECISIONS.md.

## Locked decisions
- Pluggable front-end before a FROZEN existing detector; gradients flow through the detector into the module.
- Parallel paths: degradation sensor (continuous 128-d embedding, LIVE-pretrained) and 8x8 blockwise DCT on YCbCr.
- Fluid split: gate outputs 3x64 per-coefficient weights per image, w = 2*sigmoid(z), zero-init last layer => identity at init. NO hardcoded low/mid/high bands anywhere.
- No curriculum training. Uniform random degradation sampling (PLAN 7.2).
- FF++ c23, official split, DeepfakeBench preprocessing, 32 frames/video, 256 crops. c40 test as real-compression test.
- Detectors: Xception, EfficientNet-B4, F3Net, UCF (DeepfakeBench weights). MkfaNet optional Tier 3.
- Metrics: frame + video AUC per condition, bootstrap 95% CI.

## Integrity
- Test split never used for selection/tuning. Val only.
- Paper numbers only from results/raw/*.json.
- Never cut epochs/data/frames to save time; ask the user instead.

## Token discipline
- Long jobs (download, preprocess, train, eval) go in ops/queue.txt and run via the DCFG-Runner task. Never poll, never sleep. One short check with ops/status.sh, update PROGRESS.md, end the turn.
- Never print big logs; use ops/status.sh, tail -n 40, grep, or a subagent summary.
- Delegate code writing to `implementer`, data work to `data-engineer`, reviews to `reviewer`, results to `results-analyst`.

## Environment
- WSL2 Ubuntu 24.04, venv ~/dcfg/.venv (Python 3.11). Data in ~/data, never in git, never under /mnt/c.
- Secrets in ~/.dcfg_env. Ask before sudo, before deleting anything in ~/data, before git push.
