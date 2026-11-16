# PROGRESS

**Current phase:** 2 — Data acquisition (in progress). GATE 1 PASSED 2026-11-16.
**Last updated:** 2026-11-16

## Done
- Plan moved to `docs/PLAN.md`. Repo layout (Section 6), `CLAUDE.md`, 4 agents, `.claude/settings.json`, `.gitignore`, `docs/DECISIONS.md` created. `git init` (branch `main`, no commits yet).
- uv 0.13.8 installed (`~/.local/bin`). Python 3.11 venv at `~/dcfg/.venv`.
- PyTorch 2.11.0+cu128 installed and verified: cuda available, RTX 3060, 4096² fp16 matmul OK. (Needed UV_HTTP_TIMEOUT=600; pypi.nvidia.com timed out at default.)
- numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, pillow, opencv-python-headless, scikit-image, tqdm, timm, efficientnet_pytorch, wandb, pytest installed; imports verified.

- apt packages + gh 2.104.0 installed (ops/bootstrap_sudo.sh). git identity set, remote origin = https://github.com/HarmeetKalha/dcfg.git.
- ~/.dcfg_env (chmod 600) verified: W&B login OK (entity harmeet-singh-kalha-honours), ntfy HTTP 200, healthchecks HTTP 200.

- WSL restarted: 11 GiB visible, custom agents loaded.
- dlib 20.0.1 built from source (CPU-only; fine for landmark preprocessing).
- Ops scripts built (ops/{run_queue,watchdog,notify,status}.sh, queue.txt, register_tasks.ps1, smoke_gpu.py), src/dcfg/{paths,utils/heartbeat,utils/seed}.py, pyproject.toml (editable install), tests: 11 passed. shellcheck not installed (bash -n only).
- requirements.lock frozen (98 pkgs). First commit c850f36 (not pushed).
- GATE 1 review done: BLOCKER (.gitignore `data` ignored src/dcfg/data) fixed in 5514389. SHOULD-FIXes done, tests 21 passed (watchdog restart via schtasks, job process-group kill, pinger leak, pidfile liveness, heartbeat alert tags/cooldown, CRLF, hidden task window). See DECISIONS.md.
- Task Scheduler tasks registered by user (DCFG-Runner Ready; watchdog PT30M, indefinite).
- smoke_gpu_01 DONE via Task Scheduler runner; user confirmed ntfy + W&B. GATE 1 passed. Pushed to GitHub (private HarmeetKalha/dcfg); secret scan of history clean.
- Phase 2: FF++ script present at ~/data/ffpp_raw/faceforensics_download_v4.py. Script reviewed: stock v4 downloader, safe. Server EU2 (EU refused, CA timeout). ops/jobs/ffpp_download.sh + 10 queue jobs (c23 then c40 x 5 types).
- RUNNING: FF++ downloads started 13:44 UTC 2026-11-16. Rate settled at ~2.6 s/video (516/1000 c23 original at 14:04 UTC) -> ~43 min per type, ETA all 10 downloads ~21:00 UTC, then preprocessing ~2-2.5 h.
- LIVE VERIFIED: zip from user's Downloads -> ~/data/live/ChallengeDB_release/{Data,Images} (extracted excluding __MACOSX/.DS_Store; 413 MB). live_check: 1169 .mat entries (images + MOS), calibration t1-t7.bmp in Images/trainingImages, 1162 non-calibration images all open. Zip kept at ~/data/live/ChallengeDB_release.zip.

## Out-of-order work while FF++ downloads (user-approved 2026-11-16)
Code/prep that does not need FF++ data, pulled forward from later phases. Gates stay in order; items marked DEFERRED must be done when FF++ is ready.
- [x] A. (done: DFB f188b1c pinned; wrappers ops/jobs/dfb_{preprocess,rearrange}.{py,sh}, no DFB patch; 81-pt landmarks in ~/data/weights; official split jsons in ~/data/ffpp_frames/FaceForensics++/; dry run 2 videos OK 32/32 frames; ETA ~2-2.5 h at 6 workers; jobs dfb_pre_c23, dfb_pre_c40, dfb_rearrange QUEUED after downloads) Phase 3.1-3.3 prep: clone DeepfakeBench (pin commit in DECISIONS), data-engineer report on its preprocessing/JSON pipeline, dlib landmark model -> ~/data/weights/, prepare c23 (all splits) + c40 (test only) preprocessing configs/jobs (not queued).
  - DEFERRED: 3.5 counts + QA grid, GATE 3.
- [x] B. (done: DFB v1.0.1 weights in ~/data/weights/dfb (0.85 GB); all 4 load with 0 missing/unexpected keys; prob parity with DFB atol 1e-4; all use 256 input, mean=std=0.5; fp16 bs32 img/s: xcep 785, effb4 506, f3net 741, ucf 353; video AUC = mean frame prob (matches DFB); evaluate.py --frontend not wired yet (Phase 8)) Phase 4.1-4.3 code: DFB checkpoints -> ~/data/weights/dfb/, detectors/adapter.py (zero missing/unexpected keys asserted), eval/metrics.py + evaluate.py with unit tests on synthetic data.
  - DEFERRED: 4.4 clean c23 test reproduction, GATE 4.
- [x] C. (done: 30 tests pass, 834 img/s; mixed op order blur->noise->jpeg; eval seed = sha256(name|key)) Phase 5.1: data/degradations.py (7.2 sampler + 7.3 deterministic conditions incl. WebP, motion blur), configs/eval/conditions.yaml, tests.
  - DEFERRED: 5.2 freq study, 5.3-5.4 baseline eval + curves, GATE 5.
- [~] D. (code done + reviewed, blocker (clobber) fixed, 94 tests pass; QUEUED: sensor_live after ffpp_dl_c23_original (~7 min), sensor_live_p10 at end of queue; results -> results/raw/sensor_live__*.json; then report SRCC/PLCC) Phase 6.1-6.3, 6.5: data/live.py, models/sensor.py, train/train_sensor.py, queue sensor training (runs after downloads; runner is serial), SRCC/PLCC report, save sensor_live.pt.
  - DEFERRED: 6.4 FF++ val embedding plot -> then GATE 6.
- [x] E. (code done; GATE 7 review: 0 blockers, test fixes applied; per-part LayerNorm added per user decision; 2420 img/s; awaiting user GATE 7 approval) Phase 7 complete: models/dct.py, gate.py, fixed.py, frontend.py + tests + reviewer -> GATE 7 (needs no FF++).
- [ ] F. Phase 8.1 code only: train/train_frontend.py + configs.
  - DEFERRED: 8.2 smoke/resume tests, 8.3 timing/ETA, 8.4 config/queue generation, GATE 8.
Order: A + C + E in parallel now, then B, D, F.

## Blockers (user action needed)
- none

## Next
- When downloads finish: verify 1000 videos per type for c23 and c40, record sizes -> GATE 2.
