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
- RUNNING: FF++ downloads started ~13:50 UTC 2026-11-16 (ffpp_dl_c23_original first). ETA rough: ~4-8 h total (~12-15 GB at ~0.7+ MB/s; speed unmeasured on bulk).
- LIVE VERIFIED: zip from user's Downloads -> ~/data/live/ChallengeDB_release/{Data,Images} (extracted excluding __MACOSX/.DS_Store; 413 MB). live_check: 1169 .mat entries (images + MOS), calibration t1-t7.bmp in Images/trainingImages, 1162 non-calibration images all open. Zip kept at ~/data/live/ChallengeDB_release.zip.

## Blockers (user action needed)
- none

## Next
- When downloads finish: verify 1000 videos per type for c23 and c40, record sizes -> GATE 2.
