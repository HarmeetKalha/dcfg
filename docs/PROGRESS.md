# PROGRESS

**Current phase:** 1 — Environment and skeleton (in progress)
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
- Smoke job `smoke_gpu_01` queued in ops/queue.txt (not started).

## Blockers (user action needed)
- (done) `.wslconfig` applied.
- (done) gh authenticated; user confirmed ntfy test arrived on phone.

## Next
- USER: register Task Scheduler tasks from Windows PowerShell (auto mode blocked Claude):
  `powershell -ExecutionPolicy Bypass -File \\wsl$\Ubuntu-24.04\home\harmeet\dcfg\ops\register_tasks.ps1`
- Then: `schtasks.exe /run /tn DCFG-Runner`, one ops/status.sh check; user confirms ntfy start/done + W&B smoke run.
- Reviewer pass -> GATE 1.
