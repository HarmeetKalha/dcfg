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
- (in progress) `implementer` building ops scripts + paths/heartbeat/seed utils + tests.

## Blockers (user action needed)
- (done) `.wslconfig` applied.
- (done) gh authenticated; user confirmed ntfy test arrived on phone.

## Next
- Freeze `requirements.lock` (after implementer finishes `pip install -e .`).
- `implementer`: ops scripts (Section 5) + `register_tasks.ps1` + paths/heartbeat utils + tests. Runner smoke test.
- GATE 1.
