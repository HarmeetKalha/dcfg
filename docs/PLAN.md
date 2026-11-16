# DCFG Build Plan — Degradation-Classifying Frequency Gate

**For Claude Code.** This is the master plan for building the entire honours research project from an empty Windows 11 PC to final, measured, paper-ready results. Read this whole file before doing anything. Follow the phases in order. Stop at every **GATE** and every **HUMAN STEP**.

---

## 0. How to use this file (read first, Claude)

- This file goes into the repo as `docs/PLAN.md` during Phase 1. Treat it as the source of truth. If something here turns out to be wrong in practice, do not silently deviate: record the change and the reason in `docs/DECISIONS.md`, then continue.
- Keep `docs/PROGRESS.md` updated at the end of every session: current phase, what was done, what is running, what is next, any blocker. A future session must be able to resume from that file alone.
- **Token discipline (critical):**
  - Never wait on, poll, or `sleep` through long jobs (downloads, preprocessing, training, evaluation). Put them in the job queue (Section 5), start the runner, run **one** short sanity check that the job actually started, update `PROGRESS.md` with an ETA, then **end the turn** and tell the user when to come back.
  - Never print large logs or files. Use `ops/status.sh`, `tail -n 40`, `grep`, or a subagent that returns a summary.
  - Delegate code-writing and large-file reading to subagents (Section 4). The main session plans, reviews, and decides.
- **Research integrity rules (non-negotiable):**
  - The test split is never used for model selection, early stopping, or tuning. Validation split only.
  - Every number that will go into the paper must come from a JSON file in `results/raw/`, written by the evaluation code. No hand-typed numbers.
  - Negative or null results are reported as they are. Do not tune on test to make the method look better.
  - Do not reduce epochs, data, or frames to save time. If something is too slow, report it at a GATE and let the user decide.

---

## 1. Project summary and locked design decisions

### 1.1 What we are building

A lightweight, **pluggable front-end module** placed before an **existing, unmodified deepfake detector**. Pipeline (parallel, not sequential):

```
input face crop (RGB, [0,1], 256x256)
  ├─► Degradation-Sensing CNN ──► degradation embedding (128-d)
  └─► Blockwise DCT (8x8, YCbCr) ──► 3x64 coefficient maps + log-energy stats (192-d)
                    │
        Gate (MLP on [embedding ‖ energy stats]) ──► 192 per-image trust weights
                    │
        coefficients × weights ──► inverse DCT ──► reweighted RGB image
                    │
        detector's own normalization ──► existing detector (frozen) ──► real/fake logit
```

### 1.2 Locked decisions (copy these into `CLAUDE.md`)

| # | Decision |
|---|---|
| D1 | **The DCT split is fluid, not hardcoded.** There is NO fixed "low/mid/high" band split anywhere in the code. The module uses an 8×8 blockwise orthonormal DCT-II (the same transform as JPEG) on YCbCr, and the gate outputs one weight **per DCT coefficient position per channel** (3 × 64 = 192 weights), computed fresh for every image. The effective frequency grouping is learned and varies with each image's degradation. The Part 1 frequency study (Phase 5) is used to analyze and interpret this, not to hardcode bands. |
| D2 | The degradation sensor outputs a **continuous embedding** (128-d), never a category label. It is pretrained on LIVE In-the-Wild (MOS regression), then its regression head is discarded. |
| D3 | The gate parameterization is `w = 2·sigmoid(z)`, range (0, 2). The last gate layer is **zero-initialized** so `w = 1` exactly at init ⇒ the module starts as an exact identity. A small L2 penalty `λ·mean((w−1)²)` with λ = 1e-4 is applied. |
| D4 | Detectors are **frozen** (eval mode, `requires_grad=False`, BatchNorm stats fixed) in the main setting. Gradients flow *through* the detector into the module. |
| D5 | **Curriculum training is dropped** (time). All training uses uniform random degradation sampling (Section 7.2). The paper must not claim curriculum training. |
| D6 | Primary data: FaceForensics++ **c23**, official 720/140/140 video split, 32 frames/video, 256×256 face crops, produced with **DeepfakeBench's** preprocessing so DeepfakeBench pretrained weights remain valid. |
| D7 | Detectors: **Xception, EfficientNet-B4, F3Net, UCF** via DeepfakeBench pretrained weights. **MkfaNet** is Tier 3 (optional, only if time remains). |
| D8 | Metrics: frame-level AUC and video-level AUC (mean of frame probabilities per video), per degradation condition, with 1000× bootstrap 95% CIs over videos. |
| D9 | Generalization to unseen *manipulation methods* is out of scope. Do not claim it. |
| D10 | Plain Python scripts, no notebooks. Data lives outside the repo. |

### 1.3 Known facts carried over from earlier work (do not rediscover)

- LIVE Challenge DB: `Data/AllImages_release.mat` key `AllImages_release`, shape (1169,1), filename = `arr[i][0][0]`. `Data/AllMOS_release.mat` key `AllMOS_release`, shape (1,1169), MOS = `arr[0][i]`, same order. Entries `t1.bmp`–`t7.bmp` are calibration images in `Images/trainingImages/` and are **excluded** → **1162** usable images, all other images directly in `Images/`. Images are 500×500, mixed `.bmp`/`.JPG`. The zip contains a junk `__MACOSX/` folder and may have a duplicated nested `ChallengeDB_release/ChallengeDB_release/` folder; flatten it.
- An earlier 2-epoch CPU debug run of the sensor worked end to end (MSE loss on MOS ~450–525 after 2 epochs). Expect much better after full GPU training.
- **Do not resize LIVE images to 224 for pretraining.** Resizing destroys the very degradation cues we want the sensor to learn. Use random 224×224 crops (train) and center or multi-crop averaging (eval).

---

## 2. Hardware and environment

- Windows 11, Ryzen 5 7600X (6C/12T), 16 GB RAM, **RTX 3060 12 GB**, ~740 GB free disk.
- Everything runs inside **WSL2 Ubuntu 24.04**. Reason: DeepfakeBench, dlib, and multi-worker data loading are Linux-first; native Windows would cost more debugging time than WSL setup.
- **All data and code live on the WSL ext4 filesystem** (`/home/<user>/...`), never under `/mnt/c/`. Cross-filesystem I/O is extremely slow.
- 16 GB RAM is the tightest resource. WSL is capped at 12 GB with 16 GB swap (Section 3). DataLoader workers: start at 6, reduce if RAM pressure.

---

## 3. HUMAN STEPS before Claude Code can work (Phase 0, ~1 hour)

These need admin rights, reboots, personal accounts, or email access, so the user does them.

**H0.1 — NVIDIA driver (Windows).** Install the latest NVIDIA Game Ready or Studio driver for the RTX 3060 from nvidia.com or the NVIDIA app. Do NOT install any NVIDIA driver inside WSL later; WSL uses the Windows driver.

**H0.2 — Install WSL2 + Ubuntu.** Open **PowerShell as Administrator**:
```powershell
wsl --install -d Ubuntu-24.04
```
Reboot when asked. On first launch of Ubuntu, create a Linux username and password (write them down).

**H0.3 — WSL resource limits.** In Windows, create the file `C:\Users\<WindowsUser>\.wslconfig` with:
```ini
[wsl2]
memory=12GB
processors=12
swap=16GB
```
Then in PowerShell: `wsl --shutdown`, and reopen Ubuntu.

**H0.4 — Verify GPU in WSL.** In the Ubuntu terminal: `nvidia-smi`. It must list the RTX 3060. If not, update the Windows driver and run `wsl --update` in PowerShell.

**H0.5 — Keep the PC awake and stop surprise reboots.**
- PowerShell (admin): `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`.
- Settings → Windows Update → **Pause updates** for the maximum period, and set Active hours to cover the whole day.
- After any reboot, someone must **log in to Windows** for the training runner to auto-restart (Section 5).

**H0.6 — Accounts and notifications.**
- Create a free **Weights & Biases** account; copy your API key (wandb.ai/authorize).
- Install the **ntfy** app on your phone, subscribe to a long random topic name (e.g. `dcfg-harmeet-8f3k2q9z`). This is how the watchdog pings your phone.
- (Recommended) Create a free **healthchecks.io** check with a 30-minute period and grace time, and connect it to your email/phone. Copy its ping URL. This is a dead-man switch: it alerts you if the PC dies, sleeps, or reboots without anyone logging in, which the PC itself cannot report.
- (Recommended) Create an empty **private GitHub repo** named `dcfg` for code backup.

**H0.7 — FaceForensics++ download script.** Find the FF++ approval email and save the attached/linked `download-FaceForensics.py` (or the link to it). You will give it to Claude in Phase 2. Claude cannot read your email.

**H0.8 — Install Claude Code inside WSL.** In the Ubuntu terminal, install Claude Code using the Linux native installer from the official setup page (code.claude.com/docs → Setup), then run `claude` once and log in. Run Claude Code **inside Ubuntu**, not from Windows, for every phase.

**H0.9 — Create the project folder and start.**
```bash
mkdir -p ~/dcfg/docs && cd ~/dcfg
# copy this plan file to ~/dcfg/docs/PLAN.md (e.g. from Windows: \\wsl$\Ubuntu-24.04\home\<user>\dcfg\docs\)
claude
```

---

## 4. Claude Code setup: models and subagents

- Main session: **Opus** (`/model opus`). It plans, reviews diffs, makes decisions, runs gates. Do not use the `opusplan` alias; there are reports that it overrides subagent model settings.
- Token-heavy work goes to custom subagents running **Sonnet**. Create these four files in Phase 1 under `.claude/agents/`. Verify with `/agents` that they load.

**`.claude/agents/implementer.md`**
```markdown
---
name: implementer
description: Writes and modifies Python modules, scripts, configs and pytest tests for the DCFG project from a precise spec given by the main agent. Use for any non-trivial code writing.
model: sonnet
---
You implement exactly the spec you are given for the DCFG project. Read CLAUDE.md and docs/PLAN.md sections referenced in the task first. Rules: write clean, typed, documented Python 3.11; put library code in src/dcfg/, CLI entry points as `python -m dcfg.<module>`; every new module gets a pytest test in tests/; run `pytest -q` on the touched tests before finishing; never touch data under ~/data except read-only; never print large outputs. Finish with a short report: files changed, tests run and their result, anything you were unsure about.
```

**`.claude/agents/data-engineer.md`**
```markdown
---
name: data-engineer
description: Handles dataset download scripts, preprocessing, integrity checks and the DeepfakeBench preprocessing pipeline. Use for data acquisition and data QA tasks.
model: sonnet
---
You manage data for the DCFG project. Data lives in ~/data (never in the repo). Read CLAUDE.md first. For long-running work, write a job script and add it to ops/queue.txt; never run multi-hour jobs in the foreground. Verify counts and integrity with scripts, and report results as short tables. Never delete raw data without explicit instruction.
```

**`.claude/agents/reviewer.md`**
```markdown
---
name: reviewer
description: Read-only critical review of code, configs and results for bugs, data leakage, evaluation errors and violations of the locked decisions in CLAUDE.md. Use before every GATE and before launching any long training job.
model: sonnet
tools: Read, Grep, Glob, Bash
---
You review; you never edit files. Check against CLAUDE.md locked decisions and docs/PLAN.md. Look specifically for: test-set leakage into selection; detector not truly frozen (eval mode, requires_grad, BN); normalization applied twice or not at all; degradation randomness not seeded for eval; resume logic that skips or repeats data; AUC computed on wrong labels/probabilities; frame vs video AUC confusion; any hardcoded low/mid/high band split. Return a prioritized list: BLOCKER / SHOULD-FIX / NIT, each with file:line and a one-line fix.
```

**`.claude/agents/results-analyst.md`**
```markdown
---
name: results-analyst
description: Turns results/raw/*.json into tables, figures and a findings summary, and checks numbers for internal consistency. Use in Phases 10 and 11.
model: sonnet
---
You only use numbers from results/raw/*.json. Build tables (CSV + Markdown + LaTeX) and figures (PDF + PNG, matplotlib, readable at IEEE column width) via scripts in src/dcfg/analysis/ so they can be regenerated. Flag any inconsistency, missing run, or suspicious value. Write honest summaries: report gains with CIs, report where the method does not help.
```

- Permissions: to keep human intervention low, add an allow-list in `.claude/settings.json` for routine commands (`python`, `pytest`, `uv`, `git`, `bash ops/*`, `nvidia-smi`, `ls`, `tail`, `grep`, `mkdir`, `cp`, `mv` inside `~/dcfg` and `~/data`). Ask the user before any `sudo`, any deletion under `~/data`, and any `git push`.

---

## 5. Long-job infrastructure (runner, watchdog, notifications)

Goal: training runs for days without Claude Code open and without spending tokens; it survives crashes and reboots; the user gets phone alerts.

**Files (Phase 1 builds these):**
- `ops/queue.txt` — one job per line: `<job_id> | <shell command>`. Lines starting with `#` are ignored.
- `ops/run_queue.sh` — takes a lock (`flock`) so only one runner exists; for each job without `runs/<job_id>/DONE`: runs the command (all training/eval commands auto-resume from their own checkpoints), retries up to 3 times on failure with a 60 s pause, writes `DONE` on success or `FAILED` after 3 failures, then moves on. Sends ntfy messages on job start, job done, job failed, and queue finished. Pings the healthchecks URL every 10 minutes while running. Logs to `runs/_runner.log`.
- `ops/watchdog.sh` — checks: runner alive; current job's heartbeat file updated within 20 min; GPU temp (`nvidia-smi --query-gpu=temperature.gpu`) below 85 °C; free disk > 50 GB; RAM. Sends ntfy on any problem; restarts the runner if the queue is unfinished and the runner is dead.
- `ops/notify.sh` — `curl -s -d "$1" ntfy.sh/$NTFY_TOPIC`.
- `ops/status.sh` — one compact screen: queue progress, current job, epoch/step, last val AUC, throughput, ETA, heartbeat age, GPU util/temp/VRAM, disk free. **This is the only way Claude checks on long jobs.**
- `ops/register_tasks.ps1` — registers two Windows Task Scheduler tasks for the current user:
  - `DCFG-Runner`: trigger **at logon** + on demand; action `wsl.exe -d Ubuntu-24.04 -u <linuxuser> -- bash -lc "~/dcfg/ops/run_queue.sh"`; do not stop if running long; restart on failure every 5 min, 3 times. The `wsl.exe` process staying alive is what keeps the WSL VM up while Claude Code is closed.
  - `DCFG-Watchdog`: every 30 min; runs `ops/watchdog.sh` via `wsl.exe`.
- Secrets in `~/.dcfg_env` (not in git): `WANDB_API_KEY`, `NTFY_TOPIC`, `HC_PING_URL`. Sourced by all ops scripts.
- Training scripts write `runs/<job_id>/heartbeat` (timestamp + step) every ~60 s.

**How Claude launches long work:** append job lines to `ops/queue.txt` → `schtasks.exe /run /tn DCFG-Runner` → one check ~2 minutes later with `ops/status.sh` (a single bounded command, no polling loop) → update `PROGRESS.md` with ETA → end turn.

---

## 6. Repository and data layout

```
~/dcfg/                                  (git repo)
  CLAUDE.md
  README.md
  pyproject.toml, requirements.lock
  .claude/agents/*.md, .claude/settings.json
  docs/PLAN.md, PROGRESS.md, DECISIONS.md, RUNBOOK.md
  configs/
    sensor_pretrain.yaml
    detectors/{xception,efficientnetb4,f3net,ucf}.yaml
    eval/conditions.yaml
    train/<run_id>.yaml                  (one per run in the experiment matrix)
  src/dcfg/
    paths.py                             (DATA_ROOT=~/data, RUNS_ROOT, RESULTS_ROOT)
    data/live.py, ffpp.py, degradations.py, transforms.py
    models/sensor.py, dct.py, gate.py, frontend.py, fixed.py
    detectors/adapter.py, registry.py
    train/train_sensor.py, train_frontend.py
    eval/evaluate.py, metrics.py
    analysis/freq_study.py, gate_viz.py, overhead.py, make_tables.py, make_figures.py
    utils/seed.py, checkpoint.py, logging.py, heartbeat.py
  tests/
  ops/queue.txt, run_queue.sh, watchdog.sh, notify.sh, status.sh, register_tasks.ps1
  third_party/DeepfakeBench/             (cloned at a pinned commit, gitignored)
  runs/                                  (gitignored)
  results/raw/*.json                     (tracked)
  results/tables/, results/figures/      (tracked)

~/data/                                  (never in git)
  live/ChallengeDB_release/{Data,Images}/
  ffpp_raw/                              (output of the FF++ download script)
  ffpp_frames/                           (DeepfakeBench preprocessed crops)
  dfb_json/                              (DeepfakeBench dataset JSONs)
  weights/                               (DeepfakeBench checkpoints, dlib landmark model)
```

---

## 7. Experimental protocol (fixed before any training)

### 7.1 Splits
FF++ official 720/140/140 video split as produced by DeepfakeBench JSONs. Train and val come from c23. Test: c23 test (with synthetic degradations) and **c40 test** (real heavy H.264 compression, no synthetic degradation applied).

### 7.2 Training degradation distribution (`degradations.py`)
Per training image: with p = 0.2 keep clean; otherwise apply 1–3 operations sampled without replacement, in random order, from:
- JPEG, quality ∈ U[20, 95]
- Gaussian blur, σ ∈ U[0.5, 4.0]
- Gaussian noise, σ ∈ U[2, 20] (0–255 scale)
- Downscale-then-upscale (bicubic), scale ∈ U[0.25, 0.9]

Applied to the 256×256 face crop **before** the module. Implemented with PIL/OpenCV on CPU inside DataLoader workers.

### 7.3 Evaluation conditions (`configs/eval/conditions.yaml`)
Deterministic: the RNG seed for each image is `hash(condition, image_path)`, so every model sees byte-identical degraded inputs.

| Group | Conditions |
|---|---|
| Clean | clean |
| JPEG | q90, q70, q50, q30, q10 |
| Blur | σ1, σ2, σ3, σ4 |
| Noise | σ5, σ10, σ15, σ20 |
| Resize | s0.5, s0.25 |
| Mixed | light (q70 + σblur1 + σnoise5), heavy (q30 + σblur2 + σnoise10) |
| **Held-out types (never seen in training)** | WebP q50, WebP q20, motion blur k7, motion blur k11 |
| **Real compression** | FF++ c40 test frames |

Validation (for early stopping only): clean, q50, q30, blur σ2, noise σ10, mixed-heavy, on the val split. Selection metric = mean frame AUC across these.

### 7.4 Experiment matrix

| Tier | Run ID pattern | Variant | Detectors | Seeds |
|---|---|---|---|---|
| 1 | `base_<det>` | Frozen detector, no module (evaluation only) | all 4 | — |
| 1 | `dcfg_<det>` | Full DCFG, detector frozen | all 4 | 42 |
| 1 | `fixed_<det>` | Input-independent learned weights (192 params, same parameterization), detector frozen | all 4 | 42 |
| 1 | `augft_<det>` | **Fairness control:** no module, detector fine-tuned with the same degradation augmentation and epochs | all 4 | 42 |
| 2 | `dcfg_jointft_xception` | DCFG + detector fine-tuned jointly | Xception | 42 |
| 2 | `dcfg_nolive_xception` | DCFG with sensor randomly initialized (no LIVE pretraining) | Xception | 42 |
| 2 | `gate_nosensor_xception` | Gate sees only DCT energy stats, no sensor | Xception | 42 |
| 2 | `dcfg_xception_s43/s44`, `fixed_xception_s43/s44` | Seed repeats | Xception | 43, 44 |
| 3 | `*_mkfanet` | Same as Tier 1 | MkfaNet | 42 |

`augft` answers the obvious reviewer question: "Is the gain just from training on degraded data?" `dcfg_jointft` answers: "Does the module still add something on top of augmentation fine-tuning?"

### 7.5 Optimization defaults
AdamW; module lr 1e-3 (gate/fixed), sensor lr 1e-4; detector fine-tune lr 2e-5; weight decay 1e-4; cosine schedule with 1 epoch warmup; AMP fp16 for the detector forward, **DCT and gate in fp32** (autocast disabled inside the module); batch size as large as fits (start 32, reduce for UCF/EffNet-B4 if OOM); class-balanced sampling (real:fake 1:1). Max 30 epochs, early stopping on the val selection metric with patience 6. Checkpoint every epoch **and** every 2000 steps (mid-epoch resume uses a per-epoch permutation seeded by `seed + epoch`, so batches already seen can be skipped exactly).

---

## 8. Phases

Each phase lists tasks, who does them, the acceptance criteria, and the GATE. At each GATE: have the `reviewer` subagent review, fix BLOCKERs, update `PROGRESS.md`, show the user a short summary, and wait for "continue".

### Phase 1 — Environment and skeleton (~1–2 h, interactive)
1. Ask the user for `sudo` once: `sudo apt update && sudo apt install -y build-essential cmake git ffmpeg unzip curl libgl1 libglib2.0-0 python3-dev`.
2. Install `uv`; create a **Python 3.11** venv at `~/dcfg/.venv`.
3. Install PyTorch with CUDA from the official PyTorch index (current stable, CUDA 12.x build). Verify: `torch.cuda.is_available()` is True, device name contains "3060", a 4096×4096 fp16 matmul runs.
4. Install: numpy, scipy, pandas, scikit-learn, matplotlib, pyyaml, pillow, opencv-python-headless, scikit-image, tqdm, timm, efficientnet_pytorch, wandb, pytest, dlib (builds from source with cmake; if it fails, record and try a prebuilt wheel). Freeze to `requirements.lock`.
5. Create the repo layout (Section 6), `git init`, `.gitignore` (runs/, third_party/, .venv/, *.pt, data symlinks), `CLAUDE.md` (Section 9), the four agents, `.claude/settings.json`, `docs/PROGRESS.md`, `docs/DECISIONS.md`.
6. Ask the user for `WANDB_API_KEY`, `NTFY_TOPIC`, `HC_PING_URL` → write `~/.dcfg_env` (chmod 600). Run `wandb login`.
7. Build Section 5 ops scripts. Run `powershell.exe -ExecutionPolicy Bypass -File ops/register_tasks.ps1`; if it fails due to permissions, give the user the exact command to run in PowerShell.
8. Smoke test the runner: queue a 3-minute dummy GPU job, start the runner, confirm ntfy start/finish messages arrive on the user's phone and W&B shows a dummy run.
9. Optional: add the GitHub remote and push (ask first).

**GATE 1:** GPU visible in torch; tests pass; ntfy + W&B + healthchecks confirmed by the user; Task Scheduler tasks exist.

### Phase 2 — Data acquisition (~1 h active, downloads in background)
1. **LIVE:** download the ChallengeDB release from `live.ece.utexas.edu/research/ChallengeDB/index.html` into `~/data/live/`. If the download needs a form or manual click, it's a HUMAN STEP: user downloads the zip in Windows and Claude copies it from `/mnt/c/Users/<user>/Downloads/`. Unzip, remove `__MACOSX`, flatten any nested duplicate folder, verify 1169 `.mat` entries and that all 1162 non-calibration images open.
2. **FF++ (HUMAN STEP H2.1):** user provides `download-FaceForensics.py`. Save to `~/data/ffpp_raw/download-FaceForensics.py` (not in git). Check its `--help`. The script asks for a keypress to accept the TOS; pipe `echo |` into it.
3. Queue download jobs (one per dataset type) into `ops/queue.txt`:
   - c23 videos: `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`
   - c40 videos: the same five types (only the test videos will be preprocessed, but the script downloads per type)
   - Use the server option that works fastest from India (try `EU2` first; record the choice).
4. Start the runner, confirm the first download is progressing, end the turn.
5. Next session: verify 1000 videos per type per compression level, record total size.

**GATE 2:** LIVE verified (1162 usable); FF++ c23 and c40 counts verified.

### Phase 3 — Preprocessing with DeepfakeBench (~1 h active, many hours background)
1. Clone DeepfakeBench into `third_party/DeepfakeBench`, record the commit hash in `DECISIONS.md`. The `data-engineer` subagent reads its README and `preprocessing/` code and reports how its config, landmark model, frame sampling, and JSON generation work.
2. Download the dlib landmark model that DeepfakeBench requires into `~/data/weights/`.
3. Configure DeepfakeBench preprocessing for: FF++ c23 (all splits, all five types) and FF++ c40 (**test split only**), 32 frames per video, its default face alignment and 256 crops, output to `~/data/ffpp_frames/`. Point its paths at `~/data` without editing DeepfakeBench files where possible (prefer config overrides; if a patch is needed, keep it as a `.patch` file in `ops/patches/` and record it).
4. Queue preprocessing (and the JSON rearrangement step) in the runner. It is CPU-bound; use 6–10 processes. End the turn.
5. Next session: verify per split and type: video count, frames per video (≈32), failed detections (record the rate; videos with zero faces are listed, not silently dropped). Generate `results/figures/qa_crops.png`: a 6×8 grid of random crops from each type, for the user to eyeball.

**GATE 3:** counts table + QA grid shown to the user and approved.

### Phase 4 — Detectors and reproduction (~2–3 h)
1. Download DeepfakeBench pretrained checkpoints for Xception, EfficientNet-B4, F3Net, UCF (Releases page) into `~/data/weights/dfb/`.
2. `src/dcfg/detectors/adapter.py`: a uniform wrapper `Detector(name)` with `.normalize(x01)` (each detector's own mean/std from its DeepfakeBench config), `.forward(x_normalized) -> logit (B,)`, `.freeze()`. Build each model through DeepfakeBench's own detector classes and configs so weights load with zero missing/unexpected keys (assert this). Handle any extra inputs a detector needs (UCF may need more than the image in its data dict; the adapter must hide that).
3. `src/dcfg/eval/evaluate.py` + `metrics.py`: frame AUC, video AUC, bootstrap CIs; writes one JSON per (run, condition) to `results/raw/`.
4. Evaluate all four on **clean c23 test**. Compare to DeepfakeBench's published FF++ c23 numbers (from its README or paper; record the source and value).

**GATE 4 (reproduction):** each detector within ±0.02 frame AUC of its published value. If not, stop and investigate (normalization, input size, frame selection, preprocessing mismatch) before continuing. Show the user the reproduction table.

### Phase 5 — Degradations, Part 1 frequency study, baseline robustness (~2 h active + background eval)
1. `degradations.py`: training sampler (7.2) and deterministic eval conditions (7.3), with tests: determinism (same seed → identical bytes), each op changes the image, ranges respected.
2. `analysis/freq_study.py` on 2000 clean c23 test crops: for each condition, the mean relative change in energy per 8×8 DCT position per YCbCr channel. Output 8×8 heatmaps per condition, and radial-frequency curves. Also a "fingerprint distance" matrix between degradation types. These are paper figures for the Part 1 study.
3. Queue baseline evaluation (`base_<det>`) of all four frozen detectors on every condition in 7.3 (including c40 and held-out types). End the turn.
4. Next session: `results-analyst` makes the baseline robustness curves (AUC vs severity per detector).

**GATE 5:** frequency study figures + baseline robustness table reviewed. This confirms the problem exists on our setup (AUC drops with degradation) before we invest GPU-days in training.

### Phase 6 — Degradation sensor pretraining on LIVE (~1 h active + ~1 h GPU)
1. `data/live.py` (logic from Section 1.3), `models/sensor.py`: 4 conv blocks (3×3 conv–BN–ReLU–maxpool, channels 32/64/128/128) → global average pool → linear 128 (embedding) → regression head (128→32→1). Fully convolutional, so 256 inputs work later.
2. `train/train_sensor.py`: random 224 crops + horizontal flips (no resize), MSE on MOS normalized to zero-mean/unit-std, AdamW lr 1e-3, up to 200 epochs, early stopping patience 20 on val SRCC, fixed 80/10/10 split with seed 42.
3. Report test SRCC and PLCC (center crop + 5-crop average). Also report median SRCC over 10 random 80/20 splits (standard IQA protocol) as a secondary number.
4. Sanity check: embed 500 FF++ val crops under several eval conditions, t-SNE/UMAP plot colored by condition, saved to `results/figures/sensor_embedding.png`. Degraded conditions should separate from clean.
5. Save `~/data/weights/sensor_live.pt` (backbone + embedding layer only).

**GATE 6:** SRCC/PLCC and the embedding plot. A modest SRCC (~0.5–0.7) is acceptable; a near-zero SRCC or no separation in the plot is a BLOCKER.

### Phase 7 — DCT, gate, and front-end module (~2–3 h)
1. `models/dct.py`: RGB↔YCbCr (JPEG/BT.601 full range); 8×8 blockwise orthonormal DCT-II and its inverse via a fixed 8×8 basis matrix (`C X Cᵀ`), batched and differentiable, fp32. Requires H, W divisible by 8 (256 is).
2. `models/gate.py`: input = concat(LayerNorm₁₂₈(sensor embedding 128), LayerNorm₁₉₂(per-image log mean |coefficient| energy per (channel, position) 192)), each part normalized by its own LayerNorm (learnable affine) → Linear 320→256 → GELU → Linear 256→192 (zero-init) → `w = 2·sigmoid(·)`. The nosensor variant uses LayerNorm₁₉₂ only → Linear 192→256. (Amended 2026-11-16, see DECISIONS.)
3. `models/fixed.py`: 192 learnable logits, same parameterization, identical for every image.
4. `models/frontend.py`: `FrontEnd(sensor, gate)` and variants (`fixed`, `nosensor`). forward(x01) → ycbcr → dct → multiply each 8×8 block's coefficients by w[channel, position] → idct → rgb → clamp [0,1]. Returns image and w (for logging/visualization).
5. Tests: DCT round-trip max error < 1e-5; at init the front-end output equals its input (max error < 1e-5); gradients reach the gate and sensor; no hardcoded band groupings anywhere (reviewer checks); throughput ≥ 2000 img/s at 256 on the 3060.

**GATE 7:** tests pass, reviewer finds no BLOCKER.

### Phase 8 — Training engine, smoke tests, timing (~3–4 h)
1. `train/train_frontend.py`, config-driven, supports all variants in 7.4: `dcfg`, `fixed`, `nosensor`, `dcfg_nolive`, `augft`, `dcfg_jointft`. Uses 7.2 degradations, 7.5 optimization, W&B + local CSV logging (train loss, val AUC per val condition, mean w per channel, lr, throughput), heartbeat, checkpoint/resume (7.5), `wandb.alert` on crash and on finish. Detector frozen check: assert no detector parameter has `requires_grad` unless the variant fine-tunes it, and that the detector is in eval mode every step.
2. Smoke tests on Xception (and one step on each other detector): 300 steps per variant, loss decreases, val runs, checkpoint written. **Resume test:** kill a run mid-epoch, restart it, confirm the step counter continues and no batch is repeated (log batch indices).
3. Measure throughput per detector per variant; compute the full ETA for the experiment matrix; write it to `PROGRESS.md`.
4. Generate all `configs/train/<run_id>.yaml` for Tiers 1–2 and the queue lines (train job, then its eval job, per run). Order: Tier 1 Xception first (dcfg, fixed, augft), then EffNet-B4, F3Net, UCF, then Tier 2.

**GATE 8 (launch approval):** show the user the ETA table and total GPU-days. Reviewer runs a full pre-launch review. On approval, start the runner and end the turn.

### Phase 9 — Full experiments (background, ~4–6 days GPU, rough estimate; Phase 8 replaces this with measured numbers)
- Claude is not active during this phase. The user checks the phone (ntfy, W&B).
- When the user returns and says "continue": run `ops/status.sh`, read `runs/_runner.log` tail, check for `FAILED` markers. For a failed job: diagnose from the last 40 log lines, fix, re-queue, restart the runner, end the turn.
- If time allows after Tiers 1–2: Tier 3 MkfaNet. First check whether its repo releases FF++ c23 weights trained with comparable preprocessing. If it requires training from scratch, report the estimated time to the user and ask before queueing.

### Phase 10 — Evaluation and analysis (~3–4 h active + background eval)
1. Every trained run is evaluated on all conditions in 7.3 (the queue already does this; re-run any missing).
2. `analysis/make_tables.py`:
   - **T1** Main table: per detector, AUC for baseline / fixed / augft / DCFG on clean, average over seen degradations, average over held-out types, and c40; ΔAUC vs baseline and vs augft with 95% CIs.
   - **T2** Per-condition table for Xception (all variants).
   - **T3** Ablations (Tier 2) with seed mean ± std.
   - **T4** Overhead: parameters, FLOPs if available, ms/image added (`analysis/overhead.py`, measured with CUDA events, batch 1 and 32).
   - **T5** Sensor pretraining results (SRCC/PLCC).
3. `analysis/make_figures.py`:
   - **F1** AUC vs severity curves per degradation type (Xception, all variants).
   - **F2** Learned gate weights: mean w as 8×8 heatmaps (Y channel, plus Cb/Cr) per condition (clean, q30, blur σ3, noise σ15, mixed-heavy, c40). This is the interpretability figure that shows the split is fluid and degradation-dependent.
   - **F3** Part 1 frequency-damage heatmaps (from Phase 5).
   - **F4** Cross-detector bar chart of ΔAUC.
   - **F5** Sensor embedding plot (from Phase 6).
4. Paired bootstrap tests for DCFG vs fixed and DCFG vs augft per detector.

**GATE 10:** results-analyst summary + reviewer consistency check (every table number traceable to a JSON).

### Phase 11 — Paper-ready package (~1–2 h)
1. `results/FINDINGS.md`: plain-language findings, including where the method does not help. No spin.
2. `results/PAPER_NUMBERS.md`: every number intended for the paper, with its source JSON path and run ID.
3. `docs/METHODS_FOR_PAPER.md`: exact final hyperparameters, preprocessing, commit hashes, hardware, training times, dataset counts, so the Methods and Experimental Setup sections can be written accurately.
4. Tag the repo `v1.0-results` and push (ask first).

The paper itself is written later, in a separate session, from these files only.

---

## 9. `CLAUDE.md` (create in Phase 1 with exactly this content, then keep it short)

```markdown
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
```

---

## 10. Session protocol (every session)

1. Read `CLAUDE.md`, `docs/PROGRESS.md`, and the current phase in `docs/PLAN.md`.
2. If jobs might be running: `ops/status.sh` once.
3. Do the phase work with subagents. Run tests.
4. At a GATE: reviewer pass → summary to the user → wait.
5. Before ending: update `PROGRESS.md` (done / running with ETA / next / blockers), commit code changes with a clear message.

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| DeepfakeBench old dependencies clash with modern PyTorch | Patch minimally (store patches in `ops/patches/`); if a detector cannot be made to load cleanly, use a second venv only for that detector's evaluation and record it. |
| A DeepfakeBench checkpoint missing or AUC fails GATE 4 | Investigate preprocessing first. If unresolvable, drop that detector and record why (do not train from scratch without asking). |
| dlib fails to build | Try a prebuilt wheel; record the version. |
| OOM on 12 GB VRAM (UCF, EffNet-B4) | Reduce batch size and use gradient accumulation to keep the effective batch at 32. Record it. |
| 16 GB RAM pressure | Reduce DataLoader workers, disable `persistent_workers`, keep prefetch factor at 2. |
| Windows reboot mid-training | Runner auto-restarts at logon and resumes from the last checkpoint; healthchecks alerts the user if nobody logs in. |
| Sensor SRCC very low | Check crop pipeline (no resizing), MOS normalization, lr. Do not proceed to Phase 8 with a broken sensor. |
| DCFG shows no gain over `fixed` or `augft` | That is a finding. Report it honestly in FINDINGS.md; do not tune on test. |

---

## 12. Human checkpoints summary

| When | What the user does |
|---|---|
| Phase 0 | Section 3 steps (driver, WSL, .wslconfig, power, accounts, FF++ script, Claude Code in WSL). |
| Phase 1 | Give sudo once; give W&B key, ntfy topic, healthchecks URL; confirm phone notifications arrive. |
| Phase 2 | Provide FF++ download script; download LIVE manually only if needed. |
| GATE 3 | Look at the crop QA grid. |
| GATE 4 | Approve the reproduction table. |
| GATE 8 | Approve the ETA and launch. |
| Phase 9 | Watch phone/W&B; log in after any reboot; come back and say "continue". |
| GATE 10 / Phase 11 | Review findings. |
