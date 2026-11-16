# Decisions and deviations from PLAN.md

| Date | Phase | Decision / deviation | Reason |
|---|---|---|---|
| 2026-11-16 | 1 | Plan delivered as `docs/DCFG_Build_Plan.md`; renamed to `docs/PLAN.md` as the plan specifies. | File-name mismatch only, content unchanged. |
| 2026-11-16 | 1 | PyTorch from the `cu128` index (CUDA 12.8 build). | Plan asks for the current stable CUDA 12.x build. |
| 2026-11-16 | 1 | torch 2.11.0+cu128 / torchvision 0.26.0. Installed with `UV_HTTP_TIMEOUT=600`. | Default uv timeout failed on pypi.nvidia.com. |
| 2026-11-16 | 1 | Task Scheduler tasks use LogonType Interactive (no stored password); window hidden via wscript launcher. | After reboot the user logs in (PLAN Sec 3); hidden window can't be closed by accident. |
| 2026-11-16 | 1 | Watchdog restarts runner via `schtasks.exe /run /tn DCFG-Runner`. | Long-lived Task Scheduler wsl.exe keeps the VM up. |
| 2026-11-16 | 1 | Queue tags `#noheartbeat` and `#hb=<min>` for jobs without/with slow heartbeats. | Avoid endless false stale-heartbeat alerts for download/preprocess jobs. |
| 2026-11-16 | 1 | Eval degradations must use per-sample seeded generators (hash of video/frame/condition), not global RNG. | Results must not depend on worker count or order (reviewer note for later phases). |
| 2026-11-16 | 3 | DeepfakeBench pinned at commit f188b1c105465e2e5377eb536a95022ae0e4522d (2025-08-20), cloned to third_party/DeepfakeBench. dlib model shape_predictor_81_face_landmarks.dat (sha256 8cae4375...) from the DFB v1.0.0 release; official FF++ split jsons from ondyari/FaceForensics (not bundled in DFB). | Reproducibility pin. |
| 2026-11-16 | 3 | Preprocessing/rearrange run through wrappers ops/jobs/dfb_preprocess.py and dfb_rearrange.py (no patch to DFB). They import DFB's extract_aligned_face_dlib unchanged and use its linspace 32-frame sampling; imutils 0.5.4 installed (--no-deps). | DFB preprocess.py writes next to raw videos, hardcodes paths, needs actors/FaceShifter/DFD dirs, cannot do test-only c40, does not skip done videos; rearrange.py KeyErrors without DFD/FaceShifter. Output schema kept identical. |
| 2026-11-16 | 7 | Sensor CNN runs under fp16 autocast on CUDA (flag sensor_amp=True; False = pure fp32). DCT, colour transforms and gate stay fp32; embedding cast to fp32 before the gate. DCT implemented as conv2d with 64 fixed 8x8 basis kernels, stride 8. | PLAN 7.5 only mandates fp32 for DCT+gate; fp32 sensor alone ~1.9k img/s < 2000 target. Front-end dcfg = 2428 img/s at bs64. |
| 2026-11-16 | 5 | Mixed eval conditions apply ops in order blur -> noise -> JPEG (mixed_light: blur s1, noise s5, jpeg q70; mixed_heavy: blur s2, noise s10, jpeg q30). | PLAN 7.3 lists the ops but no order; capture-then-compress order is the realistic one. Fixed before any results. |
| 2026-11-16 | 5 | Resize = cv2 INTER_CUBIC down + up, no antialiasing (PLAN "bicubic"). Motion blur is horizontal only. JPEG via PIL 4:2:0, optimize=False. | Matches PLAN wording; state "no antialias" and "horizontal" in the paper. |
| 2026-11-16 | 7 | DCT energy stats fed to the gate are NOT detached. | Smooth through log(mean|c|+1e-6); detaching would only cut a legitimate gradient path. |
| 2026-11-16 | 7 | Phase 8 engine must use torch.amp.GradScaler on the full loss and log non-finite/zero sensor grads; eval image_key = path relative to the data root. | GATE 7 reviewer: fp16 sensor + detector need loss scaling; seeds must not depend on the absolute path. |
