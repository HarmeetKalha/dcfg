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
