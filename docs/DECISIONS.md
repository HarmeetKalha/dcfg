# Decisions and deviations from PLAN.md

| Date | Phase | Decision / deviation | Reason |
|---|---|---|---|
| 2026-11-16 | 1 | Plan delivered as `docs/DCFG_Build_Plan.md`; renamed to `docs/PLAN.md` as the plan specifies. | File-name mismatch only, content unchanged. |
| 2026-11-16 | 1 | PyTorch from the `cu128` index (CUDA 12.8 build). | Plan asks for the current stable CUDA 12.x build. |
| 2026-11-16 | 1 | torch 2.11.0+cu128 / torchvision 0.26.0. Installed with `UV_HTTP_TIMEOUT=600`. | Default uv timeout failed on pypi.nvidia.com. |

## 2026-11-16 — GATE 1 review outcomes
- Task Scheduler tasks use LogonType Interactive (no stored password). After a reboot the user must log in for the runner to restart (PLAN Sec 3). The task window is hidden via a wscript launcher so it can't be closed by accident.
- Watchdog restarts the runner via `schtasks.exe /run /tn DCFG-Runner`, so the long-lived Task Scheduler wsl.exe keeps the VM up.
- Jobs without heartbeats (download/preprocess) are tagged `#noheartbeat` in queue.txt; `#hb=<min>` overrides the 20-min stale threshold.
- For later phases: eval degradations must use per-sample seeded generators (e.g. hash of video/frame/condition), not global RNG, so results don't depend on worker count or order.
