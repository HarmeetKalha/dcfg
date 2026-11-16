# Decisions and deviations from PLAN.md

| Date | Phase | Decision / deviation | Reason |
|---|---|---|---|
| 2026-11-16 | 1 | Plan delivered as `docs/DCFG_Build_Plan.md`; renamed to `docs/PLAN.md` as the plan specifies. | File-name mismatch only, content unchanged. |
| 2026-11-16 | 1 | PyTorch from the `cu128` index (CUDA 12.8 build). | Plan asks for the current stable CUDA 12.x build. |
| 2026-11-16 | 1 | torch 2.11.0+cu128 / torchvision 0.26.0. Installed with `UV_HTTP_TIMEOUT=600`. | Default uv timeout failed on pypi.nvidia.com. |
