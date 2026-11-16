#!/usr/bin/env bash
# Usage: ops/jobs/train_sensor.sh [--protocol10] [--overwrite]
# Thin wrapper; run dir comes from DCFG_RUN_DIR (set by the queue runner). Resumes from last.pt.
set -euo pipefail
cd "$HOME/dcfg"
[ -f "$HOME/.dcfg_env" ] && set -a && . "$HOME/.dcfg_env" && set +a
exec .venv/bin/python -m dcfg.train.train_sensor --config configs/sensor_pretrain.yaml "$@"
