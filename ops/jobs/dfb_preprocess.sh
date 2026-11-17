#!/usr/bin/env bash
# Usage: ops/jobs/dfb_preprocess.sh <c23|c40> [workers] [split]
# Default split: c23 -> all splits; c40 -> test only (unchanged). `dfb_preprocess.sh c40 6 val` -> the c40 VAL videos only.
# Resumable (per-video .done markers); never touches existing frames.
set -euo pipefail
CMP="${1:?c23|c40}"; W="${2:-6}"; SPLIT="${3:-}"
cd "$HOME/dcfg"
EXTRA=""
if [ -n "$SPLIT" ]; then EXTRA="--split $SPLIT"; elif [ "$CMP" = c40 ]; then EXTRA="--test-only"; fi
exec .venv/bin/python -I ops/jobs/dfb_preprocess.py --comp "$CMP" $EXTRA --workers "$W"
