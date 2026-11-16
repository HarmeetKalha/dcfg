#!/usr/bin/env bash
# Usage: ops/jobs/dfb_preprocess.sh <c23|c40> [workers]
# c23 -> all splits; c40 -> test split only. Resumable (per-video .done markers).
set -euo pipefail
CMP="${1:?c23|c40}"; W="${2:-6}"
cd "$HOME/dcfg"
EXTRA=""; [ "$CMP" = c40 ] && EXTRA="--test-only"
exec .venv/bin/python -I ops/jobs/dfb_preprocess.py --comp "$CMP" $EXTRA --workers "$W"
