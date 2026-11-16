#!/usr/bin/env bash
# Usage: ops/jobs/dfb_rearrange.sh   (idempotent; rewrites JSONs in ~/data/dfb_json)
set -euo pipefail
cd "$HOME/dcfg"; exec .venv/bin/python -I ops/jobs/dfb_rearrange.py
