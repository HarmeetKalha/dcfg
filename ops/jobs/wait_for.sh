#!/usr/bin/env bash
# Usage: ops/jobs/wait_for.sh <runs_bg/job_dir>: block (sleep loop, no polling output) until <dir>/DONE exists; exit 1 if <dir>/FAILED appears
# or after 72 h. Used by the DFD evaluation jobs that wait for the background preprocessing runner.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
D="$1"; end=$(( $(date +%s) + 72*3600 ))
while [ ! -f "$D/DONE" ]; do
  [ -f "$D/FAILED" ] && { echo "wait_for: $D FAILED"; exit 1; }
  [ "$(date +%s)" -gt "$end" ] && { echo "wait_for: timeout on $D"; exit 1; }
  sleep 300
done
