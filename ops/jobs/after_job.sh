#!/usr/bin/env bash
# Usage: ops/jobs/after_job.sh <job_id> ["<summary line>"] [extra path ...]
# Regenerates results/STATUS_FOR_MEETING.md, commits the job's results + status + (optional) extra paths, pushes (retry once;
# a push failure NEVER fails the job) and sends ONE ntfy line (the summary; computed by job_summary.py if not given).
# Only a fixed allow-list of result paths is staged (never runs/, data, secrets).
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 0
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
JOB="${1:-unknown}"; SUMMARY="${2:-}"; [ $# -ge 2 ] && shift 2 || shift $#
[ -z "$SUMMARY" ] && SUMMARY="$(python ops/jobs/job_summary.py "$JOB" 2>/dev/null)"
[ -z "$SUMMARY" ] && SUMMARY="$JOB done"
python ops/jobs/meeting_status.py >/dev/null 2>&1
# stage by job
case "$JOB" in
  gate_heatmaps_lumen2) PATHS=(results/gate/lumen2) ;;
  gate_heatmaps*) PATHS=(results/gate/lumen1) ;;
  dfd_*) PATHS=(results/dfd results/lumen2 results/gate) ;;
  sensor_train) PATHS=(results/sensor) ;;
  sensor_eval) PATHS=(results/sensor) ;;
  lumen_nogate_xception) PATHS=() ;;
  eval_lumen_nogate_xception) PATHS=(results/raw) ;;
  gate_ablation) PATHS=(results/gate) ;;
  *) PATHS=(results) ;;
esac
git add -- results/STATUS_FOR_MEETING.md "${PATHS[@]}" src configs tests ops/jobs ops/queue_meeting.txt docs "$@" 2>/dev/null
if ! git diff --cached --quiet; then
  git commit -q -m "after $JOB: $SUMMARY

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153An9KAUzGZPy7zGu25HMy" >/dev/null 2>&1
  git push -q >/dev/null 2>&1 || { sleep 20; git push -q >/dev/null 2>&1 || echo "after_job: push failed (ignored)"; }
fi
ops/notify.sh "$SUMMARY" "DCFG $JOB" >/dev/null 2>&1
exit 0
