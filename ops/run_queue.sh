#!/usr/bin/env bash
# Sequential job runner. Queue format: "<job_id> | <shell command>" (see ops/queue.txt).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO" || exit 1
# shellcheck disable=SC1090
[ -f "$HOME/.dcfg_env" ] && source "$HOME/.dcfg_env"
# shellcheck disable=SC1091
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate"

RUNS="${DCFG_RUNS_ROOT:-$REPO/runs}"
QUEUE="${DCFG_QUEUE_FILE:-$REPO/ops/queue.txt}"
PAUSE="${RETRY_PAUSE:-60}"
mkdir -p "$RUNS"

exec 9>"$RUNS/.runner.lock"
flock -n 9 || exit 0

LOG="$RUNS/_runner.log"
log() { echo "$(date -Is) $*" >> "$LOG"; }
notify() { "$HERE/notify.sh" "$@"; }

PINGER=""
cleanup() {
  [ -n "$PINGER" ] && { pkill -P "$PINGER" 2>/dev/null; kill "$PINGER" 2>/dev/null; }
  rm -f "$RUNS/_current_job"
  log "runner exit"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

if [ "${DCFG_NO_NOTIFY:-0}" != "1" ] && [ -n "${HC_PING_URL:-}" ]; then
  (
    while true; do
      curl -fsS -m 10 --retry 3 "$HC_PING_URL" >/dev/null 2>&1 || true
      sleep 600
    done
  ) 9>&- &
  PINGER=$!
fi

# Print the next unfinished job as "<id>\t<cmd>"; empty if none.
next_job() {
  local line id cmd
  [ -f "$QUEUE" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    [[ "$line" == *"|"* ]] || { log "skip malformed line: ${line:0:60}"; continue; }
    id="${line%%|*}"; cmd="${line#*|}"
    id="${id%"${id##*[![:space:]]}"}"
    cmd="${cmd#"${cmd%%[![:space:]]*}"}"
    if ! [[ "$id" =~ ^[A-Za-z0-9_.-]+$ ]] || [ -z "$cmd" ]; then
      log "skip invalid job line: ${line:0:60}"; continue
    fi
    [ -e "$RUNS/$id/DONE" ] || [ -e "$RUNS/$id/FAILED" ] && continue
    printf '%s\t%s\n' "$id" "$cmd"
    return 0
  done < "$QUEUE"
}

log "runner start pid $$"
while true; do
  job="$(next_job)"
  [ -z "$job" ] && break
  id="${job%%$'\t'*}"; cmd="${job#*$'\t'}"
  dir="$RUNS/$id"
  mkdir -p "$dir"
  echo "$id" > "$RUNS/_current_job"
  log "job start $id"
  notify "Started: $id" "DCFG"
  rc=1
  for attempt in 1 2 3; do
    echo "=== $(date -Is) attempt $attempt: $cmd" >> "$dir/log.txt"
    DCFG_JOB_ID="$id" DCFG_RUN_DIR="$dir" bash -lc "$cmd" >> "$dir/log.txt" 2>&1 9>&-
    rc=$?
    log "job $id attempt $attempt exit $rc"
    [ "$rc" -eq 0 ] && break
    [ "$attempt" -lt 3 ] && sleep "$PAUSE"
  done
  if [ "$rc" -eq 0 ]; then
    echo "$(date -Is) exit=0" > "$dir/DONE"
    notify "Done: $id" "DCFG"
  else
    echo "$(date -Is) exit=$rc" > "$dir/FAILED"
    notify "FAILED after 3 attempts: $id (exit $rc)" "DCFG"
  fi
done
rm -f "$RUNS/_current_job"
log "queue finished"
notify "Queue finished" "DCFG"
exit 0
