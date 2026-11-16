#!/usr/bin/env bash
# Sequential job runner. Queue format: "<job_id> | <shell command> [#noheartbeat] [#hb=<min>]" (see ops/queue.txt).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO" || exit 1
# shellcheck disable=SC1090
[ -f "$HOME/.dcfg_env" ] && source "$HOME/.dcfg_env"
# shellcheck disable=SC1091
[ -f "$REPO/.venv/bin/activate" ] && source "$REPO/.venv/bin/activate"
# shellcheck source=_queue_lib.sh
source "$HERE/_queue_lib.sh"

RUNS="${DCFG_RUNS_ROOT:-$REPO/runs}"
QUEUE="${DCFG_QUEUE_FILE:-$REPO/ops/queue.txt}"
PAUSE="${RETRY_PAUSE:-60}"
mkdir -p "$RUNS"
LOG="$RUNS/_runner.log"
log() { echo "$(date -Is) $*" >> "$LOG"; }
q_warn() { log "$*"; }
notify() { "$HERE/notify.sh" "$@" 9>&-; }

# Single instance: a live runner (by pidfile) wins immediately; otherwise wait up to 5 s for the lock.
runner_alive "$RUNS" && exit 0
exec 9>"$RUNS/.runner.lock"
flock -w 5 9 || exit 0
echo "$$" > "$RUNS/.runner.pid"

PINGER=""; JOBPID=""; JOBPGID=""; SLP=""

group_alive() { ps -o stat= -g "$1" 2>/dev/null | grep -qv '^Z'; }
# kill_group <pgid>: TERM, then KILL after 10 s.
kill_group() {
  local pg="$1" i
  group_alive "$pg" || return 0
  kill -TERM -- "-$pg" 2>/dev/null
  for i in $(seq 100); do group_alive "$pg" || return 0; sleep 0.1; done
  kill -KILL -- "-$pg" 2>/dev/null
}

cleanup() {
  trap '' TERM INT HUP
  [ -n "$JOBPGID" ] && kill_group "$JOBPGID"
  [ -n "$SLP" ] && kill "$SLP" 2>/dev/null
  [ -n "$PINGER" ] && { pkill -P "$PINGER" 2>/dev/null; kill "$PINGER" 2>/dev/null; }
  rm -f "$RUNS/_current_job" "$RUNS/.pinger.pid"
  [ "$(cat "$RUNS/.runner.pid" 2>/dev/null)" = "$$" ] && rm -f "$RUNS/.runner.pid"
  log "runner exit"
}
trap cleanup EXIT
trap 'log "signal received, shutting down"; exit 143' TERM INT HUP

# Pinger: pings every PINGER_EVERY_S (600), checks runner liveness every PINGER_CHECK_S (30).
if [ "${DCFG_NO_NOTIFY:-0}" != "1" ] && [ -n "${HC_PING_URL:-}" ]; then
  RUNNER_PID=$$
  (
    last=0
    while kill -0 "$RUNNER_PID" 2>/dev/null; do
      now=$(date +%s)
      if [ $((now - last)) -ge "${PINGER_EVERY_S:-600}" ]; then
        curl -fsS -m 10 --retry 3 "$HC_PING_URL" >/dev/null 2>&1 || true
        last=$now
      fi
      sleep "${PINGER_CHECK_S:-30}"
    done
  ) </dev/null >/dev/null 2>&1 9>&- &
  PINGER=$!
  echo "$PINGER" > "$RUNS/.pinger.pid"
fi

# Set JOB_ID/JOB_NOHB/JOB_CMD to the next unfinished job; return 1 if none.
next_job() {
  local id nohb hb cmd
  while IFS=$'\t' read -r id nohb hb cmd; do
    job_finished "$RUNS" "$id" && continue
    JOB_ID="$id"; JOB_NOHB="$nohb"; JOB_CMD="$cmd"
    return 0
  done < <(queue_jobs "$QUEUE")
  return 1
}

# Kill a job group orphaned by a previous (SIGKILLed) runner, if it is verifiably this job.
reap_orphan() {
  local dir="$1" id="$2" pid pg
  [ -f "$dir/job.pid" ] || return 0
  read -r pid pg < "$dir/job.pid" || return 0
  [[ "$pid" =~ ^[0-9]+$ && "$pg" =~ ^[0-9]+$ ]] || return 0
  if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "DCFG_JOB_ID=$id"; then
    log "killing orphaned job group $pg of $id"
    kill_group "$pg"
  fi
}

log "runner start pid $$"
while next_job; do
  id="$JOB_ID"; cmd="$JOB_CMD"
  dir="$RUNS/$id"
  mkdir -p "$dir"
  [ "$JOB_NOHB" = "1" ] && touch "$dir/.noheartbeat"
  reap_orphan "$dir" "$id"
  echo "$id" > "$RUNS/_current_job"
  log "job start $id"
  notify "Started: $id" "DCFG"
  rc=1
  for attempt in 1 2 3; do
    date +%s > "$dir/attempt_start"
    echo "=== $(date -Is) attempt $attempt: $cmd" >> "$dir/log.txt"
    DCFG_JOB_ID="$id" DCFG_RUN_DIR="$dir" setsid bash -lc "$cmd" </dev/null >>"$dir/log.txt" 2>&1 9>&- &
    JOBPID=$!
    JOBPGID="$(ps -o pgid= -p "$JOBPID" 2>/dev/null | tr -dc '0-9')"; JOBPGID="${JOBPGID:-$JOBPID}"
    echo "$JOBPID $JOBPGID" > "$dir/job.pid"
    wait "$JOBPID"; rc=$?
    JOBPGID=""
    log "job $id attempt $attempt exit $rc"
    [ "$rc" -eq 0 ] && break
    if [ "$attempt" -lt 3 ] && [ "$PAUSE" != "0" ]; then
      sleep "$PAUSE" & SLP=$!
      wait "$SLP"; SLP=""
    fi
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
