#!/usr/bin/env bash
# Health check; run every 30 min. Notifies on problems, restarts a dead runner if work remains.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO" || exit 1
RUNS="${DCFG_RUNS_ROOT:-$REPO/runs}"
QUEUE="${DCFG_QUEUE_FILE:-$REPO/ops/queue.txt}"
mkdir -p "$RUNS"
problems=()
problem() { problems+=("$1"); "$HERE/notify.sh" "$1" "DCFG watchdog"; }

# Runner alive? (lock held by someone else)
alive=1
if [ -e "$RUNS/.runner.lock" ] && ( flock -n 8 ) 8<"$RUNS/.runner.lock"; then alive=0; fi
[ -e "$RUNS/.runner.lock" ] || alive=0

# Unfinished jobs?
pending=0
if [ -f "$QUEUE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    [[ "$line" == *"|"* ]] || continue
    id="${line%%|*}"; id="${id%"${id##*[![:space:]]}"}"
    [[ "$id" =~ ^[A-Za-z0-9_.-]+$ ]] || continue
    [ -e "$RUNS/$id/DONE" ] || [ -e "$RUNS/$id/FAILED" ] || pending=$((pending+1))
  done < "$QUEUE"
fi

if [ "$alive" -eq 0 ] && [ "$pending" -gt 0 ]; then
  problem "Runner dead with $pending unfinished job(s); restarting"
  nohup setsid "$HERE/run_queue.sh" >/dev/null 2>&1 &
fi

# Heartbeat of current job
if [ "$alive" -eq 1 ] && [ -s "$RUNS/_current_job" ]; then
  job="$(cat "$RUNS/_current_job")"
  hb="$RUNS/$job/heartbeat"
  now=$(date +%s)
  if [ -f "$hb" ]; then
    age=$(( now - $(stat -c %Y "$hb") ))
    [ "$age" -gt 1200 ] && problem "Heartbeat of $job stale: $((age/60)) min"
  else
    jstart=$(stat -c %Y "$RUNS/_current_job")
    [ $(( now - jstart )) -gt 1200 ] && problem "No heartbeat for $job after 20 min"
  fi
fi

# GPU temp
if command -v nvidia-smi >/dev/null 2>&1; then
  t="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 | tr -dc "0-9")"
  [ -n "$t" ] && [ "$t" -ge 85 ] && problem "GPU temp ${t}C >= 85C"
fi

# Disk and RAM
free_gb=$(df --output=avail -BG "$HOME" 2>/dev/null | tail -n1 | tr -dc '0-9')
[ -n "$free_gb" ] && [ "$free_gb" -lt 50 ] && problem "Low disk: ${free_gb} GB free"
mem_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
[ -n "$mem_kb" ] && [ "$mem_kb" -lt 1048576 ] && problem "Low RAM: $((mem_kb/1024)) MB available"

echo "$(date -Is) alive=$alive pending=$pending problems=${#problems[@]} ${problems[*]:-}" >> "$RUNS/_watchdog.log"
exit 0
