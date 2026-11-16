#!/usr/bin/env bash
# Health check; run every 30 min. Notifies on problems, restarts a dead runner if work remains.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO" || exit 1
RUNS="${DCFG_RUNS_ROOT:-$REPO/runs}"
QUEUE="${DCFG_QUEUE_FILE:-$REPO/ops/queue.txt}"
mkdir -p "$RUNS"
# shellcheck source=_queue_lib.sh
source "$HERE/_queue_lib.sh"
problems=()
ALERT_DIR="$RUNS/.watchdog_alerts"
COOLDOWN="${WATCHDOG_COOLDOWN_S:-7200}"
mkdir -p "$ALERT_DIR"
# problem <key> <message>: record, and notify at most once per COOLDOWN seconds per key.
problem() {
  local key="$1" msg="$2" f last now
  problems+=("$msg")
  f="$ALERT_DIR/${key//[^A-Za-z0-9_.-]/_}"
  now=$(date +%s)
  last=$(cat "$f" 2>/dev/null || echo 0)
  [[ "$last" =~ ^[0-9]+$ ]] || last=0
  [ $((now - last)) -lt "$COOLDOWN" ] && return 0
  echo "$now" > "$f"
  "$HERE/notify.sh" "$msg" "DCFG watchdog" 9>&-
}

alive=0
runner_alive "$RUNS" && alive=1

# Unfinished jobs, and per-job heartbeat settings.
pending=0
declare -A NOHB HBMIN
while IFS=$'\t' read -r id nohb hb cmd; do
  NOHB[$id]="$nohb"; HBMIN[$id]="$hb"
  job_finished "$RUNS" "$id" || pending=$((pending+1))
done < <(queue_jobs "$QUEUE")

if [ "$alive" -eq 0 ] && [ "$pending" -gt 0 ]; then
  problem runner_dead "Runner dead with $pending unfinished job(s); restarting"
  if [ -n "${DCFG_RESTART_CMD:-}" ]; then
    bash -c "$DCFG_RESTART_CMD" >/dev/null 2>&1 9>&-
  elif command -v schtasks.exe >/dev/null 2>&1; then
    # Task Scheduler's long-lived wsl.exe owns the runner, which keeps the WSL VM up.
    schtasks.exe /run /tn DCFG-Runner >/dev/null 2>&1 9>&-
  else
    nohup setsid "$HERE/run_queue.sh" >/dev/null 2>&1 9>&- &
  fi
fi

# Heartbeat of current job
if [ "$alive" -eq 1 ] && [ -s "$RUNS/_current_job" ]; then
  job="$(cat "$RUNS/_current_job")"
  jd="$RUNS/$job"
  if [ "${NOHB[$job]:-0}" != "1" ] && [ ! -e "$jd/.noheartbeat" ]; then
    now=$(date +%s)
    thr_min="${HBMIN[$job]:--}"; [[ "$thr_min" =~ ^[0-9]+$ ]] || thr_min=20
    start=$(tr -dc '0-9' < "$jd/attempt_start" 2>/dev/null)
    [ -n "$start" ] || start=$(stat -c %Y "$RUNS/_current_job")
    hbm=""
    [ -f "$jd/heartbeat" ] && hbm=$(stat -c %Y "$jd/heartbeat")
    if [ -n "$hbm" ] && [ "$hbm" -ge "$start" ]; then
      age=$(( now - hbm ))
      [ "$age" -gt $(( thr_min * 60 )) ] && problem "hb_stale_$job" "Heartbeat of $job stale: $((age/60)) min"
    else
      age=$(( now - start ))
      [ "$age" -gt $(( thr_min * 60 )) ] && problem "hb_none_$job" "No heartbeat for $job after $thr_min min"
    fi
  fi
fi

# GPU temp
if command -v nvidia-smi >/dev/null 2>&1; then
  t="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 | tr -dc "0-9")"
  [ -n "$t" ] && [ "$t" -ge 85 ] && problem gpu_temp "GPU temp ${t}C >= 85C"
fi

# Disk and RAM
free_gb=$(df --output=avail -BG "$HOME" 2>/dev/null | tail -n1 | tr -dc '0-9')
[ -n "$free_gb" ] && [ "$free_gb" -lt 50 ] && problem low_disk "Low disk: ${free_gb} GB free"
mem_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
[ -n "$mem_kb" ] && [ "$mem_kb" -lt 1048576 ] && problem low_ram "Low RAM: $((mem_kb/1024)) MB available"

echo "$(date -Is) alive=$alive pending=$pending problems=${#problems[@]} ${problems[*]:-}" >> "$RUNS/_watchdog.log"
exit 0
