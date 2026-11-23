#!/usr/bin/env bash
# Crash supervisor for one long, resume-safe training job (used for lumen2_train only; user decision 2026-11-22).
#   ops/jobs/retry_supervised.sh <max_restarts> <cmd...>
# - Runs <cmd>; on a non-zero exit it restarts it (the trainer resumes from its last checkpoint), up to <max_restarts>
#   restarts in total. The count is kept in $DCFG_RUN_DIR/supervisor_state, so runner re-invocations do not reset it.
# - Progress check: after a crash, if the run reached no training step beyond the furthest step reached before the
#   previous crash (heartbeat step), that is a "no-progress crash". Two in a row => real bug: stop, write
#   $DCFG_RUN_DIR/SUPERVISOR_GAVEUP, notify the phone, exit 3 (later invocations exit 3 at once).
# - One phone line per crash+restart with the step it resumed from, the GPU temperature and the count of Windows
#   nvlddmkm / Display 4101 (driver reset) events in the 20 min before the crash.
# - Logs GPU temperature/power/util every 60 s to $DCFG_RUN_DIR/gpu_temp.csv; crash details to supervisor.log.
set -u
MAX="$1"; shift
HERE="$(cd "$(dirname "$0")/.." && pwd)"
RD="${DCFG_RUN_DIR:?DCFG_RUN_DIR not set (run via run_queue.sh)}"
ST="$RD/supervisor_state"; SL="$RD/supervisor.log"; LOG="$RD/log.txt"
notify() { [ -n "${DCFG_SUP_TEST:-}" ] && { echo "NOTIFY: $1" >> "$SL"; return; }; "$HERE/notify.sh" "$1" "DCFG LUMEN2" >/dev/null 2>&1 || true; }
say() { echo "$(date -Is) $*" | tee -a "$SL"; }
hb_step() { python3 -c "import json,sys;print(json.load(open('$RD/heartbeat')).get('step',-1))" 2>/dev/null || echo -1; }
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }
drv_events() {
  [ -n "${DCFG_SUP_TEST:-}" ] && { echo 0; return; }
  timeout 90 powershell.exe -NoProfile -Command "@(Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddMinutes(-20)} -ErrorAction SilentlyContinue | Where-Object { \$_.ProviderName -eq 'nvlddmkm' -or (\$_.ProviderName -eq 'Display' -and \$_.Id -eq 4101) }).Count" 2>/dev/null | tr -dc '0-9' || true
}

[ -e "$RD/SUPERVISOR_GAVEUP" ] && { say "SUPERVISOR_GAVEUP present: not retrying"; exit 3; }
restarts=0; noprog=0; prev_max=-1
[ -f "$ST" ] && . "$ST"
save_state() { printf 'restarts=%s\nnoprog=%s\nprev_max=%s\n' "$restarts" "$noprog" "$prev_max" > "$ST"; }

( while :; do echo "$(date -Is),$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" >> "$RD/gpu_temp.csv"; sleep 60; done ) &
TEMP_PID=$!
trap 'kill $TEMP_PID 2>/dev/null' EXIT

first=1
while :; do
  off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  "$@" &
  CPID=$!
  if [ "$first" = 0 ]; then
    # report the resume step of this restart (wait up to 15 min for the trainer's "resumed at step" line)
    ( for _ in $(seq 90); do
        r=$(tail -c +"$((off+1))" "$LOG" 2>/dev/null | grep -a -m1 -oE 'resumed (at|from) step [0-9]+' | grep -oE '[0-9]+$')
        [ -n "$r" ] && break; sleep 10; done
      say "restart #$restarts resumed from step ${r:-unknown}"
      notify "${DCFG_SUP_NAME:-LUMEN2} crashed (exit $rc, GPU ${tC}C, driver-reset events ${ev:-?}) -> restart #$restarts/$MAX, resumed from step ${r:-unknown}" ) &
  fi
  wait "$CPID"; rc=$?
  [ "$rc" = 0 ] && { say "command finished OK"; exit 0; }
  first=0
  tC=$(gpu_temp); ev=$(drv_events); cur=$(hb_step)
  say "CRASH exit=$rc heartbeat_step=$cur prev_max=$prev_max gpu_temp=${tC}C driver_reset_events_20min=${ev:-?}"
  if [ "$cur" -le "$prev_max" ]; then noprog=$((noprog+1)); else noprog=0; prev_max=$cur; fi
  if [ "$noprog" -ge 2 ]; then
    save_state; echo "$(date -Is) no progress beyond step $prev_max in 2 consecutive crashes" > "$RD/SUPERVISOR_GAVEUP"
    say "GAVE UP: 2 consecutive crashes without new training steps (max step $prev_max)"
    notify "${DCFG_SUP_NAME:-LUMEN2} STOPPED: crashed twice without progress past step $prev_max (exit $rc) - likely a real bug, needs a look"
    exit 3
  fi
  if [ "$restarts" -ge "$MAX" ]; then
    save_state; echo "$(date -Is) restart limit $MAX reached" > "$RD/SUPERVISOR_GAVEUP"
    say "GAVE UP: restart limit $MAX reached"
    notify "${DCFG_SUP_NAME:-LUMEN2} STOPPED: $MAX restarts used up (last exit $rc at step $cur)"
    exit 3
  fi
  restarts=$((restarts+1)); save_state
  sleep "${DCFG_SUP_SLEEP:-20}"
done
