#!/usr/bin/env bash
# One compact status screen (<= ~20 lines). Read-only.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RUNS="${DCFG_RUNS_ROOT:-$REPO/runs}"
QUEUE="${DCFG_QUEUE_FILE:-$REPO/ops/queue.txt}"
# shellcheck source=_queue_lib.sh
source "$HERE/_queue_lib.sh"
total=0; done_n=0; fail_n=0; failed_ids=()
while IFS=$'\t' read -r id nohb hb cmd; do
  total=$((total+1))
  if [ -e "$RUNS/$id/DONE" ]; then done_n=$((done_n+1))
  elif [ -e "$RUNS/$id/FAILED" ]; then fail_n=$((fail_n+1)); failed_ids+=("$id"); fi
done < <(queue_jobs "$QUEUE")
echo "queue: done=$done_n failed=$fail_n pending=$((total-done_n-fail_n)) total=$total"
[ "$fail_n" -gt 0 ] && echo "failed jobs: ${failed_ids[*]:0:5}$([ "$fail_n" -gt 5 ] && echo ' ...')"

if runner_alive "$RUNS"; then alive=yes; else alive=no; fi
echo "runner alive: $alive"

job=""; [ -s "$RUNS/_current_job" ] && job="$(cat "$RUNS/_current_job")"
echo "current job: ${job:-none}"
if [ -n "$job" ]; then
  hb="$RUNS/$job/heartbeat"
  if [ -f "$hb" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$hb") ))
    summary="$(python3 - "$hb" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
    keys = ("step", "epoch", "val_auc", "throughput", "eta")
    print(" ".join(f"{k}={d[k]}" for k in keys if k in d))
except Exception:
    print("(unparseable)")
PY
)"
    echo "heartbeat: ${summary:-?} age=${age}s"
  else
    echo "heartbeat: none"
  fi
  if [ -f "$RUNS/$job/log.txt" ]; then
    echo "log tail:"
    tail -n 3 "$RUNS/$job/log.txt" | cut -c1-160 | sed 's/^/  /'
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  g="$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)"
  [ -n "$g" ] && echo "gpu: util/temp/vram_used/vram_total = $g (%, C, MiB, MiB)" || echo "gpu: unavailable"
else
  echo "gpu: nvidia-smi not found"
fi
echo "disk free: $(df -BG --output=avail "$HOME" 2>/dev/null | tail -n1 | tr -d ' ')"
echo "MemAvailable: $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"
exit 0
