#!/usr/bin/env bash
# Shared helpers for run_queue.sh, watchdog.sh, status.sh. Source this file; do not execute.

# queue_jobs <queue_file>: print one line per valid job as
#   <id> TAB <nohb 0|1> TAB <hb_minutes or -> TAB <command>
# CRLF is stripped; trailing tags "#noheartbeat" and "#hb=<min>" are removed from the command.
# If a function q_warn is defined it is called for malformed lines.
queue_jobs() {
  local q="$1" line id cmd nohb hb
  [ -f "$q" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line//$'\r'/}"
    line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    if [[ "$line" != *"|"* ]]; then
      declare -F q_warn >/dev/null && q_warn "skip malformed line: ${line:0:60}"
      continue
    fi
    id="${line%%|*}"; cmd="${line#*|}"
    id="${id%"${id##*[![:space:]]}"}"
    cmd="${cmd#"${cmd%%[![:space:]]*}"}"
    nohb=0; hb=-
    while [[ "$cmd" =~ ^(.*[[:space:]])#(noheartbeat|hb=([0-9]+))$ ]]; do
      if [ -n "${BASH_REMATCH[3]}" ]; then hb="${BASH_REMATCH[3]}"; else nohb=1; fi
      cmd="${BASH_REMATCH[1]}"; cmd="${cmd%"${cmd##*[![:space:]]}"}"
    done
    if ! [[ "$id" =~ ^[A-Za-z0-9_.-]+$ ]] || [ -z "$cmd" ]; then
      declare -F q_warn >/dev/null && q_warn "skip invalid job line: ${line:0:60}"
      continue
    fi
    printf '%s\t%s\t%s\t%s\n' "$id" "$nohb" "$hb" "$cmd"
  done < "$q"
}

# runner_alive <runs_dir>: success iff pidfile exists, pid is alive, and its cmdline mentions run_queue.sh.
runner_alive() {
  local pf="$1/.runner.pid" pid
  [ -f "$pf" ] || return 1
  pid="$(tr -dc '0-9' < "$pf" 2>/dev/null)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'run_queue\.sh'
}

# job_finished <runs_dir> <id>: success iff DONE or FAILED marker exists.
job_finished() { [ -e "$1/$2/DONE" ] || [ -e "$1/$2/FAILED" ]; }
