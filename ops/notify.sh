#!/usr/bin/env bash
# Usage: notify.sh "message" ["title"]. Sends an ntfy push; never fails the caller.
# Test hook: DCFG_NOTIFY_LOG=<file> appends "title: message" to the file instead of sending.
if [ -n "${DCFG_NOTIFY_LOG:-}" ]; then
  echo "${2:-}: ${1:-}" >> "$DCFG_NOTIFY_LOG"
  exit 0
fi
[ "${DCFG_NO_NOTIFY:-0}" = "1" ] && exit 0
# shellcheck disable=SC1090
[ -f "$HOME/.dcfg_env" ] && source "$HOME/.dcfg_env" 2>/dev/null
if [ -n "${NTFY_TOPIC:-}" ]; then
  args=(-s -m 15 -d "${1:-}")
  [ -n "${2:-}" ] && args+=(-H "Title: $2")
  curl "${args[@]}" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
fi
exit 0
