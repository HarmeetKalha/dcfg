#!/usr/bin/env bash
# Usage: notify.sh "message" ["title"]. Sends an ntfy push; never fails the caller.
[ "${DCFG_NO_NOTIFY:-0}" = "1" ] && exit 0
# shellcheck disable=SC1090
[ -f "$HOME/.dcfg_env" ] && source "$HOME/.dcfg_env" 2>/dev/null
if [ -n "${NTFY_TOPIC:-}" ]; then
  args=(-s -m 15 -d "${1:-}")
  [ -n "${2:-}" ] && args+=(-H "Title: $2")
  curl "${args[@]}" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
fi
exit 0
