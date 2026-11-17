#!/usr/bin/env bash
# Usage: notify_file.sh <file> [message]. Sends <file> as an ntfy attachment; never fails the caller, never prints the topic.
# Test hook: DCFG_NOTIFY_LOG=<file> appends "attachment: <basename>" instead of sending. Honours DCFG_NO_NOTIFY=1.
f="${1:-}"; [ -f "$f" ] || exit 0
if [ -n "${DCFG_NOTIFY_LOG:-}" ]; then echo "attachment: $(basename "$f")" >> "$DCFG_NOTIFY_LOG"; exit 0; fi
[ "${DCFG_NO_NOTIFY:-0}" = "1" ] && exit 0
# shellcheck disable=SC1090
[ -f "$HOME/.dcfg_env" ] && source "$HOME/.dcfg_env" 2>/dev/null
if [ -n "${NTFY_TOPIC:-}" ]; then
  args=(-s -m 60 -T "$f" -H "Filename: $(basename "$f")")
  [ -n "${2:-}" ] && args+=(-H "Message: $2")
  curl "${args[@]}" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
fi
exit 0
