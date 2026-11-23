#!/usr/bin/env bash
# Waits until Tue 24 Nov 2026 07:30 IST (02:00 UTC), then writes the LUMEN-uncond snapshot into STATUS_FOR_MEETING.md, commits,
# pushes and sends one phone line (ops/jobs/after_job.sh). Does not touch the training. Relaunch after a reboot with:
#   setsid nohup ops/jobs/uncond_0730.sh >/dev/null 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
T=$(date -d '2026-11-24T02:00:00Z' +%s)
[ -f runs/uncond_0730.done ] && exit 0
echo $$ > runs/uncond_0730.pid
while [ "$(date +%s)" -lt "$T" ]; do sleep 30; done
LINE="$(.venv/bin/python ops/jobs/uncond_snapshot.py 2>&1 | tail -n 1)"
ops/jobs/after_job.sh uncond_0730 "$LINE"
date -Is > runs/uncond_0730.done
