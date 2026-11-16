"""Atomic, throttled heartbeat file written by long jobs and read by ops/status.sh."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Heartbeat:
    """Writes ``run_dir/heartbeat`` (one JSON line) at most every ``interval_s`` seconds."""

    def __init__(self, run_dir: str | os.PathLike[str], interval_s: float = 60.0) -> None:
        self.path = Path(run_dir) / "heartbeat"
        self.interval_s = interval_s
        self._last = float("-inf")

    def beat(self, step: int, force: bool = False, **fields: Any) -> bool:
        """Write the heartbeat if due (or forced). Returns True if written."""
        now = time.monotonic()
        if not force and now - self._last < self.interval_s:
            return False
        unix = time.time()
        rec = {
            "time_iso": datetime.fromtimestamp(unix, timezone.utc).isoformat(),
            "unix": unix,
            "step": step,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(rec) + "\n")
        os.replace(tmp, self.path)
        self._last = now
        return True


def read_heartbeat(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the heartbeat dict, or None if missing/unparseable."""
    try:
        return json.loads(Path(path).read_text().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
