#!/usr/bin/env python3
"""Tue 24 Nov 07:30 IST snapshot of the LUMEN-uncond training (user request): current epoch, val mean AUC per epoch next to LUMEN2's
and GenD-aug's at the same epoch, best epoch so far -> results/lumen2/uncond_0730.md (included in STATUS_FOR_MEETING.md by
meeting_status.py). Prints the one-line phone summary. Never touches the training."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IST = timezone(timedelta(hours=5, minutes=30))


def curve(run: str) -> dict[int, float]:
    p = REPO / "runs" / run / "metrics.csv"
    return {int(r["epoch"]): float(r["val_mean_auc"]) for r in csv.DictReader(p.open())} if p.exists() else {}


def main() -> int:
    u, l2, g = curve("lumen2_uncond_train"), curve("lumen2_train"), curve("gendaug_train")
    try:
        hb = json.loads((REPO / "runs/lumen2_uncond_train/heartbeat").read_text())
    except Exception:  # noqa: BLE001
        hb = {}
    done = (REPO / "runs/lumen2_uncond_train/DONE").exists()
    now = datetime.now(IST).strftime("%a %d %b %H:%M IST")
    best = max(u, key=u.get) if u else None
    f = lambda d, e: f"{d[e]:.4f}" if e in d else "-"  # noqa: E731
    cur = (f"finished ({len(u)} epochs)" if done else f"epoch {len(u) + 1} in progress (step {hb.get('step', '?')}, {len(u)} epochs completed)")
    L = [f"## LUMEN-uncond at {now} (snapshot, training NOT stopped)", "",
         f"- Current: {cur}",
         f"- Best epoch so far: {best} (val mean AUC {u[best]:.4f})" if best else "- Best epoch so far: none (no epoch completed)",
         f"- LUMEN2 best: epoch {max(l2, key=l2.get)} ({max(l2.values()):.4f}); GenD-aug best: epoch {max(g, key=g.get)} ({max(g.values()):.4f})", "",
         "Validation mean AUC (8 of 32 frames/video, 6 val conditions) at the same epoch:", "",
         "| epoch | LUMEN-uncond | LUMEN2 | GenD-aug | uncond - LUMEN2 |", "|---|---|---|---|---|"]
    for e in sorted(set(u) | set(l2) | set(g)):
        d = f"{u[e] - l2[e]:+.4f}" if e in u and e in l2 else "-"
        L.append(f"| {e} | {f(u, e)} | {f(l2, e)} | {f(g, e)} | {d} |")
    (REPO / "results/lumen2/uncond_0730.md").write_text("\n".join(L) + "\n")
    if best:
        e = len(u)
        line = (f"LUMEN-uncond 07:30: {cur}; best ep {best} val {u[best]:.4f} (LUMEN2 best {max(l2.values()):.4f}, GenD-aug {max(g.values()):.4f}); "
                f"ep{e}: uncond {u[e]:.4f} vs LUMEN2 {f(l2, e)} vs GenD-aug {f(g, e)}")
    else:
        line = f"LUMEN-uncond 07:30: {cur}, no epoch completed yet"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
