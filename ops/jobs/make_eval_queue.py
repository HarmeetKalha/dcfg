"""Print (never append) the queue lines for the Phase 5.3 baseline evaluation and the analysis jobs.

    .venv/bin/python ops/jobs/make_eval_queue.py >> ops/queue.txt    # only when the user decides

evaluate.py writes a heartbeat per batch (runs/<run>/heartbeat), so eval jobs get ``#hb=30`` (minutes).
``--conditions all`` already includes ``c40`` (``synthetic: false`` in conditions.yaml).
"""
from __future__ import annotations

PY = ".venv/bin/python"
DETECTORS = ["xception", "efficientnetb4", "f3net", "ucf"]


def queue_lines() -> list[str]:
    lines = [f"base_{d} | {PY} -m dcfg.eval.evaluate --run base_{d} --detector {d} --conditions all --split test #hb=30"
             for d in DETECTORS]
    lines += [f"preprocess_qa | {PY} -m dcfg.analysis.preprocess_qa #noheartbeat",
              f"freq_study | {PY} -m dcfg.analysis.freq_study #noheartbeat",
              f"sensor_embedding | {PY} -m dcfg.analysis.sensor_embedding #noheartbeat"]
    return lines


if __name__ == "__main__":
    print("\n".join(queue_lines()))
