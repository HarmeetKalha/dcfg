"""DFD tables from results/dfd/<run>__{clean,mixed_heavy}.json (verbatim AUC + CI) and helpers to add them to a report.

    python -m dcfg.lumen2.dfd_append --report results/lumen2/gend_aug.md --models gend_public:GenD gend_aug:GenD-aug
    python -m dcfg.lumen2.dfd_append --report results/gate/ablation_xception.md --models base_xception:base lumen_xception:LUMEN1 ...

Replaces (or appends) a section delimited by ``<!-- DFD -->`` markers, so it is idempotent. Missing runs are listed as "n/a".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dcfg.paths import RESULTS_ROOT

CONDS = ("clean", "mixed_heavy")
BEGIN, END = "<!-- DFD begin -->", "<!-- DFD end -->"


def cell(d: dict | None, m: str) -> str:
    return "n/a" if d is None else f"{d[f'{m}_auc']:.4f} [{d[f'{m}_auc_ci'][0]:.4f}, {d[f'{m}_auc_ci'][1]:.4f}]"


def dfd_section(models: list[tuple[str, str]], d: Path = RESULTS_ROOT / "dfd") -> str:
    """Markdown section: frame/video AUC [95% CI] on DFD clean and DFD mixed_heavy for ``[(run_id, label)]``."""
    docs = {}
    for rid, _ in models:
        for c in CONDS:
            p = d / f"{rid}__{c}.json"
            docs[(rid, c)] = json.loads(p.read_text()) if p.exists() else None
    L = ["## DFD (DeepFakeDetection, FF++ release; never used in training)", "",
         "All DFD videos (real actors + DFD fakes), 32 frames per video, video-level bootstrap 95% CI (1000 resamples). `mixed_heavy` uses the same "
         "degradation and seeding rule as on FF++ (per-frame key). GenD models use GenD RetinaFace crops; Xception models use DeepfakeBench crops.", "",
         "| model | frame AUC clean | video AUC clean | frame AUC mixed_heavy | video AUC mixed_heavy |", "|---|---|---|---|---|"]
    for rid, lab in models:
        L.append(f"| {lab} | " + " | ".join(cell(docs[(rid, c)], m) for c in CONDS for m in ("frame", "video")) + " |")
    return "\n".join(L) + "\n"


def apply(report: Path, section: str) -> None:
    text = report.read_text()
    block = f"{BEGIN}\n{section}{END}\n"
    if BEGIN in text:
        text = text[: text.index(BEGIN)] + block + text[text.index(END) + len(END) + 1:]
    else:
        text = text.rstrip("\n") + "\n\n" + block
    report.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.dfd_append", description=__doc__.split("\n")[0])
    ap.add_argument("--report", required=True)
    ap.add_argument("--models", nargs="+", required=True, metavar="RUN:LABEL")
    ap.add_argument("--dir", default=str(RESULTS_ROOT / "dfd"))
    a = ap.parse_args(argv)
    apply(Path(a.report), dfd_section([tuple(m.split(":", 1)) for m in a.models], Path(a.dir)))  # type: ignore[misc]
    print(f"updated {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
