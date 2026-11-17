"""Pilot sanity grid: ONE val-split real face crop under the pilot conditions, exactly as val/eval degrade it.

    .venv/bin/python ops/jobs/pilot_sanity_grid.py [--out-dir results/pilot]

Picks the first real val video (sorted) and its middle frame, applies clean, noise_s10, blur_s3, webp_q20,
mixed_heavy with ``apply_condition(img, cond, image_key=<path relative to the frames root>)`` and writes
``sanity_grid.png`` (labelled 1-row grid) plus ``sanity/<cond>.png``. A c40 panel is added only if a c40
VAL frame of the same video exists in the JSON (otherwise skipped; never falls back to test).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dcfg.data.degradations import apply_condition, load_conditions  # noqa: E402
from dcfg.data.ffpp import REAL_LABEL, DFBJsonSource  # noqa: E402

PILOT_CONDITIONS = ["clean", "noise_s10", "blur_s3", "webp_q20", "mixed_heavy"]


def pick_frame(source: DFBJsonSource, split: str = "val", comp: str = "c23"):
    """Middle frame of the first real video (sorted order) of ``split``."""
    recs = [r for r in source.records(split, comp) if r.video.startswith(REAL_LABEL + "/")]
    if not recs:
        raise RuntimeError(f"no real {split}/{comp} frames")
    vid = recs[0].video
    frames = [r for r in recs if r.video == vid]
    return frames[len(frames) // 2]


def build_panels(source: DFBJsonSource, conds: dict, names: list[str]) -> tuple[list[tuple[str, np.ndarray]], str]:
    """[(label, rgb uint8)] panels plus a note about c40."""
    rec = pick_frame(source)
    img = source.load(rec)
    panels = [(n, apply_condition(img, conds[n], rec.key)) for n in names]
    c40 = [r for r in source.records("val", "c40") if r.video == rec.video]
    if c40:
        cr = c40[len(c40) // 2]
        panels.append(("c40 (val)", source.load(cr)))
        note = "c40 val frame included"
    else:
        note = "c40 val preprocessing failed/missing -> c40 skipped"
    return panels, note


def label_panel(rgb: np.ndarray, text: str, bar: int = 28) -> np.ndarray:
    out = np.full((rgb.shape[0] + bar, rgb.shape[1], 3), 255, np.uint8)
    out[bar:] = rgb
    cv2.putText(out, text, (6, bar - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(REPO / "results" / "pilot"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--frames-root", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    (out / "sanity").mkdir(parents=True, exist_ok=True)
    source = DFBJsonSource(a.json, a.frames_root)
    conds = load_conditions(REPO / "configs" / "eval" / "conditions.yaml")
    panels, note = build_panels(source, conds, PILOT_CONDITIONS)
    for name, rgb in panels:
        cv2.imwrite(str(out / "sanity" / f"{name.split()[0]}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    grid = np.concatenate([label_panel(rgb, f"{n} [pilot]") for n, rgb in panels], axis=1)
    cv2.imwrite(str(out / "sanity_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"sanity grid: {out / 'sanity_grid.png'} ({len(panels)} panels); {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
