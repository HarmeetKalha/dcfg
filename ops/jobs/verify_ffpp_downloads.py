"""GATE 2: verify the FF++ raw downloads (read-only on ~/data/ffpp_raw).

    .venv/bin/python ops/jobs/verify_ffpp_downloads.py [--raw DIR] [--deep] [--out results/qa/ffpp_downloads.json]

Counts ``<raw>/<type>/<comp>/videos/*.mp4`` for 5 types x {c23,c40} (expects 1000 each), flags stray
``tmp*`` files, 0-byte files and (with ``--deep``, slow) files cv2 cannot read (frame count <= 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dcfg.paths import FFPP_RAW  # noqa: E402

TYPES = ["original_sequences/youtube", "manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face",
         "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures"]
COMPS = ["c23", "c40"]
EXPECTED = 1000


def scan_folder(folder: Path, deep: bool = False, expected: int = EXPECTED) -> dict:
    """Inspect one ``<type>/<comp>`` folder; never writes."""
    vids = folder / "videos"
    mp4 = sorted(vids.glob("*.mp4")) if vids.is_dir() else []
    stray = sorted(str(p.relative_to(folder)) for p in folder.rglob("tmp*")) if folder.is_dir() else []
    sizes = [p.stat().st_size for p in mp4]
    zero = [p.name for p, s in zip(mp4, sizes) if s == 0]
    unreadable: list[str] = []
    if deep:
        import cv2
        for p, s in zip(mp4, sizes):
            if s == 0:
                continue
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if n <= 0:
                unreadable.append(p.name)
    return {"exists": folder.is_dir(), "n_mp4": len(mp4), "expected": expected,
            "ok_count": len(mp4) == expected, "total_bytes": int(sum(sizes)),
            "mean_bytes": int(sum(sizes) / len(sizes)) if sizes else 0,
            "stray_tmp": stray, "zero_byte": zero, "unreadable": unreadable, "deep": deep}


def verify(raw: Path, deep: bool = False, expected: int = EXPECTED) -> dict:
    res = {f"{t}/{c}": scan_folder(raw / t / c, deep, expected) for t in TYPES for c in COMPS}
    res_ok = all(r["ok_count"] and not r["stray_tmp"] and not r["zero_byte"] and not r["unreadable"]
                 for r in res.values())
    return {"raw": str(raw), "deep": deep, "all_ok": res_ok, "folders": res}


def format_table(rep: dict) -> str:
    lines = [f"{'folder':<50}{'n':>6}{'GB':>8}{'tmp':>5}{'0B':>4}{'bad':>5}  ok"]
    for k, r in rep["folders"].items():
        lines.append(f"{k:<50}{r['n_mp4']:>6}{r['total_bytes'] / 1e9:>8.2f}{len(r['stray_tmp']):>5}"
                     f"{len(r['zero_byte']):>4}{len(r['unreadable']):>5}  {'OK' if r['ok_count'] and not (r['stray_tmp'] or r['zero_byte'] or r['unreadable']) else 'FAIL'}")
    lines.append(f"ALL OK: {rep['all_ok']} (deep={rep['deep']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(FFPP_RAW))
    ap.add_argument("--deep", action="store_true", help="open every video with cv2 (slow)")
    ap.add_argument("--expected", type=int, default=EXPECTED)
    ap.add_argument("--out", default=str(REPO / "results" / "qa" / "ffpp_downloads.json"))
    a = ap.parse_args(argv)
    rep = verify(Path(a.raw), a.deep, a.expected)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(format_table(rep))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
