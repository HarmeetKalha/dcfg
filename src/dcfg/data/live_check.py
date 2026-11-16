"""Integrity check for the LIVE ChallengeDB release.

Usage: python -m dcfg.data.live_check [~/data/live/ChallengeDB_release]
Expects Data/AllImages_release.mat, Data/AllMOS_release.mat (1169 entries each) and
Images/ with the files named in AllImages_release (first 7 = trainingImages calibration set).
Prints a JSON report; exit code 1 if any check fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

N_TOTAL, N_CALIB = 1169, 7


def _mat_vector(path: Path, key: str) -> list:
    from scipy.io import loadmat

    arr = loadmat(str(path))[key].squeeze()
    return list(arr)


def check(root: Path) -> dict:
    from PIL import Image

    rep: dict = {"root": str(root), "ok": True, "errors": []}
    names_raw = _mat_vector(root / "Data" / "AllImages_release.mat", "AllImages_release")
    mos = _mat_vector(root / "Data" / "AllMOS_release.mat", "AllMOS_release")
    names = [str(n.item() if hasattr(n, "item") else n).strip() for n in names_raw]
    rep["n_images_mat"], rep["n_mos_mat"] = len(names), len(mos)
    if len(names) != N_TOTAL or len(mos) != N_TOTAL:
        rep["errors"].append(f"expected {N_TOTAL} entries")
    rep["calibration"] = names[:N_CALIB]
    rep["calib_all_training"] = all((root / "Images" / "trainingImages" / n).is_file() for n in names[:N_CALIB])
    bad, missing = [], []
    for n in names[N_CALIB:]:
        p = root / "Images" / n
        if not p.exists():
            missing.append(n)
            continue
        try:
            with Image.open(p) as im:
                im.convert("RGB").load()
        except Exception as e:  # noqa: BLE001
            bad.append(f"{n}: {e}")
    rep["n_checked"] = len(names) - N_CALIB
    rep["missing"], rep["unreadable"] = missing, bad
    if missing or bad:
        rep["errors"].append("missing or unreadable images")
    rep["ok"] = not rep["errors"]
    return rep


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/data/live/ChallengeDB_release").expanduser()
    r = check(root)
    print(json.dumps(r, indent=1)[:4000])
    sys.exit(0 if r["ok"] else 1)
