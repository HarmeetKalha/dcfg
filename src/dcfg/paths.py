"""Central filesystem locations (PLAN Section 6). No directories are created at import."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = Path(os.environ.get("DCFG_DATA_ROOT", Path.home() / "data")).expanduser()
RUNS_ROOT: Path = Path(os.environ.get("DCFG_RUNS_ROOT", REPO_ROOT / "runs")).expanduser()
RESULTS_ROOT: Path = REPO_ROOT / "results"

WEIGHTS_DIR: Path = DATA_ROOT / "weights"
LIVE_DIR: Path = DATA_ROOT / "live" / "ChallengeDB_release"
FFPP_RAW: Path = DATA_ROOT / "ffpp_raw"
FFPP_FRAMES: Path = DATA_ROOT / "ffpp_frames"
DFB_JSON: Path = DATA_ROOT / "dfb_json"
