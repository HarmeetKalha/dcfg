from pathlib import Path

from dcfg import paths


def test_paths_are_paths_and_layout():
    for n in ("DATA_ROOT", "REPO_ROOT", "RUNS_ROOT", "RESULTS_ROOT", "WEIGHTS_DIR",
              "LIVE_DIR", "FFPP_RAW", "FFPP_FRAMES", "DFB_JSON"):
        assert isinstance(getattr(paths, n), Path)
    assert (paths.REPO_ROOT / "pyproject.toml").exists()
    assert paths.WEIGHTS_DIR == paths.DATA_ROOT / "weights"
    assert paths.RESULTS_ROOT == paths.REPO_ROOT / "results"
