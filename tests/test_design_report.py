"""Design-selection report on synthetic scores: tables, paired deltas, missing runs, notify, output guards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "ops" / "jobs"
sys.path.insert(0, str(JOBS))
import design_selection_report as ds  # noqa: E402
import pilot_report as pr  # noqa: E402
from test_pilot import _write_scores  # noqa: E402

PARAMS = {m: {"trainable": 1000 * (i + 1), "pct_of_xception": 0.5 * (i + 1), "detector_trained": m in ("augft", "jointft")}
          for i, m in enumerate(ds.MODELS)}
OVER = {"pipelines": {k: {f"batch{b}": {"ms_per_image": 5.0 + i, "added_ms_per_image": float(i)} for b in (1, 32)}
                      for i, k in enumerate(("base", "fixed", "dcfg", "film", "filmlite"))}}


def _synth(tmp_path, qual, with_c40=True):
    ed = tmp_path / "eval"
    labels = np.repeat(np.tile([0, 1], 30), 4)
    videos = np.repeat([f"v{i}" for i in range(60)], 4)
    keys = np.array([f"{v}/{i}" for v, i in zip(videos, np.tile(range(4), 60))])
    for m, q in qual.items():
        for ci, c in enumerate(["clean", *pr.DEGRADED]):
            _write_scores(ed, m, c, q if c != "clean" else 0.9, 100 * ci + len(m), keys, labels, videos)
        if with_c40:
            sel = np.arange(len(labels)) < 200
            _write_scores(ed, m, "c40", q, 5, keys[sel], labels[sel], videos[sel])
    return ed


def _runs(tmp_path, models):
    for m in models:
        d = tmp_path / "runs" / f"pilot_{m}_xception"
        d.mkdir(parents=True)
        (d / "resources_summary.json").write_text(json.dumps({
            "gpu_util_mean_train": 97.0, "gpu_util_median_train": 98.0, "data_wait_share_mean": 0.01,
            "cpu_util_mean_train": 40.0, "gpu_mem_peak_mib": 7000.0, "gpu_temp_max_c": 75.0, "wall_train_s": 5 * 600.0,
            "wall_val_s": 5 * 120.0}))
        (d / "done.json").write_text(json.dumps({"total_time_s": 3600.0 + 7 * len(m), "epochs_run": 5}))
    return tmp_path / "runs"


QUAL = {"base": 0.2, "fixed": 0.3, "augft": 0.5, "dcfg": 0.4, "jointft": 0.8, "dcfgcons": 0.45, "dcfgfilm": 0.6,
        "dcfgfilmlite": 0.42}


def test_full_report_sections_deltas_and_json(tmp_path):
    ed = _synth(tmp_path, QUAL)
    rd = _runs(tmp_path, ds.MODELS[1:])
    (tmp_path / "runs" / "_runner.log").write_text(
        "2026-11-18T10:00:00+00:00 job start pilot_jointft_xception\n"
        "2026-11-18T12:00:00+00:00 job pilot_jointft_xception attempt 1 exit 0\n"
        "2026-11-18T13:00:00+00:00 job start pilot_ds_report\n")
    md, s, rows = ds.build(ed, rd, tmp_path / "runs", PARAMS, OVER, n_boot=40)
    assert md.startswith("# Design selection study (5 epochs, validation)")
    for h in ("## 1. Cost", "## 2. Frame AUC", "## 3. Paired bootstrap", "## 4. Highlighted: blur_s3, webp_q20, c40",
              "## 5. Resource summary", "## 6. Measured epoch times", "## 7. Total wall time"):
        assert h in md
    assert "no design is selected" in md and "MISSING runs" not in md
    # deltas: each new variant vs dcfg, augft, jointft, never itself
    assert "jointft - jointft" not in md and "jointft - dcfg" in md and "dcfgfilm - jointft" in md
    assert s["deltas"]["jointft-augft|mean_degraded"]["excludes_0"] is True      # 0.8 vs 0.5 quality
    assert "jointft-jointft|mean_degraded" not in s["deltas"]
    assert set(s["deg_mean"]) == set(ds.MODELS) and "c40" in s["conditions"]
    assert s["epoch_times_min"]["dcfgfilm"]["train_min"] == pytest.approx(10.0)
    assert "3.00 h" in md or "3.0" in md  # wall span 10:00 -> 13:00
    assert "whole detector counted as trainable" in md and "+0.00" in md
    txt = ds.notify_text(s)
    assert txt.startswith("design-selection pilots done") and 4 <= len(txt.splitlines()) <= 6
    assert "jointft" in txt and "vs augft" in txt
    assert len(rows) == 8 and rows[0]["model"] == "base" and rows[4]["auc_clean"] is not None


def test_missing_runs_are_marked(tmp_path):
    q = {k: v for k, v in QUAL.items() if k in ("base", "fixed", "augft", "dcfg", "dcfgcons")}
    ed = _synth(tmp_path, q)
    rd = _runs(tmp_path, ["fixed", "augft", "dcfg", "dcfgcons"])
    md, s, rows = ds.build(ed, rd, tmp_path / "runs", PARAMS, None, n_boot=30)
    assert "MISSING runs (no eval scores): jointft, dcfgfilm, dcfgfilmlite" in md
    assert s["missing"] == ["jointft", "dcfgfilm", "dcfgfilmlite"]
    assert "dcfgfilm - dcfg | MISSING" in md and "dcfgcons - dcfg" in md
    assert "pilot_jointft_xception | MISSING" in md
    assert s["epoch_times_min"]["jointft"] is None
    assert [r["present"] for r in rows].count(False) == 3
    assert "n/a" in md  # no overhead given
    assert "MISSING runs: jointft" in ds.notify_text(s) or "MISSING" in ds.notify_text(s)


def test_partial_degraded_conditions_excluded_from_mean(tmp_path):
    ed = _synth(tmp_path, {"base": 0.2, "augft": 0.5, "dcfg": 0.4})
    (ed / "scores" / "pilot_dcfg_xception__mixed_heavy.npz").unlink()
    a = ds.analyse(ed, 20)
    assert "dcfg" not in a["res"]["deg_point"] and "augft" in a["res"]["deg_point"]


def test_main_writes_three_files_and_notifies(tmp_path, monkeypatch):
    ed = _synth(tmp_path, QUAL)
    rd = _runs(tmp_path, ds.MODELS[1:])
    out = tmp_path / "out"
    out.mkdir()
    (out / "overhead.json").write_text(json.dumps(OVER))
    log = tmp_path / "n.log"
    monkeypatch.setenv("DCFG_NOTIFY_LOG", str(log))
    rc = ds.main(["--eval-dir", str(ed), "--runs-dir", str(rd), "--runs-root", str(tmp_path / "runs"),
                  "--out-dir", str(out), "--n-boot", "20"])
    assert rc == 0
    assert (out / "report.md").exists() and (out / "results.csv").exists()
    j = json.loads((out / "results.json").read_text())
    assert "_res" not in j and j["n_boot"] == 20
    assert log.read_text().startswith("DCFG design selection: design-selection pilots done")
    monkeypatch.delenv("DCFG_NOTIFY_LOG")
    monkeypatch.setenv("DCFG_NO_NOTIFY", "1")
    ds.notify("x")  # silent


@pytest.mark.parametrize("sub", ["raw", "tables", "figures"])
def test_refuses_results_raw_tables_figures(sub):
    with pytest.raises(SystemExit, match="refusing"):
        ds.check_out_dir(ds.REPO / "results" / sub / "x")
    ds.check_out_dir(ds.REPO / "results" / "pilot" / "design_selection")


def test_overhead_design_mode_refuses_raw(tmp_path):
    from dcfg.analysis import overhead as oh
    with pytest.raises(SystemExit):
        oh.main(["--design-selection"])
    with pytest.raises(SystemExit, match="refusing"):
        oh.main(["--design-selection", "--out", str(oh.RESULTS_ROOT / "raw" / "o.json")])


def test_trainable_param_counts():
    from dcfg.analysis.overhead import trainable_params_by_model
    p = trainable_params_by_model()
    n = p["_xception_params"]["trainable"]
    gate, sensor = 320 * 256 + 256 + 256 * 192 + 192 + 2 * 192 + 2 * 128, None
    assert p["base"]["trainable"] == 0 and p["augft"]["trainable"] == n and p["augft"]["detector_trained"]
    assert p["jointft"]["trainable"] == p["dcfg"]["trainable"] + n
    assert p["dcfgcons"] == p["dcfg"]
    assert p["dcfgfilm"]["trainable"] - p["dcfg"]["trainable"] == 2_208_272
    assert p["dcfgfilmlite"]["trainable"] - p["dcfg"]["trainable"] == 369_136
    assert p["fixed"]["trainable"] == 192
    assert p["dcfg"]["trainable"] > gate
