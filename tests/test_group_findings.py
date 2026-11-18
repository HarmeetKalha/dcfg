"""group_findings: synthetic results JSON/npz in tmp_path; no real results are read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops" / "jobs"))
import group_findings as gf  # noqa: E402
from dcfg.eval.metrics import bootstrap_ci, frame_auc, video_auc  # noqa: E402

NV, NF = 40, 3
VIDS = np.repeat([f"v{i}" for i in range(NV)], NF)
LAB = np.repeat(np.tile([0, 1], NV // 2), NF)
KEYS = np.array([f"{v}/{i}" for v, i in zip(VIDS, np.tile(range(NF), NV))])


def write_run(raw: Path, run_id, q, det="xception", conds=None, pilot=False, meta=None):
    names = conds or [c["name"] for c in gf.load_conditions()]
    (raw / "scores").mkdir(parents=True, exist_ok=True)
    for ci, c in enumerate(names):
        rng = np.random.default_rng(abs(hash((run_id, c))) % 2**31)
        qq = q * (1.0 if c == "clean" else 0.85)
        prob = np.clip(0.5 + (LAB - 0.5) * qq + rng.normal(0, 0.3, len(LAB)), 0, 1)
        np.savez_compressed(raw / "scores" / f"{run_id}__{c}.npz", frame_key=KEYS, frame_video=VIDS, frame_label=LAB, frame_prob=prob)
        doc = dict(run_id=run_id, detector=det, condition=c, split="test", compression="c23", n_frames=len(LAB), n_videos=NV,
                   frame_auc=frame_auc(LAB, prob), video_auc=video_auc(LAB, prob, VIDS),
                   bootstrap=dict(n=50, seed=0, unit="video", level=0.95), scores_file=f"scores/{run_id}__{c}.npz")
        for m in ("frame", "video"):
            ci = bootstrap_ci(m, LAB, prob, VIDS, n=50)
            doc[f"{m}_auc_ci"] = [ci.lo, ci.hi]
        if pilot:
            doc["pilot"] = True
        if meta:
            doc["train"] = meta
        json.dump(doc, open(raw / f"{run_id}__{c}.json", "w"))


def test_aggregates_paired_and_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DCFG_NO_NOTIFY", "1")
    raw = tmp_path / "raw"
    meta = dict(best_epoch=7, early_stop_epoch=12, train_wall_time_s=7200, trainable_mb=3.5, detector_mb=80.0, params=900000)
    write_run(raw, "base_xception", 0.5)
    write_run(raw, "augft_xception", 0.7, meta=meta)
    write_run(raw, "lumen_xception", 0.9, meta=meta)
    write_run(raw, "pilot_xception", 0.9)
    write_run(raw, "fixed_xception", 0.8, conds=["clean"])  # incomplete run
    out = tmp_path / "out"
    g = gf.main(["--group", "1", "--raw-dir", str(raw), "--out", str(out), "--n-boot", "100"])
    x = g.runs["xception"]
    # point estimate of the synthetic mean = plain mean of JSON AUCs
    exp = np.mean([x["lumen"].docs[c]["frame_auc"] for c in g.synth])
    v = g.val("xception", "lumen", "frame", "synth_mean")
    assert abs(v[0] - exp) < 1e-6 and v[1] <= v[0] <= v[2]
    assert len(g.synth) == 17 and len(g.held) == 4
    d = g.delta("xception", "lumen", "augft", "frame", "synth_mean")
    assert d[0] > 0 and d[3]
    assert g.delta("xception", "lumen", "fixed", "frame", "synth_mean") is None  # incomplete
    assert g.val("efficientnetb4", "lumen", "frame", "clean") is None  # missing run
    txt = (out / "FINDINGS_group1.md").read_text()
    assert "MISSING RUN lumen_ucf" in txt and "MISSING CONDITIONS fixed_xception" in txt
    assert "lumen_xception__clean.json" in txt and "pilot_" not in txt.split("## Source files")[1]
    for ext in ("csv", "md", "tex"):
        assert (out / "tables" / f"group1_auc_xception.{ext}").exists()
        assert (out / "tables" / f"group1_deltas.{ext}").exists()
    assert (out / "tables" / "group1_train_meta.md").read_text().count("7200") == 0  # shown in hours
    assert "2.00" in (out / "tables" / "group1_train_meta.md").read_text()
    for stem in ("group1_main_bars", "group1_curves_xception"):
        for ext in ("png", "pdf"):
            assert (out / "figures" / f"{stem}.{ext}").stat().st_size > 0


def test_no_runs_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("DCFG_NO_NOTIFY", "1")
    (tmp_path / "raw").mkdir()
    for n in (1, 2, 3, 4):
        gf.main(["--group", str(n), "--raw-dir", str(tmp_path / "raw"), "--out", str(tmp_path / "o"), "--n-boot", "10"])
        assert "MISSING RUN" in (tmp_path / "o" / f"FINDINGS_group{n}.md").read_text()


def test_pilot_json_excluded(tmp_path, monkeypatch):
    monkeypatch.setenv("DCFG_NO_NOTIFY", "1")
    raw = tmp_path / "raw"
    write_run(raw, "lumen_xception", 0.9, pilot=True)
    g = gf.Group(1, raw, n_boot=10)
    assert not g.runs["xception"]["lumen"].docs


def test_group4_seeds_and_notify(tmp_path, monkeypatch):
    log = tmp_path / "n.log"
    monkeypatch.setenv("DCFG_NOTIFY_LOG", str(log))
    raw = tmp_path / "raw"
    write_run(raw, "base_xception", 0.5)
    for i, s in enumerate(("", "_s43", "_s44")):
        write_run(raw, f"lumen_xception{s}", 0.9 + 0.01 * i)
        write_run(raw, f"augft_xception{s}", 0.6 + 0.01 * i)
    out = tmp_path / "o"
    gf.main(["--group", "4", "--raw-dir", str(raw), "--out", str(out), "--n-boot", "30"])
    st = gf.seed_stats(gf.Group(4, raw, n_boot=10))
    assert all(r["exceeds"] for r in st) and st[0]["n"] == (3, 3)
    assert (out / "tables" / "group4_seeds.csv").exists()
    t = log.read_text()
    assert t.startswith("LUMEN group 4 done:") and "lumen_s42 vs augft_s42" in t
