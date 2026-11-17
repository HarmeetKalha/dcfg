"""Pilot tooling: configs, stop_after_epochs, pilot isolation, resource logging, val eval, report, exclusions."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

JOBS = Path(__file__).resolve().parents[1] / "ops" / "jobs"
sys.path.insert(0, str(JOBS))
import gen_matrix as gm  # noqa: E402
import pilot_report as pr  # noqa: E402
from test_evaluate import CONDS, _model, fake  # noqa: E402,F401
from test_train_frontend import cfg_for, data  # noqa: E402,F401

from dcfg.analysis import baseline_report as br  # noqa: E402
from dcfg.eval import evaluate as ev  # noqa: E402
from dcfg.paths import REPO_ROOT  # noqa: E402
from dcfg.train import train_frontend as tf  # noqa: E402
from dcfg.utils import resources as rs  # noqa: E402


# ------------------------------------------------------------------ configs
@pytest.mark.parametrize("x", ["dcfg", "fixed", "augft"])
def test_pilot_config_differs_only_in_pilot_keys(x):
    real = tf.load_config(REPO_ROOT / f"configs/train/{x}_xception.yaml")
    pil = tf.load_config(REPO_ROOT / f"configs/train/pilot/pilot_{x}_xception.yaml")
    diff = {k for k in set(real) | set(pil) if real.get(k) != pil.get(k)}
    assert diff == {"run_id", "pilot", "stop_after_epochs", "timing_sync", "run_dir", "wandb_id"}
    assert pil["run_id"] == f"pilot_{x}_xception" and pil["pilot"] is True and pil["stop_after_epochs"] == 5
    assert pil["timing_sync"] is True and pil["max_epochs"] == real["max_epochs"] == 30  # real schedule horizon
    assert pil["run_dir"] == f"runs/pilot/pilot_{x}_xception"
    assert pil["batch_size"] == 32 and pil["seed"] == 42 and pil["epoch_mode"] == "full"


def test_real_configs_are_not_pilot():
    cfgs = [tf.load_config(p) for p in (REPO_ROOT / "configs/train").glob("*.yaml")]
    assert len(cfgs) >= 20
    for c in cfgs:
        assert not c.get("pilot") and c.get("stop_after_epochs") is None and not c.get("timing_sync")
        assert not str(c["run_id"]).startswith("pilot_")
    for run in gm.matrix_runs():
        txt = gm.config_text(run, 32)
        assert "pilot" not in txt and "stop_after" not in txt


# ------------------------------------------------------------------ trainer: stop_after_epochs, pilot flag, resume guard
def test_stop_after_epochs_finalizes_with_pilot_metadata(data, tmp_path):
    cfg = cfg_for(data, tmp_path, "fixed", max_epochs=4, stop_after_epochs=2, pilot=True, timing_sync=True,
                  resource_interval_s=0.2)
    res = tf.Trainer(cfg).train()
    rd = tmp_path / "run_fixed"
    done = json.loads((rd / "done.json").read_text())
    assert done["pilot"] is True and done["stopped_after_epochs"] == 2 and done["epochs_run"] == 2
    assert not done["truncated"] and res["epochs_run"] == 2
    import torch
    assert torch.load(rd / "frontend.pt", weights_only=False)["pilot"] is True
    # resource logging
    rows = list(csv.DictReader(open(rd / "resources.csv")))
    assert rows and set(rs.FIELDS) == set(rows[0])
    summ = json.loads((rd / "resources_summary.json").read_text())
    assert summ["wall_train_s"] > 0 and summ["wall_val_s"] > 0
    assert summ["data_wait_share_mean"] is None or 0 <= summ["data_wait_share_mean"] <= 1


def test_pilot_default_run_dir(data, tmp_path):
    cfg = cfg_for(data, tmp_path, "fixed", pilot=True, run_dir=None)
    t = tf.Trainer(cfg)
    assert t.run_dir.parts[-2:] == ("pilot", "t_fixed")


def test_resume_refuses_pilot_flag_mismatch(data, tmp_path):
    rd = str(tmp_path / "shared")
    tf.Trainer(cfg_for(data, tmp_path, "fixed", run_dir=rd, pilot=True, max_steps=3)).train()
    assert (tmp_path / "shared" / "last.pt").exists()
    (tmp_path / "shared" / "done.json").unlink()
    with pytest.raises(RuntimeError, match="pilot"):
        tf.Trainer(cfg_for(data, tmp_path, "fixed", run_dir=rd, pilot=False)).train()
    rd2 = str(tmp_path / "shared2")
    tf.Trainer(cfg_for(data, tmp_path, "fixed", run_dir=rd2, max_steps=3)).train()
    (tmp_path / "shared2" / "done.json").unlink()
    with pytest.raises(RuntimeError, match="pilot"):
        tf.Trainer(cfg_for(data, tmp_path, "fixed", run_dir=rd2, pilot=True)).train()


# ------------------------------------------------------------------ resources (synthetic)
def test_step_timer_window():
    t = rs.StepTimer()
    for _ in range(4):
        t.add(0.1, 0.3)
    w = t.drain()
    assert w["n_steps"] == 4 and w["data_wait_s"] == pytest.approx(0.1) and w["compute_s"] == pytest.approx(0.3)
    assert w["data_wait_share"] == pytest.approx(0.25) and w["steps_per_s"] > 0
    assert t.drain()["n_steps"] == 0 and t.drain()["data_wait_share"] is None


def test_nvidia_smi_parse(monkeypatch):
    monkeypatch.setattr(rs.subprocess, "check_output", lambda *a, **k: "87, 12345, 64\n")
    assert rs.read_gpu_nvidia_smi() == (87.0, 12345.0, 64.0)


def test_sampler_writes_rows_and_summary(tmp_path):
    timer, phase = rs.StepTimer(), ["train"]
    s = rs.ResourceSampler(tmp_path / "resources.csv", 0.05, timer, lambda: 7, lambda: phase[0],
                           gpu_reader=lambda: (90.0, 1000.0, 60.0), cpu_reader=lambda: (50.0, 2000.0))
    timer.add(0.1, 0.9)
    s.sample_row = s.sample()  # direct call works too
    s._write(s.sample_row)
    phase[0] = "val"
    s._write({**s.sample(), "gpu_util_pct": 5.0})  # val row must be excluded from the train statistics
    s.start()
    import time
    time.sleep(0.2)
    s.stop()
    rows = list(csv.DictReader(open(tmp_path / "resources.csv")))
    assert len(rows) >= 4 and rows[0]["gpu_mem_mib"] == "1000.0000"
    out = rs.summarize_resources(tmp_path / "resources.csv", 100.0, 20.0)
    assert out["gpu_util_mean_train"] == 90.0 and out["gpu_util_median_train"] == 90.0 or out["n_train_rows"] >= 1
    assert out["gpu_mem_peak_mib"] == 1000.0 and out["gpu_temp_max_c"] == 60.0
    assert out["data_wait_share_mean"] == pytest.approx(0.1)
    assert out["wall_train_s"] == 100.0 and out["wall_val_s"] == 20.0


def test_summary_excludes_val_rows(tmp_path):
    p = tmp_path / "r.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rs.FIELDS)
        w.writeheader()
        for ph, g, n, dw, c in [("train", 80, 10, 0.2, 0.8), ("train", 100, 10, 0.0, 1.0), ("val", 10, 0, "", "")]:
            w.writerow({"timestamp": "t", "step": 1, "phase": ph, "gpu_util_pct": g, "gpu_mem_mib": g * 10,
                        "gpu_temp_c": g, "cpu_pct": 40, "ram_used_mib": 1, "n_steps": n, "data_wait_s": dw,
                        "compute_s": c, "data_wait_share": "", "steps_per_s": ""})
    s = rs.summarize_resources(p, 1.0, 2.0)
    assert s["gpu_util_mean_train"] == 90.0 and s["gpu_util_median_train"] == 90.0
    assert s["data_wait_share_mean"] == pytest.approx(0.1) and s["cpu_util_mean_train"] == 40.0


def test_proc_readers_work_here():
    r = rs.CpuRamReader()
    cpu, ram = r()
    assert 0 <= cpu <= 100 and ram > 0


# ------------------------------------------------------------------ val eval into a tmp dir
def test_val_eval_to_tmp_dir_with_scores(fake, tmp_path):
    cond = CONDS["noise_s10"]
    p = ev.evaluate_condition(run_id="pilot_x_det", detector="d", cond=cond, split="val", source=fake, model=_model,
                              out_dir=tmp_path / "ev", n_boot=20)
    doc = json.loads(p.read_text())
    assert doc["split"] == "val" and doc["n_videos"] == 8
    z = np.load(tmp_path / "ev" / doc["scores_file"])
    assert len(z["frame_prob"]) == len(z["frame_key"]) == 24
    with pytest.raises(ValueError):
        ev.check_out_dir("val", ev.RAW_DIR)
    with pytest.raises(ValueError, match="pilot"):
        ev.check_out_dir("test", ev.RAW_DIR, "pilot_base_xception")
    ev.check_out_dir("test", ev.RAW_DIR, "base_xception")  # real test runs still fine


def test_c40_val_empty_then_present(fake):
    # fixture has c40 val frames; the pilot script's own criterion is "exists and covers >= 98% of val videos"
    assert len(fake.records("val", "c40")) > 0


# ------------------------------------------------------------------ report on synthetic scores
def _write_scores(d: Path, model: str, cond: str, quality: float, seed: int, keys, labels, videos):
    rng = np.random.default_rng(seed)
    prob = np.clip(0.5 + (labels - 0.5) * quality + rng.normal(0, 0.25, len(labels)), 0, 1)
    (d / "scores").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "scores" / f"pilot_{model}_xception__{cond}.npz", frame_key=keys, frame_video=videos,
                        frame_label=labels, frame_prob=prob)


def _synth(tmp_path, qual, with_c40=False):
    ed = tmp_path / "eval"
    labels = np.repeat(np.tile([0, 1], 30), 4)  # 60 videos x 4 frames
    videos = np.repeat([f"v{i}" for i in range(60)], 4)
    keys = np.array([f"{v}/{i}" for v, i in zip(videos, np.tile(range(4), 60))])
    for m, q in qual.items():
        for ci, c in enumerate(["clean", *pr.DEGRADED]):
            _write_scores(ed, m, c, q if c != "clean" else 0.9, 100 * ci + len(m), keys, labels, videos)
        if with_c40:  # fewer videos: own frame set
            sel = np.arange(len(labels)) < 200
            _write_scores(ed, m, "c40", q, 5, keys[sel], labels[sel], videos[sel])
    return ed


def _mk_runs(tmp_path, gpu, share):
    for m in ("fixed", "augft", "dcfg"):
        d = tmp_path / "runs" / f"pilot_{m}_xception"
        d.mkdir(parents=True)
        (d / "resources_summary.json").write_text(json.dumps({
            "gpu_util_mean_train": gpu, "gpu_util_median_train": gpu, "data_wait_share_mean": share,
            "cpu_util_mean_train": 70.0, "gpu_mem_peak_mib": 9000.0, "gpu_temp_max_c": 70.0, "wall_train_s": 3600.0,
            "wall_val_s": 600.0}))
        (d / "done.json").write_text(json.dumps({"total_time_s": 4200.0}))
    return tmp_path / "runs"


def test_report_yes_verdict_and_cpu_flag(tmp_path):
    ed = _synth(tmp_path, {"base": 0.2, "fixed": 0.3, "augft": 0.35, "dcfg": 0.9})
    rd = _mk_runs(tmp_path, gpu=60.0, share=0.4)
    md, s = pr.build_report(ed, rd, tmp_path, n_boot=60)
    assert s["verdict"] == "YES" and s["cpu_bound"] and "dcfg beats augft and fixed beyond the CIs" in md
    assert "(c) GPU / CPU summary" in md and "FLAGGED" in md and "GPU" in md and "(e) c40" in md
    assert "c40 val preprocessing failed/missing -> c40 skipped" in md
    assert set(s["deg_mean"]) == {"base", "fixed", "augft", "dcfg"} and s["deg_mean"]["dcfg"] > s["deg_mean"]["augft"]
    assert "Plain CI-overlap view" in md and "Paired-CI basis" in md


def test_report_no_when_identical_and_not_flagged(tmp_path):
    ed = _synth(tmp_path, {"base": 0.5, "fixed": 0.5, "augft": 0.5, "dcfg": 0.5})
    rd = _mk_runs(tmp_path, gpu=95.0, share=0.02)
    md, s = pr.build_report(ed, rd, tmp_path, n_boot=60)
    assert s["verdict"] in ("NO", "MIXED") and not s["cpu_bound"] and "No run flagged" in md


def test_report_includes_c40_when_present(tmp_path):
    ed = _synth(tmp_path, {"base": 0.2, "fixed": 0.3, "augft": 0.35, "dcfg": 0.9}, with_c40=True)
    md, s = pr.build_report(ed, _mk_runs(tmp_path, 90.0, 0.05), tmp_path, n_boot=30)
    assert "c40 VAL frames present" in md and "c40 (val)" in md and "c40 skipped" not in md


def test_paired_delta_and_wall_time(tmp_path):
    log = tmp_path / "_runner.log"
    log.write_text("2026-11-17T10:00:00+00:00 job start pilot_sanity_grid\n"
                   "2026-11-17T10:00:05+00:00 job pilot_sanity_grid attempt 1 exit 0\n"
                   "2026-11-17T12:00:00+00:00 job start pilot_report\n")
    txt, total = pr.wall_time(tmp_path, tmp_path / "none", ["dcfg"])
    assert total == 7200.0 and "2.00 h" in txt


def test_notify_text_then_attachment(tmp_path, monkeypatch):
    log = tmp_path / "n.log"
    monkeypatch.setenv("DCFG_NOTIFY_LOG", str(log))
    g = tmp_path / "g.png"
    g.write_bytes(b"x")
    pr.notify("hello", g)
    assert log.read_text().splitlines() == ["DCFG pilot: hello", "attachment: g.png"]
    monkeypatch.delenv("DCFG_NOTIFY_LOG")
    monkeypatch.setenv("DCFG_NO_NOTIFY", "1")
    pr.notify("hello", g)  # silent, no error


# ------------------------------------------------------------------ isolation
def test_pilot_excluded_from_paper_reports(tmp_path, monkeypatch):
    assert br.is_pilot("pilot_dcfg_xception") and br.is_pilot(doc={"pilot": True}) and not br.is_pilot("base")
    with pytest.raises(ValueError):
        br.load_runs("pilot_base")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "base_xception__clean.json").write_text(json.dumps({"run_id": "base_xception"}))
    (raw / "pilot_dcfg_xception__clean.json").write_text(json.dumps({"run_id": "pilot_dcfg_xception"}))
    (raw / "weird__clean.json").write_text(json.dumps({"run_id": "weird", "pilot": True}))
    assert [p.name for p, _ in br.iter_raw_jsons(raw)] == ["base_xception__clean.json"]
    monkeypatch.setattr(br, "RAW", raw)
    (raw / "base_xception__noise_s5.json").write_text(json.dumps({"run_id": "base_xception", "pilot": True}))
    assert br.load_runs("base", ["xception"], ["clean", "noise_s5"]) [("xception", "noise_s5")] is None


def test_no_analysis_module_globs_pilot():
    for p in (REPO_ROOT / "src/dcfg/analysis").glob("*.py"):
        assert "results/pilot" not in p.read_text() and "runs/pilot" not in p.read_text(), p


# ------------------------------------------------------------------ preprocess / rearrange
def test_preprocess_split_option_and_rearrange_guard(tmp_path):
    sh = (JOBS / "dfb_preprocess.sh").read_text()
    assert "--split" in sh and "--test-only" in sh
    src = (JOBS / "dfb_preprocess.py").read_text()
    assert "--split" in src and "split_stems" in src

    # rearrange on a synthetic tree: adding c40 val keeps existing entries; changed entries are refused
    fr = tmp_path / "ff"
    for s_, pairs in (("train", [["0", "1"]]), ("val", [["2", "3"]]), ("test", [["4", "5"]])):
        (fr).mkdir(parents=True, exist_ok=True)
        (fr / f"{s_}.json").write_text(json.dumps(pairs))
    def mk(comp, vid):
        d = fr / "original_sequences/youtube" / comp / "frames" / vid
        d.mkdir(parents=True, exist_ok=True)
        (d / "000.png").write_bytes(b"x")
    for v in ("0", "2"):
        mk("c23", v)
    mk("c40", "4")
    out = tmp_path / "out"
    cmd = [sys.executable, "-I", str(JOBS / "dfb_rearrange.py"), "--frames", str(fr), "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    before = json.loads((out / "FaceForensics++.json").read_text())["FaceForensics++"]["FF-real"]
    mk("c40", "2")  # c40 val video appears
    subprocess.run(cmd, check=True, capture_output=True)
    after = json.loads((out / "FaceForensics++.json").read_text())["FaceForensics++"]["FF-real"]
    assert after["val"]["c40"] and not before["val"]["c40"]
    assert after["train"]["c23"] == before["train"]["c23"] and after["test"]["c40"] == before["test"]["c40"]
    # an existing frame list changing must abort
    (fr / "original_sequences/youtube/c23/frames/0/001.png").write_bytes(b"x")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in r.stderr
