"""Pure-logic tests for ops/jobs/phase8_smoke.py (ETA math, resume verification) and gen_matrix.py."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

JOBS = Path(__file__).resolve().parents[1] / "ops" / "jobs"
sys.path.insert(0, str(JOBS))
import gen_matrix as gm  # noqa: E402
import phase8_smoke as ps  # noqa: E402


# ------------------------------------------------------------------ gen_matrix
def test_matrix_is_exactly_the_74_list_in_order() -> None:
    ids = [r["run_id"] for r in gm.matrix_runs()]
    assert len(ids) == len(set(ids)) == 20
    exp_t1 = [f"{v}_{d}" for d in ("xception", "efficientnetb4", "f3net", "ucf") for v in ("dcfg", "fixed", "augft")]
    assert ids[:12] == exp_t1
    assert ids[12:] == ["dcfg_jointft_xception", "dcfg_nolive_xception", "gate_nosensor_xception",
                        "dcfg_frozensensor_xception", "dcfg_xception_s43", "dcfg_xception_s44",
                        "fixed_xception_s43", "fixed_xception_s44"]


def test_run_attributes() -> None:
    by = {r["run_id"]: r for r in gm.matrix_runs()}
    assert by["dcfg_ucf"]["variant"] == "dcfg" and by["dcfg_ucf"]["detector"] == "ucf"
    assert by["augft_f3net"]["variant"] == "augft" and by["augft_f3net"]["seed"] == 42
    assert by["fixed_xception_s44"]["seed"] == 44 and by["fixed_xception_s44"]["variant"] == "fixed"
    assert by["dcfg_xception_s43"]["seed"] == 43
    assert by["gate_nosensor_xception"]["variant"] == "nosensor"
    assert by["dcfg_nolive_xception"]["variant"] == "dcfg_nolive"
    assert by["dcfg_frozensensor_xception"]["freeze_sensor"] and by["dcfg_frozensensor_xception"]["variant"] == "dcfg"
    assert sum(r["freeze_sensor"] for r in by.values()) == 1
    assert all(r["detector"] == "xception" for r in by.values() if r["tier"] == 2)


def test_configs_and_queue_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bfile = tmp_path / "b.yaml"
    bfile.write_text(yaml.safe_dump({"xception": 32, "efficientnetb4": 24, "f3net": 32, "ucf": 16}))
    out = tmp_path / "cfg"
    assert gm.main(["--out-dir", str(out), "--batch-yaml", str(bfile)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 40 and len({ln.split(" | ")[0] for ln in lines}) == 40
    assert lines[0].startswith("dcfg_xception | ") and lines[1].startswith("eval_dcfg_xception | ")
    assert "--conditions all --split test --frontend runs/dcfg_xception/frontend.pt" in lines[1]
    assert lines[4].startswith("augft_xception | ") and "--detector-ckpt runs/augft_xception/detector.pt" in lines[5]
    assert "--frontend" not in lines[5]
    joint = next(ln for ln in lines if ln.startswith("eval_dcfg_jointft_xception"))
    assert "--frontend" in joint and "--detector-ckpt" in joint
    assert all(ln.endswith("#hb=30") for ln in lines)
    cfgs = sorted(out.glob("*.yaml"))
    assert len(cfgs) == 20
    c = yaml.safe_load((out / "dcfg_efficientnetb4.yaml").read_text())
    assert c["batch_size"] == 24 and c["detector"] == "efficientnetb4" and c["variant"] == "dcfg"
    assert yaml.safe_load((out / "fixed_ucf.yaml").read_text())["batch_size"] == 16
    assert yaml.safe_load((out / "fixed_xception_s44.yaml").read_text())["seed"] == 44
    assert yaml.safe_load((out / "dcfg_frozensensor_xception.yaml").read_text())["freeze_sensor"] is True


def test_gen_matrix_does_not_touch_queue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    q = JOBS.parent / "queue.txt"
    before = q.read_bytes()
    gm.main(["--no-write", "--batch-yaml", str(tmp_path / "missing.yaml")])
    assert q.read_bytes() == before
    assert "WARNING" in capsys.readouterr().err


def test_generated_configs_load_with_engine(tmp_path: Path) -> None:
    from dcfg.train.train_frontend import load_config
    (tmp_path / "base.yaml").write_text((JOBS.parents[1] / "configs/train/base.yaml").read_text())
    for r in gm.matrix_runs():
        (tmp_path / f"{r['run_id']}.yaml").write_text(gm.config_text(r, 32))
        cfg = load_config(tmp_path / f"{r['run_id']}.yaml")
        assert cfg["run_id"] == r["run_id"] and cfg["variant"] == r["variant"] and cfg["seed"] == r["seed"]


# ------------------------------------------------------------------ ETA math
def test_steps_and_run_eta() -> None:
    assert ps.steps_per_epoch(1000, 32, False) == 62
    assert ps.steps_per_epoch(1000, 32, True) == 62  # 1000 // 16
    e = ps.run_eta(spe=100, bs=32, img_s=64.0, val_epoch_s=30.0, eval_s=600.0, epochs=30)
    assert e["train_epoch_s"] == pytest.approx(50.0) and e["epoch_s"] == pytest.approx(80.0)
    assert e["total_s"] == pytest.approx(30 * 80 + 600)


def test_eval_seconds_sources() -> None:
    assert ps.eval_seconds({"a": 60.0, "b": 70.0, "c": 80.0}, 23, 1.0) == (70.0 * 23, "log (3/23 conds)")
    assert ps.eval_seconds({}, 23, 10.0) == (230.0, "estimate")
    assert ps.eval_seconds({}, 23, None)[0] is None


def test_parse_eval_log() -> None:
    t = "=== x\nbase_x: 22/23 conditions to run\n  jpeg_q90: /a/b.json (66s)\n  resize_0.5: /a/c.json (65s)\n"
    assert ps.parse_eval_log(t) == {"jpeg_q90": 66.0, "resize_0.5": 65.0}


def test_build_eta_totals_and_missing() -> None:
    runs = [r for r in gm.matrix_runs() if r["detector"] == "xception"]
    rates = {"xception": {"dcfg": 100.0, "fixed": 120.0, "augft": 50.0, "jointft": 40.0}}
    val = {"xception": {"dcfg": 600.0, "augft": 500.0}}
    rows, tot = ps.build_eta(runs, {"xception": 32}, rates, val, {"xception": (1500.0, "log")}, n_fake=64000)
    assert len(rows) == len(runs) == 11 and not tot["missing"]
    by = {r["run_id"]: r for r in rows}
    assert by["dcfg_xception"]["steps_per_epoch"] == 4000
    assert by["dcfg_xception"]["train_epoch_s"] == pytest.approx(4000 * 32 / 100)
    assert by["gate_nosensor_xception"]["timing_class"] == "fixed"  # nosensor timed as fixed
    assert by["dcfg_frozensensor_xception"]["timing_class"] == "dcfg"
    assert by["augft_xception"]["val_epoch_s"] == 500.0 and by["dcfg_jointft_xception"]["val_epoch_s"] == 600.0
    assert tot["worst_gpu_days"] == pytest.approx(sum(r["total_worst_s"] for r in rows) / 86400)
    assert tot["likely_gpu_days"] < tot["worst_gpu_days"]
    rows2, tot2 = ps.build_eta(runs + [gm.matrix_runs()[3]], {}, rates, val, {"xception": (1.0, "log")}, 64000)
    assert tot2["missing"] == ["dcfg_efficientnetb4"]
    md = ps.eta_markdown(rows, tot, ["note"])
    assert "30 ep total" in md and "GPU-days" in md and "ASSUMPTION" in md


def test_ucf_augft_uses_paired_steps() -> None:
    runs = [r for r in gm.matrix_runs() if r["run_id"] == "augft_ucf"]
    rows, _ = ps.build_eta(runs, {"ucf": 16}, {"ucf": {"augft": 10.0}}, {"ucf": {"augft": 1.0}},
                           {"ucf": (1.0, "x")}, n_fake=800)
    assert rows[0]["steps_per_epoch"] == 800 // 8


# ------------------------------------------------------------------ smoke logic
def test_loss_and_throughput_helpers() -> None:
    rows = [{"step": "50", "loss": "0.9", "img_s": "10"}, {"step": "100", "loss": "0.8", "img_s": "100"},
            {"step": "150", "loss": "0.5", "img_s": "120"}]
    assert ps.loss_first_last(rows)["decreased"] and ps.loss_first_last(rows[:1])["decreased"] is False
    assert ps.steady_img_s(rows, 100) == 110.0 and ps.steady_img_s(rows, 999) is None


def test_largest_fitting_and_pick() -> None:
    calls = []

    def fn(bs: int) -> dict:
        calls.append(bs)
        return {"ok": bs <= 24, "oom": bs > 24}
    assert ps.largest_fitting(fn)[0] == 24 and calls == [32, 24]
    assert ps.largest_fitting(lambda b: {"ok": False, "oom": False})[0] is None
    assert ps.largest_fitting(lambda b: {"ok": False, "oom": True})[0] is None
    assert ps.pick_batch({"dcfg": 24, "augft": 16}) == 16 and ps.pick_batch({"dcfg": 32, "augft": None}) is None
    assert ps.is_oom("CUDA out of memory. Tried") and not ps.is_oom("boom")


def _ref(e: int, b: int) -> str:
    return hashlib.sha1(f"{e}-{b}".encode()).hexdigest()[:12]


def test_verify_resume_ok_and_failures() -> None:
    lines = [f"0 {s} {s} {_ref(0, s)}" for s in range(250)]
    lines += ["# resumed epoch=0 step_in_epoch=200 step=200"] + [f"0 {s} {s} {_ref(0, s)}" for s in range(200, 400)]
    v = ps.verify_resume(lines, _ref, 400)
    assert v["ok"] and v["n_resumes"] == 1 and v["reexecuted_steps"] == [200, 249] and v["counter_continues"]
    # skipped batch
    bad = [ln for ln in lines if not ln.startswith("0 300 ")]
    assert ps.verify_resume(bad, _ref, 400)["missing_steps"] == [300]
    # wrong hash after resume
    bad = [("0 210 210 deadbeef0000" if i >= 251 and ln.startswith("0 210 210") else ln) for i, ln in enumerate(lines)]
    v = ps.verify_resume(bad, _ref, 400)
    assert not v["ok"] and 210 in v["hash_conflicts"]
    # counter restarts from 0 after the marker
    bad = lines[:250] + ["# resumed epoch=0 step_in_epoch=200 step=200"] + [f"0 {s} {s} {_ref(0, s)}" for s in range(0, 400)]
    assert not ps.verify_resume(bad, _ref, 400)["counter_continues"]
    # no resume at all
    assert not ps.verify_resume(lines[:250], _ref, 250)["ok"]


def test_variant_sets_and_cmd() -> None:
    assert ps.variant_sets("dcfg_frozensensor") == {"variant": "dcfg", "freeze_sensor": True}
    cmd = ps.trainer_cmd(Path("/x"), {"a": True, "b": None, "c": 3})
    assert cmd[-3:] == ["a=true", "b=null", "c=3"] and "--run-dir" in cmd
