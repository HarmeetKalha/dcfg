import json

import pytest
import torch
import yaml

from dcfg.paths import LIVE_DIR, REPO_ROOT
from dcfg.train import train_sensor as ts

pytestmark = pytest.mark.skipif(not LIVE_DIR.exists(), reason="LIVE data missing")


def _cfg(tmp_path, epochs):
    cfg = yaml.safe_load((REPO_ROOT / "configs/sensor_pretrain.yaml").read_text())
    cfg.update(subset=64, max_epochs=epochs, num_workers=2, batch_size=16, patience=5,
               split_path=str(tmp_path / "split.json"), weights_path=str(tmp_path / "w" / "sensor_live.pt"),
               results_dir=str(tmp_path / "res"), heartbeat_s=1)
    return cfg


def test_tiny_run_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("DCFG_JOB_ID", "tiny")
    run = tmp_path / "run"
    ts.run_main(_cfg(tmp_path, 1), run, False)
    assert (run / "last.pt").exists() and (run / "best.pt").exists() and (run / "heartbeat").exists()
    assert torch.load(run / "last.pt", weights_only=False)["epoch"] == 1
    # resume with a larger budget: continues at epoch 2
    res = ts.run_main(_cfg(tmp_path, 2), run, False)
    assert torch.load(run / "last.pt", weights_only=False)["epoch"] == 2
    lines = (run / "metrics.csv").read_text().strip().splitlines()
    assert [l.split(",")[0] for l in lines[1:]] == ["1", "2"]
    out = json.loads((tmp_path / "res" / "sensor_live__test.json").read_text())
    assert out["n_test"] == res["n_test"] and "five_crop" in out["test"]
    sd = torch.load(tmp_path / "w" / "sensor_live.pt", weights_only=True)
    assert all(k.startswith(("features.", "embedding.")) for k in sd)


def test_clobber_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = _cfg(tmp_path, 1)
    monkeypatch.setenv("DCFG_JOB_ID", "A")
    ts.run_main(cfg, tmp_path / "r1", False)
    monkeypatch.setenv("DCFG_JOB_ID", "B")
    with pytest.raises(FileExistsError):
        ts.run_main(cfg, tmp_path / "r2", False)
    monkeypatch.setenv("DCFG_JOB_ID", "A")  # retry of same job
    ts.run_main(cfg, tmp_path / "r1", False)
    monkeypatch.setenv("DCFG_JOB_ID", "B")
    ts.run_main(cfg, tmp_path / "r2", True)  # --overwrite
    assert json.loads((tmp_path / "res" / "sensor_live__test.json").read_text())["job_id"] == "B"
    assert (tmp_path / "w" / "sensor_live.job").read_text() == "B"
    # unknown owner (no sidecar) is refused
    (tmp_path / "w" / "sensor_live.job").unlink()
    monkeypatch.setenv("DCFG_JOB_ID", "C")
    with pytest.raises(FileExistsError):
        ts.run_main(cfg, tmp_path / "r3", False)


def test_tiny_protocol10(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("DCFG_JOB_ID", "p10")
    cfg = _cfg(tmp_path, 1)
    cfg["p10_splits"] = 2
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "sensor_live__test.json").write_text(json.dumps({"best_epoch": 1, "job_id": "main"}))
    run = tmp_path / "p10run"
    ts.run_protocol10(cfg, run)
    assert (run / "heartbeat").exists() and not (run / "split0" / "heartbeat").exists()
    hb = json.loads((run / "heartbeat").read_text())
    assert {"split", "epoch", "step"} <= set(hb)
    out = json.loads((tmp_path / "res" / "sensor_live__protocol10.json").read_text())
    assert out["n_splits"] == 2 and "five_crop" in out["median"]
    with pytest.raises(FileExistsError):
        monkeypatch.setenv("DCFG_JOB_ID", "other")
        ts.run_protocol10(cfg, tmp_path / "p10b")
