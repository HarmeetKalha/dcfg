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
