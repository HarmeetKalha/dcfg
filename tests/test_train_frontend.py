"""Tests for the front-end training engine on tiny synthetic DFB-format data and a dummy detector."""
from __future__ import annotations

import json
import os

os.environ.setdefault("WANDB_MODE", "disabled")

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from dcfg.eval import evaluate as ev
from dcfg.models.sensor import DegradationSensor
from dcfg.paths import REPO_ROOT
from dcfg.train import train_frontend as tf

SZ = 64


class DummyDetector(nn.Module):
    """Same contract as dcfg.detectors.adapter.Detector, tiny and CPU-fast."""

    n_normalize = 0

    def __init__(self, name="dummy", weights_path=None, device="cpu", load_weights=True):
        super().__init__()
        self.name = name
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))
        self.model = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4), nn.BatchNorm2d(4), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self._frozen = False
        self.to(device)
        self.eval()

    def normalize(self, x01):
        DummyDetector.n_normalize += 1
        return (x01 - self.mean) / self.std

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        self.eval()
        return self

    def train(self, mode=True):
        return super().train(False if self._frozen else mode)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    root, J = tmp_path / "frames", {}
    rng = np.random.default_rng(0)
    for label in ("FF-real", "FF-DF", "FF-F2F"):
        J[label] = {s: {c: {} for c in ("c23", "c40")} for s in ("train", "val", "test")}
        for split in ("train", "val", "test"):
            for comp in ("c23", "c40"):
                for v in range(4):
                    fs = []
                    for k in range(3):
                        p = root / label / comp / f"{split}{v}" / f"{k:03d}.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        base = 60 if label == "FF-real" else 200
                        cv2.imwrite(str(p), np.clip(base + rng.normal(0, 10, (SZ, SZ, 3)), 0, 255).astype(np.uint8))
                        fs.append(str(p))
                    J[label][split][comp][f"{split}{v}"] = {"label": label, "frames": fs}
    jp = tmp_path / "FaceForensics++.json"
    jp.write_text(json.dumps({"FaceForensics++": J}))
    sw = tmp_path / "sensor.pt"
    torch.save(DegradationSensor().backbone_state_dict(), sw)
    monkeypatch.setattr(tf, "build_detector", lambda cfg, dev: DummyDetector(device=dev))
    monkeypatch.setattr("dcfg.detectors.adapter.Detector", DummyDetector)
    return {"json": str(jp), "frames_root": str(root), "sensor_weights": str(sw)}


def cfg_for(data, tmp_path, variant="dcfg", **kw):
    cfg = tf.load_config(REPO_ROOT / "configs" / "train" / "base.yaml")
    cfg.update(run_id=f"t_{variant}", variant=variant, batch_size=4, num_workers=0, val_workers=0,
               max_epochs=2, ckpt_every_steps=5, frozen_check_every=1, log_every=3, wandb_project="dcfg-test",
               run_dir=str(tmp_path / f"run_{variant}"), **data)
    cfg.update(kw)
    return cfg


@pytest.mark.parametrize("variant", tf.VARIANTS)
def test_variant_runs_and_writes_artifacts(data, tmp_path, variant):
    cfg = cfg_for(data, tmp_path, variant)
    res = tf.Trainer(cfg).train()
    rd = tmp_path / f"run_{variant}"
    assert res["epochs_run"] == 2 and not res["truncated"]
    for f in ("last.pt", "best.pt", "metrics.csv", "train_log.csv", "batches.log", "heartbeat", "done.json"):
        assert (rd / f).exists(), f
    assert (rd / "frontend.pt").exists() == (variant != "augft")
    assert (rd / "detector.pt").exists() == (variant in tf.TRAIN_DETECTOR)
    header = (rd / "metrics.csv").read_text().splitlines()[0]
    assert "val_clean" in header and "val_mixed_heavy" in header and "w_mean_ch2" in header
    assert (rd / "w_maps.jsonl").exists()
    assert not any(p.suffix == ".json" for p in (REPO_ROOT / "results" / "raw").glob("t_*"))


def test_resume_mid_epoch_matches_uninterrupted(data, tmp_path):
    full = tf.Trainer(cfg_for(data, tmp_path, "fixed", run_dir=str(tmp_path / "full"), ckpt_every_steps=4)).train()
    assert full["epochs_run"] == 2

    class Crash(Exception):
        pass

    class Crashing(tf.Trainer):
        def after_step(self, step):
            if step == 10:  # spe = 6, ckpts at 4, 6 (epoch end), 8 -> last.pt is mid-epoch 1, batch 2
                raise Crash

    cfg = cfg_for(data, tmp_path, "fixed", run_dir=str(tmp_path / "res"), ckpt_every_steps=4)
    with pytest.raises(Crash):
        Crashing(cfg).train()
    ck = torch.load(tmp_path / "res" / "last.pt", weights_only=False)
    assert (ck["state"]["step"], ck["state"]["step_in_epoch"], ck["state"]["epoch"]) == (8, 2, 1)
    tf.Trainer(cfg).train()

    def parse(p):
        rows = [l.split() for l in (p / "batches.log").read_text().splitlines() if not l.startswith("#")]
        return [(int(e), int(b), int(s), h) for e, b, s, h in rows]

    a, b = parse(tmp_path / "full"), parse(tmp_path / "res")
    by_key: dict = {}
    for e, bi, s, h in b:  # steps 8..9 appear twice (seen before the crash): hashes must agree
        assert by_key.setdefault((e, bi), (s, h)) == (s, h)
    assert [(e, bi, s, h) for (e, bi), (s, h) in sorted(by_key.items())] == a
    assert len({h for *_, h in a}) == len(a)  # no batch repeats in the uninterrupted run
    assert "# resumed" in (tmp_path / "res" / "batches.log").read_text()
    resumed = (tmp_path / "res" / "batches.log").read_text().split("# resumed")[1].splitlines()[1]
    assert resumed.split()[:3] == ["1", "2", "8"]  # continues at epoch 1, batch 2, step 8


def test_frozen_assertion_trips(data, tmp_path):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg"))
    t.check_frozen()
    assert not any(p.requires_grad for p in t.det.parameters()) and not t.det.training
    t.det.requires_grad_(True)
    with pytest.raises(AssertionError):
        t.check_frozen()
    t.det.requires_grad_(False)
    t.det.train()  # frozen => stays eval
    t.check_frozen()
    t.det._frozen = False
    t.det.train()
    with pytest.raises(AssertionError):
        t.check_frozen()


def test_finetune_variants_train_detector_and_group_lrs(data, tmp_path):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_jointft"))
    assert {g["name"]: g["initial_lr"] for g in t.opt.param_groups} == {"gate": 1e-3, "sensor": 1e-4, "detector": 2e-5}
    t.model.train()
    assert t.det.training and all(p.requires_grad for p in t.det.parameters())
    t2 = tf.Trainer(cfg_for(data, tmp_path, "dcfg", run_dir=str(tmp_path / "r2"), freeze_sensor=True))
    assert [g["name"] for g in t2.opt.param_groups] == ["gate"]
    t3 = tf.Trainer(cfg_for(data, tmp_path, "augft", run_dir=str(tmp_path / "r3")))
    assert [g["name"] for g in t3.opt.param_groups] == ["detector"] and t3.frontend is None


def test_normalization_exactly_once(data, tmp_path):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg"))
    x = torch.rand(4, 3, SZ, SZ, device=t.dev)
    seen = {}
    t.det.model.register_forward_hook(lambda m, i, o: seen.update(inp=i[0].detach()))
    DummyDetector.n_normalize = 0
    t.model.eval()
    with torch.no_grad():
        logit, w = t.model(x)
        y, _ = t.frontend(x)
    assert DummyDetector.n_normalize == 1
    torch.testing.assert_close(seen["inp"], (y - t.det.mean) / t.det.std)
    assert logit.shape == (4,) and w.shape == (4, 3, 8, 8)


def test_fixed_w_identical_and_dcfg_identity_at_init(data, tmp_path):
    x = torch.rand(5, 3, SZ, SZ)
    fx = tf.build_frontend(cfg_for(data, tmp_path, "fixed")).eval()
    _, w = fx(x)
    assert torch.equal(w[0], w[3]) and torch.equal(w[1], w[4])
    fe = tf.build_frontend(cfg_for(data, tmp_path, "dcfg")).eval()
    y, w = fe(x)
    assert (y - x).abs().max() < 1e-5 and (w - 1).abs().max() == 0


def test_gradscaler_used(data, tmp_path, monkeypatch):
    calls = {"scale": 0, "step": 0, "unscale": 0}

    class Spy(torch.amp.GradScaler):
        def scale(self, o):
            calls["scale"] += 1
            return super().scale(o)

        def step(self, *a, **k):
            calls["step"] += 1
            return super().step(*a, **k)

        def unscale_(self, *a, **k):
            calls["unscale"] += 1
            return super().unscale_(*a, **k)

    monkeypatch.setattr(torch.amp, "GradScaler", Spy)
    t = tf.Trainer(cfg_for(data, tmp_path, "nosensor", max_steps=4))
    res = t.train()
    assert res["truncated"] and calls == {"scale": 4, "step": 4, "unscale": 4}
    assert (tmp_path / "run_nosensor" / "last.pt").exists()


def test_epoch_order_balanced_and_deterministic():
    labels = np.array([0] * 10 + [1] * 40)
    o = tf.epoch_order(labels, 42, 3)
    assert len(o) == 20 and labels[o].sum() == 10 and len(set(o)) == 20
    assert np.array_equal(o, tf.epoch_order(labels, 42, 3))
    assert not np.array_equal(o, tf.epoch_order(labels, 42, 4))
    f = tf.epoch_order(labels, 42, 0, "full")
    assert len(f) == 80 and labels[f].sum() == 40


def test_evaluate_frontend_writes_json(data, tmp_path, monkeypatch):
    t = tf.Trainer(cfg_for(data, tmp_path, "dcfg_jointft"))
    t.train()
    rd, out = tmp_path / "run_dcfg_jointft", tmp_path / "results"
    rc = ev.main(["--run", "evt", "--detector", "dummy", "--frontend", str(rd / "frontend.pt"),
                  "--detector-ckpt", str(rd / "detector.pt"), "--conditions", "clean,jpeg_q50,c40",
                  "--json", data["json"], "--frames-root", data["frames_root"], "--out-dir", str(out),
                  "--runs-dir", str(tmp_path / "runs"), "--workers", "0", "--n-boot", "20"])
    assert rc == 0
    res = json.loads((out / "evt__clean.json").read_text())
    assert res["variant"] == "dcfg_jointft" and "frontend" in res and 0 <= res["frame_auc"] <= 1
    assert (out / "evt__c40.json").exists() and (out / "evt__jpeg_q50.json").exists()
    assert not (REPO_ROOT / "results" / "raw" / "evt__clean.json").exists()


def test_main_cli_and_config_overrides(data, tmp_path):
    base = REPO_ROOT / "configs" / "train" / "dcfg_xception.yaml"
    cfg = tf.load_config(base, ["max_epochs=1", "freeze_sensor=true"])
    assert cfg["variant"] == "dcfg" and cfg["max_epochs"] == 1 and cfg["freeze_sensor"] is True
    with pytest.raises(KeyError):
        tf.load_config(base, ["nope=1"])
