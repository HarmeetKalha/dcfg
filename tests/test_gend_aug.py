"""Tests for dcfg.gend.train_gend_aug (tiny fake CLIP-like model and fake frame source; no transformers, no real data)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dcfg.data.ffpp import FrameRecord  # noqa: E402
from dcfg.gend import train_gend_aug as T  # noqa: E402
from dcfg.paths import REPO_ROOT  # noqa: E402


class TinyGend(nn.Module):
    """Names mimic GendCLIP: LayerNorms named like CLIP's, frozen projection, ``linear`` head, ``features`` method."""

    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Module()
        self.vision.pre_layrnorm = nn.LayerNorm(3)
        self.vision.layer_norm1 = nn.LayerNorm(8)
        self.vision.proj = nn.Linear(3, 8)
        self.vision.post_layernorm = nn.LayerNorm(8)
        self.linear = nn.Linear(8, 2)

    def features(self, x01):
        x = x01.mean((2, 3))
        x = self.vision.proj(self.vision.pre_layrnorm(x))
        z = torch.nn.functional.normalize(self.vision.post_layernorm(self.vision.layer_norm1(x)), dim=1)
        return self.linear(z), z

    def forward(self, x01):
        l, _ = self.features(x01)
        return l[:, 1] - l[:, 0]


class FakeSource:
    def __init__(self, nvid: int = 8, nfr: int = 12) -> None:
        self.recs = {}
        for split in ("train", "val"):
            r = []
            for v in range(nvid):
                lab = v % 2
                name = ("FF-real" if lab == 0 else "FF-DF") + f"/{v:03d}"
                for f in range(nfr):
                    r.append(FrameRecord(path="", key=f"{split}/{name}/{f:03d}.png", video=name, label=lab))
            self.recs[split] = r

    def records(self, split, comp):
        return self.recs[split]

    def load(self, rec):
        rng = np.random.default_rng(abs(hash(rec.key)) % 2**32)
        base = 60 + 120 * rec.label
        return np.clip(rng.normal(base, 20, (64, 64, 3)), 0, 255).astype(np.uint8)


def cfg(**kw):
    c = T.load_cfg("configs/train/gend_aug.yaml")
    c.update(batch_size=4, accum=2, num_workers=0, val_workers=0, max_epochs=2, val_frames_per_video=4, ckpt_every_steps=3,
             log_every=2, lr=1e-2, min_lr=1e-3, patience=6)
    c.update(kw)
    return c


def mk(tmp_path, **kw):
    return T.GendAugTrainer(cfg(**kw), tmp_path, model_fn=TinyGend, source=FakeSource(), expected_trainable=None)


def test_losses_match_official_unifalign():
    spec_path = REPO_ROOT / "third_party" / "GenD" / "src" / "losses" / "unifalign.py"
    if not spec_path.exists():
        pytest.skip("third_party/GenD not present")
    spec = importlib.util.spec_from_file_location("ua", spec_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    y = torch.randint(0, 2, (16,))
    assert torch.allclose(T.uniformity(z), m.uniformity(z))
    assert torch.allclose(T.alignment(z, y), m.alignment(z, y))


def test_gend_loss_weights_and_degenerate_batches():
    z = torch.nn.functional.normalize(torch.randn(6, 4), dim=1)
    y = torch.tensor([0, 0, 0, 0, 0, 0])
    logits = torch.randn(6, 2)
    tot, t = T.gend_loss(logits, z, y, 1.0, 0.5, 0.1)
    assert torch.isclose(tot, t["ce"] + 0.5 * t["uniformity"] + 0.1 * t["alignment"])
    assert T.alignment(z[:1], y[:1]) == 0  # single sample -> 0


def test_trainable_selection_count_and_sizes():
    m = TinyGend()
    tr = T.set_trainable(m)
    names = {n for n, _ in tr}
    assert names == {"vision.pre_layrnorm.weight", "vision.pre_layrnorm.bias", "vision.layer_norm1.weight",
                     "vision.layer_norm1.bias", "vision.post_layernorm.weight", "vision.post_layernorm.bias",
                     "linear.weight", "linear.bias"}
    assert not m.vision.proj.weight.requires_grad
    sd = T.trainable_state(m)
    m2 = TinyGend()
    T.load_trainable(m2, sd)
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(sorted(m.state_dict().items()), sorted(m2.state_dict().items()))
               if "proj" not in _)  # noqa: B007
    with pytest.raises(KeyError):
        T.load_trainable(m2, {"linear.weight": sd["linear.weight"]})


def test_lr_schedule_warmup_cosine_to_min():
    f = lambda s: T.lr_factor(s, 10, 1, 4, 3e-4, 1e-5)  # noqa: E731
    assert f(0) == pytest.approx(1e-5 / 3e-4)
    assert f(10) == pytest.approx(1.0)
    assert f(40) == pytest.approx(1e-5 / 3e-4)
    assert f(25) < f(15) < f(10)


def test_val_subsample_deterministic_8_of_32():
    recs = [FrameRecord("", f"v/a/{i:03d}.png", "a", 0) for i in range(32)] + \
           [FrameRecord("", f"v/b/{i:03d}.png", "b", 1) for i in range(31)]
    a, b = T.val_subsample(recs, 8), T.val_subsample(list(reversed(recs)), 8)
    assert [r.key for r in a] == [r.key for r in b]
    assert sum(r.video == "a" for r in a) == 8 and sum(r.video == "b" for r in a) == 8
    assert [r.key for r in a][:2] == ["v/a/000.png", "v/a/004.png"]


def test_train_end_to_end_stop_reason_and_files(tmp_path):
    res = mk(tmp_path).train()
    assert res["stop_reason"] == "epoch_cap" and res["epochs_run"] == 2 and res["epoch_base"] == 1
    assert res["best_epoch"] in (1, 2) and len(res["val_curve"]) == 2
    assert res["still_improving_at_cap"] == (res["best_epoch"] >= 1)  # cap=2: best within last 2 epochs
    assert res["trainable_file_mb"] <= res["full_model_file_mb"]
    assert (tmp_path / "best_trainable.pt").exists() and (tmp_path / "best_full.pt").exists()
    ck = torch.load(tmp_path / "best_trainable.pt", weights_only=False)
    assert set(ck["trainable"]) == set(T.trainable_state(TinyGend()))


def test_patience_stop(tmp_path):
    t = mk(tmp_path, max_epochs=6, patience=1)
    t.validate = lambda: {"aucs": {"clean": 0.5}, "mean_auc": 0.5, "n_frames": {"clean": 1}}  # never improves after epoch 1
    res = t.train()
    assert res["stop_reason"] == "patience" and res["best_epoch"] == 1 and res["stop_epoch"] == 2
    assert not res["still_improving_at_cap"]


def test_resume_reproduces_batches_and_params(tmp_path):
    a = mk(tmp_path / "a", max_steps=5, val_on_truncate=False)
    a.train()
    (tmp_path / "a" / "done.json").unlink()
    a2 = T.GendAugTrainer({**a.cfg, "max_steps": 9}, tmp_path / "a", model_fn=TinyGend, source=FakeSource(), expected_trainable=None)
    a2.train()
    s = mk(tmp_path / "s", max_steps=9)
    s.train()
    assert T._file_hashes(tmp_path / "a" / "batches.log") == T._file_hashes(tmp_path / "s" / "batches.log")
    pa, ps = T.trainable_state(a2.model), T.trainable_state(s.model)
    assert all(torch.allclose(pa[k], ps[k], atol=1e-6) for k in pa)


def test_nonfinite_step_is_skipped_and_counted(tmp_path):
    t = mk(tmp_path, max_steps=3)
    orig = T.gend_loss
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        tot, terms = orig(*a, **k)
        return (tot * float("nan") if calls["n"] == 1 else tot), terms

    T.gend_loss = bad
    try:
        t.train()
    finally:
        T.gend_loss = orig
    assert t.st["nonfinite_steps"] == 1 and t.st["step"] == 3
    assert all(torch.isfinite(p).all() for p in t.params)


def test_config_is_the_fixed_schedule():
    c = T.load_cfg("configs/train/gend_aug.yaml")
    assert (c["max_epochs"], c["patience"], c["seed"], c["val_frames_per_video"]) == (18, 6, 42, 8)
    assert c["batch_size"] * c["accum"] == 128 and c["lr"] == 3e-4 and c["weight_decay"] == 0.0
    assert (c["w_ce"], c["w_uniformity"], c["w_alignment"]) == (1.0, 0.5, 0.1) and c["epoch_mode"] == "full"


def test_report_on_public_gend_as_fake_aug(tmp_path):
    """Pipeline check: the public GenD results copied as 'gend_aug' must give exactly zero paired differences."""
    from dcfg.gend import gend_aug_report as R

    src = REPO_ROOT / "results" / "lumen2" / "gend_public"
    if not (src / "gend_public__clean.json").exists() or not (REPO_ROOT / "results" / "raw" / "base_xception__clean.json").exists():
        pytest.skip("result files not present")
    (tmp_path / "scores").mkdir()
    for j in src.glob("gend_public__*.json"):
        d = json.loads(j.read_text())
        d["run_id"] = "gend_aug"
        d["scores_file"] = d["scores_file"].replace("gend_public", "gend_aug")
        (tmp_path / j.name.replace("gend_public", "gend_aug")).write_text(json.dumps(d))
        (tmp_path / "scores" / Path(d["scores_file"]).name).write_bytes((src / "scores" / Path(d["scores_file"]).name.replace("gend_aug", "gend_public")).read_bytes())
    m = R.AugModels(src, tmp_path, 5)
    assert m.diff(R.A, R.G, "synth_mean")[0] == 0 and m.diff(R.A, R.G, "c40", "video")[2] == 0
    cal = R.calibration(m, 3)
    assert cal["ece"]["delta"] == 0 and cal["sel80"]["ci_low"] == 0
    md = R.build(m, cal, {"still_improving_at_cap": True, "max_epochs": 18, "best_epoch": 18, "stop_reason": "epoch_cap", "stop_epoch": 18,
                          "best_val_mean_auc": 0.9, "val_conditions": ["clean"], "train_s": 3600.0, "val_s": 60.0, "train_wall_time_s": 3660.0,
                          "steps": 10, "steps_per_epoch": 5, "nonfinite_steps": 0, "trainable_params": 104450, "trainable_file_mb": 0.4,
                          "full_model_file_mb": 1160.0, "val_curve": [{"epoch": 1, "val_mean_auc": 0.9}]}, 3)
    assert "FLAG" in md and "GenD-aug" in md and "did NOT drop" in md
