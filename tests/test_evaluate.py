"""Tests for the FF++ frame source and the evaluate skeleton (tiny fake JSON + PNGs)."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from dcfg.data.degradations import load_conditions
from dcfg.data.ffpp import DFBJsonSource
from dcfg.eval import evaluate as ev
from dcfg.paths import REPO_ROOT

CONDS = load_conditions(REPO_ROOT / "configs" / "eval" / "conditions.yaml")


@pytest.fixture()
def fake(tmp_path):
    root = tmp_path / "frames"
    J: dict = {}
    rng = np.random.default_rng(0)
    for label in ("FF-real", "FF-DF"):
        J[label] = {s: {c: {} for c in ("c23", "c40")} for s in ("train", "val", "test")}
        for split in ("train", "val", "test"):
            for comp in ("c23", "c40"):
                for v in range(4):
                    vid = f"{split}{v}"
                    fs = []
                    for k in range(3):
                        p = root / label / comp / vid / f"{k:03d}.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        base = 200 if label == "FF-DF" else 60  # fakes bright: trivially separable
                        img = np.clip(base + rng.normal(0, 10, (32, 32, 3)), 0, 255).astype(np.uint8)
                        cv2.imwrite(str(p), img)
                        fs.append(str(p))
                    J[label][split][comp][vid] = {"label": label, "frames": fs}
    jp = tmp_path / "FaceForensics++.json"
    jp.write_text(json.dumps({"FaceForensics++": J}))
    return DFBJsonSource(jp, root)


def _model(x01):  # brightness "detector": logit grows with mean intensity
    return (x01.mean((1, 2, 3)) - 0.5) * 20


def test_source_schema(fake):
    recs = fake.records("test", "c23")
    assert len(recs) == 2 * 4 * 3
    assert {r.label for r in recs} == {0, 1}
    assert all(r.key.startswith(("FF-real/c23/", "FF-DF/c23/")) for r in recs)
    assert len({r.video for r in recs}) == 8
    assert fake.load(recs[0]).shape == (32, 32, 3) and fake.load(recs[0]).dtype == np.uint8
    assert len(fake.records("test", "c40")) == 24 and fake.records("test", "c40")[0].key.split("/")[1] == "c40"


def test_evaluate_writes_json_and_resumes(fake, tmp_path):
    out = tmp_path / "out"
    kw = dict(run_id="r", detector="dummy", split="test", source=fake, model=_model, out_dir=out,
              workers=0, n_boot=50)
    p = ev.evaluate_condition(cond=CONDS["jpeg_q50"], **kw)
    res = json.loads(p.read_text())
    assert res["frame_auc"] == pytest.approx(1.0) and res["video_auc"] == pytest.approx(1.0)
    assert res["n_frames"] == 24 and res["n_videos"] == 8 and res["compression"] == "c23"
    for k in ("git_commit", "dfb_commit", "timestamp", "frame_auc_ci", "video_auc_ci", "scores_file"):
        assert k in res
    z = np.load(out / res["scores_file"])
    assert z["frame_prob"].shape == (24,) and z["video_prob"].shape == (8,)
    assert ev.evaluate_condition(cond=CONDS["jpeg_q50"], **kw) is None  # resumable
    assert ev.evaluate_condition(cond=CONDS["jpeg_q50"], force=True, **kw) == p


def test_c40_uses_c40_frames_without_degradation(fake, tmp_path):
    p = ev.evaluate_condition(run_id="r", detector="d", cond=CONDS["c40"], split="test", source=fake,
                              model=_model, out_dir=tmp_path / "o", workers=0, n_boot=20)
    res = json.loads(p.read_text())
    assert res["compression"] == "c40"
    z = np.load(tmp_path / "o" / res["scores_file"])
    assert all("/c40/" in k for k in z["frame_key"])
    ds = ev.ConditionDataset(fake, fake.records("test", "c40"), CONDS["c40"])
    assert np.array_equal(ds[0].permute(1, 2, 0).numpy(), fake.load(fake.records("test", "c40")[0]))


def test_degradation_applied_deterministically(fake):
    recs = fake.records("test", "c23")
    a = ev.ConditionDataset(fake, recs, CONDS["noise_s10"])[0]
    b = ev.ConditionDataset(fake, recs, CONDS["noise_s10"])[0]
    c = ev.ConditionDataset(fake, recs, CONDS["clean"])[0]
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_val_never_written_to_results_raw(fake):
    with pytest.raises(ValueError):
        ev.evaluate_condition(run_id="r", detector="d", cond=CONDS["clean"], split="val", source=fake,
                              model=_model, out_dir=ev.RAW_DIR, workers=0)
    with pytest.raises(SystemExit):
        ev.main(["--run", "r", "--detector", "xception", "--split", "val"])
    with pytest.raises(ValueError):
        ev.main(["--run", "r", "--detector", "xception", "--split", "val", "--out-dir", str(ev.RAW_DIR)])


def test_resolve_conditions():
    assert len(ev.resolve_conditions("all", CONDS)) == len(CONDS)
    assert [c.name for c in ev.resolve_conditions("clean,c40", CONDS)] == ["clean", "c40"]
    with pytest.raises(SystemExit):
        ev.resolve_conditions("nope", CONDS)
