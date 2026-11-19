"""Tests for the GenD wrapper: preprocessing determinism, input transform, face selection, result schema."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dcfg.data.degradations import apply_condition, load_conditions
from dcfg.gend import crop as gc
from dcfg.paths import REPO_ROOT

CONDS = load_conditions(REPO_ROOT / "configs" / "eval" / "conditions.yaml")


def _img(seed: int = 0, n: int = 256) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, n, 3), dtype=np.uint8)


def test_gend_resize_shape_dtype_and_determinism():
    a = _img()
    r1, r2 = gc.gend_resize(a), gc.gend_resize(a.copy())
    assert r1.shape == (224, 224, 3) and r1.dtype == np.uint8
    assert np.array_equal(r1, r2)


def test_gend_resize_is_pil_bicubic():
    a = _img(1)
    from PIL import Image
    ref = np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC))
    assert np.array_equal(gc.gend_resize(a), ref)


def test_degradation_then_resize_is_deterministic_and_keyed():
    a = _img(2)
    cond = CONDS["noise_s15"]
    x1 = gc.gend_resize(apply_condition(a, cond, "FaceForensics++/x/000.png"))
    x2 = gc.gend_resize(apply_condition(a, cond, "FaceForensics++/x/000.png"))
    x3 = gc.gend_resize(apply_condition(a, cond, "FaceForensics++/x/001.png"))
    assert np.array_equal(x1, x2) and not np.array_equal(x1, x3)


def test_align_face_identity_landmarks_and_determinism():
    pytest.importorskip("cv2")
    img = _img(3, 512)
    # landmarks that are exactly the template mapped into a 512 canvas -> well-posed fit
    lm = (gc.GEND_DST * 256 * 0.9 + 100).astype(np.float32)
    o1, o2 = gc.align_face(img, lm), gc.align_face(img, lm)
    assert o1.shape == (256, 256, 3) and np.array_equal(o1, o2)


def test_largest_face_picks_max_area():
    xyxy = np.array([[0, 0, 10, 10, 0.9], [0, 0, 50, 40, 0.6]], dtype=np.float32)
    kps = np.stack([np.zeros((5, 2)), np.ones((5, 2))])
    assert np.all(gc.largest_face(xyxy, kps) == 1)


def test_provenance_schema(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    from dcfg.gend import evaluate_gend as eg
    w = tmp_path / "hf"
    w.mkdir()
    (w / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(eg, "GEND_DIR", tmp_path)
    monkeypatch.setattr(eg, "gend_commit", lambda: "abc")
    p = eg.provenance("gend", True)
    assert p["pilot"] is False and p["lumen2"] is True and p["gend_commit"] == "abc"
    assert {"model_id", "hf_revision", "weights_sha256", "preprocessing", "degradation_placement"} <= set(p)
    json.dumps(p)


def test_evaluate_condition_with_post_transform_schema(tmp_path):
    """The standard result JSON/npz schema is produced with a post_transform (tiny fake frames)."""
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from dcfg.data.ffpp import DFBJsonSource
    from dcfg.eval.evaluate import evaluate_condition
    root = tmp_path / "fr"
    J: dict = {}
    rng = np.random.default_rng(0)
    for label in ("FF-real", "FF-DF"):
        J[label] = {s: {c: {} for c in ("c23", "c40")} for s in ("train", "val", "test")}
        for v in range(3):
            fs = []
            for k in range(2):
                p = root / label / "c23" / f"v{v}" / f"{k:03d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), np.clip((200 if label == "FF-DF" else 60) + rng.normal(0, 8, (32, 32, 3)), 0, 255).astype(np.uint8))
                fs.append(str(p))
            J[label]["test"]["c23"][f"v{v}"] = {"label": label, "frames": fs}
    jp = tmp_path / "j.json"
    jp.write_text(json.dumps({"FaceForensics++": J}))
    src = DFBJsonSource(jp, root)

    def model(x):
        assert x.shape[-2:] == (224, 224)  # post_transform (GenD resize) was applied
        return (x.mean((1, 2, 3)) - 0.5) * 10

    out = tmp_path / "o"
    p = evaluate_condition(run_id="gend_t", detector="gend_clip_l14", cond=CONDS["clean"], split="test", source=src,
                           model=model, out_dir=out, n_boot=20,
                           extra={"pilot": False, "lumen2": True}, post_transform=gc.gend_resize)
    d = json.loads(p.read_text())
    assert d["pilot"] is False and d["lumen2"] is True and d["n_frames"] == 12 and d["frame_auc"] == 1.0
    assert (out / d["scores_file"]).exists() 


def _write_video(path, n=40, size=(160, 120)):
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    rng = np.random.default_rng(0)
    for _ in range(n):
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()
    return path


def test_recrop_dfd_mode_linspace_keys_and_marker(monkeypatch, tmp_path):
    """DFD mode: DFB's linspace(0, N-1, 32) frame rule, DFB-style keys, no-face frames skipped, resumable marker."""
    pytest.importorskip("cv2")
    from dcfg.gend import recrop as rc
    vid = _write_video(tmp_path / "raw" / "manipulated_sequences" / "DeepFakeDetection" / "c23" / "videos" / "01_02__a.mp4")
    lm = (gc.GEND_DST * 100 + 20).astype(np.float32)
    calls = {"n": 0}

    def fake_detect(det, fr):
        calls["n"] += 1
        return (None, None) if calls["n"] == 3 else (lm, 0.5)

    monkeypatch.setattr(rc, "detect_landmarks", fake_detect)
    monkeypatch.setitem(rc._G, "det", None)
    out = tmp_path / "out"
    k, stats = rc.work(("FF-DFD/01_02__a", vid, None, str(out), None))
    n = int(__import__("cv2").VideoCapture(str(vid)).get(7))
    idx = sorted(set(np.linspace(0, n - 1, 32, endpoint=True, dtype=int).tolist()))
    assert [s[0] for s in stats] == [f"manipulated_sequences/DeepFakeDetection/c23/frames/01_02__a/{i:03d}.png" for i in idx]
    assert sum(s[1] == "no_face" for s in stats) == 1
    saved = sorted((out / "manipulated_sequences/DeepFakeDetection/c23/frames/01_02__a").glob("*.png"))
    assert len(saved) == len(idx) - 1
    assert rc.work(("FF-DFD/01_02__a", vid, None, str(out), None))[1] == []  # marker -> skipped
