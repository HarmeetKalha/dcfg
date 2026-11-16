"""Tests for analysis modules and ops helpers on tiny synthetic DFB-format data (no real data touched)."""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pytest
import torch

from dcfg.analysis import freq_study, overhead, preprocess_qa, sensor_embedding
from dcfg.data.ffpp import DFBJsonSource
from dcfg.models.sensor import DegradationSensor
from dcfg.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "ops" / "jobs"))
import make_eval_queue  # noqa: E402
import verify_ffpp_downloads as vfd  # noqa: E402

SEQ = preprocess_qa.SEQ_LABEL


@pytest.fixture()
def fake(tmp_path):
    """Frames root + DFB JSON + stats tsv + split jsons; one video per (split, type) is skipped (zero face)."""
    root = tmp_path / "frames"
    stats_dir = root / "FaceForensics++"
    stats_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    J: dict = {}
    splits = {s: [[f"{i:03d}", f"{i + 50:03d}"] for i in range(base, base + 3)]
              for s, base in (("train", 0), ("val", 10), ("test", 20))}
    for s, pairs in splits.items():
        (stats_dir / f"{s}.json").write_text(json.dumps(pairs))
    tsv = {"c23": [], "c40": []}
    for seq, label in SEQ.items():
        J[label] = {s: {"c23": {}, "c40": {}} for s in splits}
        for s, pairs in splits.items():
            for comp in ("c23", "c40"):
                if comp == "c40" and s != "test":
                    continue
                for k, (a, b) in enumerate(pairs):
                    stem = a if label == "FF-real" else f"{a}_{b}"
                    if k == 2:  # zero-face video
                        tsv[comp].append(f"{seq}\t{comp}\t{stem}\t100\t0\t1.0")
                        continue
                    fs = []
                    for f in range(4):
                        p = root / seq / comp / "frames" / stem / f"{f:03d}.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
                        img[:, :32] = 128 + 10 * rng.standard_normal((64, 32, 3)).clip(-5, 5)
                        cv2.imwrite(str(p), img)
                        fs.append(str(p))
                    J[label][s][comp][stem] = {"label": label, "frames": fs}
                    tsv[comp].append(f"{seq}\t{comp}\t{stem}\t100\t4\t1.0")
    jp = tmp_path / "FaceForensics++.json"
    jp.write_text(json.dumps({"FaceForensics++": J}))
    for comp, rows in tsv.items():
        (stats_dir / f"preprocess_stats_{comp}.tsv").write_text("\n".join(rows) + "\n")
    return DFBJsonSource(jp, root), stats_dir, tmp_path


def test_preprocess_counts_and_grid(fake):
    src, stats_dir, tmp = fake
    rows = preprocess_qa.compute_counts(src, stats_dir)
    r = next(x for x in rows if (x["split"], x["type"], x["comp"]) == ("train", "FF-DF", "c23"))
    assert r["videos_processed"] == 3 and r["videos_in_json"] == 2
    assert r["frames_mean"] == 4 and r["frames_min"] == 4
    assert r["n_zero_face"] == 1 and r["zero_face_videos"] == ["002_052"]
    att = len(set(np.linspace(0, 99, 32, dtype=int).tolist()))
    assert r["failed_detection_rate"] == pytest.approx(1 - 8 / (3 * att))
    assert not any(x["split"] != "test" for x in rows if x["comp"] == "c40")
    assert "002_052" in preprocess_qa.to_markdown(rows)
    out = tmp / "qa"
    assert preprocess_qa.main(["--json", str(src.json_path), "--frames-root", str(src.frames_root),
                               "--stats-dir", str(stats_dir), "--out-dir", str(out),
                               "--fig", str(tmp / "grid.png")]) == 0
    assert (out / "preprocess_counts.json").exists() and (out / "preprocess_counts.md").exists()
    assert cv2.imread(str(tmp / "grid.png")) is not None


def test_freq_helpers():
    e = np.ones((3, 8, 8))
    assert np.allclose(freq_study.relative_change(2 * e, e), 1.0)
    rc = freq_study.radial_curve(0.5 * e, e)
    assert rc.shape == (3, 10) and np.allclose(rc[:, :], -0.5)
    v = np.array([[1.0, 0], [2.0, 0], [0, 1.0]])
    d = freq_study.cosine_distance_matrix(v)
    assert d[0, 1] == pytest.approx(0, abs=1e-9) and d[0, 2] == pytest.approx(1.0)
    assert freq_study.radial_bins().max() == 9


def test_freq_study_end_to_end(fake):
    src, _, tmp = fake
    res = freq_study.run_study(src, n=12, seed=1, device="cpu")
    names = list(res["conditions"])
    assert "clean" not in names and "c40" not in names and len(names) == 21
    i = names.index("blur_s3")
    # blur removes high-frequency energy, noise adds it
    assert res["rel_change"][i, 0, 7, 7] < 0
    assert res["rel_change"][names.index("noise_s20"), 0, 7, 7] > 0
    assert res["fingerprint_dist"].shape == (21, 21) and np.allclose(np.diag(res["fingerprint_dist"]), 0, atol=1e-6)
    res2 = freq_study.run_study(src, n=12, seed=1, device="cpu")
    assert np.array_equal(res["rel_change"], res2["rel_change"])
    out = tmp / "fs.npz"
    assert freq_study.main(["--n", "8", "--out", str(out), "--fig-dir", str(tmp / "figs"),
                            "--json", str(src.json_path), "--frames-root", str(src.frames_root)]) == 0
    z = np.load(out)
    assert z["rel_change"].shape == (21, 3, 8, 8) and z["radial_rel_change"].shape == (21, 3, 10)
    assert {p.name for p in (tmp / "figs").glob("freq_*.png")} == {
        "freq_heatmaps.png", "freq_radial.png", "freq_fingerprint.png"}


def test_sensor_embedding(fake):
    src, _, tmp = fake
    w = tmp / "sensor.pt"
    torch.manual_seed(0)
    torch.save(DegradationSensor().backbone_state_dict(), w)
    out, fig = tmp / "se.json", tmp / "se.png"
    assert sensor_embedding.main(["--n", "10", "--weights", str(w), "--out", str(out), "--fig", str(fig),
                                  "--json", str(src.json_path), "--frames-root", str(src.frames_root)]) == 0
    r = json.loads(out.read_text())
    assert r["chance"] == pytest.approx(1 / 6) and 0 <= r["knn_accuracy"] <= 1 and -1 <= r["silhouette"] <= 1
    assert fig.exists() and r["n_images"] == 10


def test_separation_metrics_separable():
    rng = np.random.default_rng(0)
    cond = np.repeat(np.arange(3), 20)
    emb = rng.standard_normal((60, 8)) * 0.1 + 5 * np.eye(3, 8)[cond]
    m = sensor_embedding.separation_metrics(emb, cond, np.tile(np.arange(20), 3))
    assert m["knn_accuracy"] > 0.95 and m["silhouette"] > 0.5


def test_overhead_cpu():
    r = overhead.run("cpu", variants=("dcfg", "fixed", "nosensor"), batches=(1, 2), warmup=1, iters=2,
                     res=64, with_detectors=False)
    fe = r["frontends"]
    assert fe["fixed"]["params"] == 192 and fe["dcfg"]["params"] > fe["nosensor"]["params"] > 192
    assert fe["dcfg"]["batch2"]["ms_per_image"] > 0 and fe["dcfg"]["flops_per_image"] > 0
    json.dumps(r)


def test_verify_downloads(tmp_path):
    raw = tmp_path / "raw"
    for t in vfd.TYPES:
        for c in vfd.COMPS:
            v = raw / t / c / "videos"
            v.mkdir(parents=True)
            for i in range(3):
                (v / f"{i:03d}.mp4").write_bytes(b"x" * 10)
    rep = vfd.verify(raw, expected=3)
    assert rep["all_ok"]
    d = raw / vfd.TYPES[0] / "c23"
    (d / "videos" / "tmpabc").write_bytes(b"x")
    (d / "videos" / "000.mp4").write_bytes(b"")
    (raw / vfd.TYPES[1] / "c40" / "videos" / "002.mp4").unlink()
    rep = vfd.verify(raw, expected=3)
    f = rep["folders"]
    assert not rep["all_ok"]
    assert f[f"{vfd.TYPES[0]}/c23"]["zero_byte"] == ["000.mp4"] and len(f[f"{vfd.TYPES[0]}/c23"]["stray_tmp"]) == 1
    assert f[f"{vfd.TYPES[1]}/c40"]["n_mp4"] == 2
    deep = vfd.verify(raw, deep=True, expected=3)  # fake mp4 bytes are unreadable
    assert deep["folders"][f"{vfd.TYPES[2]}/c23"]["unreadable"]
    assert "FAIL" in vfd.format_table(rep)
    out = tmp_path / "o.json"
    assert vfd.main(["--raw", str(raw), "--expected", "3", "--out", str(out)]) == 1 and out.exists()


def test_queue_lines():
    lines = make_eval_queue.queue_lines()
    assert len(lines) == 7 and all(" | " in ln for ln in lines)
    assert lines[0].startswith("base_xception | .venv/bin/python -m dcfg.eval.evaluate --run base_xception")
    assert all(ln.endswith("#hb=30") for ln in lines[:4])
