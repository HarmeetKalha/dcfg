"""Tests for dcfg.lumen2.analysis (metric definitions), dfd_append and the report helpers (no GPU, synthetic curves)."""
from __future__ import annotations

import json

import numpy as np

from dcfg.lumen2 import analysis as A
from dcfg.lumen2 import dfd_append as D


def curves(fn):
    return {(m, c): fn(m, c) for m in ("frame", "video") for c in A.ALL_CONDS}


def test_condition_lists():
    assert len(A.SYNTH) == 17 and len(A.HELD) == 4 and len(A.DEGRADED) == 22
    assert A.HEAVY == ["jpeg_q10", "blur_s4", "noise_s20", "resize_0.25", "mixed_heavy"]  # amended list: no webp/motion
    assert set(A.HEAVY) <= set(A.SYNTH)


def test_worst_case_is_min_per_resample_and_heavy_mean():
    rng = np.random.default_rng(0)
    base = {c: 0.9 - 0.01 * rng.random(5) for c in A.ALL_CONDS}
    cur = curves(lambda m, c: base[c])
    cur[("frame", "blur_s4")] = np.array([0.5, 0.95, 0.4, 0.9, 0.9])  # the worst condition changes between resamples
    agg = A.aggregate_curves(cur)
    want = np.min([cur[("frame", c)] for c in A.DEGRADED], axis=0)
    assert np.array_equal(agg["worst_case"], want) and agg["worst_case"][0] == 0.5
    assert np.allclose(agg["heavy_mean"], np.mean([cur[("frame", c)] for c in A.HEAVY], axis=0))
    assert "webp_q20" not in A.HEAVY and "clean" not in A.DEGRADED


def test_outcome_rules():
    assert A.outcome("synth_mean", 0.01, 0.03) == "win" and A.outcome("synth_mean", -0.01, 0.03) == "tie"
    assert A.outcome("synth_mean", -0.03, -0.01) == "loss"
    assert A.outcome("ece", -0.03, -0.01) == "win" and A.outcome("ece", 0.01, 0.03) == "loss"  # lower is better
    assert A.outcome("clean", -0.02, 0.0) == "ok" and A.outcome("clean", -0.03, -0.01) == "VIOLATED"
    assert A.combine_selective("win", "tie") == "win" and A.combine_selective("loss", "loss") == "loss"
    assert A.combine_selective("loss", "tie") == "tie"


def test_pct_and_resample_matrix():
    W = A.resample_matrix(10, 50)
    assert W.shape == (51, 10) and (W[0] == 1).all() and (W[1:].sum(1) == 10).all()
    pt, lo, hi = A.pct(np.arange(101.0))
    assert pt == 0.0 and lo < hi


def test_frame_bootstrap_matches_metric_on_resample():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    vid = np.repeat(np.arange(20), 10)
    W = A.resample_matrix(20, 3)
    out = A.frame_bootstrap({"ece": A.FRAME_FNS["ece"]}, p, p, y, vid, W)["ece"][0]
    idx = np.repeat(np.arange(200), W[1][vid].astype(int))
    assert np.isclose(out[1], A.FRAME_FNS["ece"](p[idx], y[idx])) and np.isclose(out[0], A.FRAME_FNS["ece"](p, y))


def test_dfd_append_idempotent(tmp_path):
    d = tmp_path / "dfd"
    d.mkdir()
    for c in D.CONDS:
        (d / f"m__{c}.json").write_text(json.dumps({"frame_auc": 0.9, "frame_auc_ci": [0.8, 0.95], "video_auc": 0.92, "video_auc_ci": [0.8, 0.96]}))
    rep = tmp_path / "r.md"
    rep.write_text("# r\n")
    sec = D.dfd_section([("m", "M"), ("missing", "X")], d)
    D.apply(rep, sec)
    D.apply(rep, sec)
    t = rep.read_text()
    assert t.count(D.BEGIN) == 1 and "0.9000 [0.8000, 0.9500]" in t and "n/a" in t
