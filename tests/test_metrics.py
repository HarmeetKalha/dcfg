"""Tests for dcfg.eval.metrics on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dcfg.eval.metrics import bootstrap_ci, frame_auc, paired_bootstrap, video_auc


def _data(n_vid=60, fpv=8, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    vl = rng.integers(0, 2, n_vid)
    labels, groups, scores = [], [], []
    for v in range(n_vid):
        base = rng.normal(sep * vl[v], 1.0)
        for _ in range(fpv):
            labels.append(vl[v]); groups.append(f"v{v}"); scores.append(base + rng.normal(0, 1.0))
    return np.array(labels), np.array(scores), np.array(groups)


def test_frame_auc_matches_sklearn_with_ties():
    l, s, _ = _data()
    s = np.round(s, 1)  # force ties
    assert frame_auc(l, s) == pytest.approx(roc_auc_score(l, s), abs=1e-12)


def test_video_auc_matches_sklearn_mean_aggregation():
    l, s, g = _data()
    ids = np.unique(g)
    vl = np.array([l[g == i][0] for i in ids]); vs = np.array([s[g == i].mean() for i in ids])
    assert video_auc(l, s, g) == pytest.approx(roc_auc_score(vl, vs), abs=1e-12)


def test_mixed_labels_in_video_rejected():
    with pytest.raises(ValueError):
        video_auc([0, 1], [0.1, 0.9], ["a", "a"])


@pytest.mark.parametrize("metric", ["frame", "video"])
def test_bootstrap_ci_contains_point_and_deterministic(metric):
    l, s, g = _data()
    a = bootstrap_ci(metric, l, s, g, n=300, seed=1)
    b = bootstrap_ci(metric, l, s, g, n=300, seed=1)
    assert a == b
    assert a.lo < a.point < a.hi and 0 <= a.lo and a.hi <= 1 and a.n_valid == 300
    assert bootstrap_ci(metric, l, s, g, n=300, seed=2) != a


def test_bootstrap_resamples_videos_not_frames():
    # frame-level resampling would give a much narrower interval than video-level
    l, s, g = _data(n_vid=30, fpv=40, seed=3)
    ci = bootstrap_ci("frame", l, s, g, n=400, seed=0)
    rng = np.random.default_rng(0)
    narrow = [frame_auc(l[i], s[i]) for i in (rng.integers(0, len(l), len(l)) for _ in range(400))]
    assert (ci.hi - ci.lo) > 1.3 * (np.percentile(narrow, 97.5) - np.percentile(narrow, 2.5))


def test_paired_bootstrap_detects_real_difference():
    l, s, g = _data(sep=1.5)
    rng = np.random.default_rng(5)
    worse = s + rng.normal(0, 3.0, len(s))
    r = paired_bootstrap(s, worse, l, g, metric="frame", n=500, seed=0)
    assert r.delta > 0.05 and r.lo > 0 and r.p_value < 0.01
    assert r.delta == pytest.approx(frame_auc(l, s) - frame_auc(l, worse))


def test_paired_bootstrap_identical_not_significant():
    l, s, g = _data()
    noise = np.random.default_rng(1).normal(0, 1e-9, len(s))
    r = paired_bootstrap(s, s + noise, l, g, metric="video", n=300, seed=0)
    assert abs(r.delta) < 1e-6 and r.p_value > 0.05 and r.lo <= 0 <= r.hi
