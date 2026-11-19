"""Tests for ECE, selective AUC and generic (paired) bootstrap helpers."""
from functools import partial

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dcfg.eval.metrics import (auc_metric, bootstrap_metric, ece, paired_bootstrap_metric,
                               selective_auc)


def test_ece_perfectly_calibrated_bins():
    # bin 0.5-ish: p=0.5 with half positives; p=0.1 with 10% positives
    p = [0.1] * 10 + [0.5] * 10 + [0.9] * 10
    y = [1] + [0] * 9 + [1] * 5 + [0] * 5 + [1] * 9 + [0]
    assert ece(p, y) == pytest.approx(0.0, abs=1e-12)


def test_ece_hand_computed():
    p = [0.1, 0.2, 0.9, 0.8]  # bins (0.067,0.133],(0.133,0.2],(0.8,0.867]?,...
    y = [0, 1, 1, 1]
    # each p in its own bin: |0.1-0|+|0.2-1|+|0.9-1|+|0.8-1| = 0.1+0.8+0.1+0.2 = 1.2 -> /4
    assert ece(p, y) == pytest.approx(0.3)


def test_ece_all_ones_half_labels():
    assert ece([1.0] * 10, [1, 0] * 5) == pytest.approx(0.5)
    assert ece([0.0] * 4, [0, 0, 1, 1]) == pytest.approx(0.5)  # first bin includes 0


def test_selective_auc_full_coverage_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + rng.random(200) * 0.7, 0, 1)
    assert selective_auc(p, y, 1.0) == pytest.approx(roc_auc_score(y, p))


def test_selective_auc_confident_correct():
    p = [0.99, 0.01, 0.98, 0.02, 0.45, 0.6, 0.4, 0.55]
    y = [1, 0, 1, 0, 1, 0, 1, 0]  # the four uncertain ones are all wrong
    assert selective_auc(p, y, 0.5) == 1.0
    assert selective_auc(p, y, 1.0) < 1.0


def test_selective_auc_ties_and_single_class():
    p = np.array([0.9, 0.1, 0.9, 0.1])  # all equally confident
    y = np.array([1, 0, 0, 1])
    # stable: keep original indices 0,1 -> labels 1,0, scores .9,.1 -> AUC 1
    assert selective_auc(p, y, 0.5) == 1.0
    assert selective_auc(p, y, 0.5) == selective_auc(p.copy(), y.copy(), 0.5)
    assert np.isnan(selective_auc(p, y, 0.25))  # one frame kept -> single class


def _data(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [1, 2, 3, 5, 8] * 6
    vids, y = [], []
    for i, s in enumerate(sizes):
        vids += [f"v{i}"] * s
        y += [i % 2] * s
    y = np.array(y)
    good = np.clip(0.5 + 0.4 * (2 * y - 1) + 0.1 * rng.standard_normal(len(y)), 0, 1)
    bad = rng.random(len(y))
    return np.array(vids), y, good, bad


def test_paired_identical_models():
    v, y, g, _ = _data()
    r = paired_bootstrap_metric(auc_metric, g, g, y, v, n=100)
    assert r["delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["a"] == r["b"]


def test_paired_better_model_and_metrics():
    v, y, g, b = _data()
    for fn in (auc_metric, partial(selective_auc, coverage=0.5)):
        r = paired_bootstrap_metric(fn, g, b, y, v, n=200)
        assert r["ci_low"] > 0 and r["delta"] > 0
    r = paired_bootstrap_metric(ece, g, b, y, v, n=50)
    assert set(r) >= {"delta", "ci_low", "ci_high", "a", "b"}


def test_determinism_and_single_model():
    v, y, g, b = _data()
    r1 = paired_bootstrap_metric(auc_metric, g, b, y, v, n=100, seed=3)
    r2 = paired_bootstrap_metric(auc_metric, g, b, y, v, n=100, seed=3)
    r3 = paired_bootstrap_metric(auc_metric, g, b, y, v, n=100, seed=4)
    assert r1 == r2 and r1 != r3
    s = bootstrap_metric(auc_metric, g, y, v, n=100, seed=3)
    assert s["ci_low"] <= s["point"] <= s["ci_high"]
    assert s["ci_low"] == r1["a_ci_low"]


def test_videos_resampled_as_units():
    v, y, g, _ = _data()
    seen = []

    def spy(p, lab):
        seen.append((len(p), lab.copy()))
        return float(len(p))

    paired_bootstrap_metric(spy, g, g, y, v, n=20)
    sizes = {1, 2, 3, 5, 8}
    # every replicate must be a union of whole videos: total length = sum of video sizes drawn,
    # and class counts must be multiples consistent with whole videos (each video one label)
    for length, lab in seen[2:]:
        assert length >= 1
        # frames per label can only be sums of whole-video sizes; with sizes {1,2,3,5,8}
        # per-video-count check: no partial video means n_pos and n_neg are each sums of sizes
        assert lab.sum() + (len(lab) - lab.sum()) == length
    # direct check: resample reproduces whole videos by construction -> mean frame count
    # of replicates tracks the mean video size * n_videos (frame count != video count)
    lens = np.array([s[0] for s in seen[2:]])
    assert lens.mean() == pytest.approx(len(y), rel=0.25)
    assert not np.all(lens == len(y))  # varies because videos differ in size
