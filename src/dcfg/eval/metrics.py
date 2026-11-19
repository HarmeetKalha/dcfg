"""Frame / video AUC, video-level bootstrap CIs and paired bootstrap for delta-AUC.

Conventions
-----------
* ``labels``: 1 = fake, 0 = real; ``scores``: P(fake) per frame; ``groups``: video id per frame.
* Video score = **mean of the frame probabilities** of the video; video label = the (constant)
  frame label. This matches DeepfakeBench's ``metrics/utils.get_test_metrics`` (mean prob per
  video, ``label_sum/len`` as label). The only difference: DFB groups by the frames' parent
  directory name, we group by an explicit video id (identical when ids are unique).
* Bootstrap resamples **videos** (with replacement) as the unit for both frame and video AUC, so
  frame AUC intervals account for within-video correlation. Percentile interval (default 95 %).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (ties get average ranks; NaN if one class)."""
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def frame_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC over all frames."""
    return _auc(np.asarray(labels).astype(int), np.asarray(scores, dtype=np.float64))


@dataclass
class VideoTable:
    """Frames grouped by video: ids, one label per video, and frame index lists."""

    ids: np.ndarray
    labels: np.ndarray  # per video
    frame_idx: list[np.ndarray]

    @classmethod
    def build(cls, labels: Sequence[int], groups: Sequence) -> "VideoTable":
        labels = np.asarray(labels).astype(int)
        groups = np.asarray(groups)
        ids, inv = np.unique(groups, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.cumsum(np.bincount(inv, minlength=len(ids)))[:-1]
        frame_idx = np.split(order, bounds)
        vlab = np.empty(len(ids), dtype=int)
        for i, fi in enumerate(frame_idx):
            ls = np.unique(labels[fi])
            if len(ls) != 1:
                raise ValueError(f"video {ids[i]!r} has mixed frame labels")
            vlab[i] = ls[0]
        return cls(ids, vlab, frame_idx)

    def mean_scores(self, scores: np.ndarray) -> np.ndarray:
        """Mean frame score per video."""
        return np.array([scores[fi].mean() for fi in self.frame_idx])


def video_auc(labels: Sequence[int], scores: Sequence[float], groups: Sequence) -> float:
    """AUC over videos, each scored by the mean of its frame probabilities (DFB behaviour)."""
    t = VideoTable.build(labels, groups)
    return _auc(t.labels, t.mean_scores(np.asarray(scores, dtype=np.float64)))


@dataclass
class CI:
    """Point estimate with percentile interval."""

    point: float
    lo: float
    hi: float
    n_valid: int  # bootstrap replicates with both classes present

    def as_dict(self) -> dict[str, float | int]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "n_valid": self.n_valid}


def _check_kind(metric: str) -> None:
    if metric not in ("frame", "video"):
        raise ValueError("metric must be 'frame' or 'video'")


def bootstrap_ci(metric: str, labels: Sequence[int], scores: Sequence[float], groups: Sequence,
                 n: int = 1000, seed: int = 0, by: str = "video", alpha: float = 0.05) -> CI:
    """Percentile bootstrap CI of frame or video AUC, resampling videos as units.

    ``metric``: 'frame' or 'video'. Replicates lacking one class are dropped (counted in n_valid).
    """
    _check_kind(metric)
    if by != "video":
        raise ValueError("only by='video' resampling is supported")
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    t = VideoTable.build(labels, groups)
    point = frame_auc(labels, scores) if metric == "frame" else _auc(t.labels, t.mean_scores(scores))
    vm = t.mean_scores(scores) if metric == "video" else None
    rng = np.random.default_rng(seed)
    nv = len(t.ids)
    vals = []
    for _ in range(n):
        vsel = rng.integers(0, nv, nv)
        if metric == "video":
            v = _auc(t.labels[vsel], vm[vsel])
        else:
            idx = np.concatenate([t.frame_idx[i] for i in vsel])
            v = _auc(labels[idx], scores[idx])
        if not np.isnan(v):
            vals.append(v)
    arr = np.asarray(vals)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(arr) else (np.nan, np.nan)
    return CI(point, float(lo), float(hi), len(arr))


@dataclass
class PairedResult:
    """Delta = metric(a) - metric(b) with percentile CI and two-sided bootstrap p-value."""

    delta: float
    lo: float
    hi: float
    p_value: float
    n_valid: int

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def paired_bootstrap(a_scores: Sequence[float], b_scores: Sequence[float], labels: Sequence[int],
                     groups: Sequence, metric: str = "frame", n: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> PairedResult:
    """Paired video-level bootstrap of AUC(a) - AUC(b) on the same frames.

    The same resampled videos are used for both score sets. p-value is two-sided:
    ``2 * min(P(d* <= 0), P(d* >= 0))`` with add-one smoothing, capped at 1.
    """
    _check_kind(metric)
    labels = np.asarray(labels).astype(int)
    a = np.asarray(a_scores, dtype=np.float64)
    b = np.asarray(b_scores, dtype=np.float64)
    if not (len(a) == len(b) == len(labels)):
        raise ValueError("a_scores, b_scores, labels must have equal length")
    t = VideoTable.build(labels, groups)
    am, bm = (t.mean_scores(a), t.mean_scores(b)) if metric == "video" else (None, None)

    def f(vsel: np.ndarray) -> float:
        if metric == "video":
            return _auc(t.labels[vsel], am[vsel]) - _auc(t.labels[vsel], bm[vsel])
        idx = np.concatenate([t.frame_idx[i] for i in vsel])
        return _auc(labels[idx], a[idx]) - _auc(labels[idx], b[idx])

    delta = f(np.arange(len(t.ids)))
    rng = np.random.default_rng(seed)
    nv = len(t.ids)
    d = np.array([f(rng.integers(0, nv, nv)) for _ in range(n)])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))
    return PairedResult(float(delta), float(lo), float(hi), float(p), len(d))



# ---------------------------------------------------------------------------------------------
# Calibration / selective prediction / generic bootstrap helpers
# ---------------------------------------------------------------------------------------------
def auc_metric(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Frame ROC-AUC with the ``(probs, labels)`` argument order used by the helpers below."""
    return _auc(np.asarray(labels).astype(int), np.asarray(probs, dtype=np.float64))


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 15) -> float:
    """Binary expected calibration error on the predicted fake-probability ``p``.

    "Reliability on the positive-class probability": ``n_bins`` equal-width bins on [0, 1];
    ECE = sum_b (n_b / N) * |mean(p in b) - mean(label in b)|. Bins are right-closed
    ``(lo, hi]``, the first bin also includes 0 (the last includes 1). Empty bins are skipped.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(np.float64)
    if p.size == 0:
        return float("nan")
    b = np.clip(np.ceil(p * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        m = b == k
        c = int(m.sum())
        if c:
            total += c / p.size * abs(p[m].mean() - y[m].mean())
    return float(total)


def selective_auc(probs: Sequence[float], labels: Sequence[int], coverage: float) -> float:
    """ROC-AUC on the most confident ``ceil(coverage * N)`` frames (confidence = |p - 0.5|).

    Frames are ordered by confidence descending using a stable sort on the original index, so
    ties at the cutoff keep the earlier frames (deterministic). NaN if the kept set has a
    single class.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    k = int(np.ceil(coverage * p.size - 1e-12))
    order = np.argsort(-np.abs(p - 0.5), kind="stable")[:k]
    return _auc(y[order], p[order])


def bootstrap_metric(metric_fn: Callable[[np.ndarray, np.ndarray], float], scores: Sequence[float],
                     labels: Sequence[int], video_ids: Sequence, n: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, float]:
    """Video-resampled percentile bootstrap of ``metric_fn(scores, labels)`` for one model.

    Returns dict(point, ci_low, ci_high). NaN replicates are dropped.
    """
    r = paired_bootstrap_metric(metric_fn, scores, scores, labels, video_ids, n, seed, alpha)
    return dict(point=r["a"], ci_low=r["a_ci_low"], ci_high=r["a_ci_high"])


def paired_bootstrap_metric(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                            scores_a: Sequence[float], scores_b: Sequence[float],
                            labels: Sequence[int], video_ids: Sequence, n: int = 1000,
                            seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Paired video bootstrap of ``metric_fn(scores, labels)`` for models a and b.

    Each replicate samples videos with replacement (all frames of a sampled video come along,
    duplicates allowed) and evaluates both models on the same resample. Use
    ``functools.partial`` to bind extra arguments (e.g. ``partial(selective_auc, coverage=.5)``).
    Returns dict(delta = metric(a) - metric(b) on the full data, ci_low, ci_high of the delta,
    a, b = full-data metrics, plus a_ci_low/a_ci_high/b_ci_low/b_ci_high).
    """
    labels = np.asarray(labels).astype(int)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if not (len(a) == len(b) == len(labels) == len(video_ids)):
        raise ValueError("scores_a, scores_b, labels, video_ids must have equal length")
    t = VideoTable.build(labels, video_ids)
    nv = len(t.ids)
    rng = np.random.default_rng(seed)
    va, vb = metric_fn(a, labels), metric_fn(b, labels)
    reps = np.full((n, 2), np.nan)
    for i in range(n):
        idx = np.concatenate([t.frame_idx[j] for j in rng.integers(0, nv, nv)])
        reps[i] = metric_fn(a[idx], labels[idx]), metric_fn(b[idx], labels[idx])

    def ci(x: np.ndarray) -> tuple[float, float]:
        x = x[~np.isnan(x)]
        if not len(x):
            return float("nan"), float("nan")
        lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    d_lo, d_hi = ci(reps[:, 0] - reps[:, 1])
    a_lo, a_hi = ci(reps[:, 0])
    b_lo, b_hi = ci(reps[:, 1])
    return dict(delta=float(va - vb), ci_low=d_lo, ci_high=d_hi, a=float(va), b=float(vb),
                a_ci_low=a_lo, a_ci_high=a_hi, b_ci_low=b_lo, b_ci_high=b_hi)
