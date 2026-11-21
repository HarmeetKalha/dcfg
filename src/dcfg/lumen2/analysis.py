"""Pre-declared LUMEN2 metrics (docs/LUMEN2_BRIEF.md, 2026-11-21 + amendment) as pure functions over bootstrap curves.

Every metric is computed per bootstrap resample from ONE shared video-level resample matrix ``W`` (row 0 = the point estimate with unit
weights, rows 1.. = multinomial video counts), so aggregates are bootstrapped jointly and every LUMEN2 - GenD-aug difference is paired.

* synthetic-degraded mean = mean frame AUC over the 17 synthetic conditions (``SYNTH``);
* held-out mean = mean over webp_q50, webp_q20, motion_k7, motion_k11;
* worst-case = per resample, the MINIMUM frame AUC over the 22 degraded conditions (17 synthetic + 4 held-out + c40), computed for each
  model and then differenced (so the CI is of the difference of per-resample minima);
* heavy-damage mean = mean frame AUC over exactly ``HEAVY`` = {jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy};
* video AUC synth mean / c40 = video AUC (mean frame probability per video);
* ECE (15 equal-width bins) and selective AUC (80 / 90 %) over all degraded frames (22 conditions concatenated) use
  ``dcfg.eval.metrics`` on the frames of each resample (videos repeated as drawn).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from dcfg.eval.metrics import ece, selective_auc
from dcfg.paths import REPO_ROOT

SYNTH = (["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "jpeg_q10", "blur_s1", "blur_s2", "blur_s3", "blur_s4", "noise_s5", "noise_s10",
          "noise_s15", "noise_s20", "resize_0.5", "resize_0.25", "mixed_light", "mixed_heavy"])
HELD = ["webp_q50", "webp_q20", "motion_k7", "motion_k11"]
HEAVY = ["jpeg_q10", "blur_s4", "noise_s20", "resize_0.25", "mixed_heavy"]
DEGRADED = SYNTH + HELD + ["c40"]
ALL_CONDS = ["clean"] + DEGRADED

#: metric key -> (label, role, direction) ; direction +1: higher is better, -1: lower is better
METRICS: dict[str, tuple[str, str, int]] = {
    "synth_mean": ("Synthetic-degraded mean AUC (17)", "primary", 1),
    "c40": ("c40 AUC", "secondary", 1),
    "heldout_mean": ("Held-out mean AUC (WebP q50/q20, motion k7/k11)", "secondary", 1),
    "ece": ("ECE (15 bins, all degraded frames)", "secondary", -1),
    "sel80": ("Selective AUC, 80% coverage", "secondary", 1),
    "sel90": ("Selective AUC, 90% coverage", "secondary", 1),
    "worst_case": ("Worst-case AUC (min over 22 degraded conditions)", "secondary", 1),
    "heavy_mean": ("Heavy-damage mean AUC (jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy)", "secondary", 1),
    "video_synth_mean": ("Video AUC, synthetic mean (17)", "secondary", 1),
    "video_c40": ("Video AUC, c40", "secondary", 1),
    "dfd_mixed_heavy": ("DFD under damage (frame AUC, mixed_heavy)", "secondary", 1),
    "clean": ("Clean c23 AUC", "safety", 1),
    "dfd_clean": ("Clean DFD AUC", "safety", 1),
}
#: aggregates that are plain functions of the per-condition AUC curves of the FF++ test set
CURVE_METRICS = ["synth_mean", "c40", "heldout_mean", "worst_case", "heavy_mean", "video_synth_mean", "video_c40", "clean"]


def pct(arr: np.ndarray) -> tuple[float, float, float]:
    """(point estimate = row 0, 2.5 and 97.5 percentiles of rows 1..)."""
    a = arr[1:]
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [2.5, 97.5]) if len(a) else (np.nan, np.nan)
    return float(arr[0]), float(lo), float(hi)


def resample_matrix(n_videos: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot + 1, n_videos): row 0 ones, then multinomial video counts (one shared matrix per video universe)."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones((1, n_videos)), rng.multinomial(n_videos, np.full(n_videos, 1.0 / n_videos), size=n_boot).astype(float)])


def aggregate_curves(cur: dict[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
    """From ``{(frame|video, condition): (B+1,) AUC curve}`` to the curve-based pre-declared aggregates (``CURVE_METRICS``)."""
    f = lambda c: cur[("frame", c)]  # noqa: E731
    v = lambda c: cur[("video", c)]
    return {"synth_mean": np.mean([f(c) for c in SYNTH], axis=0), "heldout_mean": np.mean([f(c) for c in HELD], axis=0),
            "c40": f("c40"), "clean": f("clean"),
            "worst_case": np.min([f(c) for c in DEGRADED], axis=0),
            "heavy_mean": np.mean([f(c) for c in HEAVY], axis=0),
            "video_synth_mean": np.mean([v(c) for c in SYNTH], axis=0), "video_c40": v("c40")}


def outcome(key: str, lo: float, hi: float) -> str:
    """win / loss / tie of LUMEN2 - baseline by the pre-declared rule. Safety metrics: ``ok`` unless the CI is entirely below 0
    (``VIOLATED``). ECE: lower is better, so a CI entirely below 0 is a win."""
    role, direction = METRICS[key][1], METRICS[key][2]
    if role == "safety":
        return "ok" if hi >= 0 else "VIOLATED"
    if direction > 0:
        return "win" if lo > 0 else ("loss" if hi < 0 else "tie")
    return "win" if hi < 0 else ("loss" if lo > 0 else "tie")


def combine_selective(o80: str, o90: str) -> str:
    """Selective AUC rule: win if EITHER coverage wins; loss if both lose; else tie."""
    if "win" in (o80, o90):
        return "win"
    return "loss" if (o80 == "loss" and o90 == "loss") else "tie"


# ------------------------------------------------------------------------------------------------ frame-level metrics with resampling
def frame_bootstrap(fns: dict[str, Callable[[np.ndarray, np.ndarray], float]], pa: np.ndarray, pb: np.ndarray, y: np.ndarray,
                    vid_idx: np.ndarray, W: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``fn(prob, label)`` of two models on the frames of every resample of ``W`` (videos repeated as drawn): ``{name: (curve_a, curve_b)}``.

    Row 0 is the point estimate on the unweighted frames. Uses index repetition, i.e. exactly ``dcfg.eval.metrics`` semantics."""
    out = {k: (np.empty(len(W)), np.empty(len(W))) for k in fns}
    n = len(y)
    for i in range(len(W)):
        if i == 0:
            pa_i, pb_i, y_i = pa, pb, y
        else:
            idx = np.repeat(np.arange(n), W[i][vid_idx].astype(int))
            pa_i, pb_i, y_i = pa[idx], pb[idx], y[idx]
        for k, fn in fns.items():
            out[k][0][i], out[k][1][i] = fn(pa_i, y_i), fn(pb_i, y_i)
    return out


FRAME_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "ece": lambda p, y: ece(p, y), "sel80": lambda p, y: selective_auc(p, y, 0.8), "sel90": lambda p, y: selective_auc(p, y, 0.9)}


def load_group_findings():
    """The shared bootstrap machinery of ops/jobs/group_findings.py (Run, curves)."""
    name = "group_findings"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "ops" / "jobs" / "group_findings.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def exists_all(d: Path, run: str, conds: list[str]) -> bool:
    return all((d / f"{run}__{c}.json").exists() for c in conds)
