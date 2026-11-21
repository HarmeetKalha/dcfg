"""Ground-truth degradation labels for the training sampler (PLAN 7.2) and the eval conditions (PLAN 7.3).

Per op in ``OPS = (jpeg, blur, noise, resize)`` (fixed order, index 0..3):
  * ``applied``  in {0,1}: the op is part of the sampled degradation;
  * ``strength`` in [0,1], 1 = most damage, scaled linearly over the training sampling range of
    :class:`dcfg.data.degradations.TrainDegradation` (defaults) and clipped to [0,1]:
      jpeg   q 95 -> 0 ... q 20 -> 1       blur   sigma 0.5 -> 0 ... 4.0 -> 1
      noise  sigma 2 -> 0 ... 20 -> 1      resize scale 0.9 -> 0 ... 0.25 -> 1
    Strength of a non-applied op is 0 (and is not supervised). A clean image has all ``applied == 0``.

``apply_with_labels`` wraps the sampler WITHOUT changing any random draw: the images are bit-identical to
``deg(img, rng)``. Eval conditions outside the training range (jpeg_q10) are clipped to strength 1.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from dcfg.data.degradations import TRAIN_OPS, TrainDegradation

OPS: tuple[str, ...] = TRAIN_OPS  # ("jpeg", "blur", "noise", "resize")
N_OPS = len(OPS)


def strength_from_param(op: str, param: float, deg: TrainDegradation | None = None) -> float:
    """Map an op parameter to [0,1] (1 = most damage); clipped."""
    d = deg or TrainDegradation()
    if op == "jpeg":
        lo, hi = d.jpeg_q
        s = (hi - param) / (hi - lo)
    elif op == "blur":
        lo, hi = d.blur_sigma
        s = (param - lo) / (hi - lo)
    elif op == "noise":
        lo, hi = d.noise_sigma
        s = (param - lo) / (hi - lo)
    elif op == "resize":
        lo, hi = d.resize_scale
        s = (hi - param) / (hi - lo)
    else:
        raise KeyError(f"no label scale for op {op!r}")
    return float(min(1.0, max(0.0, s)))


def ops_to_labels(ops: Sequence[tuple[str, float]], deg: TrainDegradation | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``(applied (4,) float32 in {0,1}, strength (4,) float32 in [0,1])`` for an ordered op list.
    Ops outside ``OPS`` (webp, motion) raise KeyError: they have no label (unseen conditions)."""
    applied = np.zeros(N_OPS, np.float32)
    strength = np.zeros(N_OPS, np.float32)
    for op, param in ops:
        if op not in OPS:
            raise KeyError(f"no label for op {op!r}")
        i = OPS.index(op)
        applied[i] = 1.0
        strength[i] = strength_from_param(op, param, deg)
    return applied, strength


def apply_with_labels(deg: TrainDegradation, img: np.ndarray, rng: np.random.Generator
                      ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Same call as ``deg(img, rng)`` (identical draws, identical image) plus the labels:
    returns ``(img, info, applied (4,), strength (4,))``."""
    out, info = deg(img, rng)
    applied, strength = ops_to_labels([(o["op"], o["param"]) for o in info["ops"]], deg)
    return out, info, applied, strength
