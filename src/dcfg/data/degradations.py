"""Image degradations: training sampler (PLAN 7.2) and eval conditions (PLAN 7.3).

All images are uint8 HxWx3 RGB numpy arrays of any size. Every operation is a
pure function: randomness comes only from an explicit ``np.random.Generator``
and there is no global RNG anywhere.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, features

assert features.check("webp"), "PIL was built without WebP support"

#: JPEG settings are fixed: 4:2:0 chroma subsampling (subsampling=2), optimize=False.
JPEG_SUBSAMPLING = 2

# ----------------------------------------------------------------------------
# Elementary operations
# ----------------------------------------------------------------------------


def _pil_roundtrip(img: np.ndarray, fmt: str, **kw: Any) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, **kw)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG round trip via PIL (4:2:0 subsampling, optimize=False)."""
    return _pil_roundtrip(img, "JPEG", quality=int(quality),
                          subsampling=JPEG_SUBSAMPLING, optimize=False)


def webp(img: np.ndarray, quality: int) -> np.ndarray:
    """Lossy WebP round trip via PIL."""
    return _pil_roundtrip(img, "WEBP", quality=int(quality), lossless=False)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, kernel size 2*ceil(3*sigma)+1, BORDER_REFLECT_101."""
    k = 2 * math.ceil(3 * sigma) + 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT_101)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise, sigma on the 0-255 scale; rounded and clipped."""
    noise = rng.standard_normal(img.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(np.rint(img.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Bicubic downscale to round(H*scale) x round(W*scale), then bicubic back."""
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Horizontal line-kernel motion blur of length k (normalised), BORDER_REFLECT_101."""
    kernel = np.full((1, int(k)), 1.0 / int(k), dtype=np.float32)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


# Uniform dispatch: name -> (fn, takes_rng). The single param is passed positionally.
_OPS: dict[str, tuple[Any, bool]] = {
    "jpeg": (jpeg, False),
    "blur": (gaussian_blur, False),
    "noise": (gaussian_noise, True),
    "resize": (resize, False),
    "webp": (webp, False),
    "motion": (motion_blur, False),
}


def apply_op(img: np.ndarray, op: str, param: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply a named op with its single parameter."""
    fn, needs_rng = _OPS[op]
    if needs_rng:
        if rng is None:
            raise ValueError(f"op {op!r} needs an rng")
        return fn(img, param, rng)
    return fn(img, param)


# ----------------------------------------------------------------------------
# Training distribution (PLAN 7.2)
# ----------------------------------------------------------------------------

TRAIN_OPS: tuple[str, ...] = ("jpeg", "blur", "noise", "resize")


def make_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Independent, reproducible stream for (seed, epoch, index) via SeedSequence."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(index)]))


def worker_init_fn(worker_id: int, base_seed: int = 0) -> None:
    """DataLoader worker_init_fn placeholder: seeding is done per sample with make_rng.

    Present so datasets can rely on ``make_rng(seed, epoch, index)`` and not on any
    per-worker global state; it deliberately touches no global RNG.
    """
    return None


@dataclass(frozen=True)
class TrainDegradation:
    """Fixed training degradation distribution; no curriculum, no epoch argument."""

    p_clean: float = 0.2
    jpeg_q: tuple[int, int] = (20, 95)
    blur_sigma: tuple[float, float] = (0.5, 4.0)
    noise_sigma: tuple[float, float] = (2.0, 20.0)
    resize_scale: tuple[float, float] = (0.25, 0.9)

    def sample(self, rng: np.random.Generator) -> list[tuple[str, float]]:
        """Sample the ordered list of (op, param); empty list means clean."""
        if rng.random() < self.p_clean:
            return []
        n = int(rng.integers(1, 4))
        names = [TRAIN_OPS[i] for i in rng.permutation(len(TRAIN_OPS))[:n]]  # random order
        out: list[tuple[str, float]] = []
        for name in names:
            if name == "jpeg":
                out.append((name, int(rng.integers(self.jpeg_q[0], self.jpeg_q[1] + 1))))
            elif name == "blur":
                out.append((name, float(rng.uniform(*self.blur_sigma))))
            elif name == "noise":
                out.append((name, float(rng.uniform(*self.noise_sigma))))
            else:
                out.append((name, float(rng.uniform(*self.resize_scale))))
        return out

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        ops = self.sample(rng)
        for name, param in ops:
            img = apply_op(img, name, param, rng)
        return img, {"clean": not ops, "ops": [{"op": o, "param": p} for o, p in ops]}


# ----------------------------------------------------------------------------
# Evaluation conditions (PLAN 7.3)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCondition:
    """One evaluation condition: an ordered op list (empty = no synthetic op)."""

    name: str
    group: str
    ops: tuple[tuple[str, float], ...] = ()
    held_out: bool = False
    synthetic: bool = True
    extra: dict = field(default_factory=dict, compare=False)


def load_conditions(path: str | Path) -> dict[str, EvalCondition]:
    """Load conditions.yaml into name -> EvalCondition (file order preserved)."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    out: dict[str, EvalCondition] = {}
    for c in doc["conditions"]:
        ops = tuple((o["op"], o["param"]) for o in c.get("ops", []))
        for op, _ in ops:
            if op not in _OPS:
                raise ValueError(f"unknown op {op!r} in condition {c['name']}")
        if c["name"] in out:
            raise ValueError(f"duplicate condition {c['name']}")
        out[c["name"]] = EvalCondition(
            name=c["name"], group=c["group"], ops=ops,
            held_out=bool(c.get("held_out", False)),
            synthetic=bool(c.get("synthetic", True)),
        )
    return out


def load_val_conditions(path: str | Path) -> list[str]:
    """Names of the validation subset (early stopping only)."""
    with open(path) as f:
        return list(yaml.safe_load(f)["val_conditions"])


def condition_rng(cond: EvalCondition, image_key: str) -> np.random.Generator:
    """Stable RNG from sha256(f'{cond.name}|{image_key}'); identical across processes."""
    digest = hashlib.sha256(f"{cond.name}|{image_key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def apply_condition(img: np.ndarray, cond: EvalCondition, image_key: str) -> np.ndarray:
    """Deterministically degrade ``img`` under ``cond``; byte-identical across runs."""
    rng = condition_rng(cond, image_key)
    for op, param in cond.ops:
        img = apply_op(img, op, param, rng)
    return img
