"""GenD's own face crop and input preprocessing (public model yermandy/GenD_CLIP_L_14).

Ported from third_party/GenD (commit recorded in docs/DECISIONS.md):

* detector: RetinaFace ``det_10g.onnx`` (insightface buffalo_l), 640x640 detection canvas, NMS 0.4,
  largest-area face (``detector.py`` without masks); 5 landmarks from the detector;
* ``align_face``: GenD's destination template, scale (margin) 1.3, ``cv2.estimateAffinePartial2D(LMEDS)``,
  ``INTER_LINEAR`` warp; we use a fixed 256x256 target (GenD's README uses ``--target_size none``,
  i.e. a native-resolution crop, resized to 224 at train/test time);
* input transform (HF ``CLIPProcessor`` of openai/clip-vit-large-patch14): PIL bicubic resize to 224x224
  (shortest side -> 224, then 224 centre crop, a no-op for a square crop), /255, CLIP mean/std.

Everything here is a pure function of its arguments.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
INPUT_SIZE = 224
CROP_SIZE = 256
SCALE = 1.3

#: GenD's destination landmarks (fractions of the crop): eyes, nose, mouth corners
GEND_DST = np.array([[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32)


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, target_size: tuple[int, int] = (CROP_SIZE, CROP_SIZE),
               scale: float = SCALE) -> np.ndarray:
    """GenD ``align_face`` (detector.py) with a fixed ``target_size``; BGR in, BGR out."""
    import cv2

    dst = GEND_DST.copy()
    dst[:, 0] *= target_size[0]
    dst[:, 1] *= target_size[1]
    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)
    m = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)[0]
    if m is None:
        raise ValueError("affine estimation failed")
    return cv2.warpAffine(img_bgr, m, target_size, flags=cv2.INTER_LINEAR)


def largest_face(xyxy: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Landmarks (5x2) of the largest-area detection (GenD's rule when no mask is given)."""
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return kps[int(np.argmax(areas))]


def gend_resize(img_rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB -> uint8 224x224x3 exactly as ``CLIPImageProcessor`` (PIL bicubic, square input)."""
    h, w = img_rgb.shape[:2]
    if h != w:  # shortest side -> 224, then centre crop 224 (GenD crops are square; kept for completeness)
        s = INPUT_SIZE / min(h, w)
        nh, nw = max(INPUT_SIZE, round(h * s)), max(INPUT_SIZE, round(w * s))
        im = Image.fromarray(img_rgb).resize((nw, nh), Image.BICUBIC)
        t, l = (nh - INPUT_SIZE) // 2, (nw - INPUT_SIZE) // 2
        return np.asarray(im.crop((l, t, l + INPUT_SIZE, t + INPUT_SIZE)), dtype=np.uint8).copy()
    return np.asarray(Image.fromarray(img_rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC), dtype=np.uint8).copy()
