"""FF++ frame access behind a small interface, implemented from the DeepfakeBench JSON schema.

DFB schema (``rearrange.py`` output)::

    J['FaceForensics++'][label][split][comp][video] = {'label': <label>, 'frames': [paths]}

with ``label`` one of ``FF-real``, ``FF-DF``, ``FF-F2F``, ``FF-FS``, ``FF-NT``, ``split`` in
train/val/test and ``comp`` in c23/c40. Binary label: 0 for ``FF-real``, 1 otherwise.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from dcfg.paths import DFB_JSON, FFPP_FRAMES

REAL_LABEL = "FF-real"


@dataclass(frozen=True)
class FrameRecord:
    """One frame: absolute path, stable relative ``key`` (seed for degradations), video, label."""

    path: str
    key: str
    video: str
    label: int


class FrameSource(Protocol):
    """Interface the evaluator depends on (swap in another dataset by implementing this)."""

    def records(self, split: str, comp: str) -> list[FrameRecord]: ...

    def load(self, rec: FrameRecord) -> np.ndarray:
        """Return the frame as uint8 HxWx3 RGB."""
        ...


class DFBJsonSource:
    """Reads ``FaceForensics++.json`` (DFB schema) and frames under ``frames_root``."""

    def __init__(self, json_path: str | os.PathLike[str] | None = None,
                 frames_root: str | os.PathLike[str] | None = None) -> None:
        self.json_path = Path(json_path) if json_path else DFB_JSON / "FaceForensics++.json"
        self.frames_root = Path(frames_root) if frames_root else FFPP_FRAMES
        self._doc: dict | None = None

    def _data(self) -> dict:
        if self._doc is None:
            with open(self.json_path) as f:
                self._doc = json.load(f)["FaceForensics++"]
        return self._doc

    def _resolve(self, p: str) -> tuple[str, str]:
        path = Path(p)
        if not path.is_absolute():
            path = self.frames_root / path
        try:
            key = os.path.relpath(path, self.frames_root)
        except ValueError:
            key = str(path)
        return str(path), key

    def records(self, split: str, comp: str) -> list[FrameRecord]:
        """All frames of ``split``/``comp`` in a deterministic order (label, video, frame)."""
        out: list[FrameRecord] = []
        for label in sorted(self._data()):
            y = 0 if label == REAL_LABEL else 1
            videos = self._data()[label].get(split, {}).get(comp, {})
            for vid in sorted(videos):
                for fp in videos[vid]["frames"]:
                    path, key = self._resolve(fp)
                    out.append(FrameRecord(path, key, f"{label}/{vid}", y))
        return out

    def load(self, rec: FrameRecord) -> np.ndarray:
        bgr = cv2.imread(rec.path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rec.path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
