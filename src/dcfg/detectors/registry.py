"""Registry of the supported DeepfakeBench (DFB) detectors and access to DFB's own classes.

DFB's ``training/detectors/__init__.py`` eagerly imports every detector it ships
(CLIP, SLADD, I3D, ...), which drags in imgaug, albumentations, slowfast, ... and breaks on
numpy 2.x. We must not edit DFB, so :func:`load_dfb` installs a minimal stand-in ``detectors``
package whose ``__path__`` points at DFB's directory and which exposes DFB's own ``DETECTOR``
registry; only the four detector modules we need are then imported, unmodified, through
DFB's own registry.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dcfg.paths import REPO_ROOT, WEIGHTS_DIR

DFB_ROOT: Path = REPO_ROOT / "third_party" / "DeepfakeBench"
DFB_TRAINING: Path = DFB_ROOT / "training"
DFB_WEIGHTS_DIR: Path = WEIGHTS_DIR / "dfb"
#: ImageNet backbone weights (DFB v1.0.0 pretrained.zip, flattened) needed by DFB's builders.
DFB_BACKBONE_DIR: Path = DFB_WEIGHTS_DIR / "pretrained"


@dataclass(frozen=True)
class DetectorSpec:
    """Static description of one supported detector."""

    name: str
    config: str  #: yaml basename in training/config/detector/
    module: str  #: DFB module under ``detectors.``
    weights: str  #: checkpoint file name in DFB_WEIGHTS_DIR
    backbone_weights: str | None  #: ImageNet weights file in DFB_BACKBONE_DIR (None = none)


DETECTORS: dict[str, DetectorSpec] = {
    "xception": DetectorSpec("xception", "xception", "xception_detector",
                             "xception_best.pth", "xception-b5690688.pth"),
    # efficientnet_pytorch builds from_name() when 'pretrained' is falsy; the full DFB
    # checkpoint overwrites every weight anyway, so no ImageNet file is needed.
    "efficientnetb4": DetectorSpec("efficientnetb4", "efficientnetb4", "efficientnetb4_detector",
                                   "effnb4_best.pth", None),
    "f3net": DetectorSpec("f3net", "f3net", "f3net_detector",
                          "f3net_best.pth", "xception-b5690688.pth"),
    "ucf": DetectorSpec("ucf", "ucf", "ucf_detector",
                        "ucf_best.pth", "xception-b5690688.pth"),
}


def dfb_commit() -> str:
    """HEAD commit of the DFB checkout (``unknown`` if it is not a git repo)."""
    try:
        return subprocess.check_output(["git", "-C", str(DFB_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_dfb() -> Any:
    """Import DFB's four detector modules and return DFB's ``DETECTOR`` registry."""
    if not DFB_TRAINING.is_dir():
        raise FileNotFoundError(f"DeepfakeBench not found at {DFB_ROOT}")
    if str(DFB_TRAINING) not in sys.path:
        sys.path.insert(0, str(DFB_TRAINING))
    from metrics.registry import DETECTOR  # DFB's own registry

    if "detectors" not in sys.modules or not hasattr(sys.modules["detectors"], "DETECTOR"):
        pkg = types.ModuleType("detectors")
        pkg.__path__ = [str(DFB_TRAINING / "detectors")]  # type: ignore[attr-defined]
        pkg.DETECTOR = DETECTOR  # type: ignore[attr-defined]
        sys.modules["detectors"] = pkg
    for spec in DETECTORS.values():
        importlib.import_module(f"detectors.{spec.module}")
    return DETECTOR


def get_spec(name: str) -> DetectorSpec:
    """Look up a detector spec, raising a helpful error for unknown names."""
    if name not in DETECTORS:
        raise KeyError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}")
    return DETECTORS[name]


def load_config(name: str) -> dict[str, Any]:
    """Load DFB's detector yaml and override file-system paths for this machine."""
    spec = get_spec(name)
    with open(DFB_TRAINING / "config" / "detector" / f"{spec.config}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["pretrained"] = (str(DFB_BACKBONE_DIR / spec.backbone_weights)
                         if spec.backbone_weights else None)
    cfg["log_dir"] = str(REPO_ROOT / "runs" / "dfb_logs")
    return cfg


def default_weights_path(name: str) -> Path:
    """Default location of the DFB checkpoint for ``name``."""
    return DFB_WEIGHTS_DIR / get_spec(name).weights


def weights_available(name: str) -> bool:
    """True if the checkpoint and the backbone file DFB's builder needs both exist."""
    spec = get_spec(name)
    ok = default_weights_path(name).is_file()
    if spec.backbone_weights:
        ok = ok and (DFB_BACKBONE_DIR / spec.backbone_weights).is_file()
    return ok
