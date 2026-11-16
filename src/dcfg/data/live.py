"""LIVE In the Wild Image Quality Challenge dataset (PLAN 1.3, 6.1).

Reads ``Data/AllImages_release.mat`` / ``Data/AllMOS_release.mat``, drops the 7
calibration images ``t1``-``t7`` (1162 usable), and serves **unresized** 224x224 crops:
random crop + h-flip for training, center crop (or 4 corners + center) for evaluation.
Images are returned as float32 RGB tensors in [0, 1].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from dcfg.paths import DATA_ROOT, LIVE_DIR

CROP = 224
N_CALIB = 7
SPLIT_PATH = DATA_ROOT / "live" / "split_seed42.json"


@dataclass(frozen=True)
class LiveRecords:
    """All usable LIVE images: file names (relative to ``Images/``) and raw MOS."""

    root: Path
    names: list[str]
    mos: np.ndarray


def load_records(root: str | Path = LIVE_DIR) -> LiveRecords:
    """Parse the .mat files, excluding calibration images t1-t7 (-> 1162 entries)."""
    from scipy.io import loadmat

    root = Path(root).expanduser()
    names_raw = loadmat(str(root / "Data" / "AllImages_release.mat"))["AllImages_release"]
    mos_raw = loadmat(str(root / "Data" / "AllMOS_release.mat"))["AllMOS_release"]
    names = [str(names_raw[i][0][0]) for i in range(names_raw.shape[0])]
    mos = np.asarray(mos_raw[0], dtype=np.float64)
    assert len(names) == len(mos), "name/MOS length mismatch"
    keep = [i for i, n in enumerate(names) if not (n.lower().startswith("t") and n.lower()[1:].split(".")[0].isdigit()
                                                   and int(n.lower()[1:].split(".")[0]) <= N_CALIB)]
    return LiveRecords(root, [names[i] for i in keep], mos[keep])


def subset_records(rec: LiveRecords, n: int, seed: int = 0) -> LiveRecords:
    """Seeded random subset of ``n`` records (used for fast tests only)."""
    idx = np.sort(np.random.default_rng(seed).permutation(len(rec.names))[:n])
    return LiveRecords(rec.root, [rec.names[i] for i in idx], rec.mos[idx])


def make_split(n: int, seed: int = 42, fracs: tuple[float, ...] = (0.8, 0.1, 0.1)) -> list[list[int]]:
    """Deterministic permutation split of range(n); the last fraction takes the remainder."""
    perm = np.random.default_rng(seed).permutation(n).tolist()
    sizes = [int(round(f * n)) for f in fracs[:-1]]
    out, start = [], 0
    for s in sizes:
        out.append(sorted(perm[start:start + s]))
        start += s
    out.append(sorted(perm[start:]))
    return out


def get_split(rec: LiveRecords, path: str | Path = SPLIT_PATH, seed: int = 42) -> dict[str, list[int]]:
    """Fixed 80/10/10 split (train/val/test indices into ``rec``), saved to / reloaded from json."""
    path = Path(path).expanduser()
    if path.exists():
        d = json.loads(path.read_text())
        if d["n"] != len(rec.names) or d["seed"] != seed:
            raise ValueError(f"{path} does not match this dataset (n={d['n']}, seed={d['seed']})")
        return {k: d[k] for k in ("train", "val", "test")}
    tr, va, te = make_split(len(rec.names), seed)
    d = {"seed": seed, "n": len(rec.names), "train": tr, "val": va, "test": te}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(path)
    return {k: d[k] for k in ("train", "val", "test")}


def random_80_20_splits(n: int, seeds: Sequence[int] = tuple(range(10))) -> list[dict[str, list[int]]]:
    """Secondary IQA protocol: one 80/20 train/test split per seed (no val)."""
    out = []
    for s in seeds:
        tr, te = make_split(n, s, (0.8, 0.2))
        out.append({"seed": s, "train": tr, "test": te})
    return out


def mos_stats(rec: LiveRecords, train_idx: Sequence[int]) -> tuple[float, float]:
    """Mean/std of MOS over the **train** indices only."""
    m = rec.mos[list(train_idx)]
    return float(m.mean()), float(m.std())


def _corners(h: int, w: int, c: int = CROP) -> list[tuple[int, int]]:
    """Top-left coords of the 4 corner crops followed by the center crop."""
    return [(0, 0), (0, w - c), (h - c, 0), (h - c, w - c), ((h - c) // 2, (w - c) // 2)]


class LiveDataset(Dataset):
    """Items are ``(x01, mos_norm)``. ``mode``: ``train`` | ``center`` | ``five``.

    ``five`` yields x of shape (5, 3, 224, 224): 4 corners + center. No resizing anywhere.
    """

    def __init__(self, rec: LiveRecords, indices: Sequence[int], mode: str,
                 mean: float, std: float) -> None:
        if mode not in ("train", "center", "five"):
            raise ValueError(mode)
        self.rec, self.indices, self.mode = rec, list(indices), mode
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.indices)

    def load_image(self, i: int) -> np.ndarray:
        """Full-resolution RGB uint8 array (H, W, 3) of record ``i``."""
        with Image.open(self.rec.root / "Images" / self.rec.names[i]) as im:
            return np.asarray(im.convert("RGB"))

    @staticmethod
    def _crop(a: np.ndarray, y: int, x: int) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(a[y:y + CROP, x:x + CROP])).permute(2, 0, 1).float().div_(255.0)

    def __getitem__(self, k: int) -> tuple[Tensor, Tensor]:
        i = self.indices[k]
        a = self.load_image(i)
        h, w = a.shape[:2]
        if self.mode == "train":
            y = int(torch.randint(0, h - CROP + 1, (1,)))
            x = int(torch.randint(0, w - CROP + 1, (1,)))
            t = self._crop(a, y, x)
            if int(torch.randint(0, 2, (1,))):
                t = t.flip(-1)
        elif self.mode == "center":
            t = self._crop(a, *_corners(h, w)[4])
        else:
            t = torch.stack([self._crop(a, y, x) for y, x in _corners(h, w)])
        y_norm = torch.tensor((self.rec.mos[i] - self.mean) / self.std, dtype=torch.float32)
        return t, y_norm
