"""Degradation labels: scaling, and bit-identical images vs the plain sampler (augft + GenD-aug streams)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dcfg.data.degradations import TrainDegradation, make_rng
from dcfg.degrade.labels import OPS, apply_with_labels, ops_to_labels, strength_from_param


def test_scaling_endpoints():
    assert OPS == ("jpeg", "blur", "noise", "resize")
    assert strength_from_param("jpeg", 95) == 0 and strength_from_param("jpeg", 20) == 1
    assert strength_from_param("blur", 0.5) == 0 and strength_from_param("blur", 4.0) == 1
    assert strength_from_param("noise", 2) == 0 and strength_from_param("noise", 20) == 1
    assert strength_from_param("resize", 0.9) == 0 and strength_from_param("resize", 0.25) == 1
    assert strength_from_param("jpeg", 10) == 1  # clipped (eval jpeg_q10)
    with pytest.raises(KeyError):
        ops_to_labels([("webp", 20)])


def test_ops_to_labels_mixed():
    a, s = ops_to_labels([("blur", 2.0), ("noise", 10.0), ("jpeg", 30)])
    assert a.tolist() == [1, 1, 1, 0] and s[3] == 0
    assert ops_to_labels([])[0].sum() == 0


def _stream(n: int, use_labels: bool, seed: int = 42, epoch: int = 0) -> tuple[str, int]:
    rs = np.random.default_rng(0)
    deg = TrainDegradation()
    h, clean = hashlib.sha256(), 0
    for pos in range(n):
        img = rs.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        rng = make_rng(seed, epoch, pos)
        if use_labels:
            out, info, a, s = apply_with_labels(deg, img, rng)
            assert (a.sum() == 0) == info["clean"]
        else:
            out, info = deg(img, rng)
        clean += info["clean"]
        h.update(out.tobytes())
    return h.hexdigest(), clean


@pytest.mark.parametrize("epoch", [0, 3])
def test_bit_identical_to_plain_sampler(epoch):
    """The same (seed, epoch, position) stream used by TrainDataset (augft) and, through it, GendTrainSet (GenD-aug)."""
    a, ca = _stream(120, False, epoch=epoch)
    b, cb = _stream(120, True, epoch=epoch)
    assert a == b and ca == cb and 0 < ca < 120


def test_gend_stream_uses_same_sampler():
    import inspect
    from dcfg.gend import train_gend_aug as g
    src = inspect.getsource(g.GendTrainSet.__getitem__)
    assert "self.inner[j]" in src  # GenD-aug samples come from TrainDataset => same sampler call
