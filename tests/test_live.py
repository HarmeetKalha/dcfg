import numpy as np
import pytest
import torch

from dcfg.data.live import (CROP, LiveDataset, get_split, load_records, mos_stats,
                            random_80_20_splits)
from dcfg.paths import LIVE_DIR

pytestmark = pytest.mark.skipif(not LIVE_DIR.exists(), reason="LIVE data missing")


@pytest.fixture(scope="module")
def rec():
    return load_records()


def test_count_and_no_calibration(rec):
    import re
    assert len(rec.names) == 1162 and len(rec.mos) == 1162
    assert not any(re.fullmatch(r"t[1-7]\.bmp", n.lower()) for n in rec.names)
    assert all((rec.root / "Images" / n).is_file() for n in rec.names)


def test_split_disjoint_stable(rec, tmp_path):
    p = tmp_path / "s.json"
    a = get_split(rec, p)
    b = get_split(rec, p)  # reload
    assert a == b
    s = [set(a[k]) for k in ("train", "val", "test")]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2])
    assert sum(map(len, s)) == 1162 and len(s[0]) == 930
    for sp in random_80_20_splits(1162)[:2]:
        assert not set(sp["train"]) & set(sp["test"]) and len(sp["train"]) + len(sp["test"]) == 1162


def test_crops_no_resize(rec):
    ds = LiveDataset(rec, [0, 1], "center", 0.0, 1.0)
    src = ds.load_image(0)
    h, w = src.shape[:2]
    x, _ = ds[0]
    assert x.shape == (3, CROP, CROP) and x.dtype == torch.float32
    y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    ref = torch.from_numpy(src[y0:y0 + CROP, x0:x0 + CROP]).permute(2, 0, 1).float() / 255
    assert torch.equal(x, ref)
    f, _ = LiveDataset(rec, [0], "five", 0.0, 1.0)[0]
    assert f.shape == (5, 3, CROP, CROP)
    assert torch.equal(f[3], torch.from_numpy(src[h - CROP:, w - CROP:]).permute(2, 0, 1).float() / 255)
    t, _ = LiveDataset(rec, [0], "train", 0.0, 1.0)[0]  # random crop = (flipped) window of the source
    full = torch.from_numpy(src.copy()).permute(2, 0, 1).float() / 255
    found = False
    for c in (t, t.flip(-1)):
        ys, xs = torch.nonzero(full[0, :h - CROP + 1, :w - CROP + 1] == c[0, 0, 0], as_tuple=True)
        found = found or any(torch.equal(full[:, y:y + CROP, x:x + CROP], c) for y, x in zip(ys.tolist(), xs.tolist()))
    assert found


def test_mos_norm_train_only(rec, tmp_path):
    sp = get_split(rec, tmp_path / "s.json")
    m, s = mos_stats(rec, sp["train"])
    assert m == pytest.approx(rec.mos[sp["train"]].mean()) and s == pytest.approx(rec.mos[sp["train"]].std())
    assert m != pytest.approx(rec.mos.mean(), abs=1e-9)
    ds = LiveDataset(rec, sp["train"][:1], "center", m, s)
    assert float(ds[0][1]) == pytest.approx((rec.mos[sp["train"][0]] - m) / s, rel=1e-5)
