"""Tests for dcfg.data.degradations and configs/eval/conditions.yaml."""
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from dcfg.data import degradations as D

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "configs" / "eval" / "conditions.yaml"

EXPECTED = (
    ["clean"] + [f"jpeg_q{q}" for q in (90, 70, 50, 30, 10)]
    + [f"blur_s{s}" for s in (1, 2, 3, 4)] + [f"noise_s{s}" for s in (5, 10, 15, 20)]
    + ["resize_0.5", "resize_0.25", "mixed_light", "mixed_heavy",
       "webp_q50", "webp_q20", "motion_k7", "motion_k11", "c40"]
)


def textured(h=256, w=256, seed=0):
    r = np.random.default_rng(seed)
    base = r.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return base


def test_ops_change_image_and_preserve_dtype_shape():
    img = textured()
    rng = np.random.default_rng(1)
    outs = {
        "jpeg": D.jpeg(img, 30), "webp": D.webp(img, 30), "blur": D.gaussian_blur(img, 1.5),
        "noise": D.gaussian_noise(img, 10, rng), "resize": D.resize(img, 0.5),
        "motion": D.motion_blur(img, 7),
    }
    for name, out in outs.items():
        assert out.dtype == np.uint8 and out.shape == img.shape, name
        assert not np.array_equal(out, img), name


def test_non_square_size():
    img = textured(100, 130)
    assert D.resize(img, 0.3).shape == img.shape
    assert D.jpeg(img, 50).shape == img.shape


def test_train_determinism():
    img, t = textured(), D.TrainDegradation()
    for i in range(20):
        a, ia = t(img, D.make_rng(7, 3, i))
        b, ib = t(img, D.make_rng(7, 3, i))
        assert a.tobytes() == b.tobytes() and ia == ib
    assert D.make_rng(7, 3, 1).random() != D.make_rng(7, 3, 2).random()
    assert D.make_rng(7, 3, 1).random() != D.make_rng(7, 4, 1).random()


def test_eval_determinism_and_subprocess():
    conds = D.load_conditions(YAML)
    img = textured()
    code = (
        "import numpy as np,sys,hashlib;from dcfg.data import degradations as D;"
        "r=np.random.default_rng(0);img=r.integers(0,256,(256,256,3),dtype=np.uint8);"
        "c=D.load_conditions(sys.argv[1])['mixed_heavy'];"
        "print(hashlib.sha256(D.apply_condition(img,c,'a/b.png').tobytes()).hexdigest())"
    )
    out = subprocess.run([sys.executable, "-c", code, str(YAML)], capture_output=True,
                         text=True, check=True).stdout.strip()
    import hashlib
    here = D.apply_condition(img, conds["mixed_heavy"], "a/b.png")
    assert hashlib.sha256(here.tobytes()).hexdigest() == out
    for c in conds.values():
        assert D.apply_condition(img, c, "k").tobytes() == D.apply_condition(img, c, "k").tobytes()
    # different keys give different noise
    n = conds["noise_s10"]
    assert D.apply_condition(img, n, "x").tobytes() != D.apply_condition(img, n, "y").tobytes()


def test_sampled_params_in_range():
    t, rng = D.TrainDegradation(), np.random.default_rng(0)
    rngs = {"jpeg": (20, 95), "blur": (0.5, 4.0), "noise": (2, 20), "resize": (0.25, 0.9)}
    seen_q = set()
    for _ in range(2000):
        for op, p in t.sample(rng):
            lo, hi = rngs[op]
            assert lo <= p <= hi, (op, p)
            if op == "jpeg":
                assert isinstance(p, int)
                seen_q.add(p)
    assert min(seen_q) <= 22 and max(seen_q) >= 93


def test_clean_prob_n_uniform_no_repeats():
    t, rng = D.TrainDegradation(), np.random.default_rng(1)
    N = 4000
    samples = [t.sample(rng) for _ in range(N)]
    assert abs(sum(len(s) == 0 for s in samples) / N - 0.2) < 0.03
    ns = Counter(len(s) for s in samples if s)
    tot = sum(ns.values())
    assert set(ns) == {1, 2, 3}
    for n in (1, 2, 3):
        assert abs(ns[n] / tot - 1 / 3) < 0.05
    for s in samples:
        names = [o for o, _ in s]
        assert len(names) == len(set(names))
    orders = {tuple(o for o, _ in s) for s in samples if len(s) == 3}
    assert len(orders) > 6  # random order


def test_train_info_and_held_out_never_sampled():
    t, rng, img = D.TrainDegradation(), np.random.default_rng(2), textured(64, 64)
    seen = set()
    for _ in range(300):
        out, info = t(img, rng)
        assert out.dtype == np.uint8 and out.shape == img.shape
        assert info["clean"] == (not info["ops"])
        seen |= {o["op"] for o in info["ops"]}
    assert seen == {"jpeg", "blur", "noise", "resize"}
    held = {c.ops[0][0] for c in D.load_conditions(YAML).values() if c.held_out}
    assert held == {"webp", "motion"} and not held & seen


def test_yaml_contents():
    conds = D.load_conditions(YAML)
    assert list(conds) == EXPECTED
    assert {n for n, c in conds.items() if c.held_out} == {"webp_q50", "webp_q20", "motion_k7", "motion_k11"}
    assert conds["c40"].group == "real_compression" and not conds["c40"].synthetic and not conds["c40"].ops
    assert [o for o, _ in conds["mixed_light"].ops] == ["blur", "noise", "jpeg"]
    assert conds["mixed_heavy"].ops == (("blur", 2.0), ("noise", 10.0), ("jpeg", 30))
    val = D.load_val_conditions(YAML)
    assert val == ["clean", "jpeg_q50", "jpeg_q30", "blur_s2", "noise_s10", "mixed_heavy"]
    assert set(val) <= set(conds)
    img = textured(64, 64)
    assert D.apply_condition(img, conds["clean"], "k") is img or np.array_equal(D.apply_condition(img, conds["clean"], "k"), img)
    for c in conds.values():
        out = D.apply_condition(img, c, "k")
        assert out.dtype == np.uint8 and out.shape == img.shape


def test_throughput():
    t, img, rng = D.TrainDegradation(), textured(), np.random.default_rng(3)
    for _ in range(20):
        t(img, rng)
    n, t0 = 600, time.perf_counter()
    for _ in range(n):
        t(img, rng)
    ips = n / (time.perf_counter() - t0)
    print(f"TrainDegradation throughput: {ips:.0f} img/s")
    if ips < 300:
        warnings.warn(f"TrainDegradation only {ips:.0f} img/s (< 300)")
