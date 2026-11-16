"""Tests for dcfg.detectors (adapter + registry); skipped if DFB weights are absent."""
from __future__ import annotations

import pytest
import torch

from dcfg.detectors.adapter import Detector
from dcfg.detectors.registry import DETECTORS, weights_available

NAMES = list(DETECTORS)
CUDA = torch.cuda.is_available()


def _skip_if_absent(name: str) -> None:
    if not weights_available(name):
        pytest.skip(f"DFB weights for {name} not downloaded")


@pytest.fixture(scope="module")
def dets():
    cache: dict[str, Detector] = {}

    def get(name: str) -> Detector:
        _skip_if_absent(name)
        if name not in cache:
            cache[name] = Detector(name, device="cpu")
        return cache[name]
    return get


@pytest.mark.parametrize("name", NAMES)
def test_load_zero_missing_unexpected(dets, name):
    d = dets(name)
    assert d.load_report["missing"] == [] and d.load_report["unexpected"] == []
    assert d.load_report["n_keys"] > 100


@pytest.mark.parametrize("name", NAMES)
def test_shape_and_prob_matches_dfb(dets, name):
    d = dets(name)
    g = torch.Generator().manual_seed(0)
    x01 = torch.rand(3, 3, 256, 256, generator=g)
    xn = d.normalize(x01)
    with torch.no_grad():
        logit = d(xn)
        raw = d.dfb_forward(xn)
        # ucf's inference branch has no 'prob'; DFB's test loop collects it in model.prob
        if "prob" in raw:
            ref = raw["prob"]
        else:
            ref = torch.softmax(raw["cls"], 1)[:, 1]
    assert logit.shape == (3,) and logit.dtype == torch.float32
    torch.testing.assert_close(torch.sigmoid(logit), ref.float(), atol=1e-4, rtol=0)


def test_ucf_prob_matches_collected_list(dets):
    d = dets("ucf")
    xn = d.normalize(torch.rand(2, 3, 256, 256))
    d.model.prob.clear()
    with torch.no_grad():
        out = d.model({"image": xn, "label": torch.zeros(2, dtype=torch.long)}, inference=True)
    ref = torch.as_tensor(d.model.prob[-1]).reshape(-1)
    torch.testing.assert_close(torch.softmax(out["cls"], 1)[:, 1], ref, atol=1e-6, rtol=0)
    d.model.prob.clear()


def test_normalize_uses_config(dets):
    d = dets("xception")
    x = torch.full((1, 3, 4, 4), 0.5)
    assert torch.allclose(d.normalize(x), torch.zeros_like(x))
    assert d.normalize(torch.ones(1, 3, 2, 2)).max().item() == pytest.approx(1.0)


@pytest.mark.parametrize("name", NAMES)
def test_freeze_and_input_grad(dets, name):
    d = dets(name).freeze()
    assert not any(p.requires_grad for p in d.parameters())
    d.train()
    assert not d.training and not any(m.training for m in d.modules())
    x = torch.rand(2, 3, 256, 256, requires_grad=True)
    d(d.normalize(x)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


@pytest.mark.parametrize("name", NAMES)
def test_resize_when_not_256(dets, name):
    d = dets(name).freeze()
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    out = d(d.normalize(x))
    assert out.shape == (1,)
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("name", NAMES)
def test_fp16_autocast(name):
    _skip_if_absent(name)
    d = Detector(name, device="cuda").freeze()
    x = torch.rand(4, 3, 256, 256, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = d(d.normalize(x))
    assert out.shape == (4,) and torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    with torch.no_grad():
        ref = torch.sigmoid(d(d.normalize(x.detach())))
        with torch.autocast("cuda", dtype=torch.float16):
            half = torch.sigmoid(d(d.normalize(x.detach())))
    assert (ref - half).abs().max() < 0.05
    del d
    torch.cuda.empty_cache()


@pytest.mark.perf
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("name", NAMES)
def test_throughput(name, capsys):
    import time
    _skip_if_absent(name)
    d = Detector(name, device="cuda").freeze()
    x = torch.rand(32, 3, 256, 256, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            d(d.normalize(x))
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10):
            d(d.normalize(x))
        torch.cuda.synchronize()
    ips = 320 / (time.time() - t)
    with capsys.disabled():
        print(f"\nPERF {name}: {ips:.0f} img/s, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
