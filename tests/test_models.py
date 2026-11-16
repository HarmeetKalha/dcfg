"""Tests for dcfg.models (DCT, sensor, gate, fixed, frontend)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import scipy.fft
import torch

import dcfg.models as models_pkg
from dcfg.models.dct import BlockDCT, dct_basis, rgb_to_ycbcr, ycbcr_to_rgb
from dcfg.models.frontend import FrontEnd
from dcfg.models.sensor import DegradationSensor

KINDS = ["dcfg", "nosensor", "fixed"]


def _x(b=2, s=32, seed=0):
    return torch.rand(b, 3, s, s, generator=torch.Generator().manual_seed(seed))


def test_dct_roundtrip_and_orthonormal():
    d = BlockDCT()
    x = _x(3, 64)
    assert (d.inverse(d(x)) - x).abs().max() < 1e-5
    c = dct_basis()
    assert (c @ c.t() - torch.eye(8)).abs().max() < 1e-6
    assert d(x).shape == (3, 3, 8, 8, 8, 8)


def test_dct_matches_scipy():
    x = _x(1, 24)
    ours = BlockDCT()(x)
    for c in range(3):
        for bi, bj in [(0, 0), (1, 2), (2, 1)]:
            blk = x[0, c, bi*8:bi*8+8, bj*8:bj*8+8].numpy().astype("float64")
            ref = scipy.fft.dctn(blk, norm="ortho")
            assert abs(ours[0, c, bi, bj].numpy() - ref).max() < 1e-5


def test_dct_requires_multiple_of_8():
    with pytest.raises(AssertionError):
        BlockDCT()(torch.rand(1, 3, 12, 16))


def test_ycbcr_roundtrip():
    x = _x(2, 16)
    assert (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max() < 1e-5


@pytest.mark.parametrize("kind", KINDS)
def test_identity_at_init(kind):
    fe = FrontEnd(kind).eval()
    x = _x()
    y, w = fe(x)
    assert (y - x).abs().max() < 1e-5
    assert w.shape == (2, 3, 8, 8)
    assert torch.equal(w, torch.ones_like(w))


def test_sensor_shapes_and_load(tmp_path: Path):
    s = DegradationSensor().eval()
    emb, score = s(torch.rand(2, 3, 40, 56))
    assert emb.shape == (2, 128) and score.shape == (2,)
    p = tmp_path / "s.pt"
    torch.save(s.backbone_state_dict(), p)
    s2 = DegradationSensor()
    s2.load_backbone(p)
    assert torch.equal(s2.embedding.weight, s.embedding.weight)
    torch.save({"features.0.weight": torch.zeros(1)}, p)
    with pytest.raises(KeyError):
        s2.load_backbone(p)


def _perturb(fe):
    with torch.no_grad():
        for n, p in fe.gate.named_parameters():
            p.add_(0.05 * torch.randn_like(p))


@pytest.mark.parametrize("kind", ["dcfg", "nosensor"])
def test_gradients_reach_gate_and_sensor(kind):
    fe = FrontEnd(kind)
    x = _x(4, 32)
    target = torch.rand_like(x)
    ((fe(x)[0] - target) ** 2).mean().backward()
    assert fe.gate.fc2.weight.grad.abs().sum() > 0  # last layer at init
    _perturb(fe)
    fe.zero_grad()
    ((fe(x)[0] - target) ** 2).mean().backward()
    assert fe.gate.fc1.weight.grad.abs().sum() > 0
    if kind == "dcfg":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fe.sensor.parameters())


def test_fixed_gradient():
    fe = FrontEnd("fixed")
    x = _x(2, 32)
    ((fe(x)[0] - torch.rand_like(x)) ** 2).mean().backward()
    assert fe.gate.logits.grad.abs().sum() > 0


def test_frozen_sensor():
    fe = FrontEnd("dcfg", freeze_sensor=True)
    fe.train()
    assert not fe.sensor.training
    assert all(not p.requires_grad for p in fe.sensor.parameters())
    bn = [m for m in fe.sensor.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in bn]
    _perturb(fe)
    x = _x(2, 32).requires_grad_(True)
    _, w = fe(x)
    (w ** 2).sum().backward()  # loss on gate w only: grad reaches x via sensor and stats
    for m, (rm, rv) in zip(bn, before):
        assert torch.equal(m.running_mean, rm) and torch.equal(m.running_var, rv)
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in fe.sensor.parameters())


@pytest.mark.parametrize("c,u,v", [(0, 0, 0), (0, 1, 5), (1, 6, 2), (2, 3, 7), (2, 7, 1)])
def test_single_coefficient_layout(c, u, v):
    fe = FrontEnd("fixed").eval()
    big = -50.0  # w = 2*sigmoid(-50) ~ 0
    with torch.no_grad():
        fe.gate.logits[c, u, v] = big
    x = _x(1, 24, seed=3)
    # keep away from clamp: use mid-range image
    x = 0.25 + 0.5 * x
    y, w = fe(x)
    assert w[0, c, u, v] < 1e-6
    ycc = rgb_to_ycbcr(x)[0].double().numpy()
    for ch in range(3):
        for bi in range(3):
            for bj in range(3):
                blk = ycc[ch, bi*8:bi*8+8, bj*8:bj*8+8]
                co = scipy.fft.dctn(blk, norm="ortho")
                if ch == c:
                    co[u, v] = 0.0
                ycc[ch, bi*8:bi*8+8, bj*8:bj*8+8] = scipy.fft.idctn(co, norm="ortho")
    ref = ycbcr_to_rgb(torch.from_numpy(ycc).float()[None])
    assert (y - ref.clamp(0, 1)).abs().max() < 1e-4  # output is clamped to [0,1]


def test_w_varies_across_images():
    x = _x(4, 32)
    fe = FrontEnd("dcfg").eval()
    _perturb(fe)
    _, w = fe(x)
    assert (w[0] - w[1]).abs().max() > 1e-6
    ff = FrontEnd("fixed")
    with torch.no_grad():
        ff.gate.logits.add_(torch.randn(3, 8, 8))
    _, wf = ff(x)
    assert (wf[0] - wf[1]).abs().max() == 0 and (wf[0] - 1).abs().max() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_autocast_fp16_keeps_fp32_compute(kind):
    fe = FrontEnd(kind).eval()
    x = _x(2, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, w = fe(x)
    assert y.dtype == torch.float32 and w.dtype == torch.float32
    assert (y - x).abs().max() < 1e-5
    yd, _ = fe(x.double())
    assert yd.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_autocast_fp16_cuda():
    fe = FrontEnd("dcfg").cuda().eval()
    x = _x(2, 32).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        y, w = fe(x)
    assert y.dtype == torch.float32 and w.dtype == torch.float32
    assert (y - x).abs().max() < 1e-5


def test_no_band_constructs():
    root = Path(models_pkg.__file__).parent
    bad = ("low_band", "mid_band", "high_band", "zigzag", "lowfreq", "highfreq")
    for f in root.glob("*.py"):
        text = f.read_text().lower()
        for word in bad:
            assert word not in text, f"{word} in {f.name}"


@pytest.mark.perf
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_throughput():
    """Throughput benchmark (>= 2000 img/s). Excluded by default; run with `pytest -m perf`."""
    fe = FrontEnd("dcfg").cuda().eval()
    x = torch.rand(64, 3, 256, 256, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            fe(x)
        torch.cuda.synchronize()
        t = time.time()
        n = 20
        for _ in range(n):
            fe(x)
        torch.cuda.synchronize()
    ips = 64 * n / (time.time() - t)
    print(f"FrontEnd dcfg throughput: {ips:.0f} img/s")
    assert ips >= 2000


@pytest.mark.parametrize("kind", KINDS)
def test_identity_at_init_with_layernorm(kind):
    fe = FrontEnd(kind).eval()  # randomly initialised sensor, unperturbed gate
    imgs = [_x(2, 32, seed=5), torch.full((2, 3, 32, 32), 0.5),
            0.5 + 1e-6 * _x(2, 32, seed=6), torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32)]
    for x in imgs:
        y, w = fe(x)
        assert torch.isfinite(y).all()
        assert (y - x).abs().max() < 1e-5
        assert torch.equal(w, torch.ones_like(w))


def test_gate_param_count_and_ln_stats():
    from dcfg.models.gate import FrequencyGate, log_energy
    n = lambda g: sum(p.numel() for p in g.parameters())
    assert n(FrequencyGate(True)) == (320 * 256 + 256) + (256 * 192 + 192) + 2 * 128 + 2 * 192
    assert n(FrequencyGate(False)) == (192 * 256 + 256) + (256 * 192 + 192) + 2 * 192
    fe = FrontEnd("dcfg").eval()
    _perturb(fe)
    x = _x(4, 32)
    e = log_energy(fe.dct(rgb_to_ycbcr(x)))
    out = torch.nn.functional.layer_norm(e, (192,), eps=fe.gate.ln_energy.eps)  # affine part was perturbed
    assert out.mean(1).abs().max() < 1e-4 and (out.std(1, unbiased=False) - 1).abs().max() < 1e-3
    emb = fe.sensor.embed(x)
    o2 = torch.nn.functional.layer_norm(emb, (128,), eps=fe.gate.ln_embed.eps)
    assert o2.mean(1).abs().max() < 1e-4 and (o2.std(1, unbiased=False) - 1).abs().max() < 1e-2
