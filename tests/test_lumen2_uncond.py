"""LUMEN-uncond control: LUMEN2 with a learned constant conditioning vector (docs/DECISIONS.md 2026-11-23)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dcfg.gend.crop import gend_resize  # noqa: E402
from dcfg.gend.train_gend_aug import load_cfg  # noqa: E402
from dcfg.lumen2 import model as M  # noqa: E402
from tests.test_lumen2 import TinyBase, crops, packed_batch  # noqa: E402


def tiny(cond: str):
    torch.manual_seed(0)
    return M.Lumen2Model(TinyBase(), conditioning=cond)


def test_same_params_plus_constant_and_same_init():
    a, b = tiny("sensor"), tiny("constant")
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"const"} and set(sa) <= set(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)  # global RNG untouched: every shared tensor initialised identically
    na, nb = sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters())
    assert nb - na == M.N_COEF + 128 == 320
    assert b.part_param_counts()["LUMEN parts total"] == a.part_param_counts()["LUMEN parts total"] + 320
    assert M.Lumen2Model.is_trainable_name("const")


def test_identity_at_init_and_input_independent_conditioning():
    m = tiny("constant").eval()
    imgs = crops(4)
    x = packed_batch(imgs)
    x224 = torch.stack([torch.from_numpy(gend_resize(i)).permute(2, 0, 1) for i in imgs]).float() / 255.0
    with torch.no_grad():
        logit, sens, w = m.predict_with_sensor(x)
        assert torch.equal(w, torch.ones_like(w)) and torch.allclose(logit, m.base(x224), atol=1e-4)
        # with non-zero gate / generator, every image still gets the SAME gate weights and the same LN shift
        m.gate.fc2.weight.normal_(); m.gen.U.normal_(); m.gen.V.normal_()
        w = m.front(x)[1]
        assert torch.allclose(w, w[:1].expand_as(w)) and not torch.allclose(w, torch.ones_like(w))
        seen = []
        orig = m._condition
        m._condition = lambda s: (seen.append(s), orig(s))[1]
        m(x)
        assert torch.allclose(seen[0], seen[0][:1].expand_as(seen[0]))


def test_const_and_sensor_get_gradient():
    m = tiny("constant").train()
    with torch.no_grad():
        m.gate.fc2.weight.normal_(); m.gen.U.normal_(); m.gen.V.normal_()
    x = packed_batch(crops(4))
    logits, _, sens, _ = m.forward_train(x)
    (logits.sum() + sens["presence_logit"].sum()).backward()
    assert m.const.grad is not None and m.const.grad[: M.N_COEF].abs().sum() > 0 and m.const.grad[M.N_COEF:].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.sensor.parameters())


def test_state_roundtrip_and_eval_loader_detects_uncond(tmp_path, monkeypatch):
    m = tiny("constant")
    sd = M.trainable_state(m)
    assert "const" in sd
    m2 = tiny("constant")
    with torch.no_grad():
        m2.const.zero_()
    M.load_trainable(m2, sd)
    assert torch.equal(m2.const, m.const)
    with pytest.raises(KeyError):
        M.load_trainable(tiny("sensor"), sd)
    from dcfg.lumen2 import evaluate_lumen2 as E
    torch.save({"trainable": sd, "epoch": 1}, tmp_path / "t.pt")
    monkeypatch.setattr(E, "build_lumen2", lambda dev, conditioning="sensor": M.Lumen2Model(TinyBase(), conditioning=conditioning))
    got, meta = E.load_lumen2(tmp_path / "t.pt", "cpu")
    assert got.conditioning == "constant" and torch.equal(got.const, m.const) and meta == {"epoch": 1}


def test_config_identical_to_lumen2_except_control():
    a, b = load_cfg("configs/train/lumen2.yaml"), load_cfg("configs/train/lumen2_uncond.yaml")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"run_id", "require_pass_marker", "conditioning", "monday_deadline_utc"}
    assert b["conditioning"] == "constant" and b["run_id"] == "lumen2_uncond"


def test_bad_conditioning_rejected():
    with pytest.raises(ValueError):
        M.Lumen2Model(TinyBase(), conditioning="nope")
