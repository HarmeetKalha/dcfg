"""lumen_nogate (LUMEN1 without the DCT gate): fully gate-free, otherwise the lumen_xception config."""
from __future__ import annotations

import yaml
import torch

from dcfg.models.film import LUMEN_HIDDEN, FiLMBank
from dcfg.models.frontend import FrontEnd
from dcfg.train import train_frontend as tf

CH = {"entry": 728, "exit": 2048}


def _fe() -> FrontEnd:
    return FrontEnd("identity", film=FiLMBank("filmlite", CH, hidden=LUMEN_HIDDEN["xception"]))


def test_no_gate_params_and_identity_image():
    fe = _fe().eval()
    assert sum(p.numel() for p in fe.gate.parameters()) == 0
    assert not any(k.startswith("gate.") or "ln_energy" in k or "fc2" in k for k in fe.state_dict() if not k.startswith("film.")), \
        "no gate parameter may exist"
    assert not any(k.startswith("gate.") for k in fe.state_dict())
    x = torch.rand(3, 3, 64, 64)
    y, w, fp = fe.forward_full(x)
    assert torch.equal(y, x) and torch.equal(w, torch.ones(3, 3, 8, 8)) and fp is not None
    assert not w.requires_grad


def test_trainable_params_equal_lumen_minus_gate():
    lum = FrontEnd("dcfg", film=FiLMBank("filmlite", CH, hidden=LUMEN_HIDDEN["xception"]))
    n_gate = sum(p.numel() for p in lum.gate.parameters())
    n_all = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n_all(lum) - n_all(_fe()) == n_gate


def test_variant_registered_with_identity_gate_and_consistency():
    assert tf.canonical_variant("lumen_nogate") == "lumen_nogate"
    assert tf.GATE_KIND["lumen_nogate"] == "identity"
    assert "lumen_nogate" in tf.CONSISTENCY_VARIANTS and tf.FILM_VARIANTS["lumen_nogate"] == "filmlite"


def test_config_identical_to_lumen_except_name():
    base = yaml.safe_load(open("configs/train/lumen_xception.yaml"))
    ng = yaml.safe_load(open("configs/train/lumen_nogate_xception.yaml"))
    diff = {k for k in set(base) | set(ng) if base.get(k) != ng.get(k)}
    assert diff == {"run_id", "variant"}
    assert ng["variant"] == "lumen_nogate" and ng["seed"] == 42
