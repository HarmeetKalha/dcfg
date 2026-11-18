"""Reusable consistency loss: student(front-end(degraded)) vs frozen teacher(clean).

    total = BCE(s_logit, y) + kl * KL(p_teacher || p_student) + feat * MSE(f_student, f_teacher.detach())

* KL is the Bernoulli KL with p = sigmoid(logit), logit = cls[1] - cls[0]. This equals the 2-class softmax KL
  over the detector's two class outputs (softmax over (cls0, cls1) = (1-p, p)). Averaged over the batch.
  Computed with logsigmoid for stability; the teacher side is a constant (no grad).
* MSE is over the penultimate pooled features (mean over batch and dimensions), teacher side detached.
* The terms apply to every sample, clean ones included (degraded == clean there).
Config key: ``consistency: {kl: 1.0, feat: 1.0}``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT = {"kl": 1.0, "feat": 1.0}


def parse_config(c: dict[str, Any] | None) -> dict[str, float] | None:
    """Validate the ``consistency`` config entry (None / empty = disabled)."""
    if not c:
        return None
    unknown = set(c) - set(DEFAULT)
    if unknown:
        raise KeyError(f"unknown consistency keys {sorted(unknown)}; allowed {sorted(DEFAULT)}")
    return {k: float(c.get(k, DEFAULT[k])) for k in DEFAULT}


def bernoulli_kl(t_logit: Tensor, s_logit: Tensor) -> Tensor:
    """KL(Bernoulli(sigmoid(t)) || Bernoulli(sigmoid(s))) per sample, shape (B,). Gradient only to ``s``."""
    t, s = t_logit.detach().float(), s_logit.float()
    pt = torch.sigmoid(t)
    return pt * (F.logsigmoid(t) - F.logsigmoid(s)) + (1 - pt) * (F.logsigmoid(-t) - F.logsigmoid(-s))


def consistency_loss(s_logit: Tensor, s_feat: Tensor, t_logit: Tensor, t_feat: Tensor, y: Tensor,
                     cfg: dict[str, float]) -> tuple[Tensor, dict[str, Tensor]]:
    """``(total, terms)``; terms = unweighted ``bce``, ``kl``, ``feat`` (detached floats for logging)."""
    assert not t_logit.requires_grad and not t_feat.requires_grad, "teacher must be computed under no_grad"
    bce = F.binary_cross_entropy_with_logits(s_logit.float(), y.float())
    kl = bernoulli_kl(t_logit, s_logit).mean()
    feat = F.mse_loss(s_feat.float(), t_feat.detach().float())
    total = bce + cfg["kl"] * kl + cfg["feat"] * feat
    return total, {"bce": bce.detach(), "kl": kl.detach(), "feat": feat.detach()}
