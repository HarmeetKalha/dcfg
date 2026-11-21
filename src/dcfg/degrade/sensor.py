"""Damage sensor: a small CNN trained from scratch to detect WHICH degradations a face crop carries and how strong.

Interface (LUMEN2 reuses it)::

    s = DamageSensor()
    out = s(x01)                      # x01: (B,3,H,W) float RGB in [0,1], H,W >= 128, multiples of 8 (256 crops expected)
    out["presence_logit"]  # (B,4) logits of "op applied", op order dcfg.degrade.labels.OPS = (jpeg, blur, noise, resize)
    out["strength"]        # (B,4) in [0,1] (sigmoid), 1 = most damage; meaningful where the op is present
    out["embedding"]       # (B,128) the embedding s
    s.embed(x01) -> (B,128)           # same embedding only

Inputs inside the module: (1) the central 128x128 crop of the image at NATIVE resolution (no resize: resizing would
destroy JPEG/noise/resize traces; the centre offset (H-128)/2 is a multiple of 8 for 256 crops so the JPEG block grid is
preserved), ImageNet-normalised; (2) the 192 DCT log-energy statistics of the FULL image (YCbCr, 8x8 blockwise DCT,
per-coefficient log(mean|c|+1e-6), the same code as the DCFG gate: ``dcfg.models.gate.log_energy``).
Fusion: concat(GAP CNN feature 160, LayerNorm(192 stats)) -> Linear 352->256 -> GELU -> Linear 256->128 (= embedding).
Heads: Linear(128,4) presence, Linear(128,4) strength logit.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from dcfg.models.dct import BlockDCT, rgb_to_ycbcr
from dcfg.models.gate import N_COEF, log_energy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP = 128
EMBED_DIM = 128
N_OPS = 4


def _cbr(cin: int, cout: int, stride: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class DamageSensor(nn.Module):
    """See module docstring. < 1M parameters."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        # (cin, cout, stride) at native resolution: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        for cin, cout, st in [(3, 32, 1), (32, 32, 2), (32, 64, 2), (64, 64, 1), (64, 96, 2), (96, 96, 1),
                              (96, 128, 2), (128, 128, 1), (128, 160, 2)]:
            layers += _cbr(cin, cout, st)
        self.features = nn.Sequential(*layers)
        self.ln_stats = nn.LayerNorm(N_COEF)
        self.fc1 = nn.Linear(160 + N_COEF, 256)
        self.fc2 = nn.Linear(256, EMBED_DIM)
        self.presence = nn.Linear(EMBED_DIM, N_OPS)
        self.strength = nn.Linear(EMBED_DIM, N_OPS)
        self.dct = BlockDCT()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def embed(self, x01: Tensor) -> Tensor:
        """(B,3,H,W) in [0,1] -> (B,128)."""
        h, w = x01.shape[-2:]
        assert h >= CROP and w >= CROP and h % 8 == 0 and w % 8 == 0, f"bad input size {h}x{w}"
        with torch.autocast(device_type=x01.device.type, enabled=False):
            stats = log_energy(self.dct(rgb_to_ycbcr(x01.float())))  # (B,192) from the full image
            t, l = (h - CROP) // 2, (w - CROP) // 2
            c = x01.float()[:, :, t:t + CROP, l:l + CROP]
            f = self.features((c - self.mean) / self.std).mean(dim=(2, 3))
            z = torch.cat([f, self.ln_stats(stats)], dim=1)
            return self.fc2(nn.functional.gelu(self.fc1(z)))

    def forward(self, x01: Tensor) -> dict[str, Tensor]:
        s = self.embed(x01)
        return {"embedding": s, "presence_logit": self.presence(s), "strength": torch.sigmoid(self.strength(s))}


def damage_loss(out: dict[str, Tensor], applied: Tensor, strength: Tensor, w_strength: float = 1.0
                ) -> tuple[Tensor, dict[str, Tensor]]:
    """BCE on presence (all ops) + MSE on strength over APPLIED ops only (mean over those entries)."""
    bce = nn.functional.binary_cross_entropy_with_logits(out["presence_logit"], applied)
    m = applied > 0.5
    mse = ((out["strength"] - strength) ** 2)[m].mean() if m.any() else out["strength"].sum() * 0.0
    return bce + w_strength * mse, {"bce": bce.detach(), "mse": mse.detach()}
