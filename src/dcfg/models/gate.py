"""Per-image frequency gate producing fully per-coefficient weights (3 x 64).

No frequency-band grouping is used: every (channel, u, v) coefficient has its own weight.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

N_COEF = 3 * 64


def log_energy(coef: Tensor) -> Tensor:
    """(B,3,Hb,Wb,8,8) -> (B,192): log(mean over blocks of |c| + 1e-6), per (channel, u, v)."""
    return torch.log(coef.abs().mean(dim=(2, 3)) + 1e-6).flatten(1)


class FrequencyGate(nn.Module):
    """MLP gate: LN(embedding 128) and LN(log-energy 192) -> concat -> Linear 320->256 -> GELU
    -> Linear 256->192 -> w = 2*sigmoid(z).

    Each input has its own LayerNorm (affine, eps 1e-5) so the two differently scaled
    inputs are standardised per sample before mixing. The nosensor variant uses only
    LN(192) -> Linear 192->256.

    The last layer has zero weight AND bias, so w == 1 exactly at init.
    ``use_sensor=False`` drops the embedding input (gate_nosensor).
    """

    def __init__(self, use_sensor: bool = True, embed_dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.use_sensor = use_sensor
        din = N_COEF + (embed_dim if use_sensor else 0)
        self.ln_energy = nn.LayerNorm(N_COEF)
        self.ln_embed = nn.LayerNorm(embed_dim) if use_sensor else None
        self.fc1 = nn.Linear(din, hidden)
        self.fc2 = nn.Linear(hidden, N_COEF)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, energy: Tensor, embedding: Tensor | None = None) -> Tensor:
        """Return weights ``w`` of shape (B,3,8,8)."""
        with torch.autocast(device_type=energy.device.type, enabled=False):
            h = self.ln_energy(energy.float())
            if self.use_sensor:
                assert embedding is not None, "gate with use_sensor=True needs an embedding"
                h = torch.cat([self.ln_embed(embedding.float()), h], dim=1)
            z = self.fc2(nn.functional.gelu(self.fc1(h)))
            return (2 * torch.sigmoid(z)).view(-1, 3, 8, 8)
