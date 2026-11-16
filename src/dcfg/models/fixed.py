"""Image-independent learnable per-coefficient weights (3 x 64 logits, zero-init)."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FixedGate(nn.Module):
    """192 learnable logits; w = 2*sigmoid(logits), identical for every image (w == 1 at init)."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, 8, 8))

    def forward(self, batch_size: int) -> Tensor:
        """Return w of shape (B,3,8,8), broadcast over the batch."""
        return (2 * torch.sigmoid(self.logits.float())).unsqueeze(0).expand(batch_size, -1, -1, -1)
