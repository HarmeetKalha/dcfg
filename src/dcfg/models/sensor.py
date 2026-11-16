"""Degradation sensor (PLAN 6.1): small fully convolutional CNN -> 128-d embedding.

Input: RGB ``x`` in [0,1]. ImageNet mean/std normalisation is applied *inside*
the module, so callers always pass x01. Output of ``forward``: ``(embedding, score)``.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 128


class DegradationSensor(nn.Module):
    """4x(conv3x3-BN-ReLU-maxpool2) 32/64/128/128 -> GAP -> Linear(128,128) -> head 128-32-1."""

    def __init__(self) -> None:
        super().__init__()
        chans = [3, 32, 64, 128, 128]
        layers: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.embedding = nn.Linear(128, EMBED_DIM)
        self.head = nn.Sequential(nn.Linear(EMBED_DIM, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def embed(self, x01: Tensor) -> Tensor:
        """Return the 128-d embedding for RGB images in [0,1]."""
        x = (x01 - self.mean) / self.std
        f = self.features(x).mean(dim=(2, 3))
        return self.embedding(f)

    def forward(self, x01: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.embed(x01)
        return emb, self.head(emb).squeeze(-1)

    def backbone_state_dict(self) -> dict[str, Tensor]:
        """State dict of backbone + embedding layer only (what LIVE pretraining saves)."""
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(("features.", "embedding."))}

    def load_backbone(self, path: str | Path) -> None:
        """Load backbone + embedding weights from ``path``; strict on those keys."""
        sd = torch.load(path, map_location="cpu", weights_only=True)
        expected = set(self.backbone_state_dict())
        if set(sd) != expected:
            raise KeyError(f"backbone keys mismatch: missing={expected - set(sd)}, "
                           f"unexpected={set(sd) - expected}")
        self.load_state_dict(sd, strict=False)
