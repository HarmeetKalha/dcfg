"""Public GenD CLIP ViT-L/14 as a plain ``nn.Module``: uint8-derived [0,1] 224x224 RGB -> fake logit.

Weights: Hugging Face ``yermandy/GenD_CLIP_L_14`` (model.safetensors, 394 tensors: the whole CLIP vision tower with
the LayerNorm-tuned values + the 1024->2 linear head). Architecture = GenD ``CLIPEncoder`` (vision tower pooled
output, pre-projection) -> ``LinearNorm`` head (L2-normalise, Linear(1024, 2)). The vision tower is built from
the openai/clip-vit-large-patch14 config (config.json kept next to the weights) so no second weight download is needed.
Our logit convention: ``logit = l[1] - l[0]``, so ``sigmoid(logit) = softmax(l)[1]`` = P(fake) (label 1 = fake in GenD).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcfg.gend.crop import CLIP_MEAN, CLIP_STD

GEND_DIR = Path.home() / "data" / "weights" / "gend"


class GendCLIP(nn.Module):
    """CLIP ViT-L/14 vision tower + L2-normalised linear head (GenD ``LinearNorm``)."""

    def __init__(self, vision_config: dict | None = None) -> None:
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        if vision_config is None:
            vision_config = json.loads((GEND_DIR / "clip" / "config.json").read_text())["vision_config"]
        cfg = CLIPVisionConfig(**{k: v for k, v in vision_config.items() if k != "model_type"})
        self.vision = CLIPVisionModel(cfg)
        self.linear = nn.Linear(cfg.hidden_size, 2)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def features(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x01``: float (B,3,224,224) in [0,1], RGB. Returns ``(logits (B,2), l2 embeddings (B,1024))`` (GenD training)."""
        feats = self.vision(pixel_values=(x01 - self.mean) / self.std).pooler_output
        z = F.normalize(feats, p=2, dim=1)
        return self.linear(z), z

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """``x01``: float (B,3,224,224) in [0,1], RGB. Returns the fake logit (B,)."""
        l, _ = self.features(x01)
        return l[:, 1] - l[:, 0]


def load_gend(device: str = "cpu", weights: str | Path | None = None) -> GendCLIP:
    """Build the model and load the public checkpoint strictly (key sets must match exactly)."""
    from safetensors.torch import load_file

    sd = load_file(str(weights or GEND_DIR / "hf" / "model.safetensors"))
    out = {}
    for k, v in sd.items():
        if k.startswith("feature_extractor.vision_model."):
            out["vision.vision_model." + k[len("feature_extractor.vision_model."):]] = v
        elif k.startswith("model.linear."):
            out["linear." + k[len("model.linear."):]] = v
        elif k == "feature_extractor.visual_projection.weight":
            continue  # unused by GenD's forward (pooler_output is used before the projection)
        else:
            raise KeyError(k)
    m = GendCLIP()
    missing, unexpected = m.load_state_dict(out, strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return m.to(device).eval()
