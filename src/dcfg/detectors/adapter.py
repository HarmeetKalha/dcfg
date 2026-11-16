"""Uniform wrapper around DeepfakeBench detectors.

Contract
--------
* Input to :meth:`Detector.normalize` is an RGB float tensor in [0, 1], shape (B, 3, H, W).
* :meth:`Detector.forward` takes the *normalized* tensor and returns a ``(B,)`` logit
  ``l = cls[:, 1] - cls[:, 0]`` where ``cls`` is DFB's 2-class output ('cls'; class 1 = fake).
  Hence ``sigmoid(l) == softmax(cls, 1)[:, 1]``, which is exactly DFB's ``prob`` output
  (for UCF, whose inference branch returns no ``prob`` key, it equals the value DFB's own
  test loop collects in ``model.prob``). The logit is returned in fp32 even under autocast.
* Resolution: DFB's test pipeline always ``cv2.resize``s frames to ``config['resolution']``
  (256 for all four detectors, bicubic). Our crops are already 256 so this is a no-op; if the
  input differs, the adapter resizes bicubically to the config resolution. Gradients flow
  through the resize.
* Extra inputs (UCF needs ``data_dict['label']`` at inference) are supplied internally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dcfg.detectors.registry import (default_weights_path, get_spec, load_config,
                                     load_dfb)


class Detector(nn.Module):
    """A DFB detector with strict checkpoint loading and a differentiable ``(B,)`` logit."""

    def __init__(self, name: str, weights_path: str | Path | None = None,
                 device: str | torch.device = "cpu", load_weights: bool = True) -> None:
        super().__init__()
        self.name = name
        self.spec = get_spec(name)
        self.config: dict[str, Any] = load_config(name)
        self.resolution: int = int(self.config["resolution"])
        self.register_buffer("mean", torch.tensor(self.config["mean"], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.config["std"], dtype=torch.float32).view(1, 3, 1, 1))
        registry = load_dfb()
        self.model: nn.Module = registry[self.config["model_name"]](self.config)
        self.weights_path = Path(weights_path) if weights_path else default_weights_path(name)
        self.load_report: dict[str, Any] = {}
        if load_weights:
            self.load_checkpoint(self.weights_path)
        self._frozen = False
        self.to(device)
        self.eval()

    # ------------------------------------------------------------------ loading
    def load_checkpoint(self, path: Path) -> None:
        """Load a DFB checkpoint with strict key matching (zero missing / unexpected keys)."""
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        stripped = False
        if sd and all(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
            stripped = True
        res = self.model.load_state_dict(sd, strict=False)
        self.load_report = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys),
                            "module_prefix_stripped": stripped, "n_keys": len(sd)}
        assert not res.missing_keys, f"{self.name}: missing keys {res.missing_keys[:5]}"
        assert not res.unexpected_keys, f"{self.name}: unexpected keys {res.unexpected_keys[:5]}"

    # ------------------------------------------------------------------ inference
    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        """Map RGB [0, 1] to the detector's own normalization ``(x - mean) / std``."""
        return (x01 - self.mean) / self.std

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.resolution, self.resolution):
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bicubic",
                              align_corners=False)
        return x

    def dfb_forward(self, x_normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run DFB's own forward in inference mode and return its raw prediction dict."""
        x = self._prep(x_normalized)
        data = {"image": x, "label": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)}
        out = self.model(data, inference=True)
        # UCF's inference branch appends to python lists (and syncs); keep them from growing.
        for attr in ("prob", "label"):
            buf = getattr(self.model, attr, None)
            if isinstance(buf, list) and self.name == "ucf":
                buf.clear()
        return out

    def forward(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Return the fake-vs-real logit, shape (B,), fp32 (see module docstring)."""
        cls = self.dfb_forward(x_normalized)["cls"].float()
        return cls[:, 1] - cls[:, 0]

    @torch.no_grad()
    def predict_prob(self, x01: torch.Tensor) -> torch.Tensor:
        """Convenience: P(fake) for an RGB [0, 1] batch."""
        return torch.sigmoid(self.forward(self.normalize(x01)))

    # ------------------------------------------------------------------ freezing
    def freeze(self) -> "Detector":
        """No parameter requires grad; eval mode (BN/dropout) is kept even if ``train()`` is called.

        Gradients still flow to the *input* through the frozen network.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        self.eval()
        return self

    def train(self, mode: bool = True) -> "Detector":
        """Like ``nn.Module.train`` but a frozen detector always stays in eval mode."""
        return super().train(False if getattr(self, "_frozen", False) else mode)
