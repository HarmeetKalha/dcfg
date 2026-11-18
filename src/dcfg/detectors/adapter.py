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

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

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
        self._bn_only = False
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

    def forward_with_features(self, x_normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(logit (B,), feat (B, D))``: the logit of :meth:`forward` and the penultimate pooled features.

        Xception / F3Net: the 2048-d vector after ReLU + global average pooling, i.e. exactly the input of
        ``last_linear`` (DFB ``backbone.last_emb``); EfficientNet-B4: its 1792-d pooled ``last_emb`` (input of
        ``last_layer``; dropout is off in the DFB config). Other detectors (UCF): DFB's ``feat`` output, global-average
        pooled if 4-D (a generic fallback; check that it is the right layer before relying on it).
        Features keep the autocast dtype; callers cast to fp32."""
        out = self.dfb_forward(x_normalized)
        cls = out["cls"].float()
        bb = getattr(self.model, "backbone", None)
        if self.name in ("xception", "f3net", "efficientnetb4") and hasattr(bb, "last_emb"):
            feat = bb.last_emb
        else:
            feat = out["feat"]
            if feat.dim() == 4:
                feat = feat.mean(dim=(2, 3))
        return cls[:, 1] - cls[:, 0], feat

    def film_sites(self) -> dict[str, tuple[nn.Module, int]]:
        """``{site: (module whose OUTPUT is modulated, channels)}`` for FiLM (see ``dcfg.models.film``).

        * xception / f3net (F3Net = ONE Xception on the 12-channel FAD input, ``model.backbone``): ``entry`` =
          ``block3`` (728 ch, end of the entry flow), ``exit`` = ``bn4`` (2048 ch, pre-ReLU/pool), plus the
          design-selection sites ``mid4`` = ``block7``, ``mid8`` = ``block11`` (728 ch).
        * efficientnetb4: ``entry`` = ``efficientnet._blocks[9]`` (56 ch, end of MBConv stage 3 = DFB's own
          ``block_part1`` boundary), ``exit`` = ``efficientnet._bn1`` (1792 ch, BN after ``_conv_head``, before the
          swish and the global pooling; the analogue of Xception's bn4).
        * ucf: NOT DEFINED (ambiguous; see docs/DECISIONS.md), raises ``NotImplementedError``.

        Channels are read from the actual modules (last BatchNorm of a block)."""
        if self.name in ("xception", "f3net"):
            bb = self.model.backbone
            mods = {"entry": bb.block3, "mid4": bb.block7, "mid8": bb.block11, "exit": bb.bn4}
        elif self.name == "efficientnetb4":
            eff = self.model.backbone.efficientnet
            mods = {"entry": eff._blocks[9], "exit": eff._bn1}
        else:
            raise NotImplementedError(f"FiLM sites are not defined for {self.name!r}")

        def chans(m: nn.Module) -> int:
            if isinstance(m, nn.BatchNorm2d):
                return int(m.num_features)
            return int([x for x in m.modules() if isinstance(x, nn.BatchNorm2d)][-1].num_features)

        return {k: (m, chans(m)) for k, m in mods.items()}

    def bn_modules(self) -> dict[str, nn.modules.batchnorm._BatchNorm]:
        """All BatchNorm layers of the DFB model by module name (BN-only fine-tuning)."""
        return {n: m for n, m in self.model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}

    def bn_state_dict(self) -> dict[str, torch.Tensor]:
        """Full BN state (weight, bias, running_mean, running_var, num_batches_tracked), keys as in
        ``model.state_dict()``, cloned to CPU."""
        out: dict[str, torch.Tensor] = {}
        for n, m in self.bn_modules().items():
            for k, v in m.state_dict().items():
                out[f"{n}.{k}"] = v.detach().cpu().clone()
        return out

    def load_bn_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Apply a BN state onto this (stock) detector; the key set must match exactly."""
        want = set(self.bn_state_dict())
        if set(sd) != want:
            raise KeyError(f"BN checkpoint keys mismatch: missing={sorted(want - set(sd))[:3]}, "
                           f"unexpected={sorted(set(sd) - want)[:3]}")
        res = self.model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys, res.unexpected_keys

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
        self._frozen, self._bn_only = True, False
        self.eval()
        return self

    def freeze_except_bn(self) -> int:
        """BN-only fine-tuning (variant ``bnft``): every parameter frozen except the BatchNorm affine
        weights/biases; in ``train()`` mode ONLY the BN layers are in train mode (batch statistics, running
        stats update), every other module stays in eval mode. Returns the number of trainable parameters."""
        for p in self.parameters():
            p.requires_grad_(False)
        n = 0
        for m in self.bn_modules().values():
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad_(True)
                    n += p.numel()
        self._frozen, self._bn_only = False, True
        self.train(self.training)
        return n

    def train(self, mode: bool = True) -> "Detector":
        """Like ``nn.Module.train`` but a frozen detector always stays in eval mode, and a BN-only detector
        keeps everything except its BatchNorm layers in eval mode."""
        nn.Module.train(self, False if getattr(self, "_frozen", False) else mode)
        if getattr(self, "_bn_only", False):
            for m in self.modules():
                if not isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.training = False
        return self


class FilmHooks:
    """Forward hooks applying ``h <- h * (1 + d_gamma) + beta`` (per channel) at the detector's FiLM sites.

    The hooks hold no parameters and never touch the detector's: they read ``(d_gamma, beta)`` from
    ``self.params``, set only inside :meth:`conditioned`. Outside it (or inside :meth:`disabled`) they pass
    the activations through untouched, so a frozen eval-mode detector stays exactly the plain detector
    (used for the consistency teacher). (1 + d_gamma) is formed in fp32 and cast to the activation dtype;
    d_gamma = beta = 0 gives exactly ``h``. Gradients flow through the frozen detector to d_gamma / beta."""

    def __init__(self, detector: Any, sites: Sequence[str]) -> None:
        avail = detector.film_sites()
        self.params: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.enabled = True
        self.channels = {s: avail[s][1] for s in sites}
        self.handles = [avail[s][0].register_forward_hook(self._make(s, avail[s][1])) for s in sites]

    def _make(self, site: str, channels: int):
        def hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor | None:
            if not self.enabled or self.params is None:
                return None
            dg, b = self.params[site]
            assert out.shape[1] == channels and dg.shape == (out.shape[0], channels), \
                f"FiLM site {site}: activation {tuple(out.shape)} vs params {tuple(dg.shape)}"
            scale = (1.0 + dg.float()).to(out.dtype)[:, :, None, None]
            return out * scale + b.to(out.dtype)[:, :, None, None]
        return hook

    @contextmanager
    def conditioned(self, params: dict[str, tuple[torch.Tensor, torch.Tensor]] | None) -> Iterator[None]:
        """Apply FiLM with ``params`` for detector calls inside the block (None = plain detector)."""
        prev = self.params
        self.params = params
        try:
            yield
        finally:
            self.params = prev

    @contextmanager
    def disabled(self) -> Iterator[None]:
        """Hooks are pass-through inside the block, whatever ``params`` are set (teacher pass)."""
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
