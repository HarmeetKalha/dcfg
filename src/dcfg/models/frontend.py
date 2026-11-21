"""Pluggable front-end: RGB -> YCbCr -> block DCT -> per-coefficient weights -> IDCT -> RGB."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .dct import BlockDCT, rgb_to_ycbcr, ycbcr_to_rgb
from .film import FiLMBank
from .fixed import FixedGate
from .gate import FrequencyGate, log_energy
from .sensor import DegradationSensor

GATE_KINDS = ("dcfg", "fixed", "nosensor", "identity")


class IdentityGate(nn.Module):
    """Parameter-free stand-in for the DCT gate: weights are exactly 1 and the image is not touched."""

    def forward(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        return torch.ones(batch_size, 3, 8, 8, device=device)


class FrontEnd(nn.Module):
    """Frequency front-end placed before a frozen detector.

    ``gate_kind``: ``dcfg`` (sensor embedding + DCT energy stats), ``nosensor``
    (stats only), ``fixed`` (image-independent weights). ``dcfg_nolive`` is simply
    ``dcfg`` with a randomly initialised sensor (pass an un-loaded ``sensor``).

    Gradient policy: the DCT energy statistics fed to the gate are NOT detached.
    They are a smooth function of the input, the whole module runs under one
    objective, and detaching would only remove a legitimate (if small) gradient
    path; nothing is unstable about it since stats go through log(.+1e-6).
    If ``freeze_sensor`` the sensor has requires_grad=False and stays in eval mode
    (BatchNorm running stats fixed); gradients still flow *through* it to the input.

    ``sensor_amp`` (default True) lets the sensor CNN use fp16 autocast on CUDA only.
    The rest of the module runs in fp32 with autocast disabled and returns the input dtype.
    ``forward(x01) -> (y01, w)`` with w of shape (B,3,8,8).

    ``identity`` (variant ``lumen_nogate``): NO gate at all. Sensor + FiLM bank like ``dcfg``, but the DCT path is
    skipped entirely (no DCT, no gate module with parameters, no energy statistics): ``y01 = x01`` and the returned
    ``w`` is a constant tensor of ones (logging only, no grad). ``self.gate`` is a parameter-free :class:`IdentityGate`.

    Optional ``film`` (a :class:`FiLMBank`, dcfg gate only): maps the sensor embedding to per-site FiLM
    parameters for the detector. ``forward_full(x01) -> (y01, w, film_params|None)`` also returns them
    (the detector-side hooks that apply them live in ``dcfg.detectors.adapter.FilmHooks``).
    """

    def __init__(self, gate_kind: str = "dcfg", sensor: DegradationSensor | None = None,
                 freeze_sensor: bool = False, sensor_amp: bool = True,
                 film: FiLMBank | None = None) -> None:
        super().__init__()
        if gate_kind not in GATE_KINDS:
            raise ValueError(f"gate_kind must be one of {GATE_KINDS}")
        self.gate_kind = gate_kind
        self.sensor_amp = sensor_amp
        if film is not None and gate_kind not in ("dcfg", "identity"):
            raise ValueError("FiLM is conditioned on the sensor embedding: needs gate_kind='dcfg' or 'identity'")
        if gate_kind == "identity" and film is None:
            raise ValueError("gate_kind='identity' (lumen_nogate) is only meaningful with a FiLM bank")
        self.film = film
        self.dct = BlockDCT()
        self.sensor: DegradationSensor | None = None
        self.gate: nn.Module
        if gate_kind in ("dcfg", "identity"):
            self.sensor = sensor if sensor is not None else DegradationSensor()
            self.gate = FrequencyGate(use_sensor=True) if gate_kind == "dcfg" else IdentityGate()
        elif gate_kind == "nosensor":
            self.gate = FrequencyGate(use_sensor=False)
        else:
            self.gate = FixedGate()
        self.freeze_sensor = freeze_sensor
        if freeze_sensor and self.sensor is not None:
            for p in self.sensor.parameters():
                p.requires_grad_(False)
            self.sensor.eval()

    def train(self, mode: bool = True) -> "FrontEnd":
        super().train(mode)
        if self.freeze_sensor and self.sensor is not None:
            self.sensor.eval()
        return self

    def _embed(self, x: Tensor) -> Tensor:
        """Sensor embedding (fp32 out). On CUDA the sensor CNN alone runs in fp16 autocast
        (fp32 at 256x256 is only ~1.9k img/s on a 3060); DCT and gate stay fp32 (PLAN 7.5)."""
        assert self.sensor is not None
        if self.sensor_amp and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.sensor.embed(x).float()
        return self.sensor.embed(x)

    def forward(self, x01: Tensor) -> tuple[Tensor, Tensor]:
        y, w, _ = self.forward_full(x01)
        return y, w

    def forward_full(self, x01: Tensor) -> tuple[Tensor, Tensor, dict[str, tuple[Tensor, Tensor]] | None]:
        in_dtype = x01.dtype
        if self.gate_kind == "identity":
            with torch.autocast(device_type=x01.device.type, enabled=False):
                x = x01.float()
                emb = self._embed(x)
                film_params = self.film(emb) if self.film is not None else None
                return x01, self.gate(x.shape[0], x.device), film_params
        with torch.autocast(device_type=x01.device.type, enabled=False):
            x = x01.float()
            coef = self.dct(rgb_to_ycbcr(x))
            film_params = None
            if self.gate_kind == "dcfg":
                assert self.sensor is not None
                emb = self._embed(x)
                w = self.gate(log_energy(coef), emb)
                if self.film is not None:
                    film_params = self.film(emb)
            elif self.gate_kind == "nosensor":
                w = self.gate(log_energy(coef))
            else:
                w = self.gate(x.shape[0])
            y = ycbcr_to_rgb(self.dct.inverse(coef * w[:, :, None, None]))
            return y.clamp(0, 1).to(in_dtype), w, film_params
