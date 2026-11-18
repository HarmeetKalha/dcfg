"""Sensor-conditioned FiLM parameters for a frozen detector (design-selection variants).

``FiLMBank`` is an MLP ``embedding(128) -> hidden -> GELU -> sum_sites 2*C_site`` whose LAST layer is
zero-initialised (weight and bias), so every (delta-gamma, beta) is exactly 0 at init and the modulated
detector equals the plain one. The modulation itself, ``h <- h * (1 + d_gamma) + beta`` per channel
(broadcast over H, W), is applied by forward hooks (``dcfg.detectors.adapter.FilmHooks``).

Sites (Xception, DeepfakeBench module names; the DFB middle flow is block4..block11, so the user's
"4th and 8th middle-flow block" are block7 and block11):

* ``entry``: output of DFB ``block3`` (end of the entry flow), 728 ch
* ``mid4``: output of ``block7`` (4th middle-flow block), 728 ch
* ``mid8``: output of ``block11`` (8th middle-flow block), 728 ch
* ``exit``: output of ``bn4`` (last conv+BN before the ReLU / global pooling), 2048 ch

LUMEN (the final method, 2 sites ``entry`` + ``exit`` = kind ``filmlite``) uses per-detector sites
(``Detector.film_sites``: Xception / F3Net block3 + bn4; EfficientNet-B4 ``_blocks[9]`` + ``_bn1``) and a
per-detector MLP hidden width (``LUMEN_HIDDEN``) chosen so that gate + sensor + FiLM add ~3.5% of the
detector's parameters (:func:`choose_hidden`).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

#: kind -> sites and MLP hidden width
FILM_KINDS: dict[str, dict] = {
    "film": {"sites": ("entry", "mid4", "mid8", "exit"), "hidden": 256},
    "filmlite": {"sites": ("entry", "exit"), "hidden": 64},
}
FilmParams = dict[str, tuple[Tensor, Tensor]]
#: LUMEN FiLM MLP hidden width per detector (see :func:`choose_hidden`; Xception = the dcfg_filmlite pilot).
LUMEN_HIDDEN: dict[str, int] = {"xception": 64, "efficientnetb4": 56, "f3net": 72}
#: gate (+LayerNorms) + sensor parameters of the dcfg FrontEnd (fixed; independent of the detector)
GATE_SENSOR_PARAMS = 394_369


def film_param_count(channels: list[int], hidden: int, embed_dim: int = 128) -> int:
    """Parameters of ``FiLMBank``: Linear(embed, hidden) + Linear(hidden, 2 * sum(channels))."""
    s = 2 * sum(channels)
    return embed_dim * hidden + hidden + hidden * s + s


def choose_hidden(n_detector_params: int, channels: list[int], target_pct: float = 3.5, step: int = 8) -> int:
    """Hidden width (multiple of ``step``) whose total added trainable params (gate + sensor + FiLM) is
    closest to ``target_pct`` % of the detector's parameters."""
    best = min(range(step, 1025, step), key=lambda h: abs(
        100.0 * (GATE_SENSOR_PARAMS + film_param_count(channels, h)) / n_detector_params - target_pct))
    return best


class FiLMBank(nn.Module):
    """MLP on the sensor embedding -> per-site ``(delta_gamma, beta)``, each ``(B, C_site)``."""

    def __init__(self, kind: str, channels: dict[str, int], embed_dim: int = 128,
                 hidden: int | None = None) -> None:
        super().__init__()
        if kind not in FILM_KINDS:
            raise ValueError(f"film kind must be one of {tuple(FILM_KINDS)}")
        self.kind = kind
        self.sites: tuple[str, ...] = tuple(FILM_KINDS[kind]["sites"])
        self.channels = {s: int(channels[s]) for s in self.sites}
        total = sum(self.channels.values())
        self.hidden = int(hidden if hidden is not None else FILM_KINDS[kind]["hidden"])
        self.fc1 = nn.Linear(embed_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 2 * total)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, embedding: Tensor) -> FilmParams:
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            z = self.fc2(nn.functional.gelu(self.fc1(embedding.float())))
        out: FilmParams = {}
        off = 0
        for s in self.sites:
            c = self.channels[s]
            out[s] = (z[:, off:off + c], z[:, off + c:off + 2 * c])
            off += 2 * c
        return out

    def meta(self) -> dict:
        """What ``load_frontend`` needs to rebuild the bank."""
        return {"kind": self.kind, "channels": dict(self.channels), "hidden": self.hidden}
