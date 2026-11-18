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


class FiLMBank(nn.Module):
    """MLP on the sensor embedding -> per-site ``(delta_gamma, beta)``, each ``(B, C_site)``."""

    def __init__(self, kind: str, channels: dict[str, int], embed_dim: int = 128) -> None:
        super().__init__()
        if kind not in FILM_KINDS:
            raise ValueError(f"film kind must be one of {tuple(FILM_KINDS)}")
        self.kind = kind
        self.sites: tuple[str, ...] = tuple(FILM_KINDS[kind]["sites"])
        self.channels = {s: int(channels[s]) for s in self.sites}
        total = sum(self.channels.values())
        self.fc1 = nn.Linear(embed_dim, int(FILM_KINDS[kind]["hidden"]))
        self.fc2 = nn.Linear(int(FILM_KINDS[kind]["hidden"]), 2 * total)
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
        return {"kind": self.kind, "channels": dict(self.channels)}
