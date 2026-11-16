"""Differentiable fp32 colour transforms and 8x8 blockwise orthonormal DCT-II.

Layout: an image batch ``(B, C, H, W)`` maps to blocks ``(B, C, H/8, W/8, 8, 8)``
where the last two axes are the frequency indices ``(u, v)`` (u vertical, v
horizontal). Each block is transformed as ``C X C^T`` with a fixed orthonormal
basis ``C``. All ops run in float32 with autocast disabled.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

BLOCK = 8


def _fp32(x: Tensor) -> Tensor:
    return x.float()


def rgb_to_ycbcr(x: Tensor) -> Tensor:
    """RGB in [0,1], (B,3,H,W) -> YCbCr (JPEG/BT.601 full range), chroma offset 0.5."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = _fp32(x)
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.stack((y, cb, cr), dim=1)


def ycbcr_to_rgb(x: Tensor) -> Tensor:
    """Inverse of :func:`rgb_to_ycbcr` (no clamping)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = _fp32(x)
        y, cb, cr = x[:, 0], x[:, 1] - 0.5, x[:, 2] - 0.5
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.stack((r, g, b), dim=1)


def dct_basis(n: int = BLOCK) -> Tensor:
    """Orthonormal DCT-II matrix ``C`` (n x n): ``C[k, i] = a_k cos((2i+1)k pi / 2n)``."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    c = torch.cos((2 * i + 1) * k * math.pi / (2 * n)) * math.sqrt(2.0 / n)
    c[0] /= math.sqrt(2.0)
    return c.float()


class BlockDCT(nn.Module):
    """Blockwise 8x8 orthonormal DCT-II / inverse, batched, differentiable, fp32."""

    C: Tensor

    def __init__(self) -> None:
        super().__init__()
        c = dct_basis(BLOCK)
        self.register_buffer("C", c, persistent=False)
        # 64 fixed 8x8 kernels, K[u*8+v, 0, i, j] = C[u,i] C[v,j]; equivalent to C X C^T.
        kern = (c[:, None, :, None] * c[None, :, None, :]).reshape(64, 1, 8, 8)
        self.register_buffer("K", kern, persistent=False)

    @staticmethod
    def _check(h: int, w: int) -> None:
        assert h % BLOCK == 0 and w % BLOCK == 0, f"H,W must be divisible by 8, got {h}x{w}"

    def forward(self, x: Tensor) -> Tensor:
        """(B,C,H,W) -> coefficients (B,C,H/8,W/8,8,8) (a permuted view of a conv output)."""
        b, c, h, w = x.shape
        self._check(h, w)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = _fp32(x)
            out = F.conv2d(x.reshape(b * c, 1, h, w), self.K, stride=8)  # (BC,64,Hb,Wb)
            return out.reshape(b, c, 8, 8, h // 8, w // 8).permute(0, 1, 4, 5, 2, 3)

    def inverse(self, coef: Tensor) -> Tensor:
        """Coefficients (B,C,H/8,W/8,8,8) -> (B,C,H,W)."""
        b, c, hb, wb, _, _ = coef.shape
        with torch.autocast(device_type=coef.device.type, enabled=False):
            coef = _fp32(coef).permute(0, 1, 4, 5, 2, 3).reshape(b * c, 64, hb, wb)
            return F.conv_transpose2d(coef, self.K, stride=8).reshape(b, c, hb * 8, wb * 8)
