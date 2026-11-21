"""LUMEN2 = GenD (CLIP ViT-L/14, LayerNorm-tuned) + the LUMEN parts, the ONLY difference from GenD-aug.

Parts (all start as the identity, so step 0 == GenD):

* **damage sensor** (:class:`dcfg.degrade.sensor.DamageSensor`, from scratch, fp32). Sees the DAMAGED 256 px crop before GenD's
  resize; its 128-d embedding ``s`` conditions the two parts below. Trained jointly with ``0.1 * damage_loss`` on the
  ground-truth degradation labels (``dcfg.degrade.labels``).
* **DCT gate** (:class:`dcfg.models.gate.FrequencyGate`, as LUMEN1): 8x8 blockwise DCT of the 224 px RESIZED image in YCbCr, 192
  per-coefficient weights ``w = 2 sigmoid(z)`` (zero-initialised last layer -> ``w = 1``), driven by the 192 DCT log-energy
  statistics of that image and ``s`` (separate LayerNorms, then concat). Output clamped to [0,1]; applied after damage and resize,
  before CLIP mean/std. fp32. Penalty ``1e-4 * mean((w-1)^2)`` is added by the trainer.
* **conditional LayerNorm** for EVERY LayerNorm of CLIP's vision encoder (50 of them: pre, 2 per block x 24, post):
  ``y = LN(x) * (gamma + dgamma) + (beta + dbeta)``; ``dgamma, dbeta`` are functions of ``s`` shared across the tokens of an
  image. Low-rank generator: ``a = MLP(LayerNorm(s)) in R^8``, ``dgamma_l = U_l a``, ``dbeta_l = V_l a``; ``U, V`` zero-initialised.

Input form. CLIP needs the PIL-bicubic 224 resize of the damaged crop (not reproducible on the GPU bit-for-bit) while the sensor
needs the 256 crop, so the data pipeline packs both views of an image into ONE uint8 array ``(480, 256, 3)``
(:func:`pack_views`: rows 0..255 = damaged 256 crop; rows 256..479, cols 0..223 = ``gend_resize`` of it, the rest zero). The model
takes the packed float tensor ``(B, 3, 480, 256)`` in [0, 1] and returns the fake logit ``l[1] - l[0]`` (``forward``) or all
training tensors (``forward_train``). State-dict key names of the GenD part are unchanged (``base.vision...``, ``base.linear``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from dcfg.degrade.sensor import EMBED_DIM, DamageSensor
from dcfg.gend.crop import CROP_SIZE, INPUT_SIZE, gend_resize
from dcfg.gend.model import GendCLIP, load_gend
from dcfg.models.dct import BlockDCT, rgb_to_ycbcr, ycbcr_to_rgb
from dcfg.models.gate import FrequencyGate, log_energy

RANK = 8
GEN_HIDDEN = 128
PACK_H = CROP_SIZE + INPUT_SIZE  # 480
PACK_W = CROP_SIZE  # 256
GATE_PENALTY = 1e-4
GENDTRAIN_PARAMS = 104_450  # GenD-aug trainable set: 102,400 LayerNorm + 2,050 head
LN_KEYS = ("layer_norm", "layrnorm", "post_layernorm")


# ------------------------------------------------------------------------------------------------ input packing
def pack_views(img256: np.ndarray) -> np.ndarray:
    """uint8 256x256x3 damaged crop -> uint8 (480, 256, 3): the crop on top, its GenD 224 resize below (cols 0..223)."""
    assert img256.shape == (CROP_SIZE, CROP_SIZE, 3) and img256.dtype == np.uint8, (img256.shape, img256.dtype)
    out = np.zeros((PACK_H, PACK_W, 3), np.uint8)
    out[:CROP_SIZE] = img256
    out[CROP_SIZE:, :INPUT_SIZE] = gend_resize(img256)
    return out


def unpack_views(x: Tensor) -> tuple[Tensor, Tensor]:
    """Packed ``(B,3,480,256)`` -> ``(crop256 (B,3,256,256), resized224 (B,3,224,224))``."""
    return x[:, :, :CROP_SIZE, :], x[:, :, CROP_SIZE:, :INPUT_SIZE]


# ------------------------------------------------------------------------------------------------ conditional LayerNorm
class CondLayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` (same parameter names ``weight``/``bias``) that can be conditioned per image.

    Without condition it IS the plain LayerNorm. With ``set_cond(dgamma, dbeta)`` (each ``(B, D)``, fp32) it computes
    ``norm(x) * (weight + dgamma) + (bias + dbeta)`` with the deltas broadcast over all token positions of the same image.
    The input is normalised in fp32 (autocast would do the same for ``layer_norm``) and the result stays fp32.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5) -> None:
        super().__init__(normalized_shape, eps=eps)
        self._cond: tuple[Tensor, Tensor] | None = None

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm) -> "CondLayerNorm":
        new = cls(ln.normalized_shape, ln.eps)
        new.load_state_dict(ln.state_dict())
        return new.to(ln.weight.device, ln.weight.dtype)

    def set_cond(self, cond: tuple[Tensor, Tensor] | None) -> None:
        self._cond = cond

    def forward(self, x: Tensor) -> Tensor:
        if self._cond is None:
            return super().forward(x)
        dg, db = self._cond
        if torch.is_grad_enabled():  # recompute in backward: saves ~1.7 GB of fp32 activations over the 49 sequence-wide LayerNorms
            return checkpoint(self._modulate, x, dg, db, use_reentrant=False)
        return self._modulate(x, dg, db)

    def _modulate(self, x: Tensor, dg: Tensor, db: Tensor) -> Tensor:
        xh = F.layer_norm(x.float(), self.normalized_shape, None, None, self.eps)
        shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        return xh * (self.weight.float() + dg).view(shape) + (self.bias.float() + db).view(shape)


def convert_layernorms(module: nn.Module) -> list[CondLayerNorm]:
    """Replace every ``nn.LayerNorm`` below ``module`` by a :class:`CondLayerNorm` (same values); return them in module order."""
    out: list[CondLayerNorm] = []
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.LayerNorm) and not isinstance(child, CondLayerNorm):
                new = CondLayerNorm.from_layernorm(child)
                setattr(parent, name, new)
    for m in module.modules():
        if isinstance(m, CondLayerNorm):
            out.append(m)
    return out


class LnGenerator(nn.Module):
    """``s (B,128) -> (dgamma, dbeta)`` each ``(B, L, D)``: ``a = MLP(LN(s))`` (rank R), ``dgamma_l = U_l a``, ``dbeta_l = V_l a``.

    ``U, V`` are zero-initialised: ``dgamma = dbeta = 0`` at init for any ``s``. MLP: LN(128) -> Linear(128,128) -> GELU ->
    Linear(128,R). Runs in fp32 (autocast disabled).
    """

    def __init__(self, n_ln: int, dim: int, embed_dim: int = EMBED_DIM, hidden: int = GEN_HIDDEN, rank: int = RANK) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, rank)
        self.U = nn.Parameter(torch.zeros(n_ln, dim, rank))
        self.V = nn.Parameter(torch.zeros(n_ln, dim, rank))

    def forward(self, s: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=s.device.type, enabled=False):
            a = self.fc2(F.gelu(self.fc1(self.ln(s.float()))))
            return torch.einsum("bk,ldk->bld", a, self.U), torch.einsum("bk,ldk->bld", a, self.V)


# ------------------------------------------------------------------------------------------------ the model
class Lumen2Model(nn.Module):
    """GenD + sensor + DCT gate + conditional LayerNorm. See the module docstring."""

    def __init__(self, base: GendCLIP, sensor: DamageSensor | None = None) -> None:
        super().__init__()
        self.base = base
        self.cond_lns = convert_layernorms(base.vision)  # list of modules living inside ``base`` (not re-registered)
        dim = self.cond_lns[0].normalized_shape[0]
        assert all(m.normalized_shape == (dim,) for m in self.cond_lns)
        self.sensor = sensor if sensor is not None else DamageSensor()
        self.dct = BlockDCT()
        self.gate = FrequencyGate(use_sensor=True, embed_dim=EMBED_DIM)
        self.gen = LnGenerator(len(self.cond_lns), dim)

    # -- parameter bookkeeping
    @staticmethod
    def is_gend_name(name: str) -> bool:
        """GenD-aug's trainable set, in this model's names: LayerNorms of the vision tower + the head."""
        if not name.startswith("base."):
            return False
        n = name[len("base."):]
        return any(k in n for k in LN_KEYS) or n.startswith("linear.")

    @classmethod
    def is_trainable_name(cls, name: str) -> bool:
        return cls.is_gend_name(name) or name.startswith(("sensor.", "gate.", "gen."))

    def part_param_counts(self) -> dict[str, int]:
        n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        gend = sum(p.numel() for k, p in self.named_parameters() if self.is_gend_name(k))
        return {"gend_trainable (LayerNorms + head, = GenD-aug)": gend, "sensor": n(self.sensor), "gate (MLP + 2 LN)": n(self.gate),
                "ln_generator (MLP + U,V)": n(self.gen), "LUMEN parts total": n(self.sensor) + n(self.gate) + n(self.gen)}

    # -- forward
    def _condition(self, s: Tensor | None) -> None:
        if s is None:
            for m in self.cond_lns:
                m.set_cond(None)
            return
        dg, db = self.gen(s)
        for i, m in enumerate(self.cond_lns):
            m.set_cond((dg[:, i], db[:, i]))

    def forward_train(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
        """Packed ``x (B,3,480,256)`` in [0,1] -> ``(logits (B,2), L2 embedding z (B,1024), sensor outputs, gate weights w (B,3,8,8))``.

        Call under the outer autocast (CLIP in bf16/fp16); the LUMEN parts disable autocast themselves."""
        crop, img = unpack_views(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            sens = self.sensor(crop.float())  # embedding, presence logits, strength: all fp32
            s = sens["embedding"]
            coef = self.dct(rgb_to_ycbcr(img.float()))
            w = self.gate(log_energy(coef), s)
            y = ycbcr_to_rgb(self.dct.inverse(coef * w[:, :, None, None])).clamp(0, 1)
            self._condition(s)
        try:
            logits, z = self.base.features(y)
        finally:
            self._condition(None)
        return logits, z, sens, w

    def forward(self, x: Tensor) -> Tensor:
        """Fake logit ``l[1] - l[0]`` (B,)."""
        logits = self.forward_train(x)[0]
        return logits[:, 1] - logits[:, 0]

    def predict_with_sensor(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        logits, _, sens, w = self.forward_train(x)
        return logits[:, 1] - logits[:, 0], sens, w


def build_lumen2(device: str = "cpu", base: GendCLIP | None = None) -> Lumen2Model:
    """Public GenD weights + fresh LUMEN parts (identity at init)."""
    return Lumen2Model(base if base is not None else load_gend("cpu")).to(device)


# ------------------------------------------------------------------------------------------------ checkpoints
def trainable_state(model: Lumen2Model) -> dict[str, Tensor]:
    """CPU clones of everything that is trained or learned: GenD's LayerNorms + head, sensor (incl. BatchNorm running
    statistics), gate, LN generator. The frozen CLIP weights are NOT included (they come from the public GenD weights)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if Lumen2Model.is_trainable_name(k)}


def load_trainable(model: Lumen2Model, sd: dict[str, Tensor]) -> None:
    """Strictly load a :func:`trainable_state` dict (key set must match exactly)."""
    want = {k for k in model.state_dict() if Lumen2Model.is_trainable_name(k)}
    if set(sd) != want:
        raise KeyError(f"trainable keys differ: missing {sorted(want - set(sd))[:3]}, unexpected {sorted(set(sd) - want)[:3]}")
    own = model.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
