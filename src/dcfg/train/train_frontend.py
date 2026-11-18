"""Train the DCFG front-end (or fine-tune a detector) on FF++ c23 (PLAN 7.2, 7.4, 7.5, Phase 8.1).

    python -m dcfg.train.train_frontend --config configs/train/dcfg_xception.yaml [--set key=value ...]

Variants: ``dcfg`` (LIVE-pretrained sensor + gate, frozen detector), ``fixed``, ``nosensor``,
``dcfg_nolive`` (random sensor), ``augft`` (no module, detector fine-tuned on the degraded data),
``dcfg_jointft`` (module + detector fine-tuned), ``bnft`` (no module; ONLY the detector's BatchNorm affine params
train, BN in train mode, rest frozen; BCE), the LUMEN family (``lumen`` = ``dcfg_filmlite``, ``lumen_nocons``,
``lumen_nolive``, ``lumen_frozensensor``; see below) and the design-selection variants (frozen detector, all use
the consistency loss of ``dcfg.train.consistency``): ``dcfg_cons`` (dcfg + consistency), ``dcfg_film`` (+ FiLM at
entry/block7/block11/exit of Xception, MLP 128-256), ``dcfg_filmlite`` (FiLM at entry/exit only, MLP 128-64).
LUMEN = LIVE sensor + DCFG gate + 2-site FiLM (``Detector.film_sites``: entry/exit, per-detector MLP width
``film.LUMEN_HIDDEN``) + consistency loss (KL 1.0, feat-MSE 1.0), frozen detector; ``dcfg_filmlite`` is kept as an
alias of ``lumen`` (the pilot checkpoints/configs use it); ``lumen_nocons`` = no consistency loss, ``lumen_nolive`` =
random-init sensor, ``lumen_frozensensor`` = sensor frozen. FiLM MLP params train in the gate param group (lr_gate). The teacher (clean image, FiLM hooks disabled, no_grad,
eval mode) and the student (front-end(degraded), FiLM on) share one detector forward helper, hence the same
normalization and autocast mode.

Design notes
------------
* Pipeline: x01 -> FrontEnd (absent for augft) -> ``detector.normalize`` (exactly once, after the
  front-end) -> detector -> logit; BCE-with-logits. Autocast fp16 only around the detector forward;
  the module (DCT, gate) is fp32 (the sensor CNN has its own ``sensor_amp`` switch).
* Detector frozen (variants other than augft/dcfg_jointft): no param requires grad and the module is
  in eval mode (BN statistics fixed); asserted at start and every ``frozen_check_every`` steps.
  Fine-tuned detectors run in train mode (BN uses batch statistics), as DeepfakeBench's trainer does
  (``model.train()``); validation always uses eval mode.
* Sensor: trainable at ``lr_sensor`` unless ``freeze_sensor: true`` (PLAN 7.5 lists its lr).
* UCF (augft_ucf only): trained with UCF's own DFB loss ``get_losses(...)['overall']`` on [real | fake]
  paired batches with label_spe from the FF++ manipulation type; other detectors use BCE.
* Sampling: class-balanced 1:1 per epoch from a permutation seeded by ``[seed, epoch]`` (SeedSequence); fakes pooled
  over the 4 manipulation types. Degradations (PLAN 7.2) are applied in the DataLoader workers to the
  uint8 crop with ``make_rng(seed, epoch, position_in_epoch)`` (no curriculum).
* One ``GradScaler`` scales the FULL loss; gradients are unscaled before norm statistics.
* Checkpoints (last.pt every epoch and every ``ckpt_every_steps``; best.pt on val improvement) carry
  all state; mid-epoch resume skips the batches already consumed (same permutation). Each consumed
  batch hash goes to ``batches.log`` so resume correctness can be verified.
* Val (early stopping only) is the ``val`` split under ``val_conditions``; results stay in the run dir.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dcfg.data.degradations import (TrainDegradation, load_conditions, load_val_conditions,
                                    make_rng)
from dcfg.data.ffpp import DFBJsonSource, FrameRecord
from dcfg.eval.evaluate import ConditionDataset, condition_comp, git_commit, predict
from dcfg.eval.metrics import frame_auc
from dcfg.models.film import FILM_KINDS, LUMEN_HIDDEN, FiLMBank
from dcfg.models.frontend import FrontEnd
from dcfg.models.sensor import DegradationSensor
from dcfg.paths import REPO_ROOT, RUNS_ROOT
from dcfg.train import consistency as cons
from dcfg.utils.heartbeat import Heartbeat
from dcfg.utils.resources import ResourceSampler, StepTimer, write_summary
from dcfg.utils.seed import seed_everything

#: config keys that may differ on resume (paths, workers, logging); everything else must match.
CONFIG_RESUME_IGNORE = frozenset({"run_dir", "json", "frames_root", "sensor_weights", "detector_weights",
                                  "conditions_yaml", "num_workers", "val_workers", "heartbeat_s", "log_every",
                                  "wandb_project", "max_steps", "allow_config_change",
                                  "stop_after_epochs", "timing_sync", "resource_interval_s"})
MAX_CONSEC_SKIPS = 50


class NonFiniteAbort(RuntimeError):
    """Persistent fp16 overflow / non-finite gradients: the run is diverging, abort."""


def config_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Keys (outside CONFIG_RESUME_IGNORE) whose values differ between two configs."""
    keys = (set(old) | set(new)) - CONFIG_RESUME_IGNORE
    return {k: (old.get(k), new.get(k)) for k in sorted(keys) if old.get(k) != new.get(k)}


VARIANTS = ("dcfg", "fixed", "nosensor", "dcfg_nolive", "augft", "dcfg_jointft",
            "dcfg_cons", "dcfg_film", "dcfg_filmlite",
            "lumen", "lumen_nocons", "lumen_nolive", "lumen_frozensensor", "bnft")
#: alias -> canonical variant (behaviour is looked up on the canonical name; configs/checkpoints keep the given name)
VARIANT_ALIAS = {"dcfg_filmlite": "lumen"}
#: variants that use the consistency loss (clean teacher, frozen detector); config ``consistency`` defaults to 1/1
CONSISTENCY_VARIANTS = ("dcfg_cons", "dcfg_film", "dcfg_filmlite", "lumen", "lumen_nolive", "lumen_frozensensor")
FILM_VARIANTS = {"dcfg_film": "film", "dcfg_filmlite": "filmlite", "lumen": "filmlite", "lumen_nocons": "filmlite", "lumen_nolive": "filmlite",
                 "lumen_frozensensor": "filmlite"}
LUMEN_FAMILY = ("lumen", "lumen_nocons", "lumen_nolive", "lumen_frozensensor")
NOLIVE_VARIANTS = ("dcfg_nolive", "lumen_nolive")  #: random-init sensor (no LIVE pretraining)
FROZEN_SENSOR_VARIANTS = ("lumen_frozensensor",)  #: sensor frozen regardless of ``freeze_sensor``
TRAIN_DETECTOR = ("augft", "dcfg_jointft")
BN_ONLY = ("bnft",)  #: only the detector's BatchNorm affine params train (BN in train mode)


def canonical_variant(v: str) -> str:
    """Resolve aliases (``dcfg_filmlite`` -> ``lumen``); unknown names raise."""
    if v not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return VARIANT_ALIAS.get(v, v)
#: UCF's own specific-forgery labels (DFB ucf.yaml label_dict): real 0, DF 1, F2F 2, FS 3, NT 4.
SPE_LABEL = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 2, "FF-FS": 3, "FF-NT": 4}
GATE_KIND = {"dcfg": "dcfg", "dcfg_nolive": "dcfg", "dcfg_jointft": "dcfg", "fixed": "fixed",
             "nosensor": "nosensor", "augft": None, "dcfg_cons": "dcfg", "dcfg_film": "dcfg",
             "lumen": "dcfg", "lumen_nocons": "dcfg", "lumen_nolive": "dcfg", "lumen_frozensensor": "dcfg",
             "bnft": None}


# ----------------------------------------------------------------------------- config
def _path(p: str | os.PathLike[str]) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | os.PathLike[str], overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a yaml config (``base: other.yaml`` is merged underneath) and apply ``key=value`` overrides."""
    path = _path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        base.update(cfg)
        cfg = base
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise KeyError(f"unknown config key {k!r}")
        cfg[k] = yaml.safe_load(v)
    return cfg


# ----------------------------------------------------------------------------- model
def build_detector(cfg: dict[str, Any], device: str | torch.device) -> nn.Module:
    """The DFB detector via the adapter (monkeypatched in tests)."""
    from dcfg.detectors.adapter import Detector
    return Detector(cfg["detector"], weights_path=cfg.get("detector_weights"), device=device)


def build_frontend(cfg: dict[str, Any], detector: Any = None) -> FrontEnd | None:
    """Front-end for the variant (None for augft). Loads LIVE sensor weights for dcfg/dcfg_jointft.
    FiLM variants also need the ``detector`` (site channel counts are read from its modules)."""
    v = canonical_variant(cfg["variant"])
    kind = GATE_KIND[v]
    if kind is None:
        return None
    sensor = None
    if kind == "dcfg":
        sensor = DegradationSensor()
        if v not in NOLIVE_VARIANTS:
            sw = _path(cfg["sensor_weights"])
            if not sw.exists():
                raise FileNotFoundError(f"sensor weights {sw} missing (variant {v} needs LIVE pretraining)")
            sensor.load_backbone(sw)
    film = None
    if v in FILM_VARIANTS:
        if detector is None:
            raise ValueError(f"variant {v} needs the detector to size the FiLM bank")
        sites = detector.film_sites()
        hidden = LUMEN_HIDDEN.get(cfg["detector"]) if v in LUMEN_FAMILY else None
        film = FiLMBank(FILM_VARIANTS[v], {k: c for k, (_, c) in sites.items() if k in FILM_KINDS[FILM_VARIANTS[v]]["sites"]},
                        hidden=hidden)
    return FrontEnd(kind, sensor=sensor, freeze_sensor=bool(cfg["freeze_sensor"]) or v in FROZEN_SENSOR_VARIANTS,
                    sensor_amp=bool(cfg["sensor_amp"]), film=film)


class Pipeline(nn.Module):
    """x01 -> [front-end] -> detector.normalize -> detector -> logit. Returns ``(logit (B,), w or None)``."""

    def __init__(self, frontend: FrontEnd | None, detector: nn.Module, amp: bool, train_detector: bool,
                 native_ucf: bool = False, film_hooks: Any = None, bn_only: bool = False) -> None:
        super().__init__()
        self.frontend, self.detector = frontend, detector
        self.amp, self.train_detector, self.native_ucf = amp, train_detector, native_ucf
        self.bn_only = bn_only  # BN-only fine-tuning: the detector's own train() keeps non-BN modules in eval
        self.film_hooks = film_hooks  # adapter.FilmHooks or None (not a Module: holds no parameters)

    def _film(self, params: Any) -> Any:
        return self.film_hooks.conditioned(params) if self.film_hooks is not None else contextlib.nullcontext()

    def _detect(self, x01: Tensor) -> tuple[Tensor, Tensor, tuple]:
        """The ONE detector forward used by teacher and student: normalize (once) -> autocast -> logit + features.
        Returns ``(logit fp32, feat fp32, (z dtype, autocast flag))``; the last item lets callers assert that
        both passes ran in the same normalization / precision."""
        z = self.detector.normalize(x01)
        use = self.amp and z.is_cuda
        with torch.autocast("cuda", dtype=torch.float16, enabled=use):
            logit, feat = self.detector.forward_with_features(z)
        return logit.float(), feat.float(), (z.dtype, use)

    def consistency_forward(self, x_deg01: Tensor, x_clean01: Tensor) -> dict[str, Any]:
        """Student on front-end(degraded) with FiLM, teacher = the same frozen detector on the clean image
        (no_grad, eval mode, FiLM hooks disabled). Returns logits/features of both plus the gate weights."""
        assert not self.train_detector and not self.detector.training, "consistency needs a frozen eval-mode detector"
        assert self.frontend is not None
        with torch.no_grad():
            ctx = self.film_hooks.disabled() if self.film_hooks is not None else contextlib.nullcontext()
            with ctx:
                t_logit, t_feat, t_meta = self._detect(x_clean01)
        y, w, fp = self.frontend.forward_full(x_deg01)
        with self._film(fp):
            s_logit, s_feat, s_meta = self._detect(y)
        assert t_meta == s_meta, f"teacher/student normalization or autocast differ: {t_meta} vs {s_meta}"
        return {"s_logit": s_logit, "s_feat": s_feat, "t_logit": t_logit, "t_feat": t_feat, "w": w}

    def native_ucf_loss(self, x01: Tensor, label: Tensor, label_spe: Tensor) -> Tensor:
        """UCF's own DeepfakeBench training loss (``get_losses(...)['overall']``).

        Needs the batch laid out as DFB's pair dataset does: first half real, second half fake
        (``chunk(2)`` inside the loss); label is binary, label_spe in {0..4}. No masks/landmarks are
        used (ucf.yaml with_mask/with_landmark false)."""
        assert self.native_ucf and self.train_detector
        if self.frontend is not None:
            x01, _ = self.frontend(x01)
        z = self.detector._prep(self.detector.normalize(x01))  # normalization applied once
        h = len(label) // 2
        assert len(label) % 2 == 0 and not label[:h].any() and label[h:].all(), "UCF needs [real | fake] batches"
        data = {"image": z, "label": label.long(), "label_spe": label_spe.long(), "mask": None, "landmark": None}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and z.is_cuda):
            pred = self.detector.model(data, inference=False)
            losses = self.detector.model.get_losses(data, pred)
        return losses["overall"].float()

    def forward(self, x01: Tensor) -> tuple[Tensor, Tensor | None]:
        w, fp = None, None
        if self.frontend is not None:
            x01, w, fp = self.frontend.forward_full(x01)
        z = self.detector.normalize(x01)  # the one and only normalization
        with self._film(fp), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and z.is_cuda):
            logit = self.detector(z)
        return logit.float(), w

    def train(self, mode: bool = True) -> "Pipeline":
        super().train(mode)
        if not self.train_detector and not self.bn_only:
            self.detector.eval()
        return self


def assert_detector_state(det: nn.Module, train_detector: bool, bn_only: bool = False) -> None:
    """Frozen-detector invariant (CLAUDE.md): frozen => no grads and eval mode; fine-tuned => the opposite.

    ``bn_only`` (variant bnft, the ONLY exception that allows BN in train mode): exactly the BatchNorm
    weight/bias parameters require grad, every other parameter is frozen and every non-BN module is in eval mode
    (BN modules may be in either mode: validation switches them to eval)."""
    bn = nn.modules.batchnorm._BatchNorm
    if bn_only:
        assert not train_detector
        want = {id(p) for m in det.modules() if isinstance(m, bn) for p in (m.weight, m.bias) if p is not None}
        got = {id(p) for p in det.parameters() if p.requires_grad}
        assert got == want, f"bnft: {len(got ^ want)} params differ from the BN affine set"
        assert not any(m.training for m in det.modules() if not isinstance(m, bn)), "bnft: non-BN module in train mode"
        return
    n_req = sum(p.requires_grad for p in det.parameters())
    if train_detector:
        assert n_req == sum(1 for _ in det.parameters()), "fine-tuned detector has frozen params"
    else:
        assert n_req == 0, f"detector must be frozen but {n_req} params require grad"
        assert not det.training and not any(m.training for m in det.modules()), \
            "frozen detector must be in eval mode"


def save_frontend(path: Path, frontend: FrontEnd, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    """The artefact consumed by ``evaluate --frontend`` (weights incl. sensor; no LIVE file needed)."""
    _save_atomic({"gate_kind": frontend.gate_kind, "variant": cfg["variant"], "sensor_amp": frontend.sensor_amp,
                  "film": frontend.film.meta() if frontend.film is not None else None,
                  "state_dict": frontend.state_dict(), "run_id": cfg["run_id"], **extra}, path)


def load_frontend(path: str | os.PathLike[str], device: str | torch.device = "cpu") -> tuple[FrontEnd, dict]:
    """Rebuild a trained FrontEnd (eval mode, frozen) from ``frontend.pt``; returns (module, metadata)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fm = ck.get("film")
    fe = FrontEnd(ck["gate_kind"], sensor_amp=ck.get("sensor_amp", True),
                  film=FiLMBank(fm["kind"], fm["channels"], hidden=fm.get("hidden")) if fm else None)
    fe.load_state_dict(ck["state_dict"])
    fe.to(device).eval().requires_grad_(False)
    return fe, {k: v for k, v in ck.items() if k != "state_dict"}


# ----------------------------------------------------------------------------- data
def epoch_order(labels: np.ndarray, seed: int, epoch: int, mode: str = "balanced",
                paired_half: int | None = None) -> np.ndarray:
    """Frame indices for one epoch, class-balanced 1:1, shuffled; deterministic in ``(seed, epoch)``; seeds never collide across epochs.

    ``balanced``: all reals + as many fakes (fresh permutation per epoch, pooled over manipulations).
    ``full``: all fakes + reals re-cycled to match (each real used floor/ceil(n_fake/n_real) times,
    from fresh permutations, so repeats are spread over the epoch).
    ``paired_half=h`` (UCF native loss): every consecutive batch of 2h frames is [h reals | h fakes]
    (as DFB's pair dataset), batch order shuffled."""
    rng = np.random.default_rng([int(seed), int(epoch)])
    real, fake = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if len(real) == 0 or len(fake) == 0:
        raise ValueError("need both real and fake frames")
    m = len(real) if mode == "balanced" else len(fake)

    def take(idx: np.ndarray) -> np.ndarray:
        out = [rng.permutation(idx)]
        while sum(len(o) for o in out) < m:
            out.append(rng.permutation(idx))
        return np.concatenate(out)[:m]

    if mode not in ("balanced", "full"):
        raise ValueError(f"epoch_mode {mode!r}")
    r, f = take(real), take(fake)
    if paired_half:
        h, nb = paired_half, m // paired_half
        bs = [np.concatenate([r[i * h:(i + 1) * h], f[i * h:(i + 1) * h]]) for i in range(nb)]
        return np.concatenate([bs[i] for i in rng.permutation(nb)])
    return rng.permutation(np.concatenate([r, f]))


class TrainDataset(Dataset):
    """Degraded uint8 CHW crops for one epoch slice; sample ``j`` is position ``offset + j`` of the epoch."""

    def __init__(self, source: DFBJsonSource, records: list[FrameRecord], order: np.ndarray, offset: int,
                 seed: int, epoch: int, deg: TrainDegradation, return_clean: bool = False) -> None:
        self.source, self.records, self.order = source, records, order
        self.offset, self.seed, self.epoch, self.deg = offset, seed, epoch, deg
        self.return_clean = return_clean

    def __len__(self) -> int:
        return len(self.order) - self.offset

    def __getitem__(self, j: int) -> tuple:
        """``(degraded, label, ridx, spe_label)``; with ``return_clean`` a 5th item: the clean uint8 crop
        (the same array as the degraded one when the sample is clean). The rng/degradation are unchanged."""
        pos = self.offset + j
        ridx = int(self.order[pos])
        rec = self.records[ridx]
        clean = self.source.load(rec)
        img, _ = self.deg(clean, make_rng(self.seed, self.epoch, pos))
        out = (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
               torch.tensor(float(rec.label)), ridx, SPE_LABEL[rec.video.split("/")[0]])
        if self.return_clean:
            return out + (torch.from_numpy(np.ascontiguousarray(clean)).permute(2, 0, 1),)
        return out


def stratified_videos(records: list[FrameRecord], n: int | None, seed: int = 0) -> set[str] | None:
    """Deterministic subset of ``n`` videos, spread evenly over the video groups (real / DF / F2F / FS / NT).

    ``None`` (or n >= #videos) selects everything (returns None = no filtering). Used for the capped
    validation pass (smoke / timing runs); real training runs validate on the full val split."""
    by: dict[str, list[str]] = {}
    for v in sorted({r.video for r in records}):
        by.setdefault(v.split("/")[0], []).append(v)
    if n is None or n >= sum(len(x) for x in by.values()):
        return None
    rng = np.random.default_rng(seed)
    picks = {g: [vs[i] for i in rng.permutation(len(vs))] for g, vs in by.items()}
    out: set[str] = set()
    while len(out) < n:  # round-robin over groups
        for g in sorted(picks):
            if picks[g] and len(out) < n:
                out.add(picks[g].pop())
    return out


def batch_hash(ridx: Tensor) -> str:
    return hashlib.sha1(ridx.numpy().astype(np.int64).tobytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- helpers
def _save_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def state_dict_mb(sd: dict[str, Tensor]) -> float:
    """Size in MiB (2**20 bytes) of ``sd`` serialized with torch.save, floating tensors cast to fp32."""
    conv = {k: (v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()).contiguous()
            for k, v in sd.items()}
    buf = io.BytesIO()
    torch.save(conv, buf)
    return buf.getbuffer().nbytes / 2**20


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _lr_lambda(steps_per_epoch: int, warmup_epochs: int, max_epochs: int) -> Callable[[int], float]:
    warm, total = max(1, steps_per_epoch * warmup_epochs), steps_per_epoch * max_epochs

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
    return f


def _grad_norm(params: list[Tensor]) -> Tensor:
    gs = [p.grad for p in params if p.grad is not None]
    if not gs:
        return torch.tensor(float("nan"))
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs)))


# ----------------------------------------------------------------------------- trainer
class Trainer:
    """One training run in ``run_dir`` (resumable)."""

    def __init__(self, cfg: dict[str, Any], run_dir: Path | None = None) -> None:
        self.variant = canonical_variant(cfg["variant"])  # alias-resolved; cfg["variant"] keeps the given name
        if not cfg.get("run_id"):
            raise ValueError("config needs run_id")
        self.cfg = cfg
        if self.variant in CONSISTENCY_VARIANTS and not cfg.get("consistency"):
            cfg["consistency"] = dict(cons.DEFAULT)  # recorded in the checkpoint config
        self.cons = cons.parse_config(cfg.get("consistency"))
        if self.cons is not None and self.variant not in CONSISTENCY_VARIANTS:
            raise ValueError(f"the consistency loss is only supported for {CONSISTENCY_VARIANTS}")
        self.train_det = self.variant in TRAIN_DETECTOR
        self.bn_only = self.variant in BN_ONLY
        self.pilot = bool(cfg.get("pilot", False))
        default_dir = RUNS_ROOT / "pilot" / cfg["run_id"] if self.pilot else RUNS_ROOT / cfg["run_id"]
        self.run_dir = run_dir or Path(cfg["run_dir"] or os.environ.get("DCFG_RUN_DIR")
                                       or default_dir).expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(int(cfg["seed"]))

        self.source = DFBJsonSource(cfg["json"], cfg["frames_root"])
        self.records = self.source.records("train", cfg["comp"])
        self.labels = np.array([r.label for r in self.records])
        self.bs = int(cfg["batch_size"])
        # UCF fine-tuning uses UCF's own DFB losses (variant augft only): [real | fake] paired batches
        self.native_ucf = self.variant == "augft" and cfg["detector"] == "ucf"
        self.paired_half = self.bs // 2 if self.native_ucf else None
        if self.native_ucf and self.bs % 2:
            raise ValueError("native UCF loss needs an even batch size")
        self.spe = len(epoch_order(self.labels, 0, 0, cfg["epoch_mode"], self.paired_half)) // self.bs
        if self.spe < 1:
            raise ValueError("fewer frames than one batch")

        det = build_detector(cfg, self.dev)
        self.n_bn_trainable = 0
        if self.bn_only:
            self.n_bn_trainable = det.freeze_except_bn()
        elif not self.train_det:
            det.freeze()
        else:
            det.requires_grad_(True)
        self.frontend = build_frontend(cfg, det)
        if self.frontend is not None:
            self.frontend.to(self.dev)
        self.film_hooks = None
        if self.frontend is not None and self.frontend.film is not None:
            from dcfg.detectors.adapter import FilmHooks
            self.film_hooks = FilmHooks(det, self.frontend.film.sites)
        self.model = Pipeline(self.frontend, det, bool(cfg["amp"]), self.train_det, self.native_ucf,
                              self.film_hooks, self.bn_only).to(self.dev)
        self.det = det
        self.opt = self._make_optimizer()
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, _lr_lambda(self.spe, int(cfg["warmup_epochs"]), int(cfg["max_epochs"])))
        self.use_amp = bool(cfg["amp"]) and self.dev.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.deg = TrainDegradation()
        self.st: dict[str, Any] = {"epoch": 0, "step_in_epoch": 0, "step": 0, "best": -float("inf"),
                                   "best_epoch": 0, "bad": 0, "stopped": False}
        # heartbeat goes where the runner looks (runs/<job id>), which may differ from run_dir for pilot runs
        self.hb = Heartbeat(Path(os.environ.get("DCFG_RUN_DIR") or self.run_dir), float(cfg["heartbeat_s"]))
        self.timer = StepTimer()
        self.phase = "train"
        self.sampler: ResourceSampler | None = None
        self._wb_lock = threading.Lock()
        self._wb_step = 0
        self.wb: Any = None
        self.val_names = load_val_conditions(_path(cfg["conditions_yaml"]))
        allc = load_conditions(_path(cfg["conditions_yaml"]))
        self.val_conds = [allc[n] for n in self.val_names]
        self.t_start = time.time()
        self._sensor_params = [p for p in self.frontend.sensor.parameters() if p.requires_grad] \
            if self.frontend is not None and self.frontend.sensor is not None else []
        self._gate_params = list(self.frontend.gate.parameters()) if self.frontend is not None else []
        self._film_params = list(self.frontend.film.parameters()) \
            if self.frontend is not None and self.frontend.film is not None else []

    # -------------------------------------------------------------- setup
    def _make_optimizer(self) -> torch.optim.Optimizer:
        c, groups = self.cfg, []
        fe = self.frontend
        if fe is not None:
            # FiLM MLP (if any) shares the gate group / lr: same kind of module (small MLP, zero-init last layer)
            groups.append({"name": "gate", "params": list(fe.gate.parameters())
                           + (list(fe.film.parameters()) if fe.film is not None else []),
                           "lr": float(c["lr_gate"])})
            if fe.sensor is not None and not fe.freeze_sensor:
                groups.append({"name": "sensor", "params": list(fe.sensor.parameters()),
                               "lr": float(c["lr_sensor"])})
        if self.train_det:
            groups.append({"name": "detector", "params": list(self.det.parameters()),
                           "lr": float(c["lr_detector"])})
        if self.bn_only:  # BN affine params only (everything else has requires_grad False)
            groups.append({"name": "bn", "params": [p for p in self.det.parameters() if p.requires_grad],
                           "lr": float(c["lr_detector"])})
        return torch.optim.AdamW(groups, weight_decay=float(c["weight_decay"]))

    def check_frozen(self) -> None:
        assert_detector_state(self.det, self.train_det, self.bn_only)
        fe = self.frontend
        if fe is not None and fe.sensor is not None and fe.freeze_sensor:
            assert not any(p.requires_grad for p in fe.sensor.parameters()) and not fe.sensor.training
        detp = {id(p) for p in self.det.parameters()}
        in_opt = any(id(p) in detp for g in self.opt.param_groups for p in g["params"])
        assert in_opt == (self.train_det or self.bn_only), "detector params / optimizer mismatch"

    # -------------------------------------------------------------- checkpoints
    def _state(self) -> dict[str, Any]:
        s = {"frontend": self.frontend.state_dict() if self.frontend is not None else None,
             "detector": self.det.model.state_dict() if self.train_det else None}
        if self.bn_only:  # full BN state: affine params AND running stats / num_batches_tracked
            s["bn"] = self.det.bn_state_dict()
        return s

    def _full_ckpt(self) -> dict[str, Any]:
        return {"model": self._state(), "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(), "state": dict(self.st), "rng": _rng_state(),
                "config": self.cfg}

    def save_last(self) -> None:
        _save_atomic(self._full_ckpt(), self.run_dir / "last.pt")

    def _load_model_state(self, s: dict[str, Any]) -> None:
        if self.frontend is not None:
            self.frontend.load_state_dict(s["frontend"])
        if self.train_det:
            self.det.model.load_state_dict(s["detector"])
        if self.bn_only:
            self.det.load_bn_state_dict(s["bn"])

    def try_resume(self) -> bool:
        last = self.run_dir / "last.pt"
        if not last.exists():
            return False
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if bool(ck.get("config", {}).get("pilot", False)) != self.pilot:
            raise RuntimeError(f"refusing to resume {last}: its pilot flag "
                               f"({bool(ck.get('config', {}).get('pilot', False))}) differs from this config's "
                               f"({self.pilot}); pilot and real runs never share checkpoints")
        diff = config_diff(ck.get("config", {}), self.cfg)
        if diff:
            if not self.cfg.get("allow_config_change"):
                raise RuntimeError(f"config differs from the one in {last}: {diff}; "
                                   "pass --set allow_config_change=true to resume anyway")
            print(f"WARNING: resuming with changed config: {diff}", flush=True)
        self._load_model_state(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.st = ck["state"]
        _set_rng_state(ck["rng"])
        self._log_line("batches.log", f"# resumed epoch={self.st['epoch']} step_in_epoch="
                       f"{self.st['step_in_epoch']} step={self.st['step']}")
        print(f"resumed from step {self.st['step']} (epoch {self.st['epoch']}, "
              f"batch {self.st['step_in_epoch']})", flush=True)
        return True

    # -------------------------------------------------------------- logging
    def _log_line(self, name: str, line: str) -> None:
        with open(self.run_dir / name, "a") as f:
            f.write(line + "\n")

    def _csv_row(self, name: str, row: dict[str, Any]) -> None:
        path = self.run_dir / name
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def _wandb_init(self) -> None:
        try:
            import wandb
            self.wb = wandb.init(project=self.cfg["wandb_project"], job_type="train", name=self.cfg["run_id"],
                                 id=self.cfg.get("wandb_id") or self.cfg["run_id"], resume="allow", config=self.cfg, dir=str(self.run_dir),
                                 **({"tags": ["pilot"], "group": "pilot"} if self.pilot else {}),
                                 settings=wandb.Settings(init_timeout=60))
        except Exception as e:  # noqa: BLE001
            self.wb = None
            print(f"WARNING: wandb init failed, continuing without W&B: {e}", flush=True)

    def _wandb_log(self, d: dict[str, Any], step: int) -> None:
        if self.wb is None:
            return
        with self._wb_lock:  # called from the trainer and the resource-sampler thread; step must not go backwards
            try:
                self._wb_step = max(self._wb_step, step)
                self.wb.log(d, step=self._wb_step)
            except Exception as e:  # noqa: BLE001
                self.wb = None
                print(f"WARNING: wandb log failed, disabling W&B: {e}", flush=True)

    def _wandb_alert(self, title: str, text: str, error: bool = False) -> None:
        if self.wb is None:
            return
        try:
            import wandb
            self.wb.alert(title=title, text=text[:500],
                          level=wandb.AlertLevel.ERROR if error else wandb.AlertLevel.INFO)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: wandb alert failed: {e}", flush=True)

    # -------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Frame AUC per val condition on the val split (+ mean w per channel / 3x8x8 map per condition)."""
        was_training = self.model.training
        prev_phase, t_val = self.phase, time.time()
        self.phase = "val"
        self.model.eval()
        wsum: list[Tensor] = []
        wn = [0]

        def fn(x01: Tensor) -> Tensor:
            logit, w = self.model(x01)
            if w is not None:
                wsum.append(w.sum(0).cpu()); wn[0] += w.shape[0]
            return logit

        aucs: dict[str, float] = {}
        wmaps: dict[str, list] = {}
        times: dict[str, float] = {}
        nfr: dict[str, int] = {}
        keep = None
        if self.cfg.get("val_max_videos"):
            keep = stratified_videos(self.source.records("val", "c23"), int(self.cfg["val_max_videos"]))
        for cond in self.val_conds:
            t0 = time.time()
            recs = self.source.records("val", condition_comp(cond))
            if keep is not None:
                recs = [r for r in recs if r.video in keep]
            ld = DataLoader(ConditionDataset(self.source, recs, cond), batch_size=64, shuffle=False,
                            num_workers=int(self.cfg["val_workers"]), pin_memory=self.dev.type == "cuda")
            wsum.clear(); wn[0] = 0
            prob = predict(fn, ld, str(self.dev),
                           on_batch=lambda n, c=cond.name: self.hb.beat(self.st["step"], phase="val", cond=c, seen=n,
                                                                        epoch=self.st["epoch"]))
            aucs[cond.name] = frame_auc([r.label for r in recs], prob)
            times[cond.name], nfr[cond.name] = time.time() - t0, len(recs)
            if wn[0]:
                wmaps[cond.name] = (torch.stack(wsum).sum(0) / wn[0]).tolist()
        bad = [k for k, a in aucs.items() if not np.isfinite(a)]
        if bad:
            raise RuntimeError(f"non-finite val AUC for {bad}: {aucs}")
        mean_auc = float(np.mean(list(aucs.values())))
        self.model.train(was_training)
        self.phase = prev_phase
        self.st["val_s"] = self.st.get("val_s", 0.0) + time.time() - t_val
        return {"aucs": aucs, "mean_auc": mean_auc, "wmaps": wmaps, "times_s": times, "n_frames": nfr}

    # -------------------------------------------------------------- training
    def after_step(self, step: int) -> None:
        """Hook called after every optimizer step (tests inject faults here)."""

    def _loader(self, epoch: int, order: np.ndarray, skip: int) -> DataLoader:
        nw = int(self.cfg["num_workers"])
        ds = TrainDataset(self.source, self.records, order, skip * self.bs, int(self.cfg["seed"]), epoch, self.deg,
                          return_clean=self.cons is not None)
        return DataLoader(ds, batch_size=self.bs, shuffle=False, drop_last=True, num_workers=nw,
                          pin_memory=self.dev.type == "cuda", prefetch_factor=2 if nw else None)

    def _timed(self, loader: DataLoader) -> Any:
        """Iterate ``loader``, storing the time blocked in ``next()`` in ``self._last_wait``."""
        it = iter(loader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            self._last_wait = time.perf_counter() - t0
            yield batch

    _last_wait = 0.0

    def _train_epoch(self, epoch: int) -> bool:
        """Run the remainder of ``epoch``. Returns False if ``max_steps`` was hit."""
        cfg, st = self.cfg, self.st
        order = epoch_order(self.labels, int(cfg["seed"]), epoch, cfg["epoch_mode"],
                           self.paired_half)[: self.spe * self.bs]
        loader = self._loader(epoch, order, st["step_in_epoch"])
        self.model.train()
        loss_fn = nn.BCEWithLogitsLoss()
        log_every, accl, acc_n, acc_img, t_log = int(cfg["log_every"]), None, 0, 0, time.time()
        stats = torch.zeros(4, device=self.dev)  # nonfinite(sensor, gate), zero(sensor, gate) counts
        gn = torch.zeros(3, device=self.dev)  # grad norms: sensor, gate, film
        terms_acc: dict[str, Tensor] = {}
        self.phase = "train"
        sync = bool(cfg.get("timing_sync")) and self.dev.type == "cuda"
        t_epoch = time.time()
        for batch in self._timed(loader):
            xb, yb, ridx, spe_l = batch[:4]
            t_step = time.perf_counter()
            b = st["step_in_epoch"]
            self._log_line("batches.log", f"{epoch} {b} {st['step']} {batch_hash(ridx)}")
            x = xb.to(self.dev, non_blocking=True).float().div_(255.0)
            y = yb.to(self.dev, non_blocking=True)
            if self.native_ucf:
                loss = self.model.native_ucf_loss(x, y, spe_l.to(self.dev))
            elif self.cons is not None:
                xc = batch[4].to(self.dev, non_blocking=True).float().div_(255.0)
                o = self.model.consistency_forward(x, xc)
                loss, terms = cons.consistency_loss(o["s_logit"], o["s_feat"], o["t_logit"], o["t_feat"], y, self.cons)
                for k, v in terms.items():
                    terms_acc[k] = terms_acc.get(k, 0) + v
            else:
                logit, _ = self.model(x)
                loss = loss_fn(logit, y)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            for i, ps in enumerate((self._sensor_params, self._gate_params, self._film_params)):
                if ps:
                    n = _grad_norm(ps).to(self.dev)
                    gn[i] = n
                    if i < 2:
                        stats[i] += (~torch.isfinite(n)).float()
                        stats[2 + i] += (n == 0).float()
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.opt)
            self.scaler.update()
            if self._after_scaler(scale_before, self.scaler.get_scale()):  # skipped steps do not advance the LR schedule
                self.sched.step()
            if sync:  # accurate compute time needs the GPU work to have finished (pilot / timing only)
                torch.cuda.synchronize()
            self.timer.add(self._last_wait, time.perf_counter() - t_step)
            st["step"] += 1
            st["step_in_epoch"] += 1
            accl = loss.detach() if accl is None else accl + loss.detach()
            acc_n += 1
            acc_img += len(x)
            if st["step"] % int(cfg["frozen_check_every"]) == 0:
                self.check_frozen()
            if st["step"] % log_every == 0 or st["step_in_epoch"] == self.spe:
                self._log_train(epoch, accl, acc_n, acc_img, time.time() - t_log, stats, gn, terms_acc)
                accl, acc_n, acc_img, t_log = None, 0, 0, time.time()
                stats.zero_()
                terms_acc = {}
            if st["step"] % int(cfg["ckpt_every_steps"]) == 0:
                self.save_last()
            self.hb.beat(st["step"], epoch=epoch, val_auc=st["best"], img_s=self._img_s,
                         eta=self._eta())
            self.after_step(st["step"])
            if cfg["max_steps"] and st["step"] >= int(cfg["max_steps"]):
                st["train_s"] = st.get("train_s", 0.0) + time.time() - t_epoch
                return False
        st["train_s"] = st.get("train_s", 0.0) + time.time() - t_epoch
        return True

    _img_s = 0.0
    _consec_skips = 0

    def _after_scaler(self, before: float, after: float) -> bool:
        """Track fp16 overflow skips. Returns True if the optimizer step was applied; raises
        :class:`NonFiniteAbort` if the loss scale fell below 1 or >50 consecutive steps were skipped."""
        skipped = after < before
        if skipped:
            self.st["skipped_steps"] = self.st.get("skipped_steps", 0) + 1  # fp16 overflow skips (persisted)
        self._consec_skips = self._consec_skips + 1 if skipped else 0
        if after < 1.0 or self._consec_skips > MAX_CONSEC_SKIPS:
            msg = (f"persistent non-finite gradients at step {self.st['step']}: GradScaler scale {after}, "
                   f"{self._consec_skips} consecutive skipped steps")
            print(f"ABORT: {msg}", flush=True)
            self._log_line("batches.log", f"# ABORT {msg}")
            self._wandb_alert(f"{self.cfg['run_id']} non-finite abort", msg, error=True)
            raise NonFiniteAbort(msg)
        return not skipped

    def _eta(self) -> str:
        done = self.st["step"]
        if done == 0 or self._img_s <= 0:
            return "?"
        left = max(0, self.spe * int(self.cfg["max_epochs"]) - done)
        return f"{int(left * self.bs / self._img_s)}s"

    def _log_train(self, epoch: int, accl: Tensor | None, n: int, imgs: int, dt: float, stats: Tensor,
                   gn: Tensor, terms: dict[str, Tensor] | None = None) -> None:
        st = self.st
        loss = float(accl) / max(1, n) if accl is not None else float("nan")
        self._img_s = imgs / max(dt, 1e-9)
        s, g = stats.tolist(), gn.tolist()
        row = {"step": st["step"], "epoch": epoch, "loss": f"{loss:.5f}",
               **{f"lr_{pg['name']}": f"{pg['lr']:.3e}" for pg in self.opt.param_groups},
               "img_s": f"{self._img_s:.1f}", "scaler_scale": f"{self.scaler.get_scale():.1f}",
               "skipped_steps": st.get("skipped_steps", 0),
               "gpu_mem_mb": f"{torch.cuda.max_memory_allocated() / 2**20:.0f}" if self.dev.type == "cuda" else 0,
               "gnorm_sensor": g[0], "gnorm_gate": g[1], "nonfinite_sensor": int(s[0]), "nonfinite_gate": int(s[1]),
               "zero_sensor": int(s[2]), "zero_gate": int(s[3])}
        if self._film_params:
            row["gnorm_film"] = g[2]
        if terms:  # consistency variants: each loss term separately (unweighted means over the log window)
            row |= {f"loss_{k}": f"{float(v) / max(1, n):.5f}" for k, v in terms.items()}
        self._csv_row("train_log.csv", row)
        self._wandb_log({f"train/{k}": float(v) for k, v in row.items() if k not in ("step", "epoch")}, st["step"])
        if s[0] or s[1]:
            print(f"WARNING step {st['step']}: non-finite grad norm (sensor {int(s[0])}, gate {int(s[1])}) "
                  f"[expected only on fp16 scale back-off]", flush=True)
        if (s[3] or s[2]) and st["step"] > 100:
            print(f"WARNING step {st['step']}: zero grad norm (sensor {int(s[2])}, gate {int(s[3])})", flush=True)

    def _end_of_epoch(self, epoch: int) -> None:
        st = self.st
        v = self.validate()
        better = v["mean_auc"] > st["best"]
        if better:
            st.update(best=v["mean_auc"], best_epoch=epoch, bad=0)
        else:
            st["bad"] += 1
        st["epoch"], st["step_in_epoch"] = epoch + 1, 0
        if st["bad"] >= int(self.cfg["patience"]):
            st["stopped"] = True
            st["stop_epoch"] = epoch  # 0-based epoch at which early stopping triggered (like best_epoch)
        wm = {c: torch.tensor(m) for c, m in v["wmaps"].items()}
        wch = (torch.stack(list(wm.values())).mean(0).mean((1, 2)).tolist() if wm else [float("nan")] * 3)
        row = {"epoch": epoch, "step": st["step"], "val_mean_auc": f"{v['mean_auc']:.5f}",
               **{f"val_{k}": f"{a:.5f}" for k, a in v["aucs"].items()},
               **{f"w_mean_ch{i}": f"{x:.5f}" for i, x in enumerate(wch)},
               **{f"lr_{pg['name']}": f"{pg['lr']:.3e}" for pg in self.opt.param_groups},
               "img_s": f"{self._img_s:.1f}", "bad_epochs": st["bad"]}
        self._csv_row("metrics.csv", row)
        self._log_line("w_maps.jsonl", json.dumps({"epoch": epoch, "step": st["step"], "maps": v["wmaps"]}))
        self._wandb_log({f"val/{k}": a for k, a in v["aucs"].items()} | {"val/mean_auc": v["mean_auc"]}
                        | {f"w/mean_ch{i}": x for i, x in enumerate(wch)}, st["step"])
        if better:
            _save_atomic({**self._full_ckpt(), "metric": v["mean_auc"]}, self.run_dir / "best.pt")
        self.save_last()
        self.hb.beat(st["step"], force=True, epoch=epoch, val_auc=v["mean_auc"], img_s=self._img_s, eta=self._eta())
        print(f"epoch {epoch}: val mean AUC {v['mean_auc']:.4f} (best {st['best']:.4f} @ {st['best_epoch']})", flush=True)

    def train(self) -> dict[str, Any]:
        done = self.run_dir / "done.json"
        if done.exists():
            return json.loads(done.read_text())
        self.check_frozen()  # also at start
        self.try_resume()
        self.check_frozen()
        self._wandb_init()
        self.sampler = ResourceSampler(self.run_dir / "resources.csv", float(self.cfg.get("resource_interval_s", 30)),
                                       self.timer, lambda: self.st["step"], lambda: self.phase,
                                       on_row=self._log_resources).start()
        try:
            res = self._train_loop()
        except Exception as e:  # noqa: BLE001
            self._wandb_alert(f"{self.cfg['run_id']} crashed", f"{type(e).__name__}: {e}", error=True)
            raise
        finally:
            self.sampler.stop()
        summ = write_summary(self.run_dir, self.st.get("train_s", 0.0), self.st.get("val_s", 0.0))
        print(f"resources: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})}",
              flush=True)
        self._wandb_alert(f"{self.cfg['run_id']} finished", json.dumps(res))
        if self.wb is not None:
            try:
                self.wb.finish()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: wandb finish failed: {e}", flush=True)
        return res

    def _log_resources(self, row: dict[str, Any]) -> None:
        """Sampler callback: numeric columns to W&B under ``res/``."""
        d = {f"res/{k}": float(v) for k, v in row.items()
             if k not in ("timestamp", "phase", "step") and v is not None}
        d["res/is_train"] = float(row["phase"] == "train")
        self._wandb_log(d, int(row["step"]))

    def _pilot_stop_due(self) -> bool:
        """``stop_after_epochs`` reached (pilot runs follow the real schedule, then stop and finalize)."""
        n = self.cfg.get("stop_after_epochs")
        return bool(n) and self.st["epoch"] >= int(n)

    def _train_loop(self) -> dict[str, Any]:
        st, cfg = self.st, self.cfg
        truncated = False
        while st["epoch"] < int(cfg["max_epochs"]) and not st["stopped"] and not self._pilot_stop_due():
            epoch = st["epoch"]
            if st["step_in_epoch"] < self.spe:
                if not self._train_epoch(epoch):
                    truncated = True
                    break
            self._end_of_epoch(epoch)
        if truncated:
            self.save_last()
            res = {"run_id": cfg["run_id"], "truncated": True, "pilot": self.pilot, "step": st["step"], "img_s": self._img_s}
            if cfg.get("val_on_truncate"):  # smoke/timing: one val pass (optionally capped by val_max_videos)
                v = self.validate()
                res["val"] = {k: v[k] for k in ("aucs", "mean_auc", "times_s", "n_frames")}
            (self.run_dir / "done.json").write_text(json.dumps(res))
            return res
        return self._finalize()

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Only what this variant trains: front-end (+FiLM, fixed logits) and/or the BN state (bnft) and/or the whole
        detector (augft/jointft). Keys are prefixed by the part."""
        sd: dict[str, Tensor] = {}
        if self.frontend is not None:
            sd |= {f"frontend.{k}": v for k, v in self.frontend.state_dict().items()}
        if self.bn_only:
            sd |= {f"bn.{k}": v for k, v in self.det.bn_state_dict().items()}
        if self.train_det:
            sd |= {f"detector.{k}": v for k, v in self.det.model.state_dict().items()}
        return sd

    def train_stats(self) -> dict[str, Any]:
        """Training metadata that evaluate.py copies into every result JSON (and done.json): early-stop / best
        epoch (0-based), wall-clock training time, and checkpoint sizes (fp32 state_dict bytes, MiB)."""
        st = self.st
        n_tr = sum(p.numel() for m in (self.frontend, self.det) if m is not None
                   for p in m.parameters() if p.requires_grad)
        return {"early_stop_epoch": int(st["stop_epoch"]) if st.get("stopped") and "stop_epoch" in st else None,
                "train_wall_time_s": float(st.get("train_s", 0.0) + st.get("val_s", 0.0)),
                "trainable_mb": state_dict_mb(self.trainable_state_dict()),
                "detector_mb": state_dict_mb(self.det.model.state_dict()),
                "params": int(n_tr), "scaler_skipped_steps": int(st.get("skipped_steps", 0))}

    def _finalize(self) -> dict[str, Any]:
        """Load best.pt and write the evaluation artefacts (frontend.pt, detector.pt)."""
        st, cfg = self.st, self.cfg
        best = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=False)
        self._load_model_state(best["model"])
        meta = {"best_epoch": st["best_epoch"], "val_mean_auc": st["best"], "git_commit": git_commit(),
                "detector": cfg["detector"], "run_id": cfg["run_id"], "seed": int(cfg["seed"]),
                "train_config": dict(cfg), "pilot": self.pilot, **self.train_stats()}
        if self.frontend is not None:
            save_frontend(self.run_dir / "frontend.pt", self.frontend, cfg, meta)
        if self.bn_only:  # small artefact: the full BN state, applied onto the stock weights by evaluate --bn-ckpt
            _save_atomic({"state_dict": self.det.bn_state_dict(), "meta": {**meta, "variant": cfg["variant"]}},
                         self.run_dir / "bn.pt")
        if self.train_det:  # adapter loads {"state_dict": ...}; "meta" rides along (primitives only)
            _save_atomic({"state_dict": self.det.model.state_dict(), "meta": {**meta, "variant": cfg["variant"]}},
                         self.run_dir / "detector.pt")
        res = {"run_id": cfg["run_id"], "truncated": False, "variant": cfg["variant"], "epochs_run": st["epoch"],
               "stopped_early": bool(st["stopped"]), "total_time_s": time.time() - self.t_start,
               "train_s": st.get("train_s", 0.0), "val_s": st.get("val_s", 0.0), **meta}
        if self._pilot_stop_due() and not st["stopped"]:
            res["stopped_after_epochs"] = int(cfg["stop_after_epochs"])
        (self.run_dir / "done.json").write_text(json.dumps(res, indent=2))
        return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.train.train_frontend", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="YAML-parsed config overrides")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    res = Trainer(cfg, Path(a.run_dir) if a.run_dir else None).train()
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
