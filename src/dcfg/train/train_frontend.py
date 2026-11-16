"""Train the DCFG front-end (or fine-tune a detector) on FF++ c23 (PLAN 7.2, 7.4, 7.5, Phase 8.1).

    python -m dcfg.train.train_frontend --config configs/train/dcfg_xception.yaml [--set key=value ...]

Variants: ``dcfg`` (LIVE-pretrained sensor + gate, frozen detector), ``fixed``, ``nosensor``,
``dcfg_nolive`` (random sensor), ``augft`` (no module, detector fine-tuned on the degraded data),
``dcfg_jointft`` (module + detector fine-tuned).

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
* Sampling: class-balanced 1:1 per epoch from a permutation seeded by ``seed + epoch``; fakes pooled
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
import csv
import hashlib
import json
import math
import os
import random
import sys
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
from dcfg.models.frontend import FrontEnd
from dcfg.models.sensor import DegradationSensor
from dcfg.paths import REPO_ROOT, RUNS_ROOT
from dcfg.utils.heartbeat import Heartbeat
from dcfg.utils.seed import seed_everything

VARIANTS = ("dcfg", "fixed", "nosensor", "dcfg_nolive", "augft", "dcfg_jointft")
TRAIN_DETECTOR = ("augft", "dcfg_jointft")
#: UCF's own specific-forgery labels (DFB ucf.yaml label_dict): real 0, DF 1, F2F 2, FS 3, NT 4.
SPE_LABEL = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 2, "FF-FS": 3, "FF-NT": 4}
GATE_KIND = {"dcfg": "dcfg", "dcfg_nolive": "dcfg", "dcfg_jointft": "dcfg", "fixed": "fixed",
             "nosensor": "nosensor", "augft": None}


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


def build_frontend(cfg: dict[str, Any]) -> FrontEnd | None:
    """Front-end for the variant (None for augft). Loads LIVE sensor weights for dcfg/dcfg_jointft."""
    v = cfg["variant"]
    kind = GATE_KIND[v]
    if kind is None:
        return None
    sensor = None
    if kind == "dcfg":
        sensor = DegradationSensor()
        if v != "dcfg_nolive":
            sw = _path(cfg["sensor_weights"])
            if not sw.exists():
                raise FileNotFoundError(f"sensor weights {sw} missing (variant {v} needs LIVE pretraining)")
            sensor.load_backbone(sw)
    return FrontEnd(kind, sensor=sensor, freeze_sensor=bool(cfg["freeze_sensor"]),
                    sensor_amp=bool(cfg["sensor_amp"]))


class Pipeline(nn.Module):
    """x01 -> [front-end] -> detector.normalize -> detector -> logit. Returns ``(logit (B,), w or None)``."""

    def __init__(self, frontend: FrontEnd | None, detector: nn.Module, amp: bool, train_detector: bool,
                 native_ucf: bool = False) -> None:
        super().__init__()
        self.frontend, self.detector = frontend, detector
        self.amp, self.train_detector, self.native_ucf = amp, train_detector, native_ucf

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
        w = None
        if self.frontend is not None:
            x01, w = self.frontend(x01)
        z = self.detector.normalize(x01)  # the one and only normalization
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and z.is_cuda):
            logit = self.detector(z)
        return logit.float(), w

    def train(self, mode: bool = True) -> "Pipeline":
        super().train(mode)
        if not self.train_detector:
            self.detector.eval()
        return self


def assert_detector_state(det: nn.Module, train_detector: bool) -> None:
    """Frozen-detector invariant (CLAUDE.md): frozen => no grads and eval mode; fine-tuned => the opposite."""
    n_req = sum(p.requires_grad for p in det.parameters())
    if train_detector:
        assert n_req == sum(1 for _ in det.parameters()), "fine-tuned detector has frozen params"
    else:
        assert n_req == 0, f"detector must be frozen but {n_req} params require grad"
        assert not det.training, "frozen detector must be in eval mode"


def save_frontend(path: Path, frontend: FrontEnd, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    """The artefact consumed by ``evaluate --frontend`` (weights incl. sensor; no LIVE file needed)."""
    _save_atomic({"gate_kind": frontend.gate_kind, "variant": cfg["variant"], "sensor_amp": frontend.sensor_amp,
                  "state_dict": frontend.state_dict(), "run_id": cfg["run_id"], **extra}, path)


def load_frontend(path: str | os.PathLike[str], device: str | torch.device = "cpu") -> tuple[FrontEnd, dict]:
    """Rebuild a trained FrontEnd (eval mode, frozen) from ``frontend.pt``; returns (module, metadata)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fe = FrontEnd(ck["gate_kind"], sensor_amp=ck.get("sensor_amp", True))
    fe.load_state_dict(ck["state_dict"])
    fe.to(device).eval().requires_grad_(False)
    return fe, {k: v for k, v in ck.items() if k != "state_dict"}


# ----------------------------------------------------------------------------- data
def epoch_order(labels: np.ndarray, seed: int, epoch: int, mode: str = "balanced",
                paired_half: int | None = None) -> np.ndarray:
    """Frame indices for one epoch, class-balanced 1:1, shuffled; deterministic in ``seed + epoch``.

    ``balanced``: all reals + as many fakes (fresh permutation per epoch, pooled over manipulations).
    ``full``: all fakes + reals re-cycled to match (each real used floor/ceil(n_fake/n_real) times,
    from fresh permutations, so repeats are spread over the epoch).
    ``paired_half=h`` (UCF native loss): every consecutive batch of 2h frames is [h reals | h fakes]
    (as DFB's pair dataset), batch order shuffled."""
    rng = np.random.default_rng(seed + epoch)
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
                 seed: int, epoch: int, deg: TrainDegradation) -> None:
        self.source, self.records, self.order = source, records, order
        self.offset, self.seed, self.epoch, self.deg = offset, seed, epoch, deg

    def __len__(self) -> int:
        return len(self.order) - self.offset

    def __getitem__(self, j: int) -> tuple[Tensor, Tensor, int, int]:
        pos = self.offset + j
        ridx = int(self.order[pos])
        rec = self.records[ridx]
        img, _ = self.deg(self.source.load(rec), make_rng(self.seed, self.epoch, pos))
        return (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
                torch.tensor(float(rec.label)), ridx, SPE_LABEL[rec.video.split("/")[0]])


def batch_hash(ridx: Tensor) -> str:
    return hashlib.sha1(ridx.numpy().astype(np.int64).tobytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- helpers
def _save_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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
        if cfg["variant"] not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not cfg.get("run_id"):
            raise ValueError("config needs run_id")
        self.cfg = cfg
        self.variant = cfg["variant"]
        self.train_det = self.variant in TRAIN_DETECTOR
        self.run_dir = run_dir or Path(cfg["run_dir"] or os.environ.get("DCFG_RUN_DIR")
                                       or RUNS_ROOT / cfg["run_id"]).expanduser()
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
        if not self.train_det:
            det.freeze()
        else:
            det.requires_grad_(True)
        self.frontend = build_frontend(cfg)
        if self.frontend is not None:
            self.frontend.to(self.dev)
        self.model = Pipeline(self.frontend, det, bool(cfg["amp"]), self.train_det, self.native_ucf).to(self.dev)
        self.det = det
        self.opt = self._make_optimizer()
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, _lr_lambda(self.spe, int(cfg["warmup_epochs"]), int(cfg["max_epochs"])))
        self.use_amp = bool(cfg["amp"]) and self.dev.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.deg = TrainDegradation()
        self.st: dict[str, Any] = {"epoch": 0, "step_in_epoch": 0, "step": 0, "best": -float("inf"),
                                   "best_epoch": 0, "bad": 0, "stopped": False}
        self.hb = Heartbeat(self.run_dir, float(cfg["heartbeat_s"]))
        self.wb: Any = None
        self.val_names = load_val_conditions(_path(cfg["conditions_yaml"]))
        allc = load_conditions(_path(cfg["conditions_yaml"]))
        self.val_conds = [allc[n] for n in self.val_names]
        self.t_start = time.time()
        self._sensor_params = [p for p in self.frontend.sensor.parameters() if p.requires_grad] \
            if self.frontend is not None and self.frontend.sensor is not None else []
        self._gate_params = list(self.frontend.gate.parameters()) if self.frontend is not None else []

    # -------------------------------------------------------------- setup
    def _make_optimizer(self) -> torch.optim.Optimizer:
        c, groups = self.cfg, []
        fe = self.frontend
        if fe is not None:
            groups.append({"name": "gate", "params": list(fe.gate.parameters()), "lr": float(c["lr_gate"])})
            if fe.sensor is not None and not fe.freeze_sensor:
                groups.append({"name": "sensor", "params": list(fe.sensor.parameters()),
                               "lr": float(c["lr_sensor"])})
        if self.train_det:
            groups.append({"name": "detector", "params": list(self.det.parameters()),
                           "lr": float(c["lr_detector"])})
        return torch.optim.AdamW(groups, weight_decay=float(c["weight_decay"]))

    def check_frozen(self) -> None:
        assert_detector_state(self.det, self.train_det)
        fe = self.frontend
        if fe is not None and fe.sensor is not None and fe.freeze_sensor:
            assert not any(p.requires_grad for p in fe.sensor.parameters()) and not fe.sensor.training
        detp = {id(p) for p in self.det.parameters()}
        in_opt = any(id(p) in detp for g in self.opt.param_groups for p in g["params"])
        assert in_opt == self.train_det, "detector params / optimizer mismatch"

    # -------------------------------------------------------------- checkpoints
    def _state(self) -> dict[str, Any]:
        s = {"frontend": self.frontend.state_dict() if self.frontend is not None else None,
             "detector": self.det.model.state_dict() if self.train_det else None}
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

    def try_resume(self) -> bool:
        last = self.run_dir / "last.pt"
        if not last.exists():
            return False
        ck = torch.load(last, map_location="cpu", weights_only=False)
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
                                 id=self.cfg["run_id"], resume="allow", config=self.cfg, dir=str(self.run_dir),
                                 settings=wandb.Settings(init_timeout=60))
        except Exception as e:  # noqa: BLE001
            self.wb = None
            print(f"WARNING: wandb init failed, continuing without W&B: {e}", flush=True)

    def _wandb_log(self, d: dict[str, Any], step: int) -> None:
        if self.wb is None:
            return
        try:
            self.wb.log(d, step=step)
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
        for cond in self.val_conds:
            recs = self.source.records("val", condition_comp(cond))
            ld = DataLoader(ConditionDataset(self.source, recs, cond), batch_size=64, shuffle=False,
                            num_workers=int(self.cfg["val_workers"]), pin_memory=self.dev.type == "cuda")
            wsum.clear(); wn[0] = 0
            prob = predict(fn, ld, str(self.dev))
            aucs[cond.name] = frame_auc([r.label for r in recs], prob)
            if wn[0]:
                wmaps[cond.name] = (torch.stack(wsum).sum(0) / wn[0]).tolist()
        mean_auc = float(np.nanmean(list(aucs.values())))
        if not np.isfinite(mean_auc):
            raise RuntimeError(f"val selection metric undefined: {aucs}")
        self.model.train(was_training)
        return {"aucs": aucs, "mean_auc": mean_auc, "wmaps": wmaps}

    # -------------------------------------------------------------- training
    def after_step(self, step: int) -> None:
        """Hook called after every optimizer step (tests inject faults here)."""

    def _loader(self, epoch: int, order: np.ndarray, skip: int) -> DataLoader:
        nw = int(self.cfg["num_workers"])
        ds = TrainDataset(self.source, self.records, order, skip * self.bs, int(self.cfg["seed"]), epoch, self.deg)
        return DataLoader(ds, batch_size=self.bs, shuffle=False, drop_last=True, num_workers=nw,
                          pin_memory=self.dev.type == "cuda", prefetch_factor=2 if nw else None)

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
        gn = torch.zeros(2, device=self.dev)
        for xb, yb, ridx, spe_l in loader:
            b = st["step_in_epoch"]
            self._log_line("batches.log", f"{epoch} {b} {st['step']} {batch_hash(ridx)}")
            x = xb.to(self.dev, non_blocking=True).float().div_(255.0)
            y = yb.to(self.dev, non_blocking=True)
            if self.native_ucf:
                loss = self.model.native_ucf_loss(x, y, spe_l.to(self.dev))
            else:
                logit, _ = self.model(x)
                loss = loss_fn(logit, y)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            for i, ps in enumerate((self._sensor_params, self._gate_params)):
                if ps:
                    n = _grad_norm(ps).to(self.dev)
                    gn[i] = n
                    stats[i] += (~torch.isfinite(n)).float()
                    stats[2 + i] += (n == 0).float()
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:  # an fp16-overflow skip must not advance the LR schedule
                self.sched.step()
            st["step"] += 1
            st["step_in_epoch"] += 1
            accl = loss.detach() if accl is None else accl + loss.detach()
            acc_n += 1
            acc_img += len(x)
            if st["step"] % int(cfg["frozen_check_every"]) == 0:
                self.check_frozen()
            if st["step"] % log_every == 0 or st["step_in_epoch"] == self.spe:
                self._log_train(epoch, accl, acc_n, acc_img, time.time() - t_log, stats, gn)
                accl, acc_n, acc_img, t_log = None, 0, 0, time.time()
                stats.zero_()
            if st["step"] % int(cfg["ckpt_every_steps"]) == 0:
                self.save_last()
            self.hb.beat(st["step"], epoch=epoch, val_auc=st["best"], img_s=self._img_s,
                         eta=self._eta())
            self.after_step(st["step"])
            if cfg["max_steps"] and st["step"] >= int(cfg["max_steps"]):
                return False
        return True

    _img_s = 0.0

    def _eta(self) -> str:
        done = self.st["step"]
        if done == 0 or self._img_s <= 0:
            return "?"
        left = max(0, self.spe * int(self.cfg["max_epochs"]) - done)
        return f"{int(left * self.bs / self._img_s)}s"

    def _log_train(self, epoch: int, accl: Tensor | None, n: int, imgs: int, dt: float, stats: Tensor,
                   gn: Tensor) -> None:
        st = self.st
        loss = float(accl) / max(1, n) if accl is not None else float("nan")
        self._img_s = imgs / max(dt, 1e-9)
        s, g = stats.tolist(), gn.tolist()
        row = {"step": st["step"], "epoch": epoch, "loss": f"{loss:.5f}",
               **{f"lr_{pg['name']}": f"{pg['lr']:.3e}" for pg in self.opt.param_groups},
               "img_s": f"{self._img_s:.1f}",
               "gpu_mem_mb": f"{torch.cuda.max_memory_allocated() / 2**20:.0f}" if self.dev.type == "cuda" else 0,
               "gnorm_sensor": g[0], "gnorm_gate": g[1], "nonfinite_sensor": int(s[0]), "nonfinite_gate": int(s[1]),
               "zero_sensor": int(s[2]), "zero_gate": int(s[3])}
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
        try:
            res = self._train_loop()
        except Exception as e:  # noqa: BLE001
            self._wandb_alert(f"{self.cfg['run_id']} crashed", f"{type(e).__name__}: {e}", error=True)
            raise
        self._wandb_alert(f"{self.cfg['run_id']} finished", json.dumps(res))
        if self.wb is not None:
            try:
                self.wb.finish()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: wandb finish failed: {e}", flush=True)
        return res

    def _train_loop(self) -> dict[str, Any]:
        st, cfg = self.st, self.cfg
        truncated = False
        while st["epoch"] < int(cfg["max_epochs"]) and not st["stopped"]:
            epoch = st["epoch"]
            if st["step_in_epoch"] < self.spe:
                if not self._train_epoch(epoch):
                    truncated = True
                    break
            self._end_of_epoch(epoch)
        if truncated:
            self.save_last()
            res = {"run_id": cfg["run_id"], "truncated": True, "step": st["step"], "img_s": self._img_s}
            (self.run_dir / "done.json").write_text(json.dumps(res))
            return res
        return self._finalize()

    def _finalize(self) -> dict[str, Any]:
        """Load best.pt and write the evaluation artefacts (frontend.pt, detector.pt)."""
        st, cfg = self.st, self.cfg
        best = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=False)
        self._load_model_state(best["model"])
        meta = {"best_epoch": st["best_epoch"], "val_mean_auc": st["best"], "git_commit": git_commit()}
        if self.frontend is not None:
            save_frontend(self.run_dir / "frontend.pt", self.frontend, cfg, meta)
        if self.train_det:
            _save_atomic(self.det.model.state_dict(), self.run_dir / "detector.pt")
        res = {"run_id": cfg["run_id"], "truncated": False, "variant": self.variant, "epochs_run": st["epoch"],
               "stopped_early": bool(st["stopped"]), "total_time_s": time.time() - self.t_start, **meta}
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
