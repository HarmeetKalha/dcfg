"""GenD-aug: public GenD (CLIP ViT-L/14, LayerNorm-tuned) fine-tuned with GenD's own recipe on our degraded FF++ c23 stream.

    python -m dcfg.gend.train_gend_aug --config configs/train/gend_aug.yaml [--run-dir DIR] [--set key=value ...]
    python -m dcfg.gend.train_gend_aug --config configs/train/gend_aug.yaml --smoke   # short smoke + resume check

Schedule (fixed 2026-11-20 for GenD-aug and every later LUMEN2 run, see docs/DECISIONS.md):
  * training frames = exactly the augft frames (FF++ c23 train, all frames per video, epoch_mode ``full`` of
    ``dcfg.train.train_frontend.epoch_order``: every fake frame once, reals re-cycled to 1:1), GenD re-crops (256 px);
  * our PLAN 7.2 degradation (``TrainDegradation``, seeds ``make_rng(seed, epoch, position)``) is applied to the 256 px crop,
    THEN GenD's resize (PIL bicubic 224) and CLIP mean/std (inside the model). No other augmentation (GenD's own blur/JPEG/
    affine/jitter are replaced by our stream, so later LUMEN2 models see exactly the same inputs);
  * validation = the augft degraded val set (``val_conditions`` of configs/eval/conditions.yaml, c23 val split) subsampled
    to ``val_frames_per_video`` (8) of the ~32 frames per video with ``np.linspace`` over the sorted frame keys; selection
    metric = mean frame AUC over the val conditions (same as augft); val only, the test split is never read here;
  * AdamW (betas 0.9/0.999, weight decay 0), lr 3e-4 -> 1 epoch linear warm-up from min_lr 1e-5, then cosine to min_lr at the
    end of epoch ``max_epochs`` (18); patience 6; seed 42; bf16 autocast;
  * loss = 1.0 CE + 0.5 uniformity + 0.1 label-alignment on the L2 embeddings (GenD ``wacv-LN+L2+UnAl``);
  * trainable = LayerNorm params (``pre_layrnorm``, ``layer_norm1``, ``layer_norm2``, ``post_layernorm``) + the 1024->2 head;
  * effective batch 128 = ``accum`` (4) micro-batches of ``batch_size`` (32). Uniformity/alignment are computed per MICRO-batch
    (32 samples, 496 pairs) and averaged over the 4 micro-batches; GenD's batch-128 loss would use all 8128 pairs. CE is
    unaffected (mean over 128 either way).
Resumable: ``last.pt`` (trainable params + optimizer + scheduler + RNG + state) is written every ``ckpt_every_steps``
optimiser steps and at each epoch end; the resumed run reproduces the same batch order and degradations exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dcfg.data.degradations import TrainDegradation, load_conditions, load_val_conditions
from dcfg.data.ffpp import FrameRecord
from dcfg.eval.evaluate import ConditionDataset, condition_comp, git_commit, predict
from dcfg.eval.metrics import frame_auc
from dcfg.gend.crop import gend_resize
from dcfg.paths import DATA_ROOT, REPO_ROOT, RUNS_ROOT
from dcfg.train.train_frontend import (TrainDataset, _rng_state, _save_atomic, _set_rng_state, batch_hash, config_diff,
                                       epoch_order)
from dcfg.utils.heartbeat import Heartbeat
from dcfg.utils.seed import seed_everything

HF_ID = "yermandy/GenD_CLIP_L_14"
HF_REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"
EXPECTED_TRAINABLE = 104_450  # 102,400 LayerNorm + 2,050 head
LN_KEYS = ("layer_norm", "layrnorm", "post_layernorm")
MAX_CONSEC_NONFINITE = 10


# ---------------------------------------------------------------------------------------------- losses (GenD src/losses/unifalign.py)
def alignment(z: Tensor, labels: Tensor, alpha: float = 2.0) -> Tensor:
    """Label-aware alignment: mean ||z_i - z_j||^alpha over same-label pairs i<j (0 if none)."""
    if z.size(0) < 2:
        return z.sum() * 0.0
    same = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    idx = torch.nonzero(same, as_tuple=False)
    if idx.numel() == 0:
        return z.sum() * 0.0
    return (z[idx[:, 0]] - z[idx[:, 1]]).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(z: Tensor, t: float = 2.0, clip_value: float = 1e-6) -> Tensor:
    """log mean exp(-t * ||z_i - z_j||^2) over all pairs i<j (clamped like GenD)."""
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


def gend_loss(logits: Tensor, z: Tensor, y: Tensor, w_ce: float = 1.0, w_uni: float = 0.5,
              w_align: float = 0.1) -> tuple[Tensor, dict[str, Tensor]]:
    """GenD loss on 2-class logits and L2-normalised embeddings; fp32. Returns (total, terms)."""
    logits, z, y = logits.float(), z.float(), y.long()
    ce, uni, al = F.cross_entropy(logits, y), uniformity(z), alignment(z, y)
    return w_ce * ce + w_uni * uni + w_align * al, {"ce": ce.detach(), "uniformity": uni.detach(), "alignment": al.detach()}


# ---------------------------------------------------------------------------------------------- parameters / checkpoints
def is_trainable_name(name: str) -> bool:
    """GenD recipe: LayerNorm params of the vision tower + the linear head."""
    return any(k in name for k in LN_KEYS) or name.startswith("linear.")


def set_trainable(model: nn.Module) -> list[tuple[str, Tensor]]:
    """Freeze everything except the GenD-trainable set; return ``[(name, param)]`` of the trainable ones."""
    out = []
    for n, p in model.named_parameters():
        p.requires_grad = is_trainable_name(n)
        if p.requires_grad:
            out.append((n, p))
    return out


def trainable_state(model: nn.Module) -> dict[str, Tensor]:
    """CPU clones of the trainable parameters only."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_trainable_name(n)}


def load_trainable(model: nn.Module, sd: dict[str, Tensor]) -> None:
    """Strictly load a trainable-only state dict onto ``model`` (keys must equal the trainable set)."""
    params = dict(model.named_parameters())
    want = {n for n in params if is_trainable_name(n)}
    if set(sd) != want:
        raise KeyError(f"trainable keys differ: missing {sorted(want - set(sd))[:3]}, unexpected {sorted(set(sd) - want)[:3]}")
    with torch.no_grad():
        for n, v in sd.items():
            params[n].copy_(v.to(params[n].device, params[n].dtype))


def mb(path: Path) -> float:
    return round(path.stat().st_size / 2**20, 3)


def lr_factor(step: int, spe: int, warmup_epochs: float, max_epochs: int, lr: float, min_lr: float) -> float:
    """Multiplier of ``lr``: linear warm-up min_lr -> lr over ``warmup_epochs``, then cosine lr -> min_lr until max_epochs."""
    total, warm = spe * max_epochs, int(spe * warmup_epochs)
    r0 = min_lr / lr
    if warm > 0 and step < warm:
        return r0 + (1.0 - r0) * step / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return r0 + (1.0 - r0) * 0.5 * (1.0 + math.cos(math.pi * prog))


def val_subsample(records: list[FrameRecord], per_video: int) -> list[FrameRecord]:
    """Deterministic subset: per video, ``per_video`` frames at ``np.linspace(0, n-1, per_video)`` of the key-sorted frames."""
    by: dict[str, list[FrameRecord]] = {}
    for r in records:
        by.setdefault(r.video, []).append(r)
    out: list[FrameRecord] = []
    for v in sorted(by):
        rs = sorted(by[v], key=lambda r: r.key)
        if len(rs) <= per_video:
            out += rs
            continue
        idx = np.unique(np.round(np.linspace(0, len(rs) - 1, per_video)).astype(int))
        out += [rs[i] for i in idx]
    return out


def stratified_videos(records: list[FrameRecord], n: int | None) -> set[str] | None:
    """Smoke only: first ``n`` videos per class-of-origin directory (sorted), None = all."""
    if not n:
        return None
    by: dict[str, list[str]] = {}
    for v in sorted({r.video for r in records}):
        by.setdefault(v.split("/")[0], []).append(v)
    return {v for vs in by.values() for v in vs[: max(1, n // len(by))]}


class GendTrainSet(Dataset):
    """Degraded + GenD-resized uint8 CHW samples for one epoch slice (wraps our ``TrainDataset``: same order/seeds)."""

    def __init__(self, inner: TrainDataset) -> None:
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, j: int) -> tuple[Tensor, Tensor, int]:
        img, y, ridx, _ = self.inner[j]
        a = gend_resize(img.permute(1, 2, 0).numpy())
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1), y, ridx


def crop_source(crops_root: Path | None = None):
    from dcfg.gend.evaluate_gend import CropSource

    root = Path(crops_root) if crops_root else DATA_ROOT / "gend_crops"
    return CropSource(root)


def load_cfg(path: str | os.PathLike[str], overrides: list[str] | None = None) -> dict[str, Any]:
    p = Path(path)
    cfg = yaml.safe_load((p if p.is_absolute() else REPO_ROOT / p).read_text())
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


# ---------------------------------------------------------------------------------------------- trainer
class GendAugTrainer:
    """One resumable GenD-aug run in ``run_dir``."""

    def __init__(self, cfg: dict[str, Any], run_dir: Path, model_fn: Callable[[], nn.Module] | None = None,
                 source: Any = None, expected_trainable: int | None = EXPECTED_TRAINABLE) -> None:
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(int(cfg["seed"]))
        self.source = source if source is not None else crop_source(cfg.get("crops_root"))
        self.records = self.source.records("train", cfg["comp"])
        self.labels = np.array([r.label for r in self.records])
        self.bs, self.accum = int(cfg["batch_size"]), int(cfg["accum"])
        self.eff = self.bs * self.accum
        self.spe = len(epoch_order(self.labels, 0, 0, cfg["epoch_mode"])) // self.eff  # optimiser steps / epoch
        if self.spe < 1:
            raise ValueError("fewer frames than one effective batch")
        if model_fn is None:
            from dcfg.gend.model import load_gend

            model_fn = lambda: load_gend("cpu")  # noqa: E731
        self.model = model_fn().to(self.dev)
        self.params = [p for _, p in set_trainable(self.model)]
        self.n_trainable = sum(p.numel() for p in self.params)
        if expected_trainable is not None and self.n_trainable != expected_trainable:
            raise RuntimeError(f"trainable params {self.n_trainable} != expected {expected_trainable}")
        self.opt = torch.optim.AdamW(self.params, lr=float(cfg["lr"]), betas=tuple(cfg["betas"]),
                                     weight_decay=float(cfg["weight_decay"]))
        f = lambda s: lr_factor(s, self.spe, float(cfg["warmup_epochs"]), int(cfg["max_epochs"]),  # noqa: E731
                                float(cfg["lr"]), float(cfg["min_lr"]))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, f)
        self.deg = TrainDegradation()
        self.st: dict[str, Any] = {"epoch": 0, "step_in_epoch": 0, "step": 0, "best": -float("inf"), "best_epoch": -1,
                                   "bad": 0, "stopped": False, "history": [], "train_s": 0.0, "val_s": 0.0,
                                   "nonfinite_steps": 0}
        self._consec_nf = 0
        self.hb = Heartbeat(Path(os.environ.get("DCFG_RUN_DIR") or self.run_dir), float(cfg.get("heartbeat_s", 60)))
        allc = load_conditions(REPO_ROOT / cfg["conditions_yaml"])
        self.val_names = load_val_conditions(REPO_ROOT / cfg["conditions_yaml"])
        self.val_conds = [allc[n] for n in self.val_names]
        self._val_recs: dict[str, list[FrameRecord]] = {}
        self.t_start = time.time()

    # ---- val
    def val_records(self, comp: str) -> list[FrameRecord]:
        if comp not in self._val_recs:
            recs = val_subsample(self.source.records("val", comp), int(self.cfg["val_frames_per_video"]))
            keep = stratified_videos(recs, self.cfg.get("val_max_videos"))
            self._val_recs[comp] = [r for r in recs if keep is None or r.video in keep]
        return self._val_recs[comp]

    def val_set_hash(self) -> str:
        h = hashlib.sha256()
        for c in sorted({condition_comp(c) for c in self.val_conds}):
            h.update("\n".join(r.key for r in self.val_records(c)).encode())
        return h.hexdigest()

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        t0 = time.time()
        self.model.eval()
        aucs, nfr = {}, {}
        for cond in self.val_conds:
            recs = self.val_records(condition_comp(cond))
            ld = DataLoader(ConditionDataset(self.source, recs, cond, post_transform=gend_resize), batch_size=64,
                            shuffle=False, num_workers=int(self.cfg["val_workers"]), pin_memory=self.dev.type == "cuda")
            prob = predict(self.model, ld, str(self.dev), fp16=True,
                           on_batch=lambda n: self.hb.beat(self.st["step"], phase="val", cond=cond.name, seen=n))
            aucs[cond.name] = frame_auc([r.label for r in recs], prob)
            nfr[cond.name] = len(recs)
        bad = [k for k, a in aucs.items() if not np.isfinite(a)]
        if bad:
            raise RuntimeError(f"non-finite val AUC for {bad}: {aucs}")
        self.model.train()
        self.st["val_s"] += time.time() - t0
        return {"aucs": aucs, "mean_auc": float(np.mean(list(aucs.values()))), "n_frames": nfr}

    # ---- checkpoints
    def save_last(self) -> None:
        _save_atomic({"trainable": trainable_state(self.model), "optimizer": self.opt.state_dict(),
                      "scheduler": self.sched.state_dict(), "state": dict(self.st), "rng": _rng_state(),
                      "config": self.cfg}, self.run_dir / "last.pt")

    def try_resume(self) -> bool:
        last = self.run_dir / "last.pt"
        if not last.exists():
            return False
        ck = torch.load(last, map_location="cpu", weights_only=False)
        diff = config_diff(ck["config"], self.cfg)
        diff = {k: v for k, v in diff.items() if k not in ("max_steps", "ckpt_every_steps")}  # smoke / ops knobs
        if diff and not self.cfg.get("allow_config_change"):
            raise RuntimeError(f"config differs from {last}: {diff}; pass --set allow_config_change=true to resume anyway")
        load_trainable(self.model, ck["trainable"])
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        self.st = ck["state"]
        _set_rng_state(ck["rng"])
        self._log_line("batches.log", f"# resumed epoch={self.st['epoch']} step_in_epoch={self.st['step_in_epoch']} step={self.st['step']}")
        print(f"resumed at step {self.st['step']} (epoch {self.st['epoch']}, batch {self.st['step_in_epoch']})", flush=True)
        return True

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

    # ---- train
    def _loader(self, epoch: int, order: np.ndarray, skip: int) -> DataLoader:
        nw = int(self.cfg["num_workers"])
        ds = GendTrainSet(TrainDataset(self.source, self.records, order, skip * self.eff, int(self.cfg["seed"]), epoch, self.deg))
        return DataLoader(ds, batch_size=self.bs, shuffle=False, drop_last=True, num_workers=nw,
                          pin_memory=self.dev.type == "cuda", prefetch_factor=2 if nw else None)

    def _train_epoch(self, epoch: int) -> bool:
        """Run the rest of ``epoch``; False if ``max_steps`` stopped it (smoke)."""
        cfg, st = self.cfg, self.st
        order = epoch_order(self.labels, int(cfg["seed"]), epoch, cfg["epoch_mode"])[: self.spe * self.eff]
        it = iter(self._loader(epoch, order, st["step_in_epoch"]))
        self.model.train()
        t_epoch, t_log, n_img, acc, n_acc = time.time(), time.time(), 0, {}, 0
        while st["step_in_epoch"] < self.spe:
            mbs = [next(it) for _ in range(self.accum)]
            self._log_line("batches.log", f"{epoch} {st['step_in_epoch']} {st['step']} "
                           f"{batch_hash(torch.tensor([r for m in mbs for r in m[2].tolist()]))}")
            self.opt.zero_grad(set_to_none=True)
            finite, terms_step = True, {}
            for xb, yb, _ in mbs:
                x = xb.to(self.dev, non_blocking=True).float().div_(255.0)
                y = yb.to(self.dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                    logits, z = self.model.features(x)
                loss, terms = gend_loss(logits, z, y, float(cfg["w_ce"]), float(cfg["w_uniformity"]), float(cfg["w_alignment"]))
                finite = finite and bool(torch.isfinite(loss))
                (loss / self.accum).backward()
                for k, v in {"loss": loss.detach(), **terms}.items():
                    terms_step[k] = terms_step.get(k, 0.0) + float(v) / self.accum
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, float("inf")))  # norm only (no clipping)
            if finite and math.isfinite(gnorm):
                self.opt.step()
                self.sched.step()
                self._consec_nf = 0
            else:  # skip the update, never advance the schedule
                st["nonfinite_steps"] += 1
                self._consec_nf += 1
                self._log_line("batches.log", f"# non-finite step {st['step']} (loss finite={finite}, gnorm={gnorm})")
                if self._consec_nf > MAX_CONSEC_NONFINITE:
                    raise RuntimeError(f"ABORT: {self._consec_nf} consecutive non-finite steps at step {st['step']}")
            st["step"] += 1
            st["step_in_epoch"] += 1
            n_img += self.eff
            n_acc += 1
            for k, v in terms_step.items():
                acc[k] = acc.get(k, 0.0) + v
            if st["step"] % int(cfg["log_every"]) == 0 or st["step_in_epoch"] == self.spe:
                dt = time.time() - t_log
                self._csv_row("train_log.csv", {"step": st["step"], "epoch": epoch, **{k: f"{v / n_acc:.5f}" for k, v in acc.items()},
                                                "lr": f"{self.opt.param_groups[0]['lr']:.3e}", "gnorm": f"{gnorm:.4f}",
                                                "img_s": f"{n_img / max(dt, 1e-9):.1f}", "nonfinite_steps": st["nonfinite_steps"],
                                                "gpu_mem_mb": f"{torch.cuda.max_memory_allocated() / 2**20:.0f}" if self.dev.type == "cuda" else 0})
                self._img_s = n_img / max(dt, 1e-9)
                acc, n_acc, n_img, t_log = {}, 0, 0, time.time()
            if st["step"] % int(cfg["ckpt_every_steps"]) == 0:
                now = time.time()
                st["train_s"] += now - t_epoch
                t_epoch = now
                self.save_last()
            self.hb.beat(st["step"], epoch=epoch, best=st["best"], img_s=getattr(self, "_img_s", 0.0))
            if cfg.get("max_steps") and st["step"] >= int(cfg["max_steps"]):
                st["train_s"] += time.time() - t_epoch
                return False
        st["train_s"] += time.time() - t_epoch
        return True

    _img_s = 0.0

    def _end_of_epoch(self, epoch: int) -> None:
        st = self.st
        v = self.validate()
        better = v["mean_auc"] > st["best"]
        if better:
            st.update(best=v["mean_auc"], best_epoch=epoch, bad=0)
            _save_atomic({"trainable": trainable_state(self.model), "epoch_1based": epoch + 1, "val_mean_auc": v["mean_auc"]},
                         self.run_dir / "best_trainable.pt")
        else:
            st["bad"] += 1
        st["history"].append({"epoch": epoch + 1, "val_mean_auc": v["mean_auc"], **{f"val_{k}": a for k, a in v["aucs"].items()}})
        st["epoch"], st["step_in_epoch"] = epoch + 1, 0
        if st["bad"] >= int(self.cfg["patience"]):
            st["stopped"], st["stop_epoch"] = True, epoch
        self._csv_row("metrics.csv", {"epoch": epoch + 1, "step": st["step"], "val_mean_auc": f"{v['mean_auc']:.5f}",
                                      **{f"val_{k}": f"{a:.5f}" for k, a in v["aucs"].items()},
                                      "lr": f"{self.opt.param_groups[0]['lr']:.3e}", "bad_epochs": st["bad"], "best_epoch": st["best_epoch"] + 1})
        self.save_last()
        self.hb.beat(st["step"], force=True, epoch=epoch, val_auc=v["mean_auc"])
        print(f"epoch {epoch + 1}: val mean AUC {v['mean_auc']:.4f} (best {st['best']:.4f} @ epoch {st['best_epoch'] + 1})", flush=True)

    def train(self) -> dict[str, Any]:
        done = self.run_dir / "done.json"
        if done.exists():
            return json.loads(done.read_text())
        self.try_resume()
        st, cfg = self.st, self.cfg
        while st["epoch"] < int(cfg["max_epochs"]) and not st["stopped"]:
            epoch = st["epoch"]
            if st["step_in_epoch"] < self.spe and not self._train_epoch(epoch):
                self.save_last()
                res = {"run_id": cfg["run_id"], "truncated": True, "step": st["step"]}
                if cfg.get("val_on_truncate"):
                    v = self.validate()
                    res["val"] = v
                (self.run_dir / "done.json").write_text(json.dumps(res))
                return res
            self._end_of_epoch(epoch)
        return self._finalize()

    def _finalize(self) -> dict[str, Any]:
        st, cfg = self.st, self.cfg
        epochs_run = st["epoch"]
        stop_reason = "patience" if st["stopped"] else "epoch_cap"
        best = torch.load(self.run_dir / "best_trainable.pt", map_location="cpu", weights_only=False)
        load_trainable(self.model, best["trainable"])
        full = self.run_dir / "best_full.pt"
        torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, full)
        tr = self.run_dir / "best_trainable.pt"
        best_1, cap = st["best_epoch"] + 1, int(cfg["max_epochs"])
        res = {"run_id": cfg["run_id"], "truncated": False, "epoch_base": 1, "stop_reason": stop_reason,
               "best_epoch": best_1, "stop_epoch": epochs_run, "epochs_run": epochs_run, "max_epochs": cap,
               "still_improving_at_cap": bool(stop_reason == "epoch_cap" and best_1 >= cap - 1),
               "best_val_mean_auc": st["best"], "val_curve": st["history"], "val_conditions": self.val_names,
               "val_frames_per_video": int(cfg["val_frames_per_video"]), "val_set_sha256": self.val_set_hash(),
               "trainable_params": self.n_trainable, "trainable_file": str(tr), "trainable_file_mb": mb(tr),
               "full_model_file": str(full), "full_model_file_mb": mb(full),
               "train_s": st["train_s"], "val_s": st["val_s"], "train_wall_time_s": st["train_s"] + st["val_s"],
               "nonfinite_steps": st["nonfinite_steps"], "steps": st["step"], "steps_per_epoch": self.spe,
               "git_commit": git_commit(), "seed": int(cfg["seed"]), "train_config": dict(cfg),
               "init": {"model_id": HF_ID, "hf_revision": HF_REVISION}, "pilot": bool(cfg.get("pilot", False))}
        (self.run_dir / "done.json").write_text(json.dumps(res, indent=2))
        return res


# ---------------------------------------------------------------------------------------------- smoke
def _file_hashes(path: Path) -> list[str]:
    return [ln.split()[-1] for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


def smoke(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Few hundred micro-steps + tiny val + exact-resume check.

    Run A: N optimiser steps straight. Run B: N/2 steps, stop (checkpoint), new trainer resumes to N. Checks: identical batch
    hashes, trainable params of A and B equal (tolerance for bf16 non-determinism), finite loss, tiny val AUC finite.
    """
    n = int(cfg.get("smoke_steps", 50))
    a_dir, b_dir = run_dir / "smoke_A", run_dir / "smoke_B"
    for d in (a_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)
    base = {**cfg, "run_id": "gend_aug_smoke", "ckpt_every_steps": 1000, "val_max_videos": 10, "val_on_truncate": True,
            "log_every": max(1, min(int(cfg["log_every"]), n // 2))}
    t0 = time.time()
    ta = GendAugTrainer({**base, "max_steps": n}, a_dir)
    ra = ta.train()
    t_a = time.time() - t0
    tb = GendAugTrainer({**base, "max_steps": n // 2}, b_dir)
    tb.train()
    (b_dir / "done.json").unlink()
    tb2 = GendAugTrainer({**base, "max_steps": n}, b_dir)
    tb2.train()
    assert "# resumed" in (b_dir / "batches.log").read_text(), "run B did not resume from its checkpoint"
    ha, hb = _file_hashes(a_dir / "batches.log"), _file_hashes(b_dir / "batches.log")
    assert ha == hb, "batch order after resume differs from the straight run"
    sa, sb = trainable_state(ta.model), trainable_state(tb2.model)
    md = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    rows = list(csv.DictReader((a_dir / "train_log.csv").open()))
    losses = [float(r["loss"]) for r in rows]
    out = {"steps": n, "micro_steps": n * int(cfg["accum"]), "batch_hash_equal": True, "max_abs_param_diff_resume_vs_straight": md,
           "loss_first_last": [losses[0], losses[-1]], "all_losses_finite": bool(np.isfinite(losses).all()),
           "tiny_val": ra.get("val"), "img_s": float(rows[-1]["img_s"]), "gpu_mem_mb": float(rows[-1]["gpu_mem_mb"]),
           "trainable_params": ta.n_trainable, "straight_run_s": round(t_a, 1)}
    assert out["all_losses_finite"] and md < 5e-3, out
    (run_dir / "smoke.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.gend.train_gend_aug", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config, a.set)
    run_dir = Path(a.run_dir or os.environ.get("DCFG_RUN_DIR") or RUNS_ROOT / cfg["run_id"]).expanduser()
    if a.smoke:
        smoke(cfg, run_dir)
        return 0
    res = GendAugTrainer(cfg, run_dir).train()
    print(json.dumps({k: v for k, v in res.items() if k not in ("train_config", "val_curve")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
