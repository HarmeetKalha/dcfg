"""LUMEN2 training: GenD-aug's exact recipe + the LUMEN parts (sensor, DCT gate, conditional LayerNorm).

    python -m dcfg.lumen2.train --config configs/train/lumen2.yaml [--run-dir DIR] [--set key=value ...]

Subclass of :class:`dcfg.gend.train_gend_aug.GendAugTrainer`: SAME training frames/epoch order/degradation stream (the sampler is
called with the same ``make_rng(seed, epoch, position)``; ``dcfg.degrade.labels.apply_with_labels`` only adds the labels), SAME
validation set and selection metric, optimiser (AdamW lr 3e-4, wd 0, 1-epoch warm-up + cosine over 18 epochs, min lr 1e-5),
batch 128 = 4 x 32, CE + 0.5 uniformity + 0.1 alignment, patience 6, seed 42, max 18 epochs. Differences = the added parts:
  loss = GenD loss + 0.1 * damage_loss(sensor) + 1e-4 * mean((w-1)^2)   (no consistency loss);
  trainable = GenD-aug's set (vision LayerNorms + head, 104,450) + sensor + gate + LN generator, all in ONE AdamW group.
Checkpoints (``last.pt``, ``best_trainable.pt``) hold :func:`dcfg.lumen2.model.trainable_state` (incl. sensor BatchNorm buffers).

Monday backup (4e): at the end of an epoch, if now >= Mon 23 Nov 2026 18:00 IST (12:30 UTC), the marker ``monday_done.json`` does not
exist and training is not finished, the best checkpoint so far is evaluated on clean + the 17 synthetic conditions + c40 on a fixed
``np.linspace`` 8-of-32 frames per video subset of the TEST videos (+ sensor accuracy), written to
``results/lumen2/partial_epochN.json`` and ``results/STATUS_FOR_MEETING.md`` (labelled partial), committed/pushed, one ntfy line;
then training continues. Interim status only, never a paper number.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dcfg.data.degradations import load_conditions, make_rng
from dcfg.degrade.labels import apply_with_labels
from dcfg.degrade.sensor import damage_loss
from dcfg.gend.train_gend_aug import GendAugTrainer, load_cfg
from dcfg.lumen2.model import (GATE_PENALTY, GENDTRAIN_PARAMS, Lumen2Model, build_lumen2, load_trainable, pack_views,
                               trainable_state)
from dcfg.paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT
from dcfg.train.train_frontend import TrainDataset, _rng_state, _set_rng_state

W_DAMAGE = 0.1
MONDAY_DEADLINE_UTC = datetime(2026, 11, 23, 12, 30, tzinfo=timezone.utc)  # Mon 23 Nov 18:00 IST
PARTIAL_FRAMES_PER_VIDEO = 8


def should_run_partial(now: datetime, deadline: datetime, already_done: bool, finished: bool) -> bool:
    """Monday-backup trigger: after the deadline, once, and only while training is not finished."""
    return (not already_done) and (not finished) and now >= deadline


class LabelledTrainDataset(TrainDataset):
    """:class:`TrainDataset` that also returns the degradation labels and the packed two-view image.

    Item: ``(packed uint8 (3,480,256), label, record idx, applied (4,), strength (4,))``. The degraded crop is bit-identical to
    ``TrainDataset`` (same rng, same call: ``apply_with_labels`` only post-processes the sampled op list)."""

    def __getitem__(self, j: int) -> tuple:
        pos = self.offset + j
        ridx = int(self.order[pos])
        rec = self.records[ridx]
        clean = self.source.load(rec)
        img, _, applied, strength = apply_with_labels(self.deg, clean, make_rng(self.seed, self.epoch, pos))
        packed = torch.from_numpy(np.ascontiguousarray(pack_views(img))).permute(2, 0, 1)
        return packed, torch.tensor(float(rec.label)), ridx, torch.from_numpy(applied), torch.from_numpy(strength)


class Lumen2Trainer(GendAugTrainer):
    """Resumable LUMEN2 run (see module docstring)."""

    post_transform = staticmethod(pack_views)

    def __init__(self, cfg: dict[str, Any], run_dir: Path, model_fn: Any = None, source: Any = None,
                 expected_gend_trainable: int | None = GENDTRAIN_PARAMS, now_fn: Any = None) -> None:
        self.expected_gend_trainable = expected_gend_trainable
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        super().__init__(cfg, run_dir, model_fn or (lambda: build_lumen2("cpu")), source, expected_trainable=None)
        self.deadline = datetime.fromisoformat(str(cfg.get("monday_deadline_utc", MONDAY_DEADLINE_UTC.isoformat())))
        self.part_counts = self.model.part_param_counts()

    # ---- hooks
    def _init_params(self) -> list[Tensor]:
        m: Lumen2Model = self.model
        out, gend = [], 0
        for n, p in m.named_parameters():
            p.requires_grad = Lumen2Model.is_trainable_name(n)
            if p.requires_grad:
                out.append(p)
                gend += p.numel() if Lumen2Model.is_gend_name(n) else 0
        if self.expected_gend_trainable is not None and gend != self.expected_gend_trainable:
            raise RuntimeError(f"GenD trainable params {gend} != GenD-aug's {self.expected_gend_trainable}")
        return out

    def _trainable_state(self) -> dict[str, Tensor]:
        return trainable_state(self.model)

    def _load_trainable(self, sd: dict[str, Tensor]) -> None:
        load_trainable(self.model, sd)

    def _make_trainset(self, epoch: int, order: np.ndarray, skip: int) -> Dataset:
        return LabelledTrainDataset(self.source, self.records, order, skip * self.eff, int(self.cfg["seed"]), epoch, self.deg)

    def _micro_loss(self, mb: tuple) -> tuple[Tensor, dict[str, Tensor]]:
        from dcfg.gend.train_gend_aug import gend_loss

        cfg = self.cfg
        x = mb[0].to(self.dev, non_blocking=True).float().div_(255.0)
        y = mb[1].to(self.dev, non_blocking=True)
        applied, strength = mb[3].to(self.dev, non_blocking=True), mb[4].to(self.dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            logits, z, sens, w = self.model.forward_train(x)
        main, terms = gend_loss(logits, z, y, float(cfg["w_ce"]), float(cfg["w_uniformity"]), float(cfg["w_alignment"]))
        dl, dterms = damage_loss(sens, applied, strength)
        pen = ((w - 1.0) ** 2).mean()
        loss = main + float(cfg.get("w_damage", W_DAMAGE)) * dl + float(cfg.get("w_gate_penalty", GATE_PENALTY)) * pen
        return loss, {**terms, "main": main.detach(), "damage": dl.detach(), "sens_bce": dterms["bce"], "sens_mse": dterms["mse"],
                      "gate_pen": pen.detach()}

    # ---- epoch end + Monday backup
    def _end_of_epoch(self, epoch: int) -> None:
        super()._end_of_epoch(epoch)
        st = self.st
        finished = bool(st["stopped"]) or st["epoch"] >= int(self.cfg["max_epochs"])
        fails = self.run_dir / "monday_failures.txt"
        n_fail = len(fails.read_text().splitlines()) if fails.exists() else 0
        done = (self.run_dir / "monday_done.json").exists() or n_fail >= 2  # at most 2 failed attempts
        if should_run_partial(self.now_fn(), self.deadline, done, finished):
            try:
                self.partial_eval(st["epoch"])
            except Exception as e:  # noqa: BLE001  # the backup must never kill the training run
                self._log_line("batches.log", f"# monday partial eval FAILED: {type(e).__name__}: {e}")
                print(f"monday partial eval failed: {type(e).__name__}: {e}", flush=True)
                with open(fails, "a") as f:
                    f.write(f"epoch {st['epoch']}: {type(e).__name__}: {e}\n")

    def partial_eval(self, epoch_1based: int) -> dict[str, Any]:
        """Evaluate the BEST checkpoint so far on the 8-of-32-frames test subset; write json/status; commit+push; ntfy."""
        from dcfg.degrade.eval_damage_sensor import analyse
        from dcfg.lumen2.evaluate_lumen2 import SubsetSource, quick_eval

        t0 = time.time()
        conds = load_conditions(REPO_ROOT / "configs/eval/conditions.yaml")
        names = [n for n, c in conds.items() if not c.held_out]  # clean + 17 synthetic + c40
        synth = [n for n in names if n not in ("clean", "c40")]
        assert len(synth) == 17, synth
        best = torch.load(self.run_dir / "best_trainable.pt", map_location="cpu", weights_only=False)
        cur, rng = self._trainable_state(), _rng_state()
        self._load_trainable(best["trainable"])
        self.model.eval()
        src = SubsetSource(self.source, PARTIAL_FRAMES_PER_VIDEO)
        auc, preds, base_auc = {}, {}, {}
        try:
            for n in names:
                r = quick_eval(self.model, src, conds[n], workers=int(self.cfg["val_workers"]),
                               on_batch=lambda k, n=n: self.hb.beat(self.st["step"], force=False, phase="monday_partial", cond=n, seen=k))
                auc[n], preds[n] = r["auc"], (r["presence"], r["strength"])
                base_auc[n] = self._gend_aug_auc(src, conds[n])
        finally:
            self._load_trainable(cur)
            self.model.train()
            _set_rng_state(rng)
        sens = analyse(preds, conds)
        mean = lambda d, ks: float(np.mean([d[k] for k in ks if d.get(k) is not None]))  # noqa: E731
        res = {"label": f"partial, epoch {epoch_1based}, {PARTIAL_FRAMES_PER_VIDEO} frames/video", "epoch_trained": epoch_1based,
               "best_epoch": best.get("epoch_1based"), "best_val_mean_auc": best.get("val_mean_auc"),
               "frames_per_video": PARTIAL_FRAMES_PER_VIDEO, "clean": auc["clean"], "synth_mean": mean(auc, synth), "c40": auc["c40"],
               "per_condition": auc,
               "gend_aug_same_subset": None if any(v is None for v in base_auc.values()) else
               {"clean": base_auc["clean"], "synth_mean": mean(base_auc, synth), "c40": base_auc["c40"]},
               "sensor": {"per_op": sens["per_op"], "n_labelled_frames": sens["n_labelled_frames"]},
               "eval_s": round(time.time() - t0), "finished_utc": datetime.now(timezone.utc).isoformat(),
               "note": "interim status, test videos, fixed linspace subset of frames; not a paper number"}
        out = RESULTS_ROOT / "lumen2" / f"partial_epoch{epoch_1based}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=float))
        (self.run_dir / "monday_done.json").write_text(json.dumps({"epoch": epoch_1based, "file": str(out)}))
        b = res["gend_aug_same_subset"]
        summary = (f"LUMEN2 {res['label']}: clean {res['clean']:.3f}, synth mean {res['synth_mean']:.3f}, c40 {res['c40']:.3f}"
                   + ("" if b is None else f" (GenD-aug same subset {b['clean']:.3f}/{b['synth_mean']:.3f}/{b['c40']:.3f})"))
        self._publish(f"lumen2_partial_epoch{epoch_1based}", summary)
        return res

    def _gend_aug_auc(self, src: Any, cond: Any) -> float | None:
        """GenD-aug frame AUC on exactly the same subset of frames (from its saved scores), None if unavailable."""
        from dcfg.eval.evaluate import condition_comp
        from dcfg.eval.metrics import frame_auc

        f = RESULTS_ROOT / "lumen2" / "gend_aug" / "scores" / f"gend_aug__{cond.name}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        pk = dict(zip(z["frame_key"].tolist(), z["frame_prob"].tolist()))
        recs = src.records("test", condition_comp(cond))
        return frame_auc([r.label for r in recs], [pk[r.key] for r in recs])

    @staticmethod
    def _publish(job: str, summary: str) -> None:
        """Status file + commit + push + one ntfy line via ops/jobs/after_job.sh (never raises)."""
        try:
            subprocess.run([str(REPO_ROOT / "ops/jobs/after_job.sh"), job, summary], cwd=REPO_ROOT, timeout=600,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:  # noqa: BLE001
            pass

    def _finalize(self) -> dict[str, Any]:
        res = super()._finalize()
        res |= {"model": "lumen2", "part_params": self.part_counts, "w_damage": float(self.cfg.get("w_damage", W_DAMAGE)),
                "w_gate_penalty": float(self.cfg.get("w_gate_penalty", GATE_PENALTY))}
        (self.run_dir / "done.json").write_text(json.dumps(res, indent=2))
        return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.train", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config, a.set)
    run_dir = Path(a.run_dir or os.environ.get("DCFG_RUN_DIR") or RUNS_ROOT / cfg["run_id"]).expanduser()
    if cfg.get("require_pass_marker"):  # the queued training job must not run unless the smoke passed
        marker = REPO_ROOT / str(cfg["require_pass_marker"])
        if not marker.exists():
            print(f"ABORT: {marker} missing (smoke did not pass); not training", flush=True)
            return 3
    res = Lumen2Trainer(cfg, run_dir).train()
    print(json.dumps({k: v for k, v in res.items() if k not in ("train_config", "val_curve")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
