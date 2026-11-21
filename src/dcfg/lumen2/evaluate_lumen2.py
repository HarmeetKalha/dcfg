"""Evaluate LUMEN2 on the FF++ test conditions (same degradation code, seeds and frame keys as GenD-aug) + sensor outputs.

    python -m dcfg.lumen2.evaluate_lumen2 --run lumen2 --trainable runs/lumen2_train/best_trainable.pt \
        --conditions all --out-dir results/lumen2/lumen2 [--crops-root ~/data/gend_crops_dfd --json ~/data/gend_crops_dfd/DFD.json]

Result JSON + score npz per (run, condition): ``dcfg.eval.evaluate.evaluate_condition`` (identical schema to
``results/lumen2/gend_aug``). Additionally ``<out-dir>/sensor/<run>__<cond>.npz`` with the damage sensor's per-frame presence
probabilities (N,4) and strengths (N,4) and the frame keys (same order as the score npz). Resumable per condition.

Also exposes :func:`quick_eval` (used by the Monday partial evaluation in the trainer): a fixed ``np.linspace`` subset of
frames per video (8 of 32), AUC + sensor outputs per condition, no bootstrap.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcfg.data.degradations import EvalCondition, load_conditions
from dcfg.data.ffpp import FrameRecord
from dcfg.eval.evaluate import CONDITIONS_YAML, ConditionDataset, condition_comp, evaluate_condition, predict, resolve_conditions
from dcfg.eval.metrics import frame_auc
from dcfg.lumen2.model import Lumen2Model, build_lumen2, load_trainable, pack_views
from dcfg.paths import DATA_ROOT, RESULTS_ROOT

OUT_DEFAULT = RESULTS_ROOT / "lumen2" / "lumen2"


class SensorTap:
    """Callable ``x01 -> logit`` around a :class:`Lumen2Model` that records the sensor outputs of every batch (in order)."""

    def __init__(self, model: Lumen2Model) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.p: list[np.ndarray] = []
        self.s: list[np.ndarray] = []

    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        logit, sens, _ = self.model.predict_with_sensor(x01)
        self.p.append(torch.sigmoid(sens["presence_logit"]).float().cpu().numpy())
        self.s.append(sens["strength"].float().cpu().numpy())
        return logit

    def collected(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.concatenate(self.p), np.concatenate(self.s)) if self.p else (np.zeros((0, 4)), np.zeros((0, 4)))


class SubsetSource:
    """Wraps a frame source; ``records`` returns a fixed per-video ``np.linspace`` subset (``per_video`` frames, key-sorted)."""

    def __init__(self, source: Any, per_video: int) -> None:
        self.source, self.per_video = source, per_video

    def records(self, split: str, comp: str) -> list[FrameRecord]:
        from dcfg.gend.train_gend_aug import val_subsample

        return val_subsample(self.source.records(split, comp), self.per_video)

    def load(self, rec: FrameRecord) -> np.ndarray:
        return self.source.load(rec)


class StrideSource:
    """Smoke-test only: every k-th frame record so that about ``max_frames`` remain."""

    def __init__(self, source: Any, max_frames: int) -> None:
        self.source, self.max_frames = source, max_frames

    def records(self, split: str, comp: str) -> list[FrameRecord]:
        r = self.source.records(split, comp)
        return r[:: max(1, len(r) // self.max_frames)]

    def load(self, rec: FrameRecord) -> np.ndarray:
        return self.source.load(rec)


@torch.no_grad()
def quick_eval(model: Lumen2Model, source: Any, cond: EvalCondition, split: str = "test", batch_size: int = 64, workers: int = 5,
               on_batch: Any = None) -> dict[str, Any]:
    """Frame AUC + sensor outputs of ``model`` on one condition over ``source.records(split, comp)`` (fp16 CLIP, fp32 parts)."""
    recs = source.records(split, condition_comp(cond))
    ld = DataLoader(ConditionDataset(source, recs, cond, post_transform=pack_views), batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=torch.cuda.is_available())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap = SensorTap(model)
    prob = predict(tap, ld, dev, fp16=True, on_batch=on_batch)
    p, s = tap.collected()
    return {"auc": frame_auc([r.label for r in recs], prob), "prob": prob, "presence": p, "strength": s, "n": len(recs)}


def load_lumen2(trainable: str | Path | None, device: str = "cuda") -> tuple[Lumen2Model, dict[str, Any]]:
    """Public GenD + LUMEN parts; load a trainable-only checkpoint onto it (None = step-0 model). Returns (model.eval(), ckpt meta)."""
    model = build_lumen2("cpu")
    meta: dict[str, Any] = {}
    if trainable:
        ck = torch.load(trainable, map_location="cpu", weights_only=False)
        load_trainable(model, ck["trainable"])
        meta = {k: v for k, v in ck.items() if k != "trainable"}
    return model.to(device).eval(), meta


def main(argv: Sequence[str] | None = None) -> int:
    from dcfg.gend.evaluate_gend import CropSource, provenance, sha256_file

    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.evaluate_lumen2", description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="lumen2")
    ap.add_argument("--trainable", required=True, help="best_trainable.pt of dcfg.lumen2.train")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--json", default=None)
    ap.add_argument("--crops-root", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test only (never for results)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    conds = resolve_conditions(a.conditions, load_conditions(CONDITIONS_YAML))
    sens_dir = out_dir / "sensor"
    todo = [c for c in conds if a.force or not (out_dir / f"{a.run}__{c.name}.json").exists()
            or not (sens_dir / f"{a.run}__{c.name}.npz").exists()]
    print(f"{a.run}: {len(todo)}/{len(conds)} conditions to run", flush=True)
    if not todo:
        return 0
    root = Path(a.crops_root) if a.crops_root else DATA_ROOT / "gend_crops"
    source = CropSource(root, json_path=a.json, frames_root=root if a.json else None)
    if a.max_frames:
        source = StrideSource(source, a.max_frames)
    model, meta = load_lumen2(a.trainable, "cuda")
    tap = SensorTap(model)
    extra = provenance("gend", True) | {
        "model": "lumen2", "trainable_file": str(a.trainable), "trainable_sha256": sha256_file(Path(a.trainable)),
        "trained_epoch_1based": meta.get("epoch_1based"), "train_val_mean_auc": meta.get("val_mean_auc"),
        "init_is_public_gend": True, "part_params": model.part_param_counts(),
        "input": "packed views: damaged 256 crop (sensor) + GenD 224 resize (gate, CLIP)"}
    for cond in todo:
        t0 = time.time()
        tap.reset()
        evaluate_condition(run_id=a.run, detector="lumen2_clip_l14", cond=cond, split="test", source=source, model=tap,
                           out_dir=out_dir, device="cuda", batch_size=a.batch_size, workers=a.workers, fp16=True,
                           n_boot=a.n_boot, force=True, dfb_commit_hash="n/a", extra=extra, post_transform=pack_views)
        p, s = tap.collected()
        sens_dir.mkdir(parents=True, exist_ok=True)
        keys = np.load(out_dir / "scores" / f"{a.run}__{cond.name}.npz")["frame_key"]
        assert len(keys) == len(p), (len(keys), len(p))
        np.savez_compressed(sens_dir / f"{a.run}__{cond.name}.npz", frame_key=keys, presence=p.astype(np.float32),
                            strength=s.astype(np.float32))
        print(f"  {cond.name}: done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
