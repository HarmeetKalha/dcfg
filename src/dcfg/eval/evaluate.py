"""Evaluate a detector on FF++ test conditions; one JSON per (run, condition) in results/raw.

    python -m dcfg.eval.evaluate --run base_xception --detector xception --conditions all --split test

* ``c40`` (``synthetic: false`` in conditions.yaml) reads the c40 test frames with no synthetic
  degradation; every other condition reads c23 frames and applies
  ``apply_condition(img, cond, image_key=<frame path relative to the frames root>)``.
* Resumable: a condition whose JSON exists is skipped unless ``--force``.
* The ``val`` split is never written to ``results/raw`` (Test numbers for the paper come only
  from there); ``--split val`` requires an explicit ``--out-dir`` elsewhere.
* Scores are P(fake) = sigmoid(adapter logit) (see ``dcfg.detectors.adapter``), fp32 by default.
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
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dcfg.data.degradations import EvalCondition, apply_condition, load_conditions
from dcfg.data.ffpp import DFBJsonSource, FrameRecord, FrameSource
from dcfg.eval.metrics import bootstrap_ci, frame_auc, video_auc
from dcfg.paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT
from dcfg.utils.heartbeat import Heartbeat

RAW_DIR = RESULTS_ROOT / "raw"
CONDITIONS_YAML = REPO_ROOT / "configs" / "eval" / "conditions.yaml"


def git_commit(repo: Path = REPO_ROOT) -> str:
    """HEAD hash of ``repo`` with a ``-dirty`` suffix when the tree has uncommitted changes."""
    try:
        h = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                                    stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


class ConditionDataset(Dataset):
    """Frames of one condition; degradation is deterministic in (condition, frame key)."""

    def __init__(self, source: FrameSource, records: Sequence[FrameRecord], cond: EvalCondition) -> None:
        self.source, self.records, self.cond = source, list(records), cond

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> torch.Tensor:
        rec = self.records[i]
        img = self.source.load(rec)
        if self.cond.synthetic:
            img = apply_condition(img, self.cond, rec.key)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # uint8 CHW


def condition_comp(cond: EvalCondition) -> str:
    """c40 (real compression) uses c40 frames; everything else degrades c23 frames."""
    return "c23" if cond.synthetic else "c40"


def check_out_dir(split: str, out_dir: Path) -> None:
    """Refuse to write val results into results/raw (paper numbers come only from there)."""
    if split != "test" and out_dir.resolve() == RAW_DIR.resolve():
        raise ValueError(f"split {split!r} must not be written to results/raw; pass another --out-dir")


def predict(model: Callable[[torch.Tensor], torch.Tensor], loader: DataLoader, device: str,
            fp16: bool = False, on_batch: Callable[[int], None] | None = None) -> np.ndarray:
    """Run ``model`` (uint8 batch -> logits) over the loader; return P(fake) as float64."""
    out: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for xb in loader:
            x01 = xb.to(device, non_blocking=True).float().div_(255.0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.startswith("cuda")):
                logit = model(x01)
            out.append(torch.sigmoid(logit.double()).cpu().numpy())
            seen += len(xb)
            if on_batch:
                on_batch(seen)
    return np.concatenate(out) if out else np.zeros(0)


def evaluate_condition(*, run_id: str, detector: str, cond: EvalCondition, split: str,
                       source: FrameSource, model: Callable[[torch.Tensor], torch.Tensor],
                       out_dir: Path, device: str = "cpu", batch_size: int = 64, workers: int = 4,
                       fp16: bool = False, n_boot: int = 1000, seed: int = 0, force: bool = False,
                       heartbeat: Heartbeat | None = None, dfb_commit_hash: str = "unknown",
                       extra: dict[str, Any] | None = None) -> Path | None:
    """Evaluate one condition and write its JSON + scores npz. Returns the JSON path (None if skipped)."""
    check_out_dir(split, out_dir)
    json_path = out_dir / f"{run_id}__{cond.name}.json"
    if json_path.exists() and not force:
        return None
    comp = condition_comp(cond)
    recs = source.records(split, comp)
    if not recs:
        raise RuntimeError(f"no frames for split={split} comp={comp}")
    loader = DataLoader(ConditionDataset(source, recs, cond), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.startswith("cuda"))
    t0 = time.time()

    def beat(n: int) -> None:
        if heartbeat:
            heartbeat.beat(n, condition=cond.name, total=len(recs), elapsed_s=round(time.time() - t0, 1))

    prob = predict(model, loader, device, fp16, beat)
    labels = np.array([r.label for r in recs])
    videos = np.array([r.video for r in recs])
    f_ci = bootstrap_ci("frame", labels, prob, videos, n=n_boot, seed=seed)
    v_ci = bootstrap_ci("video", labels, prob, videos, n=n_boot, seed=seed)
    uvids, inv = np.unique(videos, return_inverse=True)
    vprob = np.bincount(inv, weights=prob) / np.bincount(inv)
    vlabel = np.array([labels[inv == i][0] for i in range(len(uvids))])

    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"{run_id}__{cond.name}.npz"
    np.savez_compressed(scores_dir / npz_name, frame_key=np.array([r.key for r in recs]),
                        frame_video=videos, frame_label=labels, frame_prob=prob,
                        video_id=uvids, video_label=vlabel, video_prob=vprob)
    res = {
        "run_id": run_id, "detector": detector, "condition": cond.name, "split": split,
        "compression": comp, "n_frames": int(len(recs)), "n_videos": int(len(uvids)),
        "frame_auc": frame_auc(labels, prob), "frame_auc_ci": [f_ci.lo, f_ci.hi],
        "video_auc": video_auc(labels, prob, videos), "video_auc_ci": [v_ci.lo, v_ci.hi],
        "bootstrap": {"n": n_boot, "seed": seed, "unit": "video", "level": 0.95},
        "fp16": fp16, "git_commit": git_commit(), "dfb_commit": dfb_commit_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores_file": f"scores/{npz_name}", **(extra or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=2) + "\n")
    os.replace(tmp, json_path)
    return json_path


def resolve_conditions(spec: str, available: dict[str, EvalCondition]) -> list[EvalCondition]:
    """'all' or a comma-separated list of condition names."""
    if spec == "all":
        return list(available.values())
    names = [s for s in spec.split(",") if s]
    bad = [n for n in names if n not in available]
    if bad:
        raise SystemExit(f"unknown conditions: {bad}; available: {list(available)}")
    return [available[n] for n in names]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.eval.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run id, e.g. base_xception")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--frontend", default=None, help="front-end checkpoint (not wired up yet)")
    ap.add_argument("--conditions", default="all", help="comma list or 'all'")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out-dir", default=None, help="default results/raw (test only)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--json", default=None, help="DFB FaceForensics++.json (default ~/data/dfb_json)")
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true", help="autocast fp16 (default fp32)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.split != "test" and a.out_dir is None:
        raise SystemExit("--split val requires --out-dir (val results never go to results/raw)")
    out_dir = Path(a.out_dir) if a.out_dir else RAW_DIR
    check_out_dir(a.split, out_dir)
    if a.frontend:
        raise SystemExit("--frontend is not wired up yet (front-end modules are being built)")

    from dcfg.detectors.adapter import Detector
    from dcfg.detectors.registry import dfb_commit

    conds = resolve_conditions(a.conditions, load_conditions(CONDITIONS_YAML))
    todo = [c for c in conds if a.force or not (out_dir / f"{a.run}__{c.name}.json").exists()]
    print(f"{a.run}: {len(todo)}/{len(conds)} conditions to run", flush=True)
    if not todo:
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = Detector(a.detector, weights_path=a.weights, device=device).freeze()
    source = DFBJsonSource(a.json, a.frames_root)
    hb = Heartbeat(RUNS_ROOT / a.run)
    commit = dfb_commit()

    def model(x01: torch.Tensor) -> torch.Tensor:
        return det(det.normalize(x01))

    for cond in todo:
        t0 = time.time()
        p = evaluate_condition(run_id=a.run, detector=a.detector, cond=cond, split=a.split, source=source,
                               model=model, out_dir=out_dir, device=device, batch_size=a.batch_size,
                               workers=a.workers, fp16=a.fp16, n_boot=a.n_boot, force=a.force,
                               heartbeat=hb, dfb_commit_hash=commit)
        print(f"  {cond.name}: {p} ({time.time() - t0:.0f}s)", flush=True)
        hb.beat(0, force=True, condition=cond.name, done=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
