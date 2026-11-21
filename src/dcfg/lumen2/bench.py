"""Inference cost of LUMEN2 vs GenD-aug (same GPU, same batch size, fp16 autocast for CLIP), plus parameter/file/training cost.

    python -m dcfg.lumen2.bench [--batch-size 64] [--iters 30] [--out results/lumen2/lumen2_cost.json]

Run it when the GPU is otherwise idle (queued after the LUMEN2 evaluation). Inputs are real val crops, preprocessing excluded:
only the model forward (GenD-aug: 224 view -> logit; LUMEN2: packed views -> logit, i.e. sensor + gate + LN generator + CLIP).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from dcfg.gend.crop import gend_resize
from dcfg.gend.model import load_gend
from dcfg.gend.train_gend_aug import load_trainable as load_gend_trainable
from dcfg.lumen2.evaluate_lumen2 import load_lumen2
from dcfg.lumen2.model import pack_views
from dcfg.paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT, RUNS_ROOT


def time_ms_per_image(fn: Callable[[], torch.Tensor], n_img: int, iters: int, warmup: int = 5) -> dict[str, float]:
    """Median / min wall time of ``fn()`` (GPU synchronised) in ms per image."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0 / n_img)
    return {"median_ms_per_image": float(np.median(ts)), "min_ms_per_image": float(np.min(ts)), "iters": iters}


def main(argv: list[str] | None = None) -> int:
    from dcfg.gend.evaluate_gend import CropSource

    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.bench", description=__doc__.split("\n")[0])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--aug-trainable", default=str(RUNS_ROOT / "gendaug_train" / "best_trainable.pt"))
    ap.add_argument("--lumen2-trainable", default=str(RUNS_ROOT / "lumen2_train" / "best_trainable.pt"))
    ap.add_argument("--lumen2-run", default=str(RUNS_ROOT / "lumen2_train"))
    ap.add_argument("--out", default=str(RESULTS_ROOT / "lumen2" / "lumen2_cost.json"))
    a = ap.parse_args(argv)
    src = CropSource(DATA_ROOT / "gend_crops")
    recs = src.records("val", "c23")
    recs = recs[:: max(1, len(recs) // a.batch_size)][: a.batch_size]
    imgs = [src.load(r) for r in recs]
    x224 = torch.stack([torch.from_numpy(gend_resize(i)).permute(2, 0, 1) for i in imgs]).cuda().float().div_(255.0)
    xpack = torch.stack([torch.from_numpy(pack_views(i)).permute(2, 0, 1) for i in imgs]).cuda().float().div_(255.0)
    aug = load_gend("cuda").eval()
    load_gend_trainable(aug, torch.load(a.aug_trainable, map_location="cpu", weights_only=False)["trainable"])
    lumen, _ = load_lumen2(a.lumen2_trainable, "cuda")

    def run(m: torch.nn.Module, x: torch.Tensor) -> Callable[[], torch.Tensor]:
        def f() -> torch.Tensor:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return m(x)
        return f

    res: dict[str, Any] = {"batch_size": len(imgs), "gpu": torch.cuda.get_device_name(0), "precision": "CLIP fp16 autocast, LUMEN parts fp32",
                           "gend_aug": time_ms_per_image(run(aug, x224), len(imgs), a.iters),
                           "lumen2": time_ms_per_image(run(lumen, xpack), len(imgs), a.iters)}
    res["overhead_ratio"] = res["lumen2"]["median_ms_per_image"] / res["gend_aug"]["median_ms_per_image"]
    counts = lumen.part_counts if hasattr(lumen, "part_counts") else lumen.part_param_counts()
    res["lumen_params"] = counts
    res["lumen_params_total"] = counts["LUMEN parts total"]
    res["trainable_file_mb"] = {"gend_aug": round(Path(a.aug_trainable).stat().st_size / 2**20, 3),
                                "lumen2": round(Path(a.lumen2_trainable).stat().st_size / 2**20, 3)}
    for key, p in (("lumen2", Path(a.lumen2_run) / "done.json"), ("gend_aug", RUNS_ROOT / "gendaug_train" / "done.json")):
        if p.exists():
            d = json.loads(p.read_text())
            res.setdefault("training", {})[key] = {k: d.get(k) for k in ("train_wall_time_s", "train_s", "val_s", "best_epoch", "stop_epoch", "stop_reason", "steps")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("gend_aug", "lumen2", "overhead_ratio")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
