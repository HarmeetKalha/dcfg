"""Evaluate public GenD (CLIP ViT-L/14) on the FF++ test conditions with OUR degradation code.

    python -m dcfg.gend.evaluate_gend --run gend_public --conditions all --out-dir results/lumen2/gend_public

Pipeline per frame (same frame keys as our DFB baselines, so degradation seeds are identical):
  GenD crop (RetinaFace + GenD align, scale 1.3, 256x256 PNG from ``dcfg.gend.recrop``)
  -> degradation ``apply_condition`` at 256 px (our code, unchanged)
  -> GenD's input transform (PIL bicubic resize to 224, CLIP mean/std)
  -> GenD CLIP ViT-L/14 + L2-norm linear head -> P(fake) = softmax[1].
Frame AUC, video AUC (mean frame probability per video), 95% video-level bootstrap CIs: ``dcfg.eval.evaluate``.
``--crops dfb`` evaluates on our DFB dlib crops instead (diagnostic only; run id must differ).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from dcfg.data.degradations import load_conditions
from dcfg.data.ffpp import DFBJsonSource, FrameRecord
from dcfg.eval.evaluate import CONDITIONS_YAML, evaluate_condition, resolve_conditions
from dcfg.gend.crop import gend_resize
from dcfg.gend.model import GEND_DIR, load_gend
from dcfg.paths import DATA_ROOT, FFPP_FRAMES, REPO_ROOT, RESULTS_ROOT

HF_REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"
HF_ID = "yermandy/GenD_CLIP_L_14"
GEND_REPO = REPO_ROOT / "third_party" / "GenD"
OUT_DEFAULT = RESULTS_ROOT / "lumen2" / "gend_public"
PREPROC = ("GenD crop: RetinaFace det_10g (largest face, det thr 0.5 then 0.1, DFB-crop fallback), GenD align_face "
           "scale 1.3 -> 256x256 re-cropped from the original videos at the SAME 32 DFB frame indices; "
           "degradation (our apply_condition, same seeds/keys) applied on the 256 px crop; then GenD/CLIP input "
           "transform: PIL bicubic 224x224, CLIP mean/std; frame prob = softmax(l)[1]; video = mean frame prob")


class CropSource(DFBJsonSource):
    """DFB JSON records; images read from a crop root that mirrors ``~/data/ffpp_frames`` (same relative keys)."""

    def __init__(self, crops_root: Path, **kw: Any) -> None:
        super().__init__(**kw)
        self.crops_root = Path(crops_root)

    def load(self, rec: FrameRecord) -> np.ndarray:
        bgr = cv2.imread(str(self.crops_root / rec.key), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.crops_root / rec.key)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def gend_commit() -> str:
    return subprocess.check_output(["git", "-C", str(GEND_REPO), "rev-parse", "HEAD"], text=True).strip()


def provenance(crops: str, fp16: bool) -> dict[str, Any]:
    """Schema extras of every GenD result JSON (on top of the standard evaluate_condition fields)."""
    return {"pilot": False, "lumen2": True, "model_id": HF_ID, "hf_revision": HF_REVISION,
            "weights_sha256": sha256_file(GEND_DIR / "hf" / "model.safetensors"), "gend_commit": gend_commit(),
            "crops": crops, "preprocessing": PREPROC if crops == "gend" else PREPROC.replace(
                "GenD crop: RetinaFace", "DIAGNOSTIC dfb crops (dlib, not GenD's): ignore RetinaFace"),
            "degradation_placement": "256 px crop, before GenD's 224 resize", "input_size": 224}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.gend.evaluate_gend", description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="gend_public")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--crops", default="gend", choices=["gend", "dfb"])
    ap.add_argument("--json", default=None, help="DFB JSON (default ~/data/dfb_json/FaceForensics++.json)")
    ap.add_argument("--crops-root", default=None, help="override the crop root (e.g. DFD crops)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--trainable", default=None, help="trainable-only checkpoint (best_trainable.pt of dcfg.gend.train_gend_aug) "
                    "loaded onto the public GenD weights (GenD-aug); the run id must differ from gend_public")
    ap.add_argument("--fp32", action="store_true", help="fp32 instead of fp16 autocast")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.crops == "dfb" and a.run == "gend_public":
        raise SystemExit("--crops dfb is a diagnostic; use another --run id")
    out_dir = Path(a.out_dir)
    conds = resolve_conditions(a.conditions, load_conditions(CONDITIONS_YAML))
    todo = [c for c in conds if a.force or not (out_dir / f"{a.run}__{c.name}.json").exists()]
    print(f"{a.run}: {len(todo)}/{len(conds)} conditions to run", flush=True)
    if not todo:
        return 0
    root = Path(a.crops_root) if a.crops_root else (DATA_ROOT / "gend_crops" if a.crops == "gend" else FFPP_FRAMES)
    source = CropSource(root, json_path=a.json, frames_root=root if a.json else None)
    model = load_gend("cuda").eval()
    extra = provenance(a.crops, not a.fp32)
    if a.trainable:
        if a.run == "gend_public":
            raise SystemExit("--trainable needs a run id other than gend_public")
        from dcfg.gend.train_gend_aug import load_trainable

        ck = torch.load(a.trainable, map_location="cpu", weights_only=False)
        load_trainable(model, ck["trainable"])
        model.eval()
        extra |= {"trainable_file": str(a.trainable), "trainable_sha256": sha256_file(Path(a.trainable)),
                  "trained_epoch_1based": ck.get("epoch_1based"), "train_val_mean_auc": ck.get("val_mean_auc"),
                  "init_is_public_gend": True}
    for cond in todo:
        t0 = time.time()
        p = evaluate_condition(run_id=a.run, detector="gend_clip_l14", cond=cond, split=a.split, source=source,
                               model=model, out_dir=out_dir, device="cuda", batch_size=a.batch_size,
                               workers=a.workers, fp16=not a.fp32, n_boot=a.n_boot, force=a.force,
                               dfb_commit_hash="n/a", extra=extra, post_transform=gend_resize)
        print(f"  {cond.name}: {p} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
