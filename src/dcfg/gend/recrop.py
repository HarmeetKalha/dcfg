"""Re-crop the FF++ test frames from the ORIGINAL videos with GenD's own pipeline (RetinaFace + GenD align, 256 px).

    python -m dcfg.gend.recrop --comp c23|c40 [--split test] [--workers 6] [--limit N]

The frame set is exactly the DFB frame set: for every frame listed in the DFB JSON (``FaceForensics++.json``) of
the split, the same video frame index (parsed from the DFB file name, np.linspace(0, N-1, 32)) is decoded and
cropped with GenD's pipeline. The output PNG keeps the DFB relative key, so degradation seeds
(``sha256(condition|key)``) are identical to our baselines.

Face selection: threshold 0.5 (GenD default / its ``FF-x1.3-th0.5`` datasets); if nothing is found, retry at 0.1
(the README's FF command); if still nothing (or the affine fit fails) the DFB dlib crop of that frame is copied
so the frame set is unchanged. Per-frame status is written to ``<out>/recrop_stats_<comp>.tsv``.
Idempotent: a marker per finished video. Reads raw videos and DFB crops only.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dcfg.data.ffpp import DFBJsonSource
from dcfg.gend.crop import align_face, largest_face
from dcfg.paths import DATA_ROOT, FFPP_FRAMES, FFPP_RAW, REPO_ROOT

GEND_REPO = REPO_ROOT / "third_party" / "GenD"
ONNX = DATA_ROOT / "weights" / "gend" / "det_10g.onnx"
OUT_ROOT = DATA_ROOT / "gend_crops"  # mirrors ~/data/ffpp_frames (keys include the "FaceForensics++/" prefix)
THRESHOLDS = (0.5, 0.1)
NUM_FRAMES = 32
DFD_ROOT = DATA_ROOT / "gend_crops_dfd"
_G: dict = {}


def _init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(1)
    spec = importlib.util.spec_from_file_location("gend_retinaface", GEND_REPO / "src" / "retinaface.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # reads GenD's RetinaFace (imports onnxruntime/cv2/numpy only)
    det = mod.RetinaFace(str(ONNX), providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, nms_thresh=0.4, input_size=(640, 640), det_thresh=0.5)
    _G["det"] = det


def detect_landmarks(det, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Landmarks of the largest face and the threshold that found it (None, None if no face)."""
    for th in THRESHOLDS:
        det.det_thresh = th
        xyxy, kps = det.detect(frame_bgr)
        if len(xyxy):
            return largest_face(xyxy, kps), th
    return None, None


def work(job: tuple) -> tuple:
    import cv2

    # frames: list of (idx, png key) matching the DFB frame set, or None (new dataset, e.g. DFD): then DFB's own
    # sampling np.linspace(0, N-1, 32) is used, keys are <seq>/<comp>/frames/<video>/<idx:03d>.png and a frame
    # without a detected face is skipped (DFB behaviour) instead of falling back to a DFB crop.
    vid_key, video, frames, out_root, dfb_root = job
    out_root = Path(out_root)
    dfb_root = Path(dfb_root) if dfb_root else None
    video = Path(video)
    comp = video.parent.parent.name  # .../<seq>/<comp>/videos/<stem>.mp4
    done = out_root / ".done" / (f"{comp}__" + vid_key.replace("/", "__") + ".txt")
    if done.exists():
        return vid_key, []
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is None:
        seq = "/".join(video.parts[-5:-3])  # e.g. manipulated_sequences/DeepFakeDetection
        idxs = np.linspace(0, n - 1, NUM_FRAMES, endpoint=True, dtype=int).tolist()
        frames = [(i, f"{seq}/{comp}/frames/{video.stem}/{i:03d}.png") for i in sorted(set(idxs))]
    want = {i: k for i, k in frames}
    stats = []
    for i in range(max(want) + 1):
        if i not in want:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        key = want[i]
        status = "fallback_dfb"
        if ok:
            try:
                lm, th = detect_landmarks(_G["det"], fr)
                if lm is not None:
                    face = align_face(fr, lm)
                    dst = out_root / key
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst), face)
                    status = f"gend_th{th}"
            except Exception:  # noqa: BLE001  (affine failure etc. -> fallback)
                status = "fallback_dfb"
        if status == "fallback_dfb" and dfb_root is None:
            stats.append((key, "no_face"))
            continue
        if status == "fallback_dfb":
            dst = out_root / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dfb_root / key, dst)
        stats.append((key, status))
    cap.release()
    missing = (set(want.values()) - {k for k, _ in stats}) if dfb_root is not None else set()
    for key in missing:  # video shorter than expected: keep the DFB crop
        dst = out_root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dfb_root / key, dst)
        stats.append((key, "fallback_dfb_short"))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(f"{n}\t{len(stats)}\n")
    return vid_key, stats


def build_jobs(split: str, comp: str, limit: int = 0) -> list[tuple]:
    src = DFBJsonSource()
    by_vid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in src.records(split, comp):
        by_vid[r.video].append((int(Path(r.key).stem), r.key))
    jobs = []
    for vid_key in sorted(by_vid):
        frames = sorted(by_vid[vid_key])
        key0 = frames[0][1]  # e.g. original_sequences/youtube/c23/frames/000/000.png
        seq, rest = key0.removeprefix("FaceForensics++/").split(f"/{comp}/frames/")
        video = FFPP_RAW / seq / comp / "videos" / (rest.split("/")[0] + ".mp4")
        jobs.append((vid_key, video, frames, str(OUT_ROOT), str(FFPP_FRAMES)))
    return jobs[:limit] if limit else jobs


DFD_SEQS = {"original_sequences/actors": "FF-real", "manipulated_sequences/DeepFakeDetection": "FF-DFD"}


def build_dfd_jobs(comp: str = "c23", limit: int = 0) -> list[tuple]:
    """All DeepFakeDetection videos (real actors + DFD fakes), frames chosen by DFB's linspace rule inside ``work``."""
    jobs = []
    for seq, label in DFD_SEQS.items():
        vids = sorted((FFPP_RAW / seq / comp / "videos").glob("*.mp4"))
        jobs += [(f"{label}/{v.stem}", v, None, str(DFD_ROOT), None) for v in (vids[:limit] if limit else vids)]
    return jobs


def write_dfd_json(comp: str = "c23") -> Path:
    """DFB-schema JSON (``FaceForensics++`` key, all videos under split ``test``) listing the DFD crops on disk.

    Paths are relative to ``DFD_ROOT`` (load with frames_root=DFD_ROOT); existing FF++ JSONs are not touched.
    """
    import json

    doc: dict = {}
    for seq, label in DFD_SEQS.items():
        doc.setdefault(label, {s: {c: {} for c in ("c23", "c40")} for s in ("train", "val", "test")})
        for vd in sorted((DFD_ROOT / seq / comp / "frames").glob("*")):
            files = sorted(str(f.relative_to(DFD_ROOT)) for f in vd.glob("*.png"))
            if files:
                doc[label]["test"][comp][vd.name] = {"label": label, "frames": files}
    out = DFD_ROOT / "DFD.json"
    out.write_text(json.dumps({"FaceForensics++": doc}))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.gend.recrop", description=__doc__.split("\n")[0])
    ap.add_argument("--comp", required=True, choices=["c23", "c40"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--dfd", action="store_true", help="DeepFakeDetection (all videos) instead of the FF++ split; "
                    "outputs to ~/data/gend_crops_dfd + DFD.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    out_root = DFD_ROOT if a.dfd else OUT_ROOT
    jobs = build_dfd_jobs(a.comp, a.limit) if a.dfd else build_jobs(a.split, a.comp, a.limit)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{a.comp}/{a.split}: {len(jobs)} videos", flush=True)
    t0 = time.time()
    k = 0
    with Pool(a.workers, initializer=_init) as p, open(out_root / f"recrop_stats_{a.comp}.tsv", "a") as sf:
        for vid, stats in p.imap_unordered(work, jobs):
            k += 1
            for key, st in stats:
                sf.write(f"{key}\t{st}\n")
            sf.flush()
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    if a.dfd:
        print(f"wrote {write_dfd_json(a.comp)}", flush=True)
    print(f"DONE {k} videos in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
