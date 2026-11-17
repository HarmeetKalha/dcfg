"""FF++ face preprocessing using DeepfakeBench's own alignment (extract_aligned_face_dlib, 256 crop,
scale 1.3, 5-pt similarity transform) and its frame sampling (np.linspace(0, N-1, 32, dtype=int)).
Wrapper instead of DFB's preprocess.py because that script (a) writes frames next to the raw videos,
(b) hardcodes ./config.yaml and ./dlib_tools/ relative paths, (c) requires actors/FaceShifter/DFD dirs
to exist, (d) uses threads, (e) cannot restrict to a split, (f) does not skip finished videos.
Idempotent/resumable: a marker per finished video under <out>/.done/. Reads raw videos only.
Usage: dfb_preprocess.py --comp c23|c40 [--test-only | --split test|val] [--raw R] [--out O] [--workers 6] [--limit N]
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import cv2, dlib, numpy as np
from imutils import face_utils  # noqa: F401 (needed by DFB module)

HOME = Path.home()
DFB = HOME / "dcfg/third_party/DeepfakeBench/preprocessing"
SEQS = ["original_sequences/youtube", "manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face",
        "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures"]
PRED = HOME / "data/weights/shape_predictor_81_face_landmarks.dat"
_G = {}

def _init():
    cv2.setNumThreads(1)
    sys.path.insert(0, str(DFB))
    import preprocess as dfb  # guarded by __main__, import is side-effect free
    _G["fn"] = dfb.extract_aligned_face_dlib
    _G["det"] = dlib.get_frontal_face_detector()
    _G["pred"] = dlib.shape_predictor(str(PRED))

def work(job):
    vid, seq, comp, out, nframes = job
    vid = Path(vid); out = Path(out)
    done = out / ".done" / seq / comp / (vid.stem + ".txt")
    if done.exists():
        return (seq, comp, vid.stem, -1, -1, 0.0)
    t0 = time.time()
    cap = cv2.VideoCapture(str(vid))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, n - 1, nframes, endpoint=True, dtype=int).tolist())
    fdir = out / seq / comp / "frames" / vid.stem
    ldir = out / seq / comp / "landmarks" / vid.stem
    saved = 0
    for i in range(n):
        if i not in idxs:
            if not cap.grab(): break
            continue
        ok, fr = cap.read()
        if not ok: break
        face, lm, _ = _G["fn"](_G["det"], _G["pred"], fr, res=256)
        if face is None: continue
        fdir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fdir / f"{i:03d}.png"), face)
        np.save(str(ldir / f"{i:03d}.npy"), lm)
        saved += 1
    cap.release()
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(f"{n}\t{saved}\n")
    return (seq, comp, vid.stem, n, saved, time.time() - t0)

def split_stems(split_dir, split="test"):
    pairs = json.load(open(split_dir / f"{split}.json"))
    s = set()
    for a, b in pairs: s |= {a, b, a + "_" + b, b + "_" + a}
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True, choices=["c23", "c40"])
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--split", choices=["test", "val"], default=None,
                    help="restrict to the videos of this official split (implies a restriction; --test-only == --split test)")
    ap.add_argument("--raw", default=str(HOME / "data/ffpp_raw"))
    ap.add_argument("--out", default=str(HOME / "data/ffpp_frames/FaceForensics++"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="per sequence type, for dry runs only")
    ap.add_argument("--only", default="", help="substring filter on sequence name")
    a = ap.parse_args()
    out = Path(a.out)
    split = a.split or ("test" if a.test_only else None)
    keep = split_stems(Path(os.environ.get("DFB_SPLITS", HOME / "data/ffpp_frames/FaceForensics++")), split) if split else None
    jobs = []
    for seq in SEQS:
        if a.only and a.only not in seq: continue
        vids = sorted(glob.glob(f"{a.raw}/{seq}/{a.comp}/videos/*.mp4"))
        if keep is not None: vids = [v for v in vids if Path(v).stem in keep]
        if a.limit: vids = vids[:a.limit]
        print(f"{seq} {a.comp}: {len(vids)} videos", flush=True)
        jobs += [(v, seq, a.comp, str(out), a.num_frames) for v in vids]
    out.mkdir(parents=True, exist_ok=True)
    stats = out / f"preprocess_stats_{a.comp}.tsv"
    t0 = time.time(); k = 0
    with Pool(a.workers, initializer=_init) as p, open(stats, "a") as sf:
        for r in p.imap_unordered(work, jobs):
            k += 1
            if r[3] >= 0:
                sf.write("\t".join(map(str, r[:5])) + f"\t{r[5]:.1f}\n"); sf.flush()
            if k % 50 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print(f"DONE {k} videos in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
