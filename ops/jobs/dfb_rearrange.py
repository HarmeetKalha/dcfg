"""Build DeepfakeBench-schema dataset JSONs for FF++ from our preprocessed frames.
Schema (training/dataset/abstract_dataset.py): J['FaceForensics++'][label][split][comp][video]={'label','frames'}.
Own wrapper because DFB rearrange.py needs actors/DFD/FaceShifter dirs (KeyError on del) and unsorted scandir.
Split source: official FF++ train/val/test.json (ondyari/FaceForensics), 360/70/70 pairs = 720/140/140 videos,
a manipulated video 'A_B' belongs to the split of pair {A,B}. Writes FaceForensics++.json and FF-<type>.json.
Usage: dfb_rearrange.py [--frames ~/data/ffpp_frames/FaceForensics++] [--out ~/data/dfb_json]
"""
import argparse, json, os
from pathlib import Path
H = Path.home()
TYPES = {"original_sequences/youtube": "FF-real", "manipulated_sequences/Deepfakes": "FF-DF",
         "manipulated_sequences/Face2Face": "FF-F2F", "manipulated_sequences/FaceSwap": "FF-FS",
         "manipulated_sequences/NeuralTextures": "FF-NT"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(H / "data/ffpp_frames/FaceForensics++"))
    ap.add_argument("--out", default=str(H / "data/dfb_json"))
    a = ap.parse_args(); root = Path(a.frames); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    v2s = {}
    for s in ("train", "val", "test"):
        for x, y in json.load(open(root / f"{s}.json")):
            for k in (x, y, x + "_" + y, y + "_" + x): v2s[k] = s
    J = {}
    for seq, lab in TYPES.items():
        J[lab] = {s: {} for s in ("train", "val", "test")}
        for comp in ("c23", "c40"):
            for s in J[lab]: J[lab][s][comp] = {}
            fr = root / seq / comp / "frames"
            if not fr.is_dir(): continue
            for vd in sorted(fr.iterdir()):
                files = sorted(str(f) for f in vd.iterdir())
                if not files: continue          # zero-face videos are reported by the QA step, not listed
                J[lab][v2s[vd.name]][comp][vd.name] = {"label": lab, "frames": files}
    json.dump({"FaceForensics++": J}, open(out / "FaceForensics++.json", "w"))
    for lab in J:
        if lab != "FF-real":
            json.dump({lab: {"FF-real": J["FF-real"], lab: J[lab]}}, open(out / f"{lab}.json", "w"))
    for lab in J:
        print(lab, {s: {c: len(J[lab][s][c]) for c in J[lab][s]} for s in J[lab]})

if __name__ == "__main__":
    main()
