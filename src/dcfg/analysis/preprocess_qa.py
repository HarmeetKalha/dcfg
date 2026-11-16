"""GATE 3 QA of DeepfakeBench preprocessing: counts table + crop grid.

    python -m dcfg.analysis.preprocess_qa [--json J] [--frames-root R] [--stats-dir D] [--out-dir results/qa]

Inputs: the DFB-schema JSON (videos with >=1 face), ``preprocess_stats_<comp>.tsv`` (one row per
processed video: ``seq comp stem n_frames saved elapsed``, written by ops/jobs/dfb_preprocess.py) and
the official ``{train,val,test}.json`` pair lists in the stats dir (to assign zero-face videos, which are
absent from the DFB JSON, to a split).

Per split x type x comp: videos processed, videos in JSON, mean/min frames per video (JSON),
zero-face videos (listed), failed-detection rate = 1 - saved / attempted over all sampled frames,
where attempted = number of distinct indices of np.linspace(0, n-1, 32, dtype=int) (as preprocess).
Without a stats file, the rate is reported as null and the video list falls back to the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dcfg.data.ffpp import DFBJsonSource
from dcfg.paths import DFB_JSON, FFPP_FRAMES, RESULTS_ROOT

SEQ_LABEL = {"original_sequences/youtube": "FF-real", "manipulated_sequences/Deepfakes": "FF-DF",
             "manipulated_sequences/Face2Face": "FF-F2F", "manipulated_sequences/FaceSwap": "FF-FS",
             "manipulated_sequences/NeuralTextures": "FF-NT"}
SPLITS = ("train", "val", "test")
NUM_FRAMES = 32


def video_split_map(stats_dir: Path) -> dict[str, str]:
    """stem -> split from the official pair lists (A, B, A_B, B_A -> split of pair)."""
    v2s: dict[str, str] = {}
    for s in SPLITS:
        p = stats_dir / f"{s}.json"
        if p.exists():
            for x, y in json.loads(p.read_text()):
                for k in (x, y, f"{x}_{y}", f"{y}_{x}"):
                    v2s[k] = s
    return v2s


def read_stats(stats_dir: Path, comp: str) -> list[dict]:
    """Rows of preprocess_stats_<comp>.tsv (duplicates by (seq, stem): last wins)."""
    p = stats_dir / f"preprocess_stats_{comp}.tsv"
    rows: dict[tuple[str, str], dict] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if len(f) < 5:
                continue
            rows[(f[0], f[2])] = {"seq": f[0], "comp": f[1], "stem": f[2], "n": int(f[3]), "saved": int(f[4])}
    return list(rows.values())


def attempted(n: int, k: int = NUM_FRAMES) -> int:
    return len(set(np.linspace(0, max(n - 1, 0), k, endpoint=True, dtype=int).tolist())) if n > 0 else 0


def compute_counts(source: DFBJsonSource, stats_dir: Path, comps: tuple[str, ...] = ("c23", "c40")) -> list[dict]:
    """One row per (split, type, comp) that has any data."""
    v2s = video_split_map(stats_dir)
    rows: list[dict] = []
    for comp in comps:
        stats = read_stats(stats_dir, comp)
        for seq, label in SEQ_LABEL.items():
            for split in SPLITS:
                vids = source._data().get(label, {}).get(split, {}).get(comp, {})  # noqa: SLF001
                st = [r for r in stats if r["seq"] == seq and v2s.get(r["stem"]) == split]
                if not vids and not st:
                    continue
                counts = [len(v["frames"]) for v in vids.values()]
                zero = sorted(r["stem"] for r in st if r["saved"] == 0)
                att = sum(attempted(r["n"]) for r in st)
                saved = sum(r["saved"] for r in st)
                rows.append({
                    "split": split, "type": label, "comp": comp,
                    "videos_processed": len(st) if st else None, "videos_in_json": len(vids),
                    "frames_mean": float(np.mean(counts)) if counts else None,
                    "frames_min": int(min(counts)) if counts else None,
                    "zero_face_videos": zero, "n_zero_face": len(zero),
                    "frames_attempted": att if st else None, "frames_saved": saved if st else None,
                    "failed_detection_rate": (1 - saved / att) if st and att else None,
                })
    return rows


def to_markdown(rows: list[dict]) -> str:
    def f(x, fmt="{:.2f}"):
        return "-" if x is None else fmt.format(x)
    out = ["| split | type | comp | videos (proc/json) | frames mean | frames min | zero-face | failed rate |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['split']} | {r['type']} | {r['comp']} | {f(r['videos_processed'], '{}')}/{r['videos_in_json']} "
                   f"| {f(r['frames_mean'], '{:.1f}')} | {f(r['frames_min'], '{}')} | {r['n_zero_face']} "
                   f"| {f(r['failed_detection_rate'], '{:.4f}')} |")
    zl = [(r["split"], r["type"], r["comp"], r["zero_face_videos"]) for r in rows if r["zero_face_videos"]]
    if zl:
        out += ["", "Zero-face videos:"] + [f"- {s}/{t}/{c}: {', '.join(v)}" for s, t, c, v in zl]
    return "\n".join(out) + "\n"


def crop_grid(source: DFBJsonSource, fig_path: Path, seed: int = 0, n_cols: int = 8, split: str = "test") -> Path:
    """6 rows (real, DF, F2F, FS, NT at c23; real at c40) x n_cols random crops, seeded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    spec = [("FF-real", "c23", "real c23"), ("FF-DF", "c23", "Deepfakes c23"), ("FF-F2F", "c23", "Face2Face c23"),
            ("FF-FS", "c23", "FaceSwap c23"), ("FF-NT", "c23", "NeuralTex c23"), ("FF-real", "c40", "real c40")]
    allrecs = {c: source.records(split, c) for c in ("c23", "c40")}
    fig, axes = plt.subplots(len(spec), n_cols, figsize=(1.6 * n_cols + 1.2, 1.6 * len(spec)), squeeze=False)
    for i, (label, comp, name) in enumerate(spec):
        recs = [r for r in allrecs[comp] if r.video.startswith(label + "/")]
        pick = rng.choice(len(recs), size=min(n_cols, len(recs)), replace=False) if recs else []
        for j in range(n_cols):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(source.load(recs[int(pick[j])]))
        axes[i][0].text(-0.08, 0.5, name, transform=axes[i][0].transAxes, ha="right", va="center", fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    return fig_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=str(DFB_JSON / "FaceForensics++.json"))
    ap.add_argument("--frames-root", default=str(FFPP_FRAMES))
    ap.add_argument("--stats-dir", default=str(FFPP_FRAMES / "FaceForensics++"))
    ap.add_argument("--out-dir", default=str(RESULTS_ROOT / "qa"))
    ap.add_argument("--fig", default=str(RESULTS_ROOT / "figures" / "qa_crops.png"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    src = DFBJsonSource(a.json, a.frames_root)
    rows = compute_counts(src, Path(a.stats_dir))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_counts.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = to_markdown(rows)
    (out / "preprocess_counts.md").write_text(md)
    crop_grid(src, Path(a.fig), a.seed)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
