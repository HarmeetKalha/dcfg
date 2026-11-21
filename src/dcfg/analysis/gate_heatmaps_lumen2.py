"""LUMEN2 gate heatmaps + side-by-side with LUMEN1 -> results/gate/lumen2/

    python -m dcfg.analysis.gate_heatmaps_lumen2 --trainable runs/lumen2_train/best_trainable.pt [--out results/gate/lumen2]

Same study as ``dcfg.analysis.gate_heatmaps`` (test split, same 13 conditions, per-coefficient mean/std of w overall and per class,
shared diverging colour scale centred at w = 1, difference-from-clean plots) for the best LUMEN2 checkpoint. The LUMEN2 gate acts on the
224 px GenD-resized damaged crop (28x28 blocks of 8x8) and is driven by the 224-px DCT statistics plus the jointly trained sensor
embedding; only the sensor + gate are evaluated here (no CLIP). Colour scales are SHARED with the LUMEN1 plots (computed over both
models), so the figures are directly comparable; ``side_by_side/`` has LUMEN1 above LUMEN2 per condition (w and difference from clean).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcfg.analysis import gate_heatmaps as GH
from dcfg.data.degradations import load_conditions
from dcfg.eval.evaluate import CONDITIONS_YAML, ConditionDataset, condition_comp
from dcfg.lumen2.model import pack_views
from dcfg.paths import DATA_ROOT, REPO_ROOT


def collect(trainable: str, conditions: Sequence[str], out_npz: Path, workers: int = 5, batch_size: int = 128,
            max_frames: int | None = None) -> None:
    """Accumulate LUMEN2 gate weights per condition on the test split (same npz schema as the LUMEN1 collector; resumable)."""
    from dcfg.gend.evaluate_gend import CropSource
    from dcfg.lumen2.evaluate_lumen2 import load_lumen2

    model, _ = load_lumen2(trainable, "cuda" if torch.cuda.is_available() else "cpu")
    dev = next(model.parameters()).device
    conds = load_conditions(CONDITIONS_YAML)
    source = CropSource(DATA_ROOT / "gend_crops")
    store: dict[str, np.ndarray] = {}
    if out_npz.exists():
        with np.load(out_npz) as z:
            store = {k: z[k] for k in z.files}
    for name in conditions:
        if f"{name}__sum" in store:
            continue
        cond = conds[name]
        recs = source.records("test", condition_comp(cond))
        if max_frames:
            recs = recs[:: max(1, len(recs) // max_frames)]
        labels = np.array([r.label for r in recs])
        dl = DataLoader(ConditionDataset(source, recs, cond, post_transform=pack_views), batch_size=batch_size, shuffle=False, num_workers=workers)
        ws = []
        with torch.no_grad():
            for xb in dl:
                _, w, _ = model.front(xb.to(dev).float().div_(255.0))
                ws.append(w.double().cpu().numpy())
        w = np.concatenate(ws)
        for tag, m in (("all", np.ones_like(labels, bool)), ("real", labels == 0), ("fake", labels == 1)):
            store[f"{name}__{tag}__sum"] = w[m].sum(0)
            store[f"{name}__{tag}__sq"] = (w[m] ** 2).sum(0)
            store[f"{name}__{tag}__n"] = np.array(int(m.sum()))
        store[f"{name}__sum"] = np.array(1)
        np.savez(out_npz, **store)
        print(f"{name}: n={len(w)} mean w={w.mean():.4f}", flush=True)


def shared_scale(*stats: dict) -> tuple[float, float]:
    """(w half-width around 1, difference half-width around 0) over several models' stats (same rule as ``gate_heatmaps.plots``)."""
    dev = max(float(np.abs(s["all"]["mean"] - 1).max()) for st in stats for s in st.values())
    dmax = max(float(np.abs(s["all"]["mean"] - st["clean"]["all"]["mean"]).max()) for st in stats for c, s in st.items() if c != "clean")
    return dev, dmax


def side_by_side(s1: dict, s2: dict, out: Path, dev: float, dmax: float) -> None:
    from matplotlib.colors import TwoSlopeNorm

    out.mkdir(parents=True, exist_ok=True)
    wn, dn = TwoSlopeNorm(vcenter=1.0, vmin=1 - dev, vmax=1 + dev), TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)
    for c in s1:
        if c not in s2:
            continue
        GH._panel_grid([(f"LUMEN1 {c}", s1[c]["all"]["mean"]), (f"LUMEN2 {c}", s2[c]["all"]["mean"])], out / f"w_{c}.png", wn, "RdBu_r",
                       "mean gate weight w (1 = unchanged)")
        if c != "clean":
            GH._panel_grid([(f"LUMEN1 {c} - clean", s1[c]["all"]["mean"] - s1["clean"]["all"]["mean"]),
                            (f"LUMEN2 {c} - clean", s2[c]["all"]["mean"] - s2["clean"]["all"]["mean"])], out / f"diff_{c}.png", dn, "PuOr_r",
                           "w(condition) - w(clean)  (0 = same as clean)")


def comparison_md(s1: dict, s2: dict) -> list[str]:
    bt1, bt2 = GH.band_table(s1, None), GH.band_table(s2, None)
    L = ["", "## LUMEN1 (Xception) vs LUMEN2 (CLIP)", "",
         "LUMEN1: gate on the 256 px DFB crop (32x32 blocks), LIVE-pretrained sensor, frozen CNN. LUMEN2: gate on the 224 px GenD-resized crop "
         "(28x28 blocks), sensor trained jointly from scratch, LayerNorm-tuned CLIP. The (u, v) coefficient grid is the same 8x8 DCT grid, "
         "but the spatial scale of a coefficient differs because the image is resized, so compare patterns, not exact values.", "",
         "Mean |w - 1| over the 192 coefficients (how far the gate departs from the identity) and mean w:", "",
         "| condition | LUMEN1 mean abs(w-1) | LUMEN2 mean abs(w-1) | LUMEN1 mean w | LUMEN2 mean w |", "|---|---|---|---|---|"]
    for c in s1:
        if c in s2:
            L.append(f"| {c} | {np.abs(s1[c]['all']['mean'] - 1).mean():.3f} | {np.abs(s2[c]['all']['mean'] - 1).mean():.3f} | "
                     f"{s1[c]['all']['mean'].mean():.3f} | {s2[c]['all']['mean'].mean():.3f} |")
    L += ["", "Change relative to clean, band means (average over Y, Cb, Cr); LUMEN1 -> LUMEN2:", "",
          "| condition | low | mid | high |", "|---|---|---|---|"]
    for c in s1:
        if c == "clean" or c not in s2:
            continue
        cells = []
        for b in GH.BANDS:
            d1 = np.mean([bt1[c][ch][b] - bt1["clean"][ch][b] for ch in GH.CHANNELS])
            d2 = np.mean([bt2[c][ch][b] - bt2["clean"][ch][b] for ch in GH.CHANNELS])
            cells.append(f"{d1:+.3f} -> {d2:+.3f}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "Files: `side_by_side/w_<cond>.png` and `side_by_side/diff_<cond>.png` (LUMEN1 on top, LUMEN2 below, identical colour scales); "
          "`w_*.png` / `diff_*.png` in this directory are the LUMEN2 plots on that same scale."]
    return L


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.analysis.gate_heatmaps_lumen2", description=__doc__.split("\n")[0])
    ap.add_argument("--trainable", default=str(REPO_ROOT / "runs/lumen2_train/best_trainable.pt"))
    ap.add_argument("--lumen1-dir", default=str(REPO_ROOT / "results/gate/lumen1"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results/gate/lumen2"))
    ap.add_argument("--conditions", default=",".join(GH.HEATMAP_CONDITIONS))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test only")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = [c for c in a.conditions.split(",") if c]
    npz = out / "gate_means.npz"
    if not a.plot_only:
        collect(a.trainable, conds, npz, workers=a.workers, max_frames=a.max_frames)
    s2 = GH.load_stats(npz, conds)
    s1 = GH.load_stats(Path(a.lumen1_dir) / "gate_means.npz", conds)
    dev, dmax = shared_scale(s1, s2)
    GH.write_csv(s2, None, out / "gate_means.csv")
    scale = GH.plots(s2, None, out, dev=dev, dmax=dmax)
    side_by_side(s1, s2, out / "side_by_side", dev, dmax)
    GH.write_summary(s2, None, scale, out / "summary.md", "lumen2", title="LUMEN2 (CLIP ViT-L/14) gate heatmaps")
    with open(out / "summary.md", "a") as f:
        f.write("\n".join(comparison_md(s1, s2)) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
