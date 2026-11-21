"""Gate heatmaps: what does the trained DCT gate do per damage type? (eval only, test split)

    python -m dcfg.analysis.gate_heatmaps --run lumen_xception --fixed-run fixed_xception --out results/gate/lumen1

For each condition (same degradation code and per-frame seeding as ``results/raw``) the front-end of
``runs/<run>/frontend.pt`` is run on every test frame and the per-coefficient gate weights ``w`` (3x8x8) are
accumulated: mean and std over all frames, plus per-class (real/fake) means. No detector is needed.
Outputs: ``gate_means.csv`` (long format), ``gate_means.npz`` (cache), PNG grids (shared colour scale, diverging
map centred at w=1) and ``summary.md``. Frequency bands (u+v position in the 8x8 grid) appear ONLY in this
reporting step; the gate itself has no bands.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dcfg.data.degradations import load_conditions
from dcfg.data.ffpp import DFBJsonSource
from dcfg.eval.evaluate import CONDITIONS_YAML, ConditionDataset, condition_comp
from dcfg.paths import REPO_ROOT, RUNS_ROOT

#: conditions of the heatmap study (medium + heaviest per synthetic type, mixed_heavy, c40, held-out heavy ones)
HEATMAP_CONDITIONS = ("clean", "jpeg_q50", "jpeg_q10", "blur_s2", "blur_s4", "noise_s10", "noise_s20",
                      "resize_0.5", "resize_0.25", "mixed_heavy", "c40", "webp_q20", "motion_k11")
CHANNELS = ("Y", "Cb", "Cr")
#: reporting-only bands on u+v (u,v in 0..7): low 0-3, mid 4-7, high 8-14
BANDS = {"low": (0, 3), "mid": (4, 7), "high": (8, 14)}


def band_mask(name: str) -> np.ndarray:
    """Boolean 8x8 mask of the coefficients whose u+v lies in the band (reporting only)."""
    lo, hi = BANDS[name]
    s = np.add.outer(np.arange(8), np.arange(8))
    return (s >= lo) & (s <= hi)


def collect(run: str, conditions: Sequence[str], out_npz: Path, workers: int = 5, batch_size: int = 128,
            max_frames: int | None = None) -> None:
    """Accumulate gate weights per condition on the test split; writes ``out_npz`` (resumable per condition)."""
    from dcfg.train.train_frontend import load_frontend
    device = "cuda" if torch.cuda.is_available() else "cpu"
    front, meta = load_frontend(RUNS_ROOT / run / "frontend.pt", device)
    conds = load_conditions(CONDITIONS_YAML)
    source = DFBJsonSource()
    store: dict[str, np.ndarray] = {}
    if out_npz.exists():
        with np.load(out_npz) as z:
            store = {k: z[k] for k in z.files}
    for name in conditions:
        if f"{name}__sum" in store:
            continue
        cond = conds[name]
        recs = source.records("test", condition_comp(cond))
        if max_frames:  # smoke test only
            recs = recs[:: max(1, len(recs) // max_frames)]
        labels = np.array([r.label for r in recs])
        dl = DataLoader(ConditionDataset(source, recs, cond), batch_size=batch_size, shuffle=False, num_workers=workers)
        ws = []
        with torch.no_grad():
            for xb in dl:
                x01 = xb.to(device).float().div_(255.0)
                _, w, _ = front.forward_full(x01)
                ws.append(w.double().cpu().numpy())
        w = np.concatenate(ws)  # (N,3,8,8)
        for tag, m in (("all", np.ones_like(labels, bool)), ("real", labels == 0), ("fake", labels == 1)):
            store[f"{name}__{tag}__sum"] = w[m].sum(0)
            store[f"{name}__{tag}__sq"] = (w[m] ** 2).sum(0)
            store[f"{name}__{tag}__n"] = np.array(int(m.sum()))
        store[f"{name}__sum"] = np.array(1)  # done marker
        np.savez(out_npz, **store)
        print(f"{name}: n={len(w)} mean w={w.mean():.4f}", flush=True)


def fixed_weights(fixed_run: str) -> np.ndarray:
    """Static weights (3,8,8) of the fixed gate from ``runs/<fixed_run>/frontend.pt``."""
    from dcfg.train.train_frontend import load_frontend
    front, _ = load_frontend(RUNS_ROOT / fixed_run / "frontend.pt", "cpu")
    return front.gate(1)[0].detach().double().numpy()


def load_stats(npz: Path, conditions: Sequence[str]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """name -> class tag -> {mean, std, n}."""
    out: dict = {}
    with np.load(npz) as z:
        for c in conditions:
            out[c] = {}
            for tag in ("all", "real", "fake"):
                n = float(z[f"{c}__{tag}__n"])
                mean = z[f"{c}__{tag}__sum"] / n
                var = np.maximum(z[f"{c}__{tag}__sq"] / n - mean**2, 0)
                out[c][tag] = {"mean": mean, "std": np.sqrt(var), "n": n}
    return out


def write_csv(stats: dict, fixed: np.ndarray | None, path: Path) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "class", "channel", "u", "v", "mean_w", "std_w", "n_frames"])
        for c, d in stats.items():
            for tag, s in d.items():
                for ch in range(3):
                    for u in range(8):
                        for v in range(8):
                            wr.writerow([c, tag, CHANNELS[ch], u, v, f"{s['mean'][ch, u, v]:.6f}",
                                         f"{s['std'][ch, u, v]:.6f}", int(s["n"])])
        if fixed is not None:
            for ch in range(3):
                for u in range(8):
                    for v in range(8):
                        wr.writerow(["fixed_gate", "static", CHANNELS[ch], u, v, f"{fixed[ch, u, v]:.6f}", "0", 0])


def _panel_grid(arrs: list[tuple[str, np.ndarray]], path: Path, norm, cmap: str, cbar_label: str, ncols: int = 1) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(arrs)
    fig, axes = plt.subplots(rows, 3, figsize=(7.5, 2.3 * rows), squeeze=False)
    im = None
    for r, (title, a) in enumerate(arrs):
        for ch in range(3):
            ax = axes[r, ch]
            im = ax.imshow(a[ch], cmap=cmap, norm=norm)
            ax.set_xticks(range(8)); ax.set_yticks(range(8))
            ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(CHANNELS[ch], fontsize=9)
            if ch == 0:
                ax.set_ylabel(f"{title}\nu (vertical)", fontsize=7)
            if r == rows - 1:
                ax.set_xlabel("v (horizontal)", fontsize=7)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.92, 0.1, 0.02, 0.8))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plots(stats: dict, fixed: np.ndarray | None, out: Path) -> dict[str, float]:
    """Write PNGs; returns the shared scale limits."""
    from matplotlib.colors import TwoSlopeNorm
    all_means = [s["all"]["mean"] for s in stats.values()] + ([fixed] if fixed is not None else [])
    dev = max(float(np.abs(a - 1).max()) for a in all_means)
    wnorm = TwoSlopeNorm(vcenter=1.0, vmin=1 - dev, vmax=1 + dev)
    lab = "mean gate weight w (1 = unchanged)"
    for c, s in stats.items():
        _panel_grid([(c, s["all"]["mean"])], out / f"w_{c}.png", wnorm, "RdBu_r", lab)
    _panel_grid([(c, s["all"]["mean"]) for c, s in stats.items()], out / "w_all_conditions.png", wnorm, "RdBu_r", lab)
    if fixed is not None:
        _panel_grid([("fixed gate (static)", fixed)], out / "w_fixed.png", wnorm, "RdBu_r", lab)
    clean = stats["clean"]["all"]["mean"]
    diffs = [(c, s["all"]["mean"] - clean) for c, s in stats.items() if c != "clean"]
    dmax = max(float(np.abs(d).max()) for _, d in diffs)
    dnorm = TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)
    dlab = "w(condition) - w(clean)  (0 = same as clean)"
    for c, d in diffs:
        _panel_grid([(f"{c} - clean", d)], out / f"diff_{c}.png", dnorm, "PuOr_r", dlab)
    _panel_grid([(f"{c} - clean", d) for c, d in diffs], out / "diff_all_conditions.png", dnorm, "PuOr_r", dlab)
    return {"w_dev": dev, "diff_max": dmax}


def band_table(stats: dict, fixed: np.ndarray | None) -> dict[str, dict[str, dict[str, float]]]:
    """condition -> channel -> band -> mean w over the band's coefficients."""
    out: dict = {}
    items = [(c, s["all"]["mean"]) for c, s in stats.items()] + ([("fixed_gate", fixed)] if fixed is not None else [])
    for c, m in items:
        out[c] = {CHANNELS[ch]: {b: float(m[ch][band_mask(b)].mean()) for b in BANDS} for ch in range(3)}
    return out


def write_summary(stats: dict, fixed: np.ndarray | None, scale: dict[str, float], path: Path, run: str) -> None:
    bt = band_table(stats, fixed)
    L = [f"# LUMEN1 (Xception) gate heatmaps: `{run}`", "",
         "Test split, all frames, same degradation code and per-frame seeds as `results/raw`. w = 2*sigmoid(z); w = 1 means "
         "the coefficient is passed unchanged, w < 1 attenuated, w > 1 amplified. The gate has no bands; low / mid / high "
         "below are an after-the-fact grouping of the 8x8 grid by u+v (low 0-3, mid 4-7, high 8-14; u = vertical, "
         "v = horizontal frequency index, DC at (0,0)).", "",
         f"Shared colour scale for all w plots: 1 +/- {scale['w_dev']:.3f}; difference plots: +/- {scale['diff_max']:.3f}.",
         "", "Files: `gate_means.csv` (mean, std, per class real/fake, per coefficient), `w_<cond>.png`, `w_all_conditions.png`, "
         "`w_fixed.png`, `diff_<cond>.png` (condition minus clean), `diff_all_conditions.png`.", "",
         "## Mean w per band (all test frames)", "",
         "| condition | Y low | Y mid | Y high | Cb low | Cb mid | Cb high | Cr low | Cr mid | Cr high |", "|---|" + "---|" * 9]
    for c, d in bt.items():
        L.append(f"| {c} | " + " | ".join(f"{d[ch][b]:.3f}" for ch in CHANNELS for b in BANDS) + " |")
    L += ["", "## Change relative to clean (band mean of w(condition) - w(clean))", "",
          "| condition | Y low | Y mid | Y high | Cb low | Cb mid | Cb high | Cr low | Cr mid | Cr high |", "|---|" + "---|" * 9]
    for c in stats:
        if c == "clean":
            continue
        L.append(f"| {c} | " + " | ".join(f"{bt[c][ch][b] - bt['clean'][ch][b]:+.3f}" for ch in CHANNELS for b in BANDS) + " |")
    L += ["", "## In plain language", ""]
    thr = 0.02
    for c in stats:
        if c == "clean":
            continue
        down, up = [], []
        for ch in CHANNELS:
            for b in BANDS:
                d = bt[c][ch][b] - bt["clean"][ch][b]
                (down if d < -thr else up if d > thr else []).append(f"{ch} {b} ({d:+.3f})")
        base = ", ".join(f"{ch} {b} {bt['clean'][ch][b]:.2f}" for ch in CHANNELS for b in BANDS if abs(bt['clean'][ch][b] - 1) > 0.1)
        dd = stats[c]["all"]["mean"] - stats["clean"]["all"]["mean"]
        i = np.unravel_index(np.abs(dd).argmax(), dd.shape)
        L.append(f"- **{c}**: lowered vs clean: {', '.join(down) or 'none above 0.02'}; raised: {', '.join(up) or 'none above 0.02'}. "
                 f"Largest single change: {CHANNELS[i[0]]} (u={i[1]}, v={i[2]}) {dd[i]:+.3f}.")
    cl = bt["clean"]
    L += ["", f"Clean-image baseline of the gate (what it does with no damage): " +
          ", ".join(f"{ch} {b} {cl[ch][b]:.2f}" for ch in CHANNELS for b in BANDS) + ".", ""]
    # image dependence: how much of the variation is between images, and real vs fake
    L += ["## Image dependence", "", "| condition | mean per-coefficient std over frames | max |real - fake| mean-w gap |", "|---|---|---|"]
    for c, s in stats.items():
        gap = np.abs(s["real"]["mean"] - s["fake"]["mean"]).max()
        L.append(f"| {c} | {s['all']['std'].mean():.4f} | {gap:.4f} |")
    if fixed is not None:
        f_ = bt["fixed_gate"]
        L += ["", "## Fixed gate (static, `fixed_xception`)", "", "Band means: " +
              ", ".join(f"{ch} {b} {f_[ch][b]:.3f}" for ch in CHANNELS for b in BANDS) + "."]
    path.write_text("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.analysis.gate_heatmaps")
    ap.add_argument("--run", default="lumen_xception")
    ap.add_argument("--fixed-run", default="fixed_xception")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/gate/lumen1"))
    ap.add_argument("--conditions", default=",".join(HEATMAP_CONDITIONS))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test only")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = [c for c in a.conditions.split(",") if c]
    npz = out / "gate_means.npz"
    if not a.plot_only:
        collect(a.run, conds, npz, workers=a.workers, max_frames=a.max_frames)
    stats = load_stats(npz, conds)
    fixed = fixed_weights(a.fixed_run) if (RUNS_ROOT / a.fixed_run / "frontend.pt").exists() else None
    write_csv(stats, fixed, out / "gate_means.csv")
    scale = plots(stats, fixed, out)
    write_summary(stats, fixed, scale, out / "summary.md", a.run)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
