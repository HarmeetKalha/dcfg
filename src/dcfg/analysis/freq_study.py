"""Phase 5.2 frequency-damage study (Part 1 of the paper).

    python -m dcfg.analysis.freq_study [--n 2000] [--out results/raw/freq_study.npz] [--fig-dir results/figures]

Sample N clean c23 test crops (seeded), apply every synthetic condition of conditions.yaml, and measure
how the energy in each 8x8 DCT position of each YCbCr channel changes.

Definitions (all fixed here, documented for the paper):
* Transform: ``rgb_to_ycbcr`` (BT.601 full range, chroma offset 0.5, no level shift) then ``BlockDCT``
  (orthonormal 8x8), exactly the front-end's transform. Note the DC of Cb/Cr therefore contains the offset.
* Energy ``E[c,u,v]`` = mean of ``|coef|`` over all blocks of all images (L1 magnitude, not squared).
* Relative change ``(E_deg - E_clean) / E_clean`` per position (``E_clean`` floored at 1e-8).
* Radial curve: radius = sqrt(u^2+v^2) in [0, 9.9], bins [k,k+1) for k=0..9; per bin the relative change of
  the bin-summed energies (sum E_deg - sum E_clean)/sum E_clean.
* Fingerprint distance: cosine distance 1 - cos between the flattened 192-d relative-change vectors.
* c40 (real compression) is not synthetic and is excluded; ``clean`` is the reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from dcfg.data.degradations import apply_condition, load_conditions
from dcfg.data.ffpp import DFBJsonSource
from dcfg.models.dct import BlockDCT, rgb_to_ycbcr
from dcfg.paths import REPO_ROOT, RESULTS_ROOT

CONDITIONS_YAML = REPO_ROOT / "configs" / "eval" / "conditions.yaml"
N_BINS = 10
CHANNELS = ("Y", "Cb", "Cr")


def radial_bins() -> np.ndarray:
    """(8,8) int array: radial bin index floor(sqrt(u^2+v^2)) in 0..9."""
    u, v = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    return np.floor(np.sqrt(u ** 2 + v ** 2)).astype(int)


@torch.no_grad()
def mean_abs_coef(imgs: np.ndarray, dct: BlockDCT, device: str, batch: int = 128) -> np.ndarray:
    """imgs uint8 (N,H,W,3) -> (3,8,8) float64 mean |coef| per position over images and blocks."""
    total = torch.zeros(3, 8, 8, dtype=torch.float64, device=device)
    nblocks = 0
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        c = dct(rgb_to_ycbcr(x))  # (B,3,Hb,Wb,8,8)
        total += c.abs().double().sum(dim=(0, 2, 3))
        nblocks += c.shape[0] * c.shape[2] * c.shape[3]
    return (total / nblocks).cpu().numpy()


def relative_change(e_deg: np.ndarray, e_clean: np.ndarray) -> np.ndarray:
    return (e_deg - e_clean) / np.maximum(e_clean, 1e-8)


def radial_curve(e_deg: np.ndarray, e_clean: np.ndarray) -> np.ndarray:
    """(3,8,8),(3,8,8) -> (3,N_BINS) relative change of bin-summed energy."""
    rb = radial_bins()
    out = np.zeros((3, N_BINS))
    for k in range(N_BINS):
        m = rb == k
        if m.any():
            d, c = e_deg[:, m].sum(1), e_clean[:, m].sum(1)
            out[:, k] = (d - c) / np.maximum(c, 1e-8)
    return out


def cosine_distance_matrix(vecs: np.ndarray) -> np.ndarray:
    """(n,d) -> (n,n) 1 - cosine similarity."""
    nrm = np.linalg.norm(vecs, axis=1, keepdims=True)
    u = vecs / np.maximum(nrm, 1e-12)
    return np.clip(1.0 - u @ u.T, 0.0, 2.0)


def run_study(source: DFBJsonSource, n: int, seed: int, device: str, conditions_yaml: Path = CONDITIONS_YAML,
              split: str = "test", comp: str = "c23") -> dict[str, np.ndarray]:
    """Compute all study arrays (no I/O besides reading frames)."""
    recs = source.records(split, comp)
    if not recs:
        raise RuntimeError(f"no frames for {split}/{comp}")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(recs), size=min(n, len(recs)), replace=False))
    sel = [recs[i] for i in idx]
    imgs = np.stack([source.load(r) for r in sel])
    conds = [c for c in load_conditions(conditions_yaml).values() if c.synthetic and c.name != "clean"]
    dct = BlockDCT().to(device)
    e_clean = mean_abs_coef(imgs, dct, device)
    e_deg, rel, rad = [], [], []
    for c in conds:
        deg = np.stack([apply_condition(im, c, r.key) for im, r in zip(imgs, sel)])
        e = mean_abs_coef(deg, dct, device)
        e_deg.append(e)
        rel.append(relative_change(e, e_clean))
        rad.append(radial_curve(e, e_clean))
    rel_a = np.stack(rel)
    return {
        "conditions": np.array([c.name for c in conds]), "groups": np.array([c.group for c in conds]),
        "channels": np.array(CHANNELS), "e_clean": e_clean, "e_deg": np.stack(e_deg), "rel_change": rel_a,
        "radial_rel_change": np.stack(rad), "radial_bin_lo": np.arange(N_BINS),
        "fingerprint_dist": cosine_distance_matrix(rel_a.reshape(len(conds), -1)),
        "n_images": np.array(len(sel)), "seed": np.array(seed), "sample_keys": np.array([r.key for r in sel]),
    }


def make_figures(res: dict[str, np.ndarray], fig_dir: Path) -> list[Path]:
    """Heatmaps (conditions x Y/Cb/Cr), radial curves, fingerprint distance matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    fig_dir.mkdir(parents=True, exist_ok=True)
    names, groups = list(res["conditions"]), list(res["groups"])
    rel, rad = res["rel_change"], res["radial_rel_change"]
    n = len(names)
    paths = []

    vmax = max(float(np.abs(rel).max()), 1.0)
    norm = SymLogNorm(linthresh=0.1, vmin=-vmax, vmax=vmax, base=10)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow * 3, ncol, figsize=(2.0 * ncol, 2.0 * 3 * nrow), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for i, nm in enumerate(names):
        r0, c0 = (i // ncol) * 3, i % ncol
        for ch in range(3):
            ax = axes[r0 + ch][c0]
            im = ax.imshow(rel[i, ch], cmap="RdBu_r", norm=norm)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis("on")
            ax.set_title(f"{nm} {CHANNELS[ch]}", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.4, label="relative energy change (symlog)")
    p = fig_dir / "freq_heatmaps.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    uniq = sorted(set(groups), key=groups.index)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)
    for ch, ax in enumerate(axes):
        for i, nm in enumerate(names):
            ax.plot(res["radial_bin_lo"], rad[i, ch], color=cmap(uniq.index(groups[i]) % 10),
                    alpha=0.8, lw=1.2, label=nm)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(CHANNELS[ch]), ax.set_xlabel("radius sqrt(u^2+v^2) bin"), ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("relative energy change")
    axes[-1].legend(fontsize=5, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    p = fig_dir / "freq_radial.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(res["fingerprint_dist"], cmap="viridis", vmin=0)
    ax.set_xticks(range(n)), ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6), ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(im, label="cosine distance")
    ax.set_title("Degradation fingerprint distance (192-d relative change)")
    p = fig_dir / "freq_fingerprint.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS_ROOT / "raw" / "freq_study.npz"))
    ap.add_argument("--fig-dir", default=str(RESULTS_ROOT / "figures"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_study(DFBJsonSource(a.json, a.frames_root), a.n, a.seed, device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    for p in make_figures(res, Path(a.fig_dir)):
        print(p)
    print(f"{out}: {len(res['conditions'])} conditions, {int(res['n_images'])} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
