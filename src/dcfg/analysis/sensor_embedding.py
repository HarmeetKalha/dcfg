"""Phase 6.4: does the LIVE-pretrained sensor separate degradation types on FF++ faces?

    python -m dcfg.analysis.sensor_embedding [--n 500] [--weights ~/data/weights/sensor_live.pt]

Embeds N seeded FF++ *val* c23 crops under clean, jpeg_q30, blur_s3, noise_s15, resize_0.25, mixed_heavy
(deterministic eval conditions), then reports t-SNE (figure), silhouette score of the 128-d embeddings by
condition (euclidean), and 5-fold kNN (k=5) condition classification accuracy vs chance (1/#conditions).
Folds are grouped by source image so the same crop never lands in train and test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dcfg.data.degradations import apply_condition, load_conditions
from dcfg.data.ffpp import DFBJsonSource
from dcfg.models.sensor import DegradationSensor
from dcfg.paths import REPO_ROOT, RESULTS_ROOT, WEIGHTS_DIR

CONDITIONS_YAML = REPO_ROOT / "configs" / "eval" / "conditions.yaml"
CONDS = ("clean", "jpeg_q30", "blur_s3", "noise_s15", "resize_0.25", "mixed_heavy")


@torch.no_grad()
def embed_conditions(source: DFBJsonSource, sensor: DegradationSensor, n: int, seed: int, device: str,
                     conds: tuple[str, ...] = CONDS, split: str = "val", comp: str = "c23",
                     conditions_yaml: Path = CONDITIONS_YAML) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings (N*C,128), condition index (N*C,), image index (N*C,))."""
    recs = source.records(split, comp)
    if not recs:
        raise RuntimeError(f"no frames for {split}/{comp}")
    rng = np.random.default_rng(seed)
    sel = [recs[i] for i in np.sort(rng.choice(len(recs), size=min(n, len(recs)), replace=False))]
    imgs = [source.load(r) for r in sel]
    table = load_conditions(conditions_yaml)
    sensor.eval().to(device)
    embs, cl, im = [], [], []
    for ci, name in enumerate(conds):
        c = table[name]
        deg = np.stack([apply_condition(x, c, r.key) if c.synthetic else x for x, r in zip(imgs, sel)])
        for i in range(0, len(deg), 128):
            x = torch.from_numpy(deg[i:i + 128]).to(device).permute(0, 3, 1, 2).float() / 255.0
            embs.append(sensor.embed(x).float().cpu().numpy())
        cl += [ci] * len(sel)
        im += list(range(len(sel)))
    return np.concatenate(embs), np.array(cl), np.array(im)


def separation_metrics(emb: np.ndarray, cond: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    n_cond = len(np.unique(cond))
    folds = min(5, len(np.unique(groups)))
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=k), emb, cond, cv=GroupKFold(n_splits=folds),
                             groups=groups)
    return {"silhouette": float(silhouette_score(emb, cond)), "knn_k": k, "knn_folds": folds,
            "knn_accuracy": float((pred == cond).mean()), "chance": 1.0 / n_cond}


def plot_tsne(emb: np.ndarray, cond: np.ndarray, names: tuple[str, ...], path: Path, seed: int = 0) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    perp = float(max(2, min(30, (len(emb) - 1) // 3)))
    xy = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cmap = plt.get_cmap("tab10")
    for i, nm in enumerate(names):
        m = cond == i
        ax.scatter(xy[m, 0], xy[m, 1], s=6, color=cmap(i), label=nm, alpha=0.75)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_title("Sensor embedding (FF++ val crops), t-SNE")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--weights", default=str(WEIGHTS_DIR / "sensor_live.pt"))
    ap.add_argument("--fig", default=str(RESULTS_ROOT / "figures" / "sensor_embedding.png"))
    ap.add_argument("--out", default=str(RESULTS_ROOT / "qa" / "sensor_embedding.json"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensor = DegradationSensor()
    sensor.load_backbone(a.weights)
    emb, cond, img = embed_conditions(DFBJsonSource(a.json, a.frames_root), sensor, a.n, a.seed, device)
    res = {"n_images": int(img.max()) + 1, "conditions": list(CONDS), "weights": str(a.weights),
           "split": "val", "seed": a.seed, **separation_metrics(emb, cond, img, seed=a.seed)}
    plot_tsne(emb, cond, CONDS, Path(a.fig), a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
