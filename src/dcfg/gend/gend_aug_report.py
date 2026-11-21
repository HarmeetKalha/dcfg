"""GenD-aug report -> results/lumen2/gend_aug.md

    python -m dcfg.gend.gend_aug_report [--aug-dir results/lumen2/gend_aug] [--train-dir runs/gendaug_train] [--n-boot 1000]

Numbers: result JSONs (single conditions verbatim) and score npz files (aggregates, paired differences) with ONE shared
video-level bootstrap resample matrix for every model/condition (``dcfg.gend.room_report.Models`` machinery), so aggregate
means are bootstrapped jointly and every difference is paired. Models: public GenD, GenD-aug (results/lumen2/gend_aug),
Xception base / augft / LUMEN (results/raw). ECE (15 bins) and selective AUC (80/90%) are computed over all degraded test frames
(21 synthetic conditions + c40) with paired video bootstrap of GenD-aug minus GenD.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np

from dcfg.eval.metrics import auc_metric, ece, paired_bootstrap_metric, selective_auc
from dcfg.gend.room_report import COL_LABEL, COLS, RAW, Models, f4, gf
from dcfg.paths import REPO_ROOT

G, A = "GenD (public)", "GenD-aug"
SUMMARY = ["clean", "synth_mean", "heldout_mean", "c40"]


class AugModels(Models):
    """Same machinery as ``Models`` with GenD-aug added next to the public GenD."""

    def __init__(self, gend_dir: Path, aug_dir: Path, n_boot: int, seed: int = 0) -> None:
        cl = gf.load_conditions()
        self.names = [c["name"] for c in cl]
        self.synth = [c["name"] for c in cl if c["name"] != "clean" and not c.get("held_out") and c.get("synthetic", True)]
        self.held = [c["name"] for c in cl if c.get("held_out")]
        spec = [(G, "gend_public", gend_dir), (A, "gend_aug", aug_dir), ("base_xception", "base_xception", RAW),
                ("augft_xception", "augft_xception", RAW), ("lumen_xception", "lumen_xception", RAW)]
        self.runs = {lab: gf.Run(lab, rid, d, self.names) for lab, rid, d in spec}
        vids = sorted({v for r in self.runs.values() for s in r.scores.values() for v in set(s["video"])})
        rng = np.random.default_rng(seed)
        nv = len(vids)
        self.W = np.vstack([np.ones((1, nv)), rng.multinomial(nv, np.full(nv, 1.0 / nv), size=n_boot).astype(float)])
        self.cur = {lab: gf.curves(r, np.array(vids), self.W) for lab, r in self.runs.items()}

    def diff(self, a: str, b: str, item: str, metric: str = "frame") -> tuple[float, float, float] | None:
        x, y = self.arr(a, item, metric), self.arr(b, item, metric)
        return None if x is None or y is None else gf.pct(x - y)


def degraded_arrays(m: AugModels, lab: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """probs, labels, video ids, frame keys of all non-clean conditions concatenated (fixed condition order)."""
    conds = [c for c in m.names if c != "clean" and c in m.runs[lab].docs]
    z = [np.load(Path(m.runs[lab].raw) / m.runs[lab].docs[c]["scores_file"]) for c in conds]
    cat = lambda k: np.concatenate([a[k] for a in z])  # noqa: E731
    return cat("frame_prob"), cat("frame_label").astype(int), cat("frame_video"), cat("frame_key")


def calibration(m: AugModels, n_boot: int) -> dict:
    pg, yg, vg, kg = degraded_arrays(m, G)
    pa, ya, va, ka = degraded_arrays(m, A)
    if not (np.array_equal(kg, ka) and np.array_equal(yg, ya) and np.array_equal(vg, va)):
        raise RuntimeError("GenD and GenD-aug frame sets differ; paired comparison invalid")
    out = {"n_frames": int(len(pg))}
    for name, fn in (("ece", ece), ("sel80", partial(selective_auc, coverage=0.8)), ("sel90", partial(selective_auc, coverage=0.9)),
                     ("auc_all", auc_metric)):
        out[name] = paired_bootstrap_metric(fn, pa, pg, ya, vg, n=n_boot)  # a = GenD-aug, b = GenD
    return out


def fmt(v: float, lo: float | None = None, hi: float | None = None, signed: bool = False) -> str:
    s = "+" if signed else ""
    return f"{v:{s}.4f}" + ("" if lo is None else f" [{lo:{s}.4f}, {hi:{s}.4f}]")


def ist(sec: float) -> str:
    return f"{sec / 3600:.2f} h ({sec / 86400:.2f} d)"


def build(m: AugModels, cal: dict, train: dict | None, n_boot: int) -> str:
    L: list[str] = []
    P = L.append
    P("# GenD-aug (LUMEN2 baseline): GenD fine-tuned with its own recipe on our degraded stream\n")
    if train and train.get("still_improving_at_cap"):
        P(f"> **FLAG: the run hit the {train['max_epochs']}-epoch cap while validation was still improving** (best epoch "
          f"{train['best_epoch']} of {train['max_epochs']}). GenD-aug may be under-trained relative to a longer schedule.\n")
    P(f"Frame/video AUC on the FF++ c23 test split (+ c40), 95% video-level bootstrap CI ({n_boot} resamples, one shared resample "
      "matrix: aggregates are joint and every difference is paired). Same GenD crops, degradation code, seeds and frame keys as "
      "`results/lumen2/gend_public`. Sources: results/lumen2/gend_public, results/lumen2/gend_aug, results/raw (Xception).\n")
    P("## Frame AUC [95% CI]\n")
    hdr = ["model"] + [COL_LABEL.get(c, c) for c in COLS]
    L += gf.md_table(hdr, [[lab] + [f4(m.value(lab, c)) for c in COLS] for lab in m.runs])
    P("\n## Video AUC [95% CI]\n")
    L += gf.md_table(hdr, [[lab] + [f4(m.value(lab, c, "video")) for c in COLS] for lab in m.runs])
    P("\n## Paired difference GenD-aug minus GenD (public) [95% CI]\n")
    rows = [[COL_LABEL[c], f4(m.diff(A, G, c), True), f4(m.diff(A, G, c, "video"), True)] for c in SUMMARY]
    L += gf.md_table(["aggregate", "frame AUC", "video AUC"], rows)
    d = m.diff(A, G, "clean")
    P("")
    if d:
        verdict = ("**clean AUC did NOT drop** (CI not entirely below 0)" if d[2] >= 0 else "**clean AUC DROPPED** (paired CI entirely below 0)")
        P(f"Clean frame AUC change: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] -> {verdict}.")
    P("\n## Drop from own clean score (frame AUC, paired bootstrap) [95% CI]\n")
    dc = [c for c in SUMMARY if c != "clean"]
    L += gf.md_table(["model"] + [COL_LABEL[c] for c in dc], [[lab] + [f4(m.drop(lab, c), True) for c in dc] for lab in (G, A, "augft_xception")])
    P("\n## Calibration and selective prediction over all degraded test frames\n")
    P(f"All non-clean conditions concatenated (21 synthetic + c40 = 22 conditions, {cal['n_frames']} frames). ECE: 15 equal-width bins, "
      "predicted P(fake) vs label (lower is better). Selective AUC: keep the most confident 80%/90% by |p-0.5|. CIs: video-level "
      "bootstrap; difference = GenD-aug minus GenD, paired.\n")
    rows = []
    for name, lab in (("ECE", "ece"), ("selective AUC 80%", "sel80"), ("selective AUC 90%", "sel90"), ("AUC (all frames)", "auc_all")):
        r = cal[lab]
        rows.append([name, fmt(r["b"], r["b_ci_low"], r["b_ci_high"]), fmt(r["a"], r["a_ci_low"], r["a_ci_high"]),
                     fmt(r["delta"], r["ci_low"], r["ci_high"], True)])
    L += gf.md_table(["metric", "GenD (public)", "GenD-aug", "difference"], rows)
    if train:
        P("\n## Training\n")
        P(f"- stop reason: **{train['stop_reason']}**; best epoch {train['best_epoch']}, stop epoch {train['stop_epoch']} "
          f"(1-based; cap {train['max_epochs']}, patience 6); best val mean AUC {train['best_val_mean_auc']:.4f} "
          f"(degraded val set, 8 of 32 frames per video, {len(train['val_conditions'])} conditions: {', '.join(train['val_conditions'])})")
        P(f"- wall time: train {ist(train['train_s'])}, val {ist(train['val_s'])}, total {ist(train['train_wall_time_s'])} (excludes crop preparation); "
          f"{train['steps']} optimiser steps ({train['steps_per_epoch']}/epoch, effective batch 128), non-finite skipped steps: {train['nonfinite_steps']}")
        P(f"- trainable parameters: {train['trainable_params']:,}; trainable-only file {train['trainable_file_mb']:.3f} MB vs full model file "
          f"{train['full_model_file_mb']:.1f} MB (fp32)")
        P("- val curve (mean frame AUC over the val conditions): " + ", ".join(f"{h['epoch']}: {h['val_mean_auc']:.4f}" for h in train["val_curve"]))
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.gend.gend_aug_report")
    ap.add_argument("--gend-dir", default=str(REPO_ROOT / "results" / "lumen2" / "gend_public"))
    ap.add_argument("--aug-dir", default=str(REPO_ROOT / "results" / "lumen2" / "gend_aug"))
    ap.add_argument("--train-dir", default=str(REPO_ROOT / "runs" / "gendaug_train"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "lumen2" / "gend_aug.md"))
    a = ap.parse_args(argv)
    aug = Path(a.aug_dir)
    m = AugModels(Path(a.gend_dir), aug, a.n_boot)
    cal = calibration(m, a.n_boot)
    td = Path(a.train_dir)
    train = json.loads((td / "done.json").read_text()) if (td / "done.json").exists() else None
    if train:  # keep the training record next to the results
        aug.mkdir(parents=True, exist_ok=True)
        (aug / "train_summary.json").write_text(json.dumps({k: v for k, v in train.items() if k != "train_config"} | {"train_config": train["train_config"]}, indent=2))
        for f in ("metrics.csv", "train_log.csv"):
            if (td / f).exists():
                shutil.copy(td / f, aug / f"train_{f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(m, cal, train, a.n_boot))
    (aug / "calibration.json").write_text(json.dumps(cal, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
