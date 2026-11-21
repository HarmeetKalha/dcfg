"""Gate ablation table for Xception -> results/gate/ablation_xception.md (+ .json)

    python -m dcfg.analysis.gate_ablation [--n-boot 1000]

Models (all Xception, FF++ c23 test, results/raw): base, fixed, LUMEN1-without-gate (lumen_nogate), LUMEN1 (lumen), augft.
Aggregates: clean; synthetic mean (17 synthetic conditions); held-out mean (webp_q50/q20, motion_k7/k11); c40; heavy-damage mean =
mean over exactly {jpeg_q10, blur_s4, noise_s20, resize_0.25, mixed_heavy} (DECISIONS.md 2026-11-21 amendment); worst case = min
over the 22 degraded conditions (incl. c40). Frame AUC and video AUC, 95% CIs from ONE shared video-level bootstrap
(1,000 resamples) applied to every model, so aggregates are jointly resampled and differences are paired:
gate contribution = LUMEN1 - LUMEN1-without-gate; module-without-gate contribution = LUMEN1-without-gate - base.
Per-condition appendix: single-condition AUCs verbatim from the JSONs with their CIs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from dcfg.gend.room_report import gf
from dcfg.paths import REPO_ROOT

RAW = REPO_ROOT / "results" / "raw"
OUT = REPO_ROOT / "results" / "gate"
HEAVY = ["jpeg_q10", "blur_s4", "noise_s20", "resize_0.25", "mixed_heavy"]
MODELS = [("Xception base", "base_xception"), ("Fixed gate", "fixed_xception"),
          ("LUMEN1 without gate", "lumen_nogate_xception"), ("LUMEN1", "lumen_xception"), ("augft (full fine-tune)", "augft_xception")]
ITEMS = ["clean", "synth_mean", "heldout_mean", "c40", "heavy_mean", "worst_case"]
ITEM_LABEL = {"clean": "clean", "synth_mean": "synthetic mean (17)", "heldout_mean": "held-out mean (4)", "c40": "c40",
              "heavy_mean": "heavy-damage mean (5)", "worst_case": "worst case (min of 22 degraded)"}


class Ablation:
    """Shared-bootstrap curves of the compared runs."""

    def __init__(self, n_boot: int, seed: int = 0) -> None:
        cl = gf.load_conditions()
        self.names = [c["name"] for c in cl]
        self.synth = [c["name"] for c in cl if c["name"] != "clean" and not c.get("held_out") and c.get("synthetic", True)]
        self.held = [c["name"] for c in cl if c.get("held_out")]
        self.degraded = [n for n in self.names if n != "clean"]
        self.runs = {rid: gf.Run(lab, rid, RAW, self.names) for lab, rid in MODELS}
        vids = sorted({v for r in self.runs.values() for s in r.scores.values() for v in set(s["video"])})
        rng = np.random.default_rng(seed)
        nv = len(vids)
        self.W = np.vstack([np.ones((1, nv)), rng.multinomial(nv, np.full(nv, 1.0 / nv), size=n_boot).astype(float)])
        self.cur = {rid: gf.curves(r, np.array(vids), self.W) for rid, r in self.runs.items()}

    def arr(self, rid: str, item: str, metric: str) -> np.ndarray | None:
        def get(c: str) -> np.ndarray | None:
            return self.cur[rid].get((metric, c))
        groups = {"synth_mean": self.synth, "heldout_mean": self.held, "heavy_mean": HEAVY}
        if item in groups:
            xs = [get(c) for c in groups[item]]
            return None if any(x is None for x in xs) else np.mean(xs, axis=0)
        if item == "worst_case":
            xs = [get(c) for c in self.degraded]
            return None if any(x is None for x in xs) else np.min(xs, axis=0)
        return get(item)

    def val(self, rid: str, item: str, metric: str) -> tuple[float, float, float] | None:
        a = self.arr(rid, item, metric)
        if a is None:
            return None
        return gf.pct(a)

    def diff(self, a: str, b: str, item: str, metric: str) -> tuple[float, float, float] | None:
        x, y = self.arr(a, item, metric), self.arr(b, item, metric)
        return None if x is None or y is None else gf.pct(x - y)


def f4(v: tuple[float, float, float] | None, signed: bool = False) -> str:
    if v is None:
        return "n/a (run missing)"
    s = "+" if signed else ""
    return f"{v[0]:{s}.4f} [{v[1]:{s}.4f}, {v[2]:{s}.4f}]"


def build(m: Ablation) -> tuple[str, dict]:
    L = ["# Gate ablation (Xception, FF++ c23 test)", "", __doc__.split("\n\n", 2)[2].strip(), ""]
    js: dict = {"values": {}, "diffs": {}}
    for metric in ("frame", "video"):
        L += [f"## {metric.capitalize()} AUC (95% CI)", "", "| model | " + " | ".join(ITEM_LABEL[i] for i in ITEMS) + " |",
              "|---|" + "---|" * len(ITEMS)]
        for lab, rid in MODELS:
            vs = [m.val(rid, i, metric) for i in ITEMS]
            js["values"][f"{metric}/{rid}"] = {i: v for i, v in zip(ITEMS, vs)}
            L.append(f"| {lab} | " + " | ".join(f4(v) for v in vs) + " |")
        L.append("")
    pairs = [("Gate contribution: LUMEN1 minus LUMEN1-without-gate", "lumen_xception", "lumen_nogate_xception"),
             ("LUMEN1-without-gate minus base", "lumen_nogate_xception", "base_xception"),
             ("(context) LUMEN1 minus base", "lumen_xception", "base_xception"),
             ("(context) LUMEN1 minus augft", "lumen_xception", "augft_xception")]
    for metric in ("frame", "video"):
        L += [f"## Paired differences, {metric} AUC (95% CI, paired video bootstrap)", "",
              "| difference | " + " | ".join(ITEM_LABEL[i] for i in ITEMS) + " |", "|---|" + "---|" * len(ITEMS)]
        for title, a, b in pairs:
            ds = [m.diff(a, b, i, metric) for i in ITEMS]
            js["diffs"][f"{metric}/{a}-{b}"] = {i: d for i, d in zip(ITEMS, ds)}
            L.append(f"| {title} | " + " | ".join(f4(d, True) for d in ds) + " |")
        L.append("")
    L += ["## Appendix: per-condition AUC (frame / video, point [95% CI] from the result JSONs)", "",
          "| condition | " + " | ".join(lab for lab, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for c in m.names:
        cells = []
        for _, rid in MODELS:
            d = m.runs[rid].docs.get(c)
            cells.append("n/a" if d is None else f"{d['frame_auc']:.3f} [{d['frame_auc_ci'][0]:.3f}, {d['frame_auc_ci'][1]:.3f}] / "
                         f"{d['video_auc']:.3f} [{d['video_auc_ci'][0]:.3f}, {d['video_auc_ci'][1]:.3f}]")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n", js


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.analysis.gate_ablation")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    m = Ablation(a.n_boot)
    md, js = build(m)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_xception.md").write_text(md)
    (out / "ablation_xception.json").write_text(json.dumps(js, indent=2, default=float) + "\n")
    print("wrote", out / "ablation_xception.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
