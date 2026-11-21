"""Final LUMEN2 pilot report -> results/lumen2/lumen2_pilot.md (+ results/lumen2/final_line.txt, the one-line ntfy summary).

    python -m dcfg.lumen2.pilot_report

Built from results/lumen2/lumen2_eval.json (paired metrics, written by ``dcfg.lumen2.report``), the per-condition result JSONs of GenD /
GenD-aug / LUMEN2 (point estimates of the main table), results/dfd (DFD), results/raw (Xception context) and the cost/sensor files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dcfg.lumen2 import analysis as A
from dcfg.lumen2.dfd_append import dfd_section
from dcfg.lumen2.report import f4, load_json
from dcfg.paths import RESULTS_ROOT

R = RESULTS_ROOT / "lumen2"
MODELS = [("LUMEN2", "lumen2", R / "lumen2"), ("GenD-aug", "gend_aug", R / "gend_aug"), ("GenD (public)", "gend_public", R / "gend_public")]
XCEPTION = [("base", "base_xception"), ("augft (full fine-tune)", "augft_xception"), ("LUMEN1", "lumen_xception"), ("fixed gate", "fixed_xception"),
            ("LUMEN1 without gate", "lumen_nogate_xception")]


def point_aucs(run: str, d: Path, conds: list[str]) -> dict[str, float] | None:
    out = {}
    for c in conds:
        j = load_json(d / f"{run}__{c}.json")
        if j is None:
            return None
        out[c] = j["frame_auc"]
    return out


def summary_row(auc: dict[str, float]) -> list[float]:
    f = lambda cs: float(np.mean([auc[c] for c in cs]))  # noqa: E731
    return [auc["clean"], f(A.SYNTH), f(A.HELD), auc["c40"], f(A.HEAVY), min(auc[c] for c in A.DEGRADED)]


def dfd_points(run: str, d: Path = RESULTS_ROOT / "dfd") -> list[str]:
    out = []
    for c in ("clean", "mixed_heavy"):
        j = load_json(d / f"{run}__{c}.json")
        out.append("n/a" if j is None else f"{j['frame_auc']:.4f}")
    return out


def md(hdr: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows]


def build(ev: dict, cost: dict | None) -> tuple[str, str]:
    m, ver, dfd = ev["metrics"], ev["verdict"], ev.get("dfd")
    L = ["# LUMEN2 pilot report", "",
         "LUMEN2 = GenD-aug (GenD LayerNorm tuning on our degraded stream, the fair baseline) + the LUMEN parts (jointly trained damage sensor, DCT gate, "
         "degradation-conditioned LayerNorm). Everything else (data, degradation stream, schedule, seed) is identical. Fixed pre-declared metrics: "
         "docs/LUMEN2_BRIEF.md. Full detail: `lumen2_eval.md`, `gend_aug.md`.", "", "## Main table (frame AUC, point estimates)", ""]
    rows = []
    for lab, rid, d in MODELS:
        auc = point_aucs(rid, d, A.ALL_CONDS)
        rows.append([lab] + (["n/a"] * 6 if auc is None else [f"{v:.4f}" for v in summary_row(auc)]) + dfd_points(rid))
    L += md(["model", "clean", "synthetic mean (17)", "held-out (4)", "c40", "heavy-damage (5)", "worst-case (22)", "DFD clean", "DFD mixed_heavy"], rows)
    L += ["", "## Pre-declared metrics: LUMEN2 minus GenD-aug (paired, 1000 video-level bootstrap resamples, 95% CI)", "",
          "Outcome: win = CI entirely favourable; loss = entirely unfavourable; tie otherwise. Safety: ok unless CI entirely below 0.", ""]
    rows, counts = [], {"win": 0, "loss": 0, "tie": 0}
    for k, (label, role, _) in A.METRICS.items():
        o = ver["outcomes"][k]
        src = dfd if k.startswith("dfd") else m
        if src is None or k not in src:
            rows.append([label, role, "n/a", o])
            continue
        rows.append([label, role, f4(src[k]["diff"], True), o])
        if o in counts and role != "safety":
            counts[o] += 1
    L += md(["metric", "role", "LUMEN2 - GenD-aug [95% CI]", "outcome"], rows)
    L += ["", f"Secondary + primary tally (selective counted per coverage): {counts['win']} win, {counts['loss']} loss, {counts['tie']} tie.", "",
          "## Exploratory (not claims)", "", "AP, EER and every single condition are in `lumen2_eval.md` (point estimates / paired per-condition CIs).", "",
          "## Context: Xception results (DIFFERENT BACKBONE, not comparable as a head-to-head)", "",
          "Frame AUC point estimates from results/raw (FF++ c23 test + c40, DFB crops) and results/dfd.", ""]
    rows = []
    for lab, rid in XCEPTION:
        auc = point_aucs(rid, RESULTS_ROOT / "raw", A.ALL_CONDS)
        rows.append([f"{lab} (different backbone: Xception)"] + (["n/a"] * 6 if auc is None else [f"{v:.4f}" for v in summary_row(auc)]) + dfd_points(rid))
    L += md(["model", "clean", "synthetic mean", "held-out", "c40", "heavy-damage", "worst-case", "DFD clean", "DFD mixed_heavy"], rows)
    L += ["", dfd_section([(rid, lab) for lab, rid, _ in MODELS] + [(rid, f"{lab} (Xception)") for lab, rid in XCEPTION]).replace("## DFD", "### DFD", 1)]
    L += ["## Damage sensor", "", "Accuracy per op, per condition and on c40 / WebP / motion blur: `lumen2_eval.md` (section Damage sensor inside LUMEN2), "
          "next to the standalone sensor (`results/sensor/standalone.md`).", "", "## Cost", ""]
    if cost:
        L += [f"- LUMEN parameters {cost['lumen_params_total']:,}; trainable file {cost['trainable_file_mb']['lumen2']:.2f} MB (GenD-aug {cost['trainable_file_mb']['gend_aug']:.2f} MB)",
              f"- inference {cost['lumen2']['median_ms_per_image']:.2f} ms/image vs {cost['gend_aug']['median_ms_per_image']:.2f} (x{cost['overhead_ratio']:.2f})"]
        for k, t in cost.get("training", {}).items():
            L.append(f"- training {k}: best epoch {t['best_epoch']}, stop epoch {t['stop_epoch']} ({t['stop_reason']}), {t['train_wall_time_s'] / 3600:.1f} h")
    else:
        L.append("(cost benchmark not available)")
    pend = ver["pending"]
    rule = ver["rule_passes_before_uncond"]
    L += ["", "## Verdict", "",
          f"- Primary (synthetic-degraded mean): {ver['outcomes']['synth_mean']}; wins: {', '.join(ver['wins']) or 'none'}",
          f"- Safety: clean c23 {ver['safety']['clean']}, clean DFD {ver['safety']['dfd_clean']}"]
    if rule is None:
        L.append(f"- 'Worth publishing' rule undecided (pending {', '.join(pend)}).")
    else:
        L.append(f"- 'Worth publishing' rule (wins primary or >= 1 secondary AND passes both safety checks): **{'PASSES' if rule else 'FAILS'}**.")
    L.append("- The LUMEN-uncond control (same architecture, constant conditioning input) is NOT run: the rule additionally requires LUMEN2 to beat it on "
             "the won metric. " + ("Running it is justified only if the rule passes." if rule else "Not justified unless the rule can pass.") if rule is not None else
             "- LUMEN-uncond pending the rule outcome.")
    ds = lambda k: m[k]["diff"] if k in m else (0, 0, 0)  # noqa: E731
    s, c4 = ds("synth_mean"), ds("c40")
    d = dfd["dfd_clean"] if dfd else None
    d_aug = f"{d['base'][0]:.3f}" if d else "n/a"
    d_new = f"{d['new'][0]:.3f}" if d else "n/a"
    v = "rule PASSES (LUMEN-uncond pending)" if rule else ("rule FAILS" if rule is False else "rule undecided")
    line = (f"LUMEN2-GenD-aug: synth mean {s[0]:+.4f} [{s[1]:+.4f},{s[2]:+.4f}], c40 {c4[0]:+.4f} [{c4[1]:+.4f},{c4[2]:+.4f}]; "
            f"verdict {v}; DFD AUC GenD-aug {d_aug}, LUMEN2 {d_new}")
    return "\n".join(L) + "\n", line


def main() -> int:
    ev = load_json(R / "lumen2_eval.json")
    if ev is None:
        print("results/lumen2/lumen2_eval.json missing (run dcfg.lumen2.report first)")
        return 1
    text, line = build(ev, load_json(R / "lumen2_cost.json"))
    (R / "lumen2_pilot.md").write_text(text)
    (R / "final_line.txt").write_text(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
