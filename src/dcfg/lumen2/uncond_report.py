"""LUMEN-uncond control report -> results/lumen2/uncond.md (+ uncond.json, + uncond_line.txt = the one-line phone summary).

    python -m dcfg.lumen2.uncond_report [--n-boot 1000]

Every pre-declared metric (docs/LUMEN2_BRIEF.md) as paired LUMEN2 - LUMEN-uncond with 95% video-level bootstrap CIs from ONE shared
resample matrix over the three models (GenD-aug, LUMEN-uncond, LUMEN2), the same machinery as ``dcfg.lumen2.report``. The control
question (docs/DECISIONS.md 2026-11-23): does LUMEN2 beat uncond (CI entirely favourable) on the metrics it won against GenD-aug
(synthetic mean, heavy-damage mean, video AUC synthetic mean)?
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from dcfg.lumen2 import analysis as A
from dcfg.lumen2.report import DFD_DIR, R, Study, check_same_frames, dfd_metrics, f4, load_json, paired_metrics
from dcfg.paths import RUNS_ROOT

AUG, UNC, NEW = "GenD-aug", "LUMEN-uncond", "LUMEN2"
WON = ["synth_mean", "heavy_mean", "video_synth_mean"]  # LUMEN2's wins vs GenD-aug (results/lumen2/lumen2_eval.md)


def outcomes(res: dict[str, Any], dfd: dict[str, Any] | None) -> dict[str, str]:
    out = {}
    for k in A.METRICS:
        src = dfd if k.startswith("dfd") else res
        out[k] = A.outcome(k, *src[k]["diff"][1:]) if src and k in src else "pending"
    return out


def val_curves() -> list[list[str]]:
    rd = lambda p: {int(r["epoch"]): float(r["val_mean_auc"]) for r in csv.DictReader(p.open())} if p.exists() else {}  # noqa: E731
    u, l2, g = rd(RUNS_ROOT / "lumen2_uncond_train/metrics.csv"), rd(RUNS_ROOT / "lumen2_train/metrics.csv"), rd(RUNS_ROOT / "gendaug_train/metrics.csv")
    s = lambda d, e: f"{d[e]:.4f}" if e in d else "-"  # noqa: E731
    return [[str(e), s(u, e), s(l2, e), s(g, e)] for e in sorted(set(u) | set(l2) | set(g))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dcfg.lumen2.uncond_report", description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(R / "uncond.md"))
    a = ap.parse_args(argv)
    dirs = {AUG: ("gend_aug", R / "gend_aug"), UNC: ("lumen2_uncond", R / "lumen2_uncond"), NEW: ("lumen2", R / "lumen2")}
    st = Study(dirs, A.ALL_CONDS, a.n_boot)
    check_same_frames(None, None, A.ALL_CONDS, dirs[UNC][1], dirs[NEW][1], "lumen2_uncond", "lumen2")
    res = paired_metrics(st, UNC, NEW, a.n_boot)  # LUMEN2 - uncond
    res_u = paired_metrics(st, AUG, UNC, a.n_boot)  # uncond - GenD-aug (context)
    runs = {AUG: "gend_aug", UNC: "lumen2_uncond", NEW: "lumen2"}
    dfd = dfd_metrics(UNC, NEW, runs, a.n_boot)
    dfd_u = dfd_metrics(AUG, UNC, runs, a.n_boot)
    out, out_u = outcomes(res, dfd), outcomes(res_u, dfd_u)
    beats = {k: out[k] == "win" for k in WON}
    train = load_json(RUNS_ROOT / "lumen2_uncond_train/done.json") or {}

    L: list[str] = []
    P = L.append
    P("# LUMEN-uncond control: LUMEN2 minus LUMEN-uncond (pre-declared metrics)\n")
    P("LUMEN-uncond = LUMEN2 with the per-image conditioning input of the DCT gate and the LayerNorm generator replaced by ONE learned "
      "constant vector (definition: docs/DECISIONS.md 2026-11-23); the sensor still trains with its damage loss. Same code, data, "
      "degradation stream, seed and schedule. All numbers frame AUC unless stated, FF++ c23 test + c40, GenD crops, identical frames/"
      f"degradations for all three models. Paired differences, 95% video-level bootstrap CI ({a.n_boot} resamples, one shared resample "
      "matrix for the three models). Win = CI entirely on the favourable side of 0 (ECE: lower is better).\n")
    if train:
        P(f"Training: stop reason **{train.get('stop_reason')}**, best epoch {train.get('best_epoch')} of {train.get('epochs_run')} "
          f"(val mean AUC {train.get('best_val_mean_auc', float('nan')):.4f}), parameters {train.get('part_params', {}).get('LUMEN parts total')} LUMEN parts.\n")
    P("## The control question\n")
    P("Does LUMEN2 beat LUMEN-uncond on the metrics it won against GenD-aug?\n")
    for k in WON:
        P(f"- {A.METRICS[k][0]}: LUMEN2 - uncond {f4(res[k]['diff'], True)} -> **{'yes' if beats[k] else 'no'}** ({out[k]})")
    allb, anyb = all(beats.values()), any(beats.values())
    concl = ("LUMEN2 beats uncond on ALL its won metrics: the gain needs per-image conditioning." if allb else
             f"LUMEN2 beats uncond on {sum(beats.values())} of {len(WON)} won metrics ({', '.join(k for k in WON if beats[k])})." if anyb else
             "LUMEN2 does NOT beat uncond on any won metric: the gain vs GenD-aug is not shown to come from per-image conditioning.")
    P(f"\n**Verdict: {concl}**\n")
    P("## Every pre-declared metric\n")
    rows = []
    for k, (label, role, _) in A.METRICS.items():
        src, src_u = (dfd, dfd_u) if k.startswith("dfd") else (res, res_u)
        if not src or k not in src:
            rows.append([label, role, "n/a", "n/a", "n/a", "n/a", "pending", "n/a"])
            continue
        r = src[k]
        rows.append([label, role, f4(src_u[k]["base"]), f4(r["base"]), f4(r["new"]), f4(r["diff"], True), out[k], f"{f4(src_u[k]['diff'], True)} ({out_u[k]})"])
    L += st.gf.md_table(["metric", "role", AUG, UNC, NEW, f"{NEW} - {UNC}", "outcome", f"{UNC} - {AUG} (context)"], rows)
    P("\nOutcome for safety rows: ok unless the CI is entirely below 0. Selective AUC: win if either coverage wins.\n")
    P("## Validation curve (val mean AUC per epoch, 8 of 32 frames/video)\n")
    L += st.gf.md_table(["epoch", UNC, NEW, AUG], val_curves())
    out_p = Path(a.out)
    out_p.write_text("\n".join(L) + "\n")
    out_p.with_suffix(".json").write_text(json.dumps({"lumen2_minus_uncond": res, "uncond_minus_gend_aug": res_u, "dfd": dfd, "dfd_uncond_vs_aug": dfd_u,
                                                      "outcomes": out, "beats_on_won": beats, "n_boot": a.n_boot}, indent=2, default=float))
    d = lambda k: res[k]["diff"][0]  # noqa: E731
    ci = lambda k: f"{d(k):+.4f} [{res[k]['diff'][1]:+.4f},{res[k]['diff'][2]:+.4f}]"  # noqa: E731
    line = (f"LUMEN2-uncond: synth {ci('synth_mean')}, heavy {ci('heavy_mean')}, video {ci('video_synth_mean')}; beats uncond on "
            f"{sum(beats.values())}/3 won metrics" + (f"; DFD clean {dfd['dfd_clean']['diff'][0]:+.4f}" if dfd else ""))
    (R / "uncond_line.txt").write_text(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
