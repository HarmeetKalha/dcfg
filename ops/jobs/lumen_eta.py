"""GPU-time ETA of the LUMEN matrix (ops/jobs/gen_matrix.py): per run and per group, worst (30 epochs) and likely
(15 epochs, an ASSUMPTION) case, in GPU-days.

    .venv/bin/python ops/jobs/lumen_eta.py [--smoke results/qa/lumen_smoke.json] [--out results/qa/lumen_eta.md]

Per-epoch time (train + val, minutes) = the MEASURED Xception pilot epoch time of the variant class
(results/pilot/design_selection/report.md section 6 and the earlier pilots), scaled to another detector by
``img/s(Xception) / img/s(detector)`` of the same class:
  * lumen / lumen_nolive / lumen_frozensensor: lumen smoke img/s per detector (results/qa/lumen_smoke.json).
    UCF has no lumen smoke (FiLM sites undecided): ASSUMPTION = phase-8 dcfg ratio ucf/xception.
  * lumen_nocons: pilot lumen epoch x (lumen / lumen_nocons Xception smoke img/s).
  * fixed / augft: phase-8 img/s per detector (results/qa/phase8_smoke.json); jointft, dcfg, dcfgcons: Xception only.
  * bnft: pilot augft epoch x (phase-8 augft Xception img/s / bnft smoke img/s).
Test-eval time per run = mean per-condition seconds of runs/base_<det>/log.txt x 23 conditions (no module overhead).
Total per run = epochs x epoch + eval. Seeds / ablations reuse the Xception numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))
import gen_matrix as gm  # noqa: E402
from phase8_smoke import eval_seconds, parse_eval_log  # noqa: E402

REPO = gm.REPO
N_CONDS = 23
EPOCHS_WORST, EPOCHS_LIKELY = 30, 15
#: measured Xception pilot minutes per epoch incl. val (report.md section 6; lumen = pilot dcfgfilmlite)
PILOT_EPOCH_MIN = {"lumen": 22.2, "fixed": 14.0, "augft": 17.8, "jointft": 22.9, "dcfg": 18.8, "dcfgcons": 22.1}
#: variant -> (pilot class, rate class used for the cross-detector / ablation scaling)
CLASS = {"lumen": "lumen", "lumen_nolive": "lumen", "lumen_frozensensor": "lumen", "lumen_nocons": "lumen_nocons",
         "fixed": "fixed", "augft": "augft", "dcfg_jointft": "jointft", "dcfg": "dcfg", "dcfg_cons": "dcfgcons",
         "bnft": "bnft"}
GROUP_TITLE = {1: "core: lumen/fixed/augft x 4 detectors", 2: "jointft_xception, bnft_xception",
               3: "Xception ablation chain", 4: "seeds (lumen, augft Xception s43/s44)"}


def epoch_minutes(run: dict[str, Any], r: dict[str, dict[str, float]]) -> tuple[float, str]:
    """``(minutes per epoch, how it was derived)``; ``r`` = rates {class: {detector: img/s}}."""
    det, cls = run["detector"], CLASS[run["variant"]]
    if cls == "lumen_nocons":
        return PILOT_EPOCH_MIN["lumen"] * r["lumen"]["xception"] / r["lumen_nocons"]["xception"], "pilot lumen x smoke ratio nocons"
    if cls == "bnft":
        return PILOT_EPOCH_MIN["augft"] * r["augft"]["xception"] / r["bnft"]["xception"], "pilot augft x smoke ratio bnft"
    base = PILOT_EPOCH_MIN[cls]
    if det == "xception":
        return base, "pilot (measured)"
    if cls == "lumen" and det not in r["lumen"]:  # no lumen smoke (UCF): phase-8 dcfg ratio
        return base * r["dcfg"]["xception"] / r["dcfg"][det], "ASSUMPTION: pilot lumen x phase-8 dcfg ratio (no lumen smoke)"
    return base * r[cls]["xception"] / r[cls][det], "pilot x img/s ratio"


def build(rates: dict[str, dict[str, float]], eval_min: dict[str, float]) -> dict[str, Any]:
    """Per-run rows and per-group / cumulative totals (GPU-days)."""
    rows = []
    for run in gm.matrix_runs():
        em, how = epoch_minutes(run, rates)
        ev = eval_min[run["detector"]]
        rows.append({"run_id": run["run_id"], "group": run["group"], "detector": run["detector"], "variant": run["variant"],
                     "epoch_min": em, "how": how, "eval_min": ev,
                     "worst_min": EPOCHS_WORST * em + ev, "likely_min": EPOCHS_LIKELY * em + ev})
    groups, cw, cl = {}, 0.0, 0.0
    for g in gm.GROUPS:
        w = sum(r["worst_min"] for r in rows if r["group"] == g) / 1440
        l = sum(r["likely_min"] for r in rows if r["group"] == g) / 1440
        cw, cl = cw + w, cl + l
        groups[g] = {"n": sum(r["group"] == g for r in rows), "worst": w, "likely": l, "cum_worst": cw, "cum_likely": cl}
    return {"rows": rows, "groups": groups, "worst": cw, "likely": cl}


def markdown(res: dict[str, Any], notes: list[str]) -> str:
    L = ["# LUMEN matrix ETA", "",
         f"Worst case = {EPOCHS_WORST} full epochs (no early stop). Likely case = {EPOCHS_LIKELY} epochs (ASSUMPTION, not measured). "
         "Per run: epochs x (train + val per epoch) + test evaluation over 23 conditions. Units: GPU-days (24 h).", "",
         "## Per group (you can stop after any group)", "",
         "| group | contents | runs | worst | likely | cumulative worst | cumulative likely |", "|---|---|---|---|---|---|---|"]
    for g, v in res["groups"].items():
        L.append(f"| G{g} | {GROUP_TITLE[g]} | {v['n']} | {v['worst']:.2f} | {v['likely']:.2f} | {v['cum_worst']:.2f} | {v['cum_likely']:.2f} |")
    L += ["", f"**Total ({len(res['rows'])} runs): worst {res['worst']:.2f} GPU-days; likely (15 ep, assumption) {res['likely']:.2f} GPU-days.**", "",
          "## Per run", "", "| run | group | epoch (min) | eval (min) | worst (GPU-days) | likely (GPU-days) | epoch time from |",
          "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['run_id']} | G{r['group']} | {r['epoch_min']:.1f} | {r['eval_min']:.1f} | {r['worst_min'] / 1440:.2f} | "
                 f"{r['likely_min'] / 1440:.2f} | {r['how']} |")
    return "\n".join(L + ["", "## Notes", ""] + [f"- {n}" for n in notes]) + "\n"


def load_rates(smoke_json: Path, phase8_json: Path) -> dict[str, dict[str, float]]:
    sm = {s["name"]: s["img_s_steady"] for s in json.loads(smoke_json.read_text())["smokes"]}
    p8 = json.loads(phase8_json.read_text())["d_timing"]
    r: dict[str, dict[str, float]] = {"lumen": {}, "lumen_nocons": {}, "bnft": {}, "fixed": {}, "augft": {}, "dcfg": {}}
    for k, v in sm.items():
        if k.startswith("lumen_nocons_"):
            r["lumen_nocons"][k[len("lumen_nocons_"):]] = v
        elif k.startswith("lumen_"):
            r["lumen"][k[len("lumen_"):]] = v
        elif k.startswith("bnft_"):
            r["bnft"][k[len("bnft_"):]] = v
    for v in p8.values():
        if v["variant"] in ("fixed", "augft", "dcfg"):
            r[v["variant"]][v["detector"]] = v["img_s"]
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", default=str(REPO / "results/qa/lumen_smoke.json"))
    ap.add_argument("--phase8", default=str(REPO / "results/qa/phase8_smoke.json"))
    ap.add_argument("--out", default=str(REPO / "results/qa/lumen_eta.md"))
    a = ap.parse_args(argv)
    rates = load_rates(Path(a.smoke), Path(a.phase8))
    eval_min, srcs = {}, []
    for det in gm.DETECTORS:
        lg = REPO / "runs" / f"base_{det}" / "log.txt"
        sec, src = eval_seconds(parse_eval_log(lg.read_text()) if lg.exists() else {}, N_CONDS, None)
        if sec is None:
            raise SystemExit(f"no test-eval timing for {det} in {lg}")
        eval_min[det] = sec / 60
        srcs.append(f"{det} {sec / 60:.1f} min ({src})")
    res = build(rates, eval_min)
    notes = ["Epoch time includes val; scaling by steady img/s ratios applies to the whole epoch (val share is similar across detectors).",
             "Test eval time = base_<det> logs; module variants add a small per-image cost not included. " + "; ".join(srcs),
             "lumen_frozensensor and lumen_nolive are timed as lumen (frozen sensor is slightly cheaper: conservative).",
             "lumen_ucf: no smoke (UCF FiLM sites undecided), epoch time estimated from the phase-8 dcfg ucf/xception ratio. "
             "augft_ucf uses UCF's native loss (phase-8 img/s).",
             "img/s: " + json.dumps({k: {d: round(x, 1) for d, x in v.items()} for k, v in rates.items()})]
    Path(a.out).write_text(markdown(res, notes))
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(f"worst {res['worst']:.2f} / likely {res['likely']:.2f} GPU-days -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
