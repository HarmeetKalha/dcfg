"""Design selection study (5 epochs, VALIDATION split only) -> results/pilot/design_selection/{report.md,results.json,results.csv}.

    .venv/bin/python ops/jobs/design_selection_report.py [--out-dir results/pilot/design_selection] [--no-notify]
        [--no-overhead-run]

PILOT / appendix material, NOT paper main numbers (never in results/raw|tables|figures). Rows: base, fixed, augft,
dcfg (existing pilot evals), jointft, dcfgcons, dcfgfilm, dcfgfilmlite. Reads ``results/pilot/eval/scores/`` (per-frame
scores of ``dcfg.eval.evaluate --split val``), ``runs/pilot/<run>/{resources_summary,done}.json`` and
``<out-dir>/overhead.json`` (measured here via ``python -m dcfg.analysis.overhead --design-selection`` if missing).
Bootstraps over VIDEOS (one resample shared by all models and conditions, as pilot_report.py). The report does NOT
select a design. Ends with an ntfy summary (honours DCFG_NO_NOTIFY). Missing runs are marked as missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

JOBS = Path(__file__).resolve().parent
REPO = JOBS.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(JOBS))
import pilot_report as pr  # noqa: E402

MODELS = ["base", "fixed", "augft", "dcfg", "jointft", "dcfgcons", "dcfgfilm", "dcfgfilmlite"]
NEW = ["jointft", "dcfgcons", "dcfgfilm", "dcfgfilmlite"]
BASELINES = ["dcfg", "augft", "jointft"]
FOCUS = ["blur_s3", "webp_q20", "c40"]
#: row -> which measured inference pipeline it costs (augft/jointft/dcfgcons have the same inference graph as these)
OVERHEAD_OF = {"base": "base", "fixed": "fixed", "augft": "base", "dcfg": "dcfg", "jointft": "dcfg", "dcfgcons": "dcfg",
               "dcfgfilm": "film", "dcfgfilmlite": "filmlite"}
#: queue job ids of the design-selection study (wall-time window)
DS_JOBS = {f"pilot_{m}_xception" for m in NEW} | {f"pilot_eval_{m}" for m in NEW} | {"pilot_ds_report"}
REPORT_JOB = "pilot_ds_report"
FORBIDDEN = ("raw", "tables", "figures")


def _fmt(x: float | None, f: str = "{:.4f}") -> str:
    return "n/a" if x is None else f.format(x)


def star(d: tuple[float, float, float] | None) -> str:
    if d is None:
        return "n/a"
    return f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" + (" *" if d[1] > 0 or d[2] < 0 else "")


def analyse(eval_dir: Path, n_boot: int, seed: int = 0) -> dict[str, Any]:
    """Joint bootstrap over all available models/conditions. Returns res (pilot_report format) + bookkeeping."""
    conds_seen = sorted({p.stem.split("__", 1)[1] for p in (eval_dir / "scores").glob("pilot_*__*.npz")})
    conds = [c for c in ["clean", *pr.DEGRADED, "c40"] if c in conds_seen]
    deg = [c for c in conds if c not in ("clean", "c40")]
    probs, meta = pr.load_scores(eval_dir, MODELS, conds)
    main_c = [c for c in conds if c != "c40"]
    labels, videos = meta[main_c[0]][1], meta[main_c[0]][2]
    if not all(np.array_equal(meta[main_c[0]][0], meta[c][0]) for c in main_c):
        raise ValueError("synthetic conditions must share the same frames (c23 val)")
    res = pr.joint_bootstrap({k: v for k, v in probs.items() if k[1] != "c40"}, labels, videos, deg, n_boot, seed)
    for m in list(res["deg_point"]):  # degraded mean only for models that have EVERY degraded condition
        if any((m, c) not in probs for c in deg):
            res["deg_point"].pop(m)
            res["deg_boot"].pop(m)
    if "c40" in conds:
        r40 = pr.joint_bootstrap({k: v for k, v in probs.items() if k[1] == "c40"}, meta["c40"][1], meta["c40"][2],
                                 [], n_boot, seed)
        res["point"] |= r40["point"]
        res["boot"] |= r40["boot"]
        res["n_videos_c40"] = r40["n_videos"]
    present = [m for m in MODELS if any((m, c) in probs for c in conds)]
    return {"res": res, "conds": conds, "deg": deg, "present": present, "missing": [m for m in MODELS if m not in present],
            "n_frames": len(labels)}


def accuracy_table(res: dict, models: list[str], conds: list[str], deg: list[str]) -> list[str]:
    cols = ["clean", "d clean vs base", *[c for c in conds if c not in ("clean", "c40")], "mean(degraded)", "c40"]
    rows = ["| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for m in MODELS:
        if m not in models:
            rows.append(f"| {m} | " + " | ".join(["MISSING (no eval scores)"] + [""] * (len(cols) - 1)) + " |")
            continue
        cells = []
        for c in cols:
            if c == "d clean vs base":
                d = pr.paired_delta(res, m, "base", "clean") if m != "base" else None
                cells.append("-" if m == "base" else star(d))
            elif c == "mean(degraded)":
                cells.append(pr.fmt_ci(res["deg_point"][m], *pr.pct(res["deg_boot"][m])) if m in res["deg_point"] else "n/a")
            else:
                k = (m, c)
                cells.append(pr.fmt_ci(res["point"][k], *pr.pct(res["boot"][k])) if k in res["point"] else "n/a")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return rows


def delta_rows(res: dict, models: list[str], cols: list[str | None]) -> list[str]:
    head = [("mean(degraded)" if c is None else c) for c in cols]
    rows = ["| new variant vs | " + " | ".join(head) + " |", "|" + "---|" * (len(cols) + 1)]
    for m in NEW:
        for b in BASELINES:
            if m == b:
                continue
            if m not in models or b not in models:
                rows.append(f"| {m} - {b} | " + " | ".join(["MISSING"] * len(cols)) + " |")
                continue
            rows.append(f"| {m} - {b} | " + " | ".join(star(pr.paired_delta(res, m, b, c)) for c in cols) + " |")
    return rows


def cost_table(params: dict[str, Any] | None, overhead: dict[str, Any] | None) -> list[str]:
    head = ["model", "trainable params added", "% of Xception", "ms/img b32 (total / added)", "ms/img b1 (total / added)",
            "inference graph measured as"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in MODELS:
        p = params.get(m) if params else None
        ptxt = ("n/a", "n/a")
        if p:
            note = " (whole detector counted as trainable)" if p["detector_trained"] else ""
            ptxt = (f"{p['trainable']:,}{note}", f"{p['pct_of_xception']:.2f}%")
        pipe = (overhead or {}).get("pipelines", {}).get(OVERHEAD_OF[m])
        o = ["n/a", "n/a"]
        if pipe:
            o = [f"{pipe[f'batch{b}']['ms_per_image']:.2f} / +{pipe[f'batch{b}']['added_ms_per_image']:.2f}" for b in (32, 1)]
        rows.append(f"| {m} | {ptxt[0]} | {ptxt[1]} | {o[0]} | {o[1]} | {OVERHEAD_OF[m]} |")
    return rows


def epoch_times(runs_dir: Path) -> dict[str, dict[str, float] | None]:
    """Measured per-epoch minutes (train / val / total) from resources_summary.json + done.json; None if missing."""
    out: dict[str, dict[str, float] | None] = {}
    for m in MODELS[1:]:
        rs, dn = runs_dir / pr.run_id(m) / "resources_summary.json", runs_dir / pr.run_id(m) / "done.json"
        if not (rs.exists() and dn.exists()):
            out[m] = None
            continue
        s, d = json.loads(rs.read_text()), json.loads(dn.read_text())
        n = max(1, int(d.get("epochs_run", 5)))
        out[m] = {"epochs": n, "train_min": s["wall_train_s"] / n / 60, "val_min": s["wall_val_s"] / n / 60,
                  "total_min": (s["wall_train_s"] + s["wall_val_s"]) / n / 60}
    return out


def wall_time(runs_root: Path, runs_dir: Path) -> tuple[str, float | None]:
    """Design-selection wall time: runner-log span of the DS jobs (first start -> report start / last end),
    plus the sum of the 4 trainings' total_time_s."""
    parts, total = [], None
    log = runs_root / "_runner.log"
    if log.exists():
        starts, ends, rep = [], [], []
        for line in log.read_text().splitlines():
            f = line.split()
            if len(f) >= 4 and f[1] == "job" and f[2] == "start" and f[3] in DS_JOBS:
                (rep if f[3] == REPORT_JOB else starts).append(pr._ts(f[0]))
            elif len(f) >= 7 and f[1] == "job" and f[2] in DS_JOBS and f[-2] == "exit":
                ends.append(pr._ts(f[0]))
        if starts:
            t1 = max(rep) if rep else (max(ends) if ends else None)
            if t1:
                total = (t1 - min(starts)).total_seconds()
                parts.append(f"runner log span (first design-selection job start to report start): {total / 3600:.2f} h")
    tr = sum(float(json.loads((runs_dir / pr.run_id(m) / "done.json").read_text()).get("total_time_s", 0.0))
             for m in NEW if (runs_dir / pr.run_id(m) / "done.json").exists())
    if tr:
        parts.append(f"sum of the training total_time_s of the new runs (done.json): {tr / 3600:.2f} h")
        total = total if total is not None else tr
    return ("; ".join(parts) or "n/a (no runner log / done.json found)"), total


def build(eval_dir: Path, runs_dir: Path, runs_root: Path, params: dict | None, overhead: dict | None,
          n_boot: int = 1000) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    a = analyse(eval_dir, n_boot)
    res, conds, deg, present = a["res"], a["conds"], a["deg"], a["present"]
    et = epoch_times(runs_dir)
    wt_txt, wall = wall_time(runs_root, runs_dir)
    rrows, flagged, why = pr.resource_section(runs_dir, [m for m in MODELS[1:] if m in present or (runs_dir / pr.run_id(m)).exists()])
    dcols: list[str | None] = [*deg, None, *(["c40"] if "c40" in conds else [])]
    focus_cols: list[str | None] = [c for c in FOCUS if c in conds]
    L = ["# Design selection study (5 epochs, validation)", "",
         "**PILOT / appendix material. VALIDATION split only; not paper main numbers; no design is selected here "
         "(the choice is the user's).** Xception detector, seed 42, batch 32, same TrainDegradation, 5 epochs of the "
         "real 30-epoch LR schedule. Models: base = frozen detector; fixed / dcfg = front-end + frozen detector; augft = "
         "detector fine-tuned; jointft = dcfg module + detector fine-tuned (lr 1e-3 / 1e-4 / 2e-5); dcfgcons = dcfg + "
         "consistency loss (frozen detector); dcfgfilm = dcfgcons + FiLM at entry/block7/block11/exit; dcfgfilmlite = "
         "FiLM at entry/exit only (small MLP).", "",
         f"{res['n_videos']} val videos, {a['n_frames']} frames per condition; {n_boot} video-level bootstrap replicates "
         "(95% percentile CI, one resample shared by all models/conditions, so deltas are paired).", ""]
    if a["missing"]:
        L += [f"**MISSING runs (no eval scores): {', '.join(a['missing'])}.** Their rows are marked and excluded from comparisons.", ""]
    L += ["## 1. Cost: trainable parameters and inference overhead", "", *cost_table(params, overhead), "",
          "Trainable params ADDED to the frozen Xception (sensor counted as trainable: it is fine-tuned at lr 1e-4); for "
          "augft/jointft the whole detector counts as trainable. Overhead: inference, fp32 detector, no_grad, CUDA events, "
          "median; 'added' = pipeline minus the plain detector (front-end, FiLM MLP and hooks included). Training cost of "
          "dcfgcons/dcfgfilm/dcfgfilmlite additionally includes one extra no-grad detector forward per step (the teacher).", "",
          "## 2. Frame AUC [95% CI] (validation)", "", *accuracy_table(res, present, conds, deg), "",
          f"Degraded mean = mean over {', '.join(deg)}; c40 = real H.264 compression (own video set), not in the mean. "
          "'d clean vs base' is a paired delta; * = CI excludes 0.", "",
          "## 3. Paired bootstrap delta-AUC of each new variant vs dcfg, augft, jointft; `*` = CI excludes 0", "",
          *delta_rows(res, present, dcols), "",
          "## 4. Highlighted: blur_s3, webp_q20, c40", ""]
    focus = [c for c in FOCUS if c in conds]
    hrows = ["| model | " + " | ".join(focus) + " |", "|" + "---|" * (len(focus) + 1)]
    for m in MODELS:
        hrows.append(f"| **{m}** | " + " | ".join(
            (pr.fmt_ci(res['point'][(m, c)], *pr.pct(res['boot'][(m, c)])) if (m, c) in res["point"] else "n/a") for c in focus) + " |")
    L += [*hrows, "", *delta_rows(res, present, focus_cols), "",
          "## 5. Resource summary per run (training phase only)", "", *rrows, "",
          f"Flag rule: CPU-bound if mean GPU util < {pr.GPU_UTIL_MIN:.0f}% or data-wait share > {100 * pr.WAIT_SHARE_MAX:.0f}%. "
          + (f"**FLAGGED: {'; '.join(why)}.**" if flagged else "No run flagged."), "",
          "## 6. Measured epoch times (for the matrix ETA)", "",
          "| run | train min/epoch | val min/epoch | total min/epoch | epochs |", "|---|---|---|---|---|"]
    for m in MODELS[1:]:
        e = et[m]
        L.append(f"| {pr.run_id(m)} | " + (" | ".join([f"{e['train_min']:.1f}", f"{e['val_min']:.1f}", f"{e['total_min']:.1f}",
                                                       str(e["epochs"])]) if e else "MISSING | MISSING | MISSING | MISSING") + " |")
    L += ["", "From resources_summary.json (wall_train_s / wall_val_s) divided by epochs_run.", "",
          "## 7. Total wall time", "", wt_txt, ""]

    # ----- machine-readable
    pts = {f"{m}|{c}": {"auc": res["point"][(m, c)], "ci": list(pr.pct(res["boot"][(m, c)]))}
           for m in present for c in conds if (m, c) in res["point"]}
    degm = {m: {"auc": res["deg_point"][m], "ci": list(pr.pct(res["deg_boot"][m]))} for m in present if m in res["deg_point"]}
    dl: dict[str, Any] = {}
    for m in NEW:
        for b in BASELINES:
            if m == b or m not in present or b not in present:
                continue
            for c in [*deg, None, *(["c40"] if "c40" in conds else [])]:
                d = pr.paired_delta(res, m, b, c)
                if d:
                    dl[f"{m}-{b}|{'mean_degraded' if c is None else c}"] = {"delta": d[0], "ci": [d[1], d[2]],
                                                                          "excludes_0": bool(d[1] > 0 or d[2] < 0)}
    summary = {"missing": a["missing"], "conditions": conds, "n_videos": res["n_videos"], "n_boot": n_boot,
               "auc": pts, "deg_mean": degm, "deltas": dl, "params": params, "overhead_file_used": bool(overhead),
               "epoch_times_min": et, "wall_s": wall, "cpu_bound": flagged, "cpu_bound_why": why,
               "note": "design-selection pilot, validation split, not paper numbers; no design selected"}
    csv_rows = []
    for m in MODELS:
        p = (params or {}).get(m) or {}
        pipe = ((overhead or {}).get("pipelines") or {}).get(OVERHEAD_OF[m]) or {}
        row = {"model": m, "present": m in present, "trainable_params": p.get("trainable"),
               "pct_of_xception": p.get("pct_of_xception"), "detector_trained": p.get("detector_trained"),
               "ms_img_b32": (pipe.get("batch32") or {}).get("ms_per_image"),
               "added_ms_img_b32": (pipe.get("batch32") or {}).get("added_ms_per_image"),
               "ms_img_b1": (pipe.get("batch1") or {}).get("ms_per_image"),
               "added_ms_img_b1": (pipe.get("batch1") or {}).get("added_ms_per_image")}
        for c in conds:
            k = (m, c)
            row[f"auc_{c}"], row[f"auc_{c}_lo"], row[f"auc_{c}_hi"] = (
                (res["point"][k], *pr.pct(res["boot"][k])) if k in res["point"] else (None, None, None))
        row["auc_mean_degraded"], row["auc_mean_degraded_lo"], row["auc_mean_degraded_hi"] = (
            (res["deg_point"][m], *pr.pct(res["deg_boot"][m])) if m in res["deg_point"] else (None, None, None))
        e = et.get(m)
        row["epoch_train_min"], row["epoch_val_min"] = (e["train_min"], e["val_min"]) if e else (None, None)
        csv_rows.append(row)
    summary["_res"] = res  # for notify text (removed before json dump)
    return "\n".join(L) + "\n", summary, csv_rows


def notify_text(s: dict[str, Any]) -> str:
    deg = s["deg_mean"]
    lines = ["design-selection pilots done",
             "Degraded-mean val frame AUC: " + ", ".join(f"{m} {v['auc']:.3f}" for m, v in deg.items())]
    cand = {m: v for m, v in deg.items() if m in NEW}
    if cand and "augft" in deg:
        best = max(cand, key=lambda m: cand[m]["auc"])
        d = s["deltas"].get(f"{best}-augft|mean_degraded")
        if d:
            lines.append(f"Highest degraded mean among new variants: {best}; vs augft {d['delta']:+.4f} "
                         f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]{' *' if d['excludes_0'] else ''}")
    if s["missing"]:
        lines.append("MISSING runs: " + ", ".join(s["missing"]))
    wall = f"{s['wall_s'] / 3600:.1f} h" if s["wall_s"] else "n/a"
    lines.append(f"Wall time {wall}; report results/pilot/design_selection/report.md (no design selected)")
    return "\n".join(lines[:6])


def notify(text: str) -> None:
    subprocess.run(["bash", str(REPO / "ops" / "notify.sh"), text, "DCFG design selection"], check=False, timeout=60)


def check_out_dir(out_dir: Path) -> None:
    for f in FORBIDDEN:
        if out_dir.resolve().is_relative_to((REPO / "results" / f).resolve()):
            raise SystemExit(f"refusing to write design-selection output into results/{f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", default=str(REPO / "results/pilot/eval"))
    ap.add_argument("--runs-dir", default=str(REPO / "runs/pilot"))
    ap.add_argument("--runs-root", default=str(REPO / "runs"))
    ap.add_argument("--out-dir", default=str(REPO / "results/pilot/design_selection"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-overhead-run", action="store_true", help="do not measure overhead.json if missing")
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    ov_path = out / "overhead.json"
    if not ov_path.exists() and not a.no_overhead_run:
        subprocess.run([sys.executable, "-m", "dcfg.analysis.overhead", "--design-selection", "--out", str(ov_path)],
                       check=False, cwd=REPO, timeout=900)
    overhead = json.loads(ov_path.read_text()) if ov_path.exists() else None
    try:
        from dcfg.analysis.overhead import trainable_params_by_model
        params = trainable_params_by_model()
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: parameter counts unavailable: {e}", file=sys.stderr)
        params = None
    md, s, rows = build(Path(a.eval_dir), Path(a.runs_dir), Path(a.runs_root), params, overhead, a.n_boot)
    s.pop("_res")
    (out / "report.md").write_text(md)
    (out / "results.json").write_text(json.dumps(s, indent=2, default=float) + "\n")
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = notify_text(s)
    print(txt)
    if not a.no_notify:
        notify(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
